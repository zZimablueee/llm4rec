# python /root/autodl-tmp/qwenrec/gr/train_gr.py --config /root/autodl-tmp/qwenrec/gr/gr_train.json


import json
import os
import re
import math
import argparse
import random
import torch
import numpy as np
from collections import defaultdict
from transformers import (
    HfArgumentParser, TrainingArguments, Trainer,
    TrainerCallback, TrainerState, TrainerControl
)
from transformers import default_data_collator
from arguments import ModelArguments, DataTrainingArguments
from utils import load_model_and_tokenizer, load_item2token_dict, BASE_CACHE
from custom_dataset import CustomTrainDataset
from transformers import Trainer
from torch.optim import AdamW

# ── 兼容 PyTorch 2.6 rng_state 加载 ──────────────────────
import torch
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load
# ─────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# 中途评估 Callback
# ══════════════════════════════════════════════════════════════
class RecommendEvalCallback(TrainerCallback):
    """
    每隔 eval_every_steps 步，对 eval_users 做一次真实推荐评估。
    只取前 num_eval_users 个用户，速度快（约2~5分钟）。
    """
    def __init__(
        self,
        eval_data,           # dict: {user_id: item_seq}
        item2token_dict,
        token2item_dict,
        tokenizer,
        eval_every_steps=500,
        num_eval_users=200,
        beam_size=20,
        top_k=10,
    ):
        self.eval_data        = eval_data
        self.item2token_dict  = item2token_dict
        self.token2item_dict  = token2item_dict
        self.tokenizer        = tokenizer
        self.eval_every_steps = eval_every_steps
        self.num_eval_users   = num_eval_users
        self.beam_size        = beam_size
        self.top_k            = top_k

        # 构建 Trie
        vocab = tokenizer.get_vocab()
        self.trie = {}
        for item_id, token_str in item2token_dict.items():
            tokens = re.findall(r'<[a-c]_\d+>', token_str)
            if len(tokens) != 3:
                continue
            a = vocab.get(tokens[0])
            b = vocab.get(tokens[1])
            c = vocab.get(tokens[2])
            if None in (a, b, c):
                continue
            self.trie.setdefault(a, {}).setdefault(b, set()).add(c)
        self.valid_a_ids = list(self.trie.keys())
        self.eos_id      = tokenizer.eos_token_id
        print(f"[EvalCallback] Trie built: {len(self.valid_a_ids)} a-nodes")

        # 预先构造好测试样本，避免每次重复计算
        self.test_samples = []
        for user_id, item_seq in list(eval_data.items())[:num_eval_users * 3]:
            target_item, target_idx = None, -1
            for i in range(len(item_seq) - 1, -1, -1):
                if str(item_seq[i]) in item2token_dict:
                    target_item = str(item_seq[i])
                    target_idx  = i
                    break
            if target_item is None or target_idx == 0:
                continue
            history = [
                item2token_dict[str(x)]
                for x in item_seq[:target_idx]
                if str(x) in item2token_dict
            ][-50:]
            if not history:
                continue
            self.test_samples.append((user_id, history, target_item))
            if len(self.test_samples) >= num_eval_users:
                break
        print(f"[EvalCallback] 准备好 {len(self.test_samples)} 个评估用户")

    def _prefix_fn(self, input_len):
        trie    = self.trie
        eos_id  = self.eos_id
        valid_a = self.valid_a_ids

        def fn(batch_id, input_ids):
            gen     = input_ids[input_len:]
            gen_len = len(gen)
            if gen_len == 0:
                return valid_a
            elif gen_len == 1:
                return list(trie.get(gen[0].item(), {}).keys())
            elif gen_len == 2:
                return list(trie.get(gen[0].item(), {}).get(gen[1].item(), set()))
            else:
                return [eos_id]
        return fn

    @torch.no_grad()
    def _run_eval(self, model):
        model.eval()
        device = next(model.parameters()).device

        # 推理时用 left padding
        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        hit1 = hit5 = hit10 = 0
        ndcg5 = ndcg10 = 0.0
        n = len(self.test_samples)

        # 逐条推理，彻底杜绝 batch padding 错位
        for user_id, history, gt in self.test_samples:
            prompt = '<|hist_clk_start|>' + ''.join(history) + '<|hist_clk_end|>'
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            ).to(device)

            base_len = inputs["input_ids"].shape[1]   # 真实长度

            trie   = self.trie
            eos_id = self.eos_id
            valid_a = self.valid_a_ids

            def prefix_fn(batch_id, input_ids, _len=base_len):
                gen = input_ids[_len:]
                gen_len = len(gen)
                if gen_len == 0:
                    return valid_a
                elif gen_len == 1:
                    return list(trie.get(gen[0].item(), {}).keys())
                elif gen_len == 2:
                    return list(trie.get(gen[0].item(), {}).get(gen[1].item(), set()))
                else:
                    return [eos_id]

            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                num_beams=self.beam_size,
                num_return_sequences=self.beam_size,
                prefix_allowed_tokens_fn=prefix_fn,
                early_stopping=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            pred_items = []
            for k in range(self.beam_size):
                gen_ids = outputs[k][base_len:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                m = re.search(r'<a_\d+><b_\d+><c_\d+>', gen_text)
                if m:
                    item = self.token2item_dict.get(m.group(0))
                    if item and item not in pred_items:
                        pred_items.append(item)
                if len(pred_items) >= self.top_k:
                    break

            if gt in pred_items[:1]:
                hit1 += 1
            if gt in pred_items[:5]:
                hit5 += 1
                rank = pred_items[:5].index(gt) + 1
                ndcg5 += 1 / math.log2(rank + 1)
            if gt in pred_items[:10]:
                hit10 += 1
                rank = pred_items[:10].index(gt) + 1
                ndcg10 += 1 / math.log2(rank + 1)

        self.tokenizer.padding_side = orig_side
        model.train()

        return {
            "Hit@1":   hit1  / n,
            "Hit@5":   hit5  / n,
            "Hit@10":  hit10 / n,
            "NDCG@5":  ndcg5 / n,
            "NDCG@10": ndcg10 / n,
        }

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_every_steps != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return

        print(f"\n{'='*55}")
        print(f"[推荐评估 @ step {state.global_step}]  ({len(self.test_samples)} 用户)")
        metrics = self._run_eval(model)
        for k, v in metrics.items():
            print(f"  {k:8s} = {v:.4f}")
        print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════
# 原有逻辑（不动）
# ══════════════════════════════════════════════════════════════
def parse_args_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config_dict = {}
    for _, value_dict in config.items():
        for k, v in value_dict.items():
            config_dict[k] = v

    known_training_keys = {
        "output_dir", "max_steps", "num_train_epochs",
        "per_device_train_batch_size", "save_steps", "logging_steps",
        "fp16", "bf16",
        "dataloader_pin_memory", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "dataloader_prefetch_factor",
        "dataloader_num_workers", "learning_rate", "weight_decay",
        "save_strategy", "save_total_limit", "logging_strategy",
        "eval_steps", "eval_strategy",
        "resume_from_checkpoint",
        "warmup_ratio", "lr_scheduler_type", "max_grad_norm",
    }
    known_model_keys = {
        "from_pretrained", "from_checkpoint", "pretrained_path",
        "checkpoint_path", "tokenizer_path", "se_id_space_width",
    }
    known_data_keys = {
        "train_file", "test_file", "max_seq_length", "max_imp_seq_length",
        "max_click_seq_length", "token_depth", "padding_side",
        "item2token_dict", "token2item_dict", "token2ads_dict",
        "response_flag",
    }
    known_keys = known_training_keys | known_model_keys | known_data_keys
    filtered_dict = {k: v for k, v in config_dict.items() if k in known_keys}

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(filtered_dict)
    return model_args, data_args, training_args


def main(model_args, data_args, training_args):
    model, tokenizer = load_model_and_tokenizer(model_args)
    print("pad_token_id:", tokenizer.pad_token_id)

    item2token_dict = load_item2token_dict(
        os.path.join(BASE_CACHE, data_args.item2token_dict)
    )
    token2item_dict = {v: k for k, v in item2token_dict.items()}

    # 修正 output_dir
    output_dir = os.path.join(BASE_CACHE, "checkpoints", training_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir

    # 续训
    resume_ckpt = None
    if training_args.resume_from_checkpoint:
        ckpt_dirs = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if ckpt_dirs:
            latest     = max(ckpt_dirs, key=lambda x: int(x.split("-")[-1]))
            resume_ckpt = os.path.join(output_dir, latest)
            print(f"[续训] 从 checkpoint 恢复: {resume_ckpt}")
        else:
            print("[续训] 未找到任何 checkpoint，从头开始训练")

    # 构造数据集
    with open(os.path.join(BASE_CACHE, data_args.train_file)) as f:
        all_data = json.load(f)

    all_users = list(all_data.keys())
    random.seed(42)
    val_users      = set(random.sample(all_users, len(all_users) // 100))
    train_users    = {u: all_data[u] for u in all_users if u not in val_users}
    val_users_data = {u: all_data[u] for u in val_users}
    print(f"训练用户: {len(train_users)}, 验证用户: {len(val_users_data)}")

    train_tmp = os.path.join(BASE_CACHE, "_train_split.json")
    val_tmp   = os.path.join(BASE_CACHE, "_val_split.json")
    with open(train_tmp, 'w') as f:
        json.dump(train_users, f)
    with open(val_tmp, 'w') as f:
        json.dump(val_users_data, f)

    train_dataset = CustomTrainDataset(
        json_path=train_tmp,
        item2token_dict=item2token_dict,
        tokenizer=tokenizer,
        data_args=data_args
    )
    eval_dataset = CustomTrainDataset(
        json_path=val_tmp,
        item2token_dict=item2token_dict,
        tokenizer=tokenizer,
        data_args=data_args
    )

    # ── 构造评估 Callback ──────────────────────────────────
    rec_eval_callback = RecommendEvalCallback(
        eval_data       = val_users_data,
        item2token_dict = item2token_dict,
        token2item_dict = token2item_dict,
        tokenizer       = tokenizer,
        eval_every_steps= 500,   # ← 每500步评估一次，可按需调整
        num_eval_users  = 200,   # ← 只用200个用户，约3分钟
        beam_size       = 20,
        top_k           = 10,
    )

    training_args.report_to           = []
    training_args.dataloader_pin_memory = False
    # 在 trainer = Trainer(...) 之前加入


    class CustomTrainer(Trainer):
        def create_optimizer(self):
            opt_model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    # 新嵌入层和输出层，用更高学习率
                    "params": [p for n, p in opt_model.named_parameters() 
                            if ("embed_tokens" in n or "lm_head" in n) and p.requires_grad],
                    "lr": self.args.learning_rate * 5,  # 5倍于基座学习率，即 1.5e-4
                },
                {
                    # 基座模型其余部分，用低学习率
                    "params": [p for n, p in opt_model.named_parameters() 
                            if "embed_tokens" not in n and "lm_head" not in n and p.requires_grad],
                    "lr": self.args.learning_rate,  # 3e-5
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
    trainer = CustomTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = default_data_collator,
        callbacks     = [rec_eval_callback],   # ← 挂上去
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    model_args, data_args, training_args = parse_args_from_json(args.config)
    main(model_args, data_args, training_args)

    # cd /root/autodl-tmp/qwenrec/gr && python train_gr.py --config /root/autodl-tmp/qwenrec/gr/gr_train.json

