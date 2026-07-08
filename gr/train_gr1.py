import json
import os
import argparse
from transformers import (
    HfArgumentParser, TrainingArguments, Trainer
)
from transformers import default_data_collator
from arguments import ModelArguments, DataTrainingArguments
from utils import load_model_and_tokenizer, load_item2token_dict, BASE_CACHE
from custom_dataset import CustomTrainDataset
import random


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
    "fp16", "bf16",                                    # ← 加 bf16
    "dataloader_pin_memory", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "dataloader_prefetch_factor",
    "dataloader_num_workers", "learning_rate", "weight_decay",
    "save_strategy", "save_total_limit", "logging_strategy",
    "eval_steps", "eval_strategy",                     # ← 加 eval_strategy
    "resume_from_checkpoint",
    "warmup_ratio", "lr_scheduler_type", "max_grad_norm",  # ← 加这三个
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

    item2token_dict = load_item2token_dict(
        os.path.join(BASE_CACHE, data_args.item2token_dict)
    )

    # 修正 output_dir 为绝对路径
    output_dir = os.path.join(BASE_CACHE, "checkpoints", training_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = output_dir

    # ✅ 自动找最新 checkpoint 用于续训
    resume_ckpt = None
    if training_args.resume_from_checkpoint:
        # 找 output_dir 下 step 数最大的 checkpoint
        ckpt_dirs = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if ckpt_dirs:
            latest = max(ckpt_dirs, key=lambda x: int(x.split("-")[-1]))
            resume_ckpt = os.path.join(output_dir, latest)
            print(f"[续训] 从 checkpoint 恢复: {resume_ckpt}")
        else:
            print("[续训] 未找到任何 checkpoint，从头开始训练")

    # ── 构造训练集 ──────────────────────────────────────────
    with open(os.path.join(BASE_CACHE, data_args.train_file)) as f:
        all_data = json.load(f)

    all_users = list(all_data.keys())
    random.seed(42)
    val_users = set(random.sample(all_users, len(all_users) // 100))  # 1% 做验证
    train_users = {u: all_data[u] for u in all_users if u not in val_users}
    val_users_data = {u: all_data[u] for u in val_users}

    print(f"训练用户: {len(train_users)}, 验证用户: {len(val_users_data)}")
    # 写成临时文件给 CustomTrainDataset 用（保持接口不变）
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

    data_collator = default_data_collator

    training_args.report_to = []
    training_args.dataloader_pin_memory = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,          # ← 加这行
        data_collator=data_collator,
    )


    # ✅ 关键修复：把 resume_ckpt 传给 trainer.train()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    model_args, data_args, training_args = parse_args_from_json(args.config)
    main(model_args, data_args, training_args)
