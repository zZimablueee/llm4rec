import os
import re
import json
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig
from tqdm import tqdm
import math

BASE_CACHE      = "/root/autodl-tmp/qwenrec/cache"
QWEN_PATH       = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_BASE = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2"

ITEM2TOKEN_DIR  = os.path.join(BASE_CACHE, "emb_infer/sinkhorn10")
TRAIN_DATA      = os.path.join(BASE_CACHE, "train_data.json")
OUTPUT_FILE     = os.path.join(BASE_CACHE, "inference_result.json")
CHECKPOINT_FILE = os.path.join(BASE_CACHE, "inference_checkpoint.json")  # 断点存档

EVAL_EVERY      = 500
SAVE_EVERY      = 500  # 每1000个用户存一次断点
TOP_K           = 10
BEAM_SIZE       = 30
NUM_RETURN      = 30
MAX_NEW_TOKENS  = 4
BATCH_SIZE      = 16
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. 自动找最大步数的checkpoint ────────────────────────────
ckpt_dirs = [
    d for d in os.listdir(CKPT_BASE)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(CKPT_BASE, d))
]
latest_ckpt = os.path.join(
    CKPT_BASE,
    sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))[-1]
)
print(f"Using checkpoint: {latest_ckpt}")

# ── 2. tokenizer & model ──────────────────────────────────────
print("Loading tokenizer & model...")
# tokenizer = AutoTokenizer.from_pretrained(CKPT_BASE, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/qwenrec/qwen_init2", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(latest_ckpt,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    CKPT_BASE,     # ← 这一层才有 tokenizer.json
    trust_remote_code=True
)

tokenizer.padding_side = 'left'

config = AutoConfig.from_pretrained(QWEN_PATH)
model = Qwen2ForCausalLM(config)

ckpt_bin  = os.path.join(latest_ckpt, "pytorch_model.bin")
ckpt_safe = os.path.join(latest_ckpt, "model.safetensors")
if os.path.exists(ckpt_bin):
    state_dict = torch.load(ckpt_bin, map_location="cpu", weights_only=True)
elif os.path.exists(ckpt_safe):
    from safetensors.torch import load_file
    state_dict = load_file(ckpt_safe)
else:
    raise FileNotFoundError(f"找不到权重文件: {latest_ckpt}")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[Checkpoint] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print("  MISSING (前5个):", missing[:5])
if unexpected:
    print("  UNEXPECTED (前5个):", unexpected[:5])
model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── 3. item2token_dict ────────────────────────────────────────
print("Loading item2token_dict...")
item2token_dict = {}
for fname in os.listdir(ITEM2TOKEN_DIR):
    fpath = os.path.join(ITEM2TOKEN_DIR, fname)
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            item_id, token = parts
            item2token_dict[item_id] = token

token2item_dict = {v: k for k, v in item2token_dict.items()}
print(f"item2token size: {len(item2token_dict)}")

# ── 4. 留一法构造测试集 ───────────────────────────────────────
print("Building test samples...")
with open(TRAIN_DATA) as f:
    train_data = json.load(f)

test_samples = []
for user_id, item_seq in train_data.items():
    if len(item_seq) < 2:
        continue
    target_item, target_idx = None, -1
    for i in range(len(item_seq) - 1, -1, -1):
        if str(item_seq[i]) in item2token_dict:
            target_item = str(item_seq[i])
            target_idx = i
            break
    if target_item is None:
        continue
    history_tokens = [
        item2token_dict[str(item)]
        for item in item_seq[:target_idx]
        if str(item) in item2token_dict and str(item) != target_item
    ]
    history_tokens = history_tokens[-50:]  # ← 加这行，与训练对齐

    if not history_tokens:
        continue
    test_samples.append((user_id, history_tokens, target_item))

print(f"有效测试用户数: {len(test_samples)}")

# ── 5. 加载断点（如果有）────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    print(f"发现断点存档，从上次进度继续...")
    with open(CHECKPOINT_FILE, 'r') as f:
        results = json.load(f)
    finished_users = set(results.keys())
    test_samples = [(uid, h, gt) for uid, h, gt in test_samples if uid not in finished_users]
    print(f"已完成: {len(finished_users)} 用户，剩余: {len(test_samples)} 用户")
else:
    results = {}
    print("无断点存档，从头开始。")

# ── 6. 评估函数 ───────────────────────────────────────────────
def evaluate_partial(results):
    hit1, hit5, hit10 = 0, 0, 0
    ndcg5, ndcg10 = 0.0, 0.0
    n = len(results)
    if n == 0:
        print("[Eval] 暂无数据")
        return
    for uid, val in results.items():
        gt   = val["gt"]
        pred = val["pred"]
        if gt in pred[:1]:
            hit1 += 1
        if gt in pred[:5]:
            hit5 += 1
            rank  = pred[:5].index(gt) + 1        # rank 从1开始
            ndcg5 += 1 / math.log2(rank + 1)
        if gt in pred[:10]:
            hit10 += 1
            rank   = pred[:10].index(gt) + 1
            ndcg10 += 1 / math.log2(rank + 1)
    print(f"\n[Partial Eval @ {n} users]")
    print(f"  Hit@1={hit1/n:.4f}  Hit@5={hit5/n:.4f}  Hit@10={hit10/n:.4f}")
    print(f"  NDCG@5={ndcg5/n:.4f}  NDCG@10={ndcg10/n:.4f}\n")


# ── 7. 推理主循环 ─────────────────────────────────────────────
# ── 在第7步推理循环之前，加这段初始化 ──────────────────────
print("Building prefix token id lists...")
vocab = tokenizer.get_vocab()
a_ids = [vocab[f"<a_{i}>"] for i in range(2048) if f"<a_{i}>" in vocab]
b_ids = [vocab[f"<b_{i}>"] for i in range(2048) if f"<b_{i}>" in vocab]
c_ids = [vocab[f"<c_{i}>"] for i in range(1024) if f"<c_{i}>" in vocab]
eos_id = tokenizer.eos_token_id
print(f"Valid token counts: a={len(a_ids)}, b={len(b_ids)}, c={len(c_ids)}")
# 预期输出: a=2048, b=2048, c=1024


def build_prompt(history_tokens):
    return '<|hist_clk_start|>' + ''.join(history_tokens) + '<|hist_clk_end|>'

print("Running inference...")
processed = 0

for i in tqdm(range(0, len(test_samples), BATCH_SIZE)):
    batch = test_samples[i: i + BATCH_SIZE]
    prompts = [build_prompt(h) for _, h, _ in batch]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)
    current_input_len = inputs["input_ids"].shape[1]

    #real_input_lens = inputs["attention_mask"].sum(dim=1).tolist()  # shape: [batch_size]
    #ef prefix_fn(batch_id, input_ids, _real_lens=real_input_lens):
     #   real_len = _real_lens[batch_id // NUM_RETURN]  # beam search时batch_id会扩展
      #  gen_len = len(input_ids) - real_len
       # if gen_len <= 0: return a_ids
        #if gen_len == 1: return b_ids
        #if gen_len == 2: return c_ids
        #return [eos_id]
    def prefix_fn(batch_id, input_ids, _len=current_input_len):   # ← 加 _len 默认参数
        gen_len = len(input_ids) - _len
        if gen_len == 0: return a_ids
        if gen_len == 1: return b_ids
        if gen_len == 2: return c_ids
        return [eos_id]

    with torch.no_grad():
        outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,                      # 3个SID token + 1个EOS余量
                    num_beams=BEAM_SIZE,                   # 30
                    num_return_sequences=NUM_RETURN,       # 30
                    prefix_allowed_tokens_fn=prefix_fn,
                    early_stopping=False,                  # False 更稳，避免版本兼容问题
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    )


    input_len = inputs["input_ids"].shape[1]
    for j, (user_id, _, gt_item) in enumerate(batch):
        pred_items = []
        for k in range(NUM_RETURN):                        # ← 遍历全部30个
            gen_ids = outputs[j * NUM_RETURN + k][input_len:]   # ← 注意乘NUM_RETURN
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            match = re.search(r'<a_\d+><b_\d+><c_\d+>', gen_text)
            if match:
                pred_item = token2item_dict.get(match.group(0))
                if pred_item and pred_item not in pred_items:
                    pred_items.append(pred_item)
            if len(pred_items) >= TOP_K:                   # ← 凑够10个立即停
                break
        results[user_id] = {"pred": pred_items[:TOP_K], "gt": gt_item}


    processed += len(batch)

    # 每1000个用户存一次断点 + 打印评估
    if processed % SAVE_EVERY < BATCH_SIZE:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(results, f)
        evaluate_partial(results)
        print(f"  [断点已保存 @ {len(results)} users]")

# ── 8. 全部完成，保存最终结果 ─────────────────────────────────
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

# 删掉断点文件，避免下次误读
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"推理完成，结果保存至: {OUTPUT_FILE}")
print(f"总用户: {len(results)}")
evaluate_partial(results)
