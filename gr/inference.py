#!/usr/bin/env python3
# cd /root/autodl-tmp/qwenrec/gr && python inference.py
# cd /root/autodl-tmp/qwenrec/gr && python -u inference.py 2>&1 | tee infer_log.txt

"""
推理脚本 v2：
- 支持断点续跑
- 完整保存原始 pred 列表（最多 TOP_K_SAVE 个），方便事后补算任意指标
- 评估 Hit/Recall @10/20/30 + nDCG @10/20/30 + 多样性
- 参考值对标 R³-VAE 论文 RQ-VAE baseline
"""
import os, re, json, math, torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, Qwen2ForCausalLM, AutoConfig
from safetensors.torch import load_file
from tqdm import tqdm

# ==================== 配置 ====================
BASE_CACHE      = "/root/autodl-tmp/qwenrec/cache"
QWEN_PATH       = "/root/autodl-tmp/qwenrec/qwen_init2"
CKPT_BASE       = "/root/autodl-tmp/qwenrec/cache/checkpoints/global_step_ep10_v2_seq100"
ITEM2TOKEN_DIR  = os.path.join(BASE_CACHE, "emb_infer/sinkhorn10")
TRAIN_DATA      = os.path.join(BASE_CACHE, "train_data.json")

OUTPUT_FILE     = os.path.join(BASE_CACHE, "inference_result.json")      # 最终结果
CHECKPOINT_FILE = os.path.join(BASE_CACHE, "inference_checkpoint.json")  # 断点

# 推理参数
BEAM_SIZE       = 50    # beam 数量，决定候选上限
NUM_RETURN      = 50    # 返回序列数，必须 <= BEAM_SIZE
TOP_K_SAVE      = 30    # 每个用户最多保存多少个 pred（决定事后能算到 @几）
MAX_NEW_TOKENS  = 4
MAX_USERS       = None  # None = 全量；设置整数如 10000 = 小批量验证

EVAL_EVERY      = 1000 # 每处理多少用户打印一次中间指标
SAVE_EVERY      = 500   # 每处理多少用户保存一次断点

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# 论文 RQ-VAE 参考值（R³-VAE Table 1）
PAPER_RQVAE = {
    "Beauty":  {"R@10": 0.059, "R@20": 0.091, "R@30": 0.111, "N@10": 0.033, "N@20": 0.041, "N@30": 0.047},
    "Sports":  {"R@10": 0.031, "R@20": 0.048, "R@30": 0.062, "N@10": 0.017, "N@20": 0.021, "N@30": 0.024},
    "Toys":    {"R@10": 0.051, "R@20": 0.074, "R@30": 0.089, "N@10": 0.028, "N@20": 0.035, "N@30": 0.038},
}

# ==================== 工具函数 ====================
def ndcg_at_k(hit_position, k):
    if hit_position is None or hit_position > k:
        return 0.0
    return 1.0 / math.log2(hit_position + 1)

def evaluate_and_print(results, label=""):
    """
    从保存的 results 字典计算完整指标。
    results 格式：{user_id: {"pred": [...最多30个...], "gt": "item_id"}}
    """
    K_LIST = [10, 20, 30]
    hits  = {k: 0   for k in K_LIST}
    ndcgs = {k: 0.0 for k in K_LIST}
    all_top1       = []
    all_preds_flat = []
    n = len(results)

    if n == 0:
        print("  [Eval] 暂无数据")
        return

    for uid, val in results.items():
        gt   = val["gt"]
        pred = val["pred"]   # 已经是字符串列表，最多 TOP_K_SAVE 个

        all_top1.append(pred[0] if pred else "__empty__")
        all_preds_flat.extend(pred[:30])

        # 找命中位置（1-based）
        hit_pos = None
        for rank, p in enumerate(pred, start=1):
            if p == gt:
                hit_pos = rank
                break

        for k in K_LIST:
            if hit_pos is not None and hit_pos <= k:
                hits[k]  += 1
                ndcgs[k] += ndcg_at_k(hit_pos, k)

    tag = f"[{label}] " if label else ""
    tqdm.write(f"\n  {tag}评估用户数: {n}  |  数据集: Electronics")
    tqdm.write(f"\n  {'':4} {'Recall@10':>10} {'Recall@20':>10} {'Recall@30':>10} "
          f"{'NDCG@10':>10} {'NDCG@20':>10} {'NDCG@30':>10}")
    tqdm.write(f"  {'-'*66}")
    tqdm.write(f"  {'你的模型':<4} "
          f"{hits[10]/n:>10.4f} {hits[20]/n:>10.4f} {hits[30]/n:>10.4f} "
          f"{ndcgs[10]/n:>10.4f} {ndcgs[20]/n:>10.4f} {ndcgs[30]/n:>10.4f}")
    tqdm.write(f"  {'-'*66}")

    for ds, vals in PAPER_RQVAE.items():
        tqdm.write(f"  {'RQ/'+ds:<14} "
              f"{vals['R@10']:>10.4f} {vals['R@20']:>10.4f} {vals['R@30']:>10.4f} "
              f"{vals['N@10']:>10.4f} {vals['N@20']:>10.4f} {vals['N@30']:>10.4f}")
    tqdm.write(f"  {'-'*66}")
    tqdm.write(f"  ⚠️  论文数据集规模 < 36K 用户，Electronics = 728K，数值仅供参考")

    # 多样性
    unique_top1 = len(set(all_top1))
    diversity   = unique_top1 / n
    tqdm.write(f"\n  Top1 多样性 : {unique_top1}/{n} = {diversity:.2%}")
    if diversity < 0.3:
        tqdm.write("  ❌ 多样性低，热门坍缩明显")
    elif diversity < 0.6:
        tqdm.write("  🟡 多样性尚可")
    else:
        tqdm.write("  ✅ 多样性良好")

    tqdm.write(f"\n  最常推荐的 10 个物品:")
    for item, cnt in Counter(all_preds_flat).most_common(10):
        tqdm.write(f"    item {item}: {cnt} 次")

    # 决策建议（只在最终评估时有意义）
    r20 = hits[20] / n
    r10 = hits[10] / n
    ref_r10 = PAPER_RQVAE["Sports"]["R@10"] * 0.5
    ref_r20 = PAPER_RQVAE["Sports"]["R@20"] * 0.5
    tqdm.write(f"\n  决策参考（难度修正后，Sports RQ-VAE × 0.5）:")
    tqdm.write(f"  Recall@10 = {r10:.4f}  参考线 ≥ {ref_r10:.4f}  {'✅' if r10 >= ref_r10 else '🟡'}")
    tqdm.write(f"  Recall@20 = {r20:.4f}  参考线 ≥ {ref_r20:.4f}  {'✅' if r20 >= ref_r20 else '🟡'}")
    tqdm.write(f"  {'决策参考':<4} "
      f"{hits[10]/n:>10.4f} {hits[20]/n:>10.4f} {hits[30]/n:>10.4f} "
      f"{ndcgs[10]/n:>10.4f} {ndcgs[20]/n:>10.4f} {ndcgs[30]/n:>10.4f}")


# ==================== 1. 找 checkpoint ====================
ckpt_dirs = [
    d for d in os.listdir(CKPT_BASE)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(CKPT_BASE, d))
]
latest_ckpt = os.path.join(
    CKPT_BASE,
    sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))[-1]
)
print(f"Using checkpoint: {latest_ckpt}")

# ==================== 2. 加载模型 ====================
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

config = AutoConfig.from_pretrained(QWEN_PATH, trust_remote_code=True)
model  = Qwen2ForCausalLM(config)

ckpt_bin  = os.path.join(latest_ckpt, "pytorch_model.bin")
ckpt_safe = os.path.join(latest_ckpt, "model.safetensors")
if os.path.exists(ckpt_bin):
    state_dict = torch.load(ckpt_bin, map_location="cpu", weights_only=True)
elif os.path.exists(ckpt_safe):
    state_dict = load_file(ckpt_safe)
else:
    raise FileNotFoundError(f"找不到权重文件: {latest_ckpt}")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if "lm_head.weight" in missing:
    print("⚠️  从 embed_tokens 恢复 lm_head 权重（tied weights）")
    model.lm_head.weight = model.model.embed_tokens.weight
    missing = [k for k in missing if k != "lm_head.weight"]
if missing:
    print(f"  Missing keys (前5): {missing[:5]}")

model.to(DEVICE)
model.eval()
print("Model ready.")

# ==================== 3. 加载 item2token ====================
print("Loading item2token...")
item2token_dict = {}
for fname in os.listdir(ITEM2TOKEN_DIR):
    fpath = os.path.join(ITEM2TOKEN_DIR, fname)
    if not os.path.isfile(fpath): continue
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                item2token_dict[parts[0]] = parts[1]
token2item_dict = {v: k for k, v in item2token_dict.items()}
print(f"item2token size: {len(item2token_dict)}")

# ==================== 4. 构造测试集（留一法）====================
print("Building test samples...")
with open(TRAIN_DATA) as f:
    train_data = json.load(f)

test_samples = []
for user_id, item_seq in train_data.items():
    if len(item_seq) < 2:
        continue
    # 从后往前找最后一个有 token 的物品作为 ground truth
    target_item, target_idx = None, -1
    for i in range(len(item_seq) - 1, -1, -1):
        if str(item_seq[i]) in item2token_dict:
            target_item = str(item_seq[i])
            target_idx  = i
            break
    if target_item is None:
        continue
    history_tokens = [
        item2token_dict[str(item)]
        for item in item_seq[:target_idx]
        if str(item) in item2token_dict and str(item) != target_item
    ][-50:]
    if not history_tokens:
        continue
    test_samples.append((user_id, history_tokens, target_item))

print(f"有效测试用户数: {len(test_samples)}")
if MAX_USERS is not None:
    test_samples = test_samples[:MAX_USERS]
    print(f"截取前 {MAX_USERS} 用户进行推理")

# ==================== 5. 断点续跑 ====================
if os.path.exists(CHECKPOINT_FILE):
    print("发现断点，从上次进度继续...")
    with open(CHECKPOINT_FILE) as f:
        results = json.load(f)
    finished = set(results.keys())
    test_samples = [(uid, h, gt) for uid, h, gt in test_samples if uid not in finished]
    print(f"已完成: {len(finished)}，剩余: {len(test_samples)}")
else:
    results = {}
    print("无断点，从头开始。")

# ==================== 6. 构建 Trie ====================
print("Building Trie...")
vocab = tokenizer.get_vocab()
trie  = {}
for token_str in item2token_dict.values():
    tokens = re.findall(r'<[a-c]_\d+>', token_str)
    if len(tokens) != 3: continue
    a = vocab.get(tokens[0])
    b = vocab.get(tokens[1])
    c = vocab.get(tokens[2])
    if None in (a, b, c): continue
    trie.setdefault(a, {}).setdefault(b, set()).add(c)
valid_a_ids = list(trie.keys())
eos_id = tokenizer.eos_token_id
print(f"Trie built: {len(valid_a_ids)} root nodes")

# ==================== 7. 推理主循环 ====================
print("Running inference...")
processed = 0

for user_id, history_tokens, gt_item in tqdm(test_samples):
    prompt = "<|hist_clk_start|>" + "".join(history_tokens) + "<|hist_clk_end|>"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512
    ).to(DEVICE)
    base_len = inputs["input_ids"].shape[1]

    def prefix_fn(batch_id, input_ids, _len=base_len):
        gen = input_ids[_len:]
        if len(gen) == 0:   return valid_a_ids
        elif len(gen) == 1: return list(trie.get(gen[0].item(), {}).keys())
        elif len(gen) == 2: return list(trie.get(gen[0].item(), {}).get(gen[1].item(), set()))
        else:               return [eos_id]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=BEAM_SIZE,
            num_return_sequences=NUM_RETURN,
            prefix_allowed_tokens_fn=prefix_fn,
            early_stopping=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )

    pred_items = []
    for k in range(NUM_RETURN):
        gen_text = tokenizer.decode(outputs[k][base_len:], skip_special_tokens=False)
        match    = re.search(r"<a_\d+><b_\d+><c_\d+>", gen_text)
        if match:
            item = token2item_dict.get(match.group(0))
            if item and item not in pred_items:
                pred_items.append(item)
        if len(pred_items) >= TOP_K_SAVE:
            break

    # ✅ 保存完整 pred 列表（最多 TOP_K_SAVE 个），gt 单独存
    # 这样事后可以重新算任意 @K 的指标，不需要重跑推理
    results[user_id] = {
        "pred": pred_items[:TOP_K_SAVE],
        "gt":   gt_item
    }
    processed += 1

    # 前 3 条打印调试信息
    if processed <= 3:
        print(f"\n  [Debug] User {user_id}")
        print(f"    gt_token : {item2token_dict.get(gt_item, '?')}")
        print(f"    pred[:5] : {pred_items[:5]}")
        print(f"    beam[0]  : {tokenizer.decode(outputs[0][base_len:], skip_special_tokens=False)}")

    # 定期保存断点 + 打印中间指标
    if processed % SAVE_EVERY == 0:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(results, f)

    if processed % EVAL_EVERY == 0:
        tqdm.write(f"\n{'='*66}")
        evaluate_and_print(results, label=f"中间评估 @ {len(results)} users")
        tqdm.write(f"{'='*66}")

# ==================== 8. 保存最终结果 ====================
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print(f"\n推理完成，结果已保存至: {OUTPUT_FILE}")
print(f"总用户数: {len(results)}")

print(f"\n{'='*66}")
evaluate_and_print(results, label="最终评估")
print(f"{'='*66}")
