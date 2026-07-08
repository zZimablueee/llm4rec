#!/usr/bin/env python3
# python reeval.py
# python /root/autodl-tmp/qwenrec/gr/inference_result.py
"""
事后重算指标脚本：从已保存的 inference_result.json 重新计算任意 @K 指标。
不需要加载模型，秒级完成。
"""
import json, math
from collections import Counter

RESULT_FILE = "/root/autodl-tmp/qwenrec/cache/inference_result.json"

PAPER_RQVAE = {
    "Beauty":  {"R@10": 0.059, "R@20": 0.091, "R@30": 0.111, "N@10": 0.033, "N@20": 0.041, "N@30": 0.047},
    "Sports":  {"R@10": 0.031, "R@20": 0.048, "R@30": 0.062, "N@10": 0.017, "N@20": 0.021, "N@30": 0.024},
    "Toys":    {"R@10": 0.051, "R@20": 0.074, "R@30": 0.089, "N@10": 0.028, "N@20": 0.035, "N@30": 0.038},
}

def ndcg_at_k(hit_position, k):
    if hit_position is None or hit_position > k:
        return 0.0
    return 1.0 / math.log2(hit_position + 1)

with open(RESULT_FILE) as f:
    results = json.load(f)

K_LIST = [10, 20, 30]
hits  = {k: 0   for k in K_LIST}
ndcgs = {k: 0.0 for k in K_LIST}
all_top1       = []
all_preds_flat = []
n = len(results)

for uid, val in results.items():
    gt   = val["gt"]
    pred = val["pred"]
    all_top1.append(pred[0] if pred else "__empty__")
    all_preds_flat.extend(pred[:30])

    hit_pos = None
    for rank, p in enumerate(pred, start=1):
        if p == gt:
            hit_pos = rank
            break

    for k in K_LIST:
        if hit_pos is not None and hit_pos <= k:
            hits[k]  += 1
            ndcgs[k] += ndcg_at_k(hit_pos, k)

print(f"\n总用户数: {n}  |  数据集: Electronics")
print(f"\n  {'':4} {'Recall@10':>10} {'Recall@20':>10} {'Recall@30':>10} "
      f"{'NDCG@10':>10} {'NDCG@20':>10} {'NDCG@30':>10}")
print(f"  {'-'*66}")
print(f"  {'你的模型':<4} "
      f"{hits[10]/n:>10.4f} {hits[20]/n:>10.4f} {hits[30]/n:>10.4f} "
      f"{ndcgs[10]/n:>10.4f} {ndcgs[20]/n:>10.4f} {ndcgs[30]/n:>10.4f}")
print(f"  {'-'*66}")
for ds, vals in PAPER_RQVAE.items():
    print(f"  {'RQ/'+ds:<14} "
          f"{vals['R@10']:>10.4f} {vals['R@20']:>10.4f} {vals['R@30']:>10.4f} "
          f"{vals['N@10']:>10.4f} {vals['N@20']:>10.4f} {vals['N@30']:>10.4f}")
print(f"  {'-'*66}")
print(f"  ⚠️  论文数据集规模 < 36K 用户，Electronics = 728K，数值仅供参考")

# 多样性
unique_top1 = len(set(all_top1))
print(f"\n  Top1 多样性: {unique_top1}/{n} = {unique_top1/n:.2%}")
print(f"\n  最常推荐的 10 个物品:")
for item, cnt in Counter(all_preds_flat).most_common(10):
    print(f"    item {item}: {cnt} 次")
