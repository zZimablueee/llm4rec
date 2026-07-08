import json, math

RESULT_FILE = "/root/autodl-tmp/qwenrec/cache/inference_result.json"
TOP_K_LIST  = [1, 5, 10]

with open(RESULT_FILE) as f:
    results = json.load(f)

def hit_at_k(pred, gt, k):
    return 1.0 if gt in pred[:k] else 0.0

def ndcg_at_k(pred, gt, k):
    if gt in pred[:k]:
        return 1.0 / math.log2(pred[:k].index(gt) + 2)
    return 0.0

metrics = {k: {"hit": 0.0, "ndcg": 0.0} for k in TOP_K_LIST}
total = 0

for user_id, data in results.items():
    pred, gt = data["pred"], data["gt"]
    if not pred:
        continue
    total += 1
    for k in TOP_K_LIST:
        metrics[k]["hit"]  += hit_at_k(pred, gt, k)
        metrics[k]["ndcg"] += ndcg_at_k(pred, gt, k)

print(f"\n{'='*40}")
print(f"评估用户数: {total}")
print(f"{'='*40}")
for k in TOP_K_LIST:
    print(f"Hit@{k:<3} = {metrics[k]['hit']/total:.4f}    NDCG@{k:<3} = {metrics[k]['ndcg']/total:.4f}")
print(f"{'='*40}\n")
