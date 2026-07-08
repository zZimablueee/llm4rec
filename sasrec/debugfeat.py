# 新建文件 debug_feat.py，放在 sasrec 目录下
import json
import pickle
from pathlib import Path

data_dir = Path("/root/autodl-tmp/qwenrec/dataprepare/electronic")

with open(data_dir / "indexer.pkl", "rb") as f:
    indexer = pickle.load(f)

with open(data_dir / "item_feat_dict.json", "r") as f:
    item_feat_dict = json.load(f)

feat_ids = ['100', '101', '102', '103']

print("=== indexer['f'] 中各特征的值数量 ===")
for feat_id in feat_ids:
    if feat_id in indexer.get('f', {}):
        n = len(indexer['f'][feat_id])
        print(f"  feat '{feat_id}': indexer 中有 {n} 个值，Embedding 表大小应为 {n+1}")
    else:
        print(f"  feat '{feat_id}': 不在 indexer['f'] 中！")

print("\n=== item_feat_dict 中各特征的实际最大值 ===")
max_vals = {fid: 0 for fid in feat_ids}
for item_id, feat in item_feat_dict.items():
    for fid in feat_ids:
        if fid in feat:
            v = feat[fid]
            if isinstance(v, int) and v > max_vals[fid]:
                max_vals[fid] = v

for fid in feat_ids:
    emb_size = len(indexer['f'].get(fid, {})) + 1
    actual_max = max_vals[fid]
    status = "✅" if actual_max < emb_size else "❌ 越界！"
    print(f"  feat '{fid}': 实际最大值={actual_max}, Embedding表大小={emb_size}  {status}")
