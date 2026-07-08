import pickle
from pathlib import Path
import numpy as np

mm_path = Path('/root/autodl-tmp/qwenrec/dataprepare/electronic')   # 按你的实际路径调整
feat_ids = ['512']                         # 你要检查的特征

for feat_id in feat_ids:
    path = mm_path / f'item_img_emb_{feat_id}.pkl'
    if not path.exists():
        print(f"文件不存在: {path}")
        continue
    with open(path, 'rb') as f:
        emb_dict = pickle.load(f)
    
    nan_count = 0
    inf_count = 0
    for k, v in emb_dict.items():
        arr = np.array(v)
        if np.isnan(arr).any():
            nan_count += 1
            if nan_count <= 5:
                print(f"NaN 样本: key={k}, array_preview={arr[:5]}...")
        if np.isinf(arr).any():
            inf_count += 1
            if inf_count <= 5:
                print(f"Inf 样本: key={k}, array_preview={arr[:5]}...")
    print(f"特征 {feat_id}: 总数 {len(emb_dict)}, NaN数量 {nan_count}, Inf数量 {inf_count}")