# debug_nan.py
import pickle
import numpy as np
from pathlib import Path

data_dir = Path("/root/autodl-tmp/qwenrec/dataprepare/electronic")

print("=== 检查图像 Embedding ===")
with open(data_dir / "item_img_emb_512.pkl", "rb") as f:
    emb_dict = pickle.load(f)

all_embs = np.stack(list(emb_dict.values()))
print(f"shape: {all_embs.shape}")
print(f"NaN 数量: {np.isnan(all_embs).sum()}")
print(f"Inf 数量: {np.isinf(all_embs).sum()}")
print(f"最大值: {all_embs.max():.4f}")
print(f"最小值: {all_embs.min():.4f}")
print(f"均值: {all_embs.mean():.4f}")
print(f"全零向量数量: {(all_embs == 0).all(axis=1).sum()}")
