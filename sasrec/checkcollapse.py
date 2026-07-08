# check_collapse.py
import numpy as np

data = np.load("/root/autodl-tmp/qwenrec/cache/emb/embeddings.npz")
embs = data['embs']

# L2 归一化
norms = np.linalg.norm(embs, axis=1, keepdims=True)
embs_norm = embs / (norms + 1e-8)

# 随机抽 1000 对计算余弦相似度
idx = np.random.choice(len(embs_norm), size=(1000, 2), replace=False)
sims = (embs_norm[idx[:, 0]] * embs_norm[idx[:, 1]]).sum(axis=1)

print(f"余弦相似度 均值: {sims.mean():.4f}")
print(f"余弦相似度 标准差: {sims.std():.4f}")
print(f"余弦相似度 最大值: {sims.max():.4f}")
print(f"余弦相似度 最小值: {sims.min():.4f}")


import numpy as np
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ========== 路径设置 ==========
base = "/root/autodl-tmp/qwenrec"
npz_path = f"{base}/cache/emb/embeddings.npz"
ckpt_path = f"{base}/output/electronic/checkpoints/epoch_1.pt"   # 新训练的只有 epoch_1
mm_path = f"{base}/dataprepare/electronic/item_img_emb_512.pkl"
indexer_path = f"{base}/dataprepare/electronic/indexer.pkl"

print("="*60)
print("📦 1. 检查新生成的 npz 文件")
print("="*60)
try:
    data = np.load(npz_path)
    embs = data["embs"]
    ids = data["ids"]
    print(f"npz 向量数量: {len(embs)}, 维度: {embs.shape[1]}")
    var = embs.var(axis=0)
    print(f"每维方差 min: {var.min():.6f}  max: {var.max():.6f}  mean: {var.mean():.6f}")
    # 随机500个计算余弦相似度
    n_sample = min(500, len(embs))
    idx = np.random.choice(len(embs), n_sample, replace=False)
    sim = cosine_similarity(embs[idx], embs[idx])
    print(f"平均余弦相似度: {sim.mean():.6f}")
except Exception as e:
    print(f"❌ 读取 npz 失败: {e}")

print("\n" + "="*60)
print("🧠 2. 检查新模型 (epoch_1) 的物品 ID embedding")
print("="*60)
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    item_emb = ckpt["item_emb.weight"].numpy()   # (itemnum+1, hidden_dim)
    # 去掉 padding 行 0
    item_emb_nopad = item_emb[1:, :]
    var = item_emb_nopad.var(axis=0)
    print(f"ID emb 数量: {len(item_emb_nopad)}, 维度: {item_emb_nopad.shape[1]}")
    print(f"每维方差 min: {var.min():.6f}  max: {var.max():.6f}  mean: {var.mean():.6f}")
    n_sample = min(500, len(item_emb_nopad))
    idx = np.random.choice(len(item_emb_nopad), n_sample, replace=False)
    sim = cosine_similarity(item_emb_nopad[idx], item_emb_nopad[idx])
    print(f"ID embedding 平均余弦相似度: {sim.mean():.6f}")
except Exception as e:
    print(f"❌ 加载模型失败: {e}")

print("\n" + "="*60)
print("🖼️  3. 检查原始多模态特征 (512维)")
print("="*60)
try:
    with open(mm_path, "rb") as f:
        mm_dict = pickle.load(f)
    all_vecs = np.stack(list(mm_dict.values()))
    print(f"多模态向量数量: {all_vecs.shape[0]}, 维度: {all_vecs.shape[1]}")
    var = all_vecs.var(axis=0)
    print(f"每维方差 min: {var.min():.6f}  max: {var.max():.6f}  mean: {var.mean():.6f}")
    n_sample = min(500, all_vecs.shape[0])
    idx = np.random.choice(all_vecs.shape[0], n_sample, replace=False)
    sim = cosine_similarity(all_vecs[idx], all_vecs[idx])
    print(f"原始多模态平均余弦相似度: {sim.mean():.6f}")
except Exception as e:
    print(f"❌ 读取多模态失败: {e}")

print("\n" + "="*60)
print("📊 结论：")
print("  - 如果 npz 或 ID emb 的相似度 > 0.95，说明 SASRec 生成的物品向量坍塌。")
print("  - 如果原始多模态正常（相似度 < 0.7），说明问题在 SASRec 融合，")
print("    可用 PCA 直接降维多模态特征作为输入。")
print("="*60)