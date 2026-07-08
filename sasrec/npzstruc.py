import numpy as np

data = np.load("/root/autodl-tmp/qwenrec/cache/emb/embeddings.npz", allow_pickle=True)

print("keys:", list(data.keys()))
print("ids shape:", data['ids'].shape)
print("embs shape:", data['embs'].shape)
print("ids 前5个:", data['ids'][:5])
print("embs 前2行:", data['embs'][:2])
import numpy as np

data = np.load("/root/autodl-tmp/qwenrec/cache/emb/embeddings.npz", allow_pickle=True)

print("ids 最小值:", data['ids'].min())
print("ids 最大值:", data['ids'].max())
print("总条数:", len(data['ids']))
