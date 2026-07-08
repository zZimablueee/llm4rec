import numpy as np
data = np.load("/root/autodl-tmp/qwenrec/cache/emb/embeddings.npz")
print(data['embs'].shape)
# npz向量是64维的