# eval_only.py
import torch, random, numpy as np, os
from torch.utils.data import DataLoader
from rqvae_model import RQVAE
from trainer import Trainer
from datasets import CustomNpzFile
from utils import *

# ===== 路径 =====
NPZ_PATH = "/root/autodl-tmp/qwenrec/cache/emb"
CKPT_DIR = "/root/autodl-tmp/qwenrec/cache/rqvae_ckpt/2048_2048_1024"

# 找best_collision_model.pth
import glob
ckpt_files = glob.glob(os.path.join(CKPT_DIR, "**/best_collision_model.pth"), recursive=True)
print(f"找到ckpt: {ckpt_files}")
CKPT_PATH = ckpt_files[0]

# 加载数据
data_path_list = [os.path.join(NPZ_PATH, i) for i in os.listdir(NPZ_PATH) if '.npz' in i]
dataset = CustomNpzFile(data_path_list)
loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)

# 加载模型
device = torch.device("cuda:0")
ckpt = torch.load(CKPT_PATH, map_location=device)
saved_args = ckpt['args']

model = RQVAE(
    in_dim=dataset.dim,
    num_emb_list=saved_args.num_emb_list,
    e_dim=saved_args.e_dim,
    layers=saved_args.layers,
    dropout_prob=saved_args.dropout_prob,
    bn=saved_args.bn,
    loss_type=saved_args.loss_type,
    quant_loss_weight=saved_args.quant_loss_weight,
    beta=saved_args.beta,
    kmeans_init=False,   # 评估时不需要初始化
    kmeans_iters=saved_args.kmeans_iters,
    sk_epsilons=saved_args.sk_epsilons,
    sk_iters=saved_args.sk_iters,
).to(device)
model.load_state_dict(ckpt['state_dict'])

# ===== 新增这一行，模拟优化前 =====. 如果想要优化后的就把这个for循环注释掉
for vq in model.rq.vq_layers:
    vq.sk_epsilon = 0.0


# 创建trainer只是为了用_valid_epoch
trainer = Trainer(saved_args, model, len(loader), device)

print("===== 优化后指标 =====")
collision_rate = trainer._valid_epoch(loader)
print(f"冲突率: {collision_rate*100:.4f}%")

# python eval_only.py
# 得到：优化后冲突率 + 优化后每层码本利用率
