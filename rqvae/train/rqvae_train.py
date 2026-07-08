import argparse
import random
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader

from rqvae_model import RQVAE
from trainer import  Trainer
from utils import *
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import CustomNpzFile
def cleanup():
    dist.destroy_process_group()
def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=20, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="/jy/code/tx_comp/TencentGR_1k/creative_emb/emb_81_32",
                        help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.000, 0.000, 0.003], help="sinkhorn epsilons")  # [0, 0, 0]的效果其实也还行
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    # parser.add_argument('--num_emb_list', type=int, nargs='+', default=[512,2048,4096], help='emb num of every vq')
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[2048,2048,1024], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[4096,2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=100)
    parser.add_argument("--ckpt_dir", type=str, default="/jy/code/tx_comp/emb_test/256_4", help="output directory for model")
    parser.add_argument('--world_size', type=int, default=1, help='GPU数量')

    return parser.parse_args()


def main_single(args, data_path_list):

    """构建数据集"""
    data = CustomNpzFile(data_path_list)

    # 单卡直接使用普通DataLoader
    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    device = torch.device(args.device)
    """创建模型"""
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    ).to(device)


    """创建训练器"""
    trainer = Trainer(args, model, len(data_loader),device)

    # 训练循环
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print(best_loss)
    print(best_collision_rate)



if __name__ == '__main__':
    args = parse_args()
    args.data_path = "/root/autodl-tmp/qwenrec/cache/emb"
    args.ckpt_dir = "/root/autodl-tmp/qwenrec/cache/rqvae_ckpt/2048_2048_1024"
    # ===== 适配你的向量维度 64 =====
    # encoder: 64 → 32，decoder 自动反转
    args.layers = [128, 64]          # 64维输入不需要太深的网络
    args.e_dim = 32                  # 码本向量维度
    # ===== 码本大小（保持原作者设置即可）=====
    args.num_emb_list = [2048, 2048, 1024]

    # ===== 其他保持默认 =====
    args.batch_size = 4096           # 159k数据，适当缩小
    args.epochs = 1000
    args.kmeans_init = True          # 一定要开，防坍塌


    # 设置随机种子
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 准备数据路径

    data_path_list = [os.path.join(args.data_path, i)
                      for i in os.listdir(args.data_path) if '.npz' in i]
    print(f"找到 {len(data_path_list)} 个 npz 文件: {data_path_list}")


    # 启动多进程训练
    main_single(args,data_path_list)