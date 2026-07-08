import os
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from qwenrec.sasrec.dataset0 import MyDataset
from qwenrec.sasrec.model0 import BaselineModel

# --- 1. 最小路径适配 (保持你的路径逻辑，确保能读到 electronic) ---
current_script_path = os.path.abspath(__file__) 
project_root = os.path.dirname(os.path.dirname(current_script_path))
target_dataset = 'electronic' # 或通过 args 获取

os.environ['TRAIN_DATA_PATH'] = os.path.join(project_root, "dataprepare", target_dataset)
output_root = os.path.join(project_root, "output", target_dataset)
os.environ['TRAIN_LOG_PATH'] = os.path.join(output_root, "logs")
os.environ['TRAIN_TF_EVENTS_PATH'] = os.path.join(output_root, "tf_events")
os.environ['TRAIN_CKPT_PATH'] = os.path.join(output_root, "checkpoints")
os.environ['USER_CACHE_PATH'] = os.path.join(output_root, "cache")

for p in [os.environ['TRAIN_LOG_PATH'], os.environ['TRAIN_TF_EVENTS_PATH'], 
          os.environ['TRAIN_CKPT_PATH'], os.environ['USER_CACHE_PATH']]:
    os.makedirs(p, exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='electronic', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0001, type=float) # 建议先用这个稳健的学习率
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str) # AutoDL 有卡记得用 cuda
    parser.add_argument('--mm_emb_id', nargs='+', default=['512'], type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    dataset = MyDataset(os.environ.get('TRAIN_DATA_PATH'), args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = BaselineModel(dataset.usernum, dataset.itemnum, dataset.feat_statistics, dataset.feature_types, args).to(args.device)

    # 初始化权重并清零 Padding
    for name, param in model.named_parameters():
        try: torch.nn.init.xavier_normal_(param.data)
        except: pass
    model.item_emb.weight.data[0, :] = 0
    # ... 其他 embedding 清零保持原样 ...

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    global_step = 0
    print("🚀 Start training...")
    init_item_emb = model.item_emb.weight.data.clone()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 1. 确保数据全部上云（上 GPU）
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = [
                b.to(args.device) if isinstance(b, torch.Tensor) else b for b in batch
            ]
            
            # 🚀 救命第一行：必须清空梯度，否则梯度会像雪球一样滚炸
            optimizer.zero_grad()

            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            # 2. 找到有效标签（物品 Token）
            indices = torch.where(next_token_type == 1)
            
            # 🚀 救命第二行：加一个保险。如果这个 batch 没抓到有效 item，直接跳过，防止 0/0 导致 NaN
            if len(indices[0]) == 0:
                continue

            pos_labels = torch.ones(pos_logits[indices].shape, device=args.device)
            neg_labels = torch.zeros(neg_logits[indices].shape, device=args.device)
            
            # 只算有效位的 Loss
            loss = bce_criterion(pos_logits[indices], pos_labels)
            loss += bce_criterion(neg_logits[indices], neg_labels)

            # 3. 安全版 L2 正则（如果 args.l2_emb > 0 才加）
            if args.l2_emb > 0:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.sum(param ** 2)

            # 4. 反向传播与防爆处理
            loss.backward()
            if global_step % 10 == 0:
                item_grad_norm = model.item_emb.weight.grad.norm().item()
                print(f"    [Grad] item_emb grad norm: {item_grad_norm:.6f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 打印日志（这里加个判断，防止打印太频繁）
            if global_step % 10 == 0:
                log_json = json.dumps({'step': global_step, 'loss': loss.item(), 'epoch': epoch})
                log_file.write(log_json + '\n')
                log_file.flush()
                print(log_json)

            global_step += 1
        # 计算并打印物品 ID embedding 的总变化量
        weight_change = (model.item_emb.weight.data - init_item_emb).norm().item()
        print(f"    [Weight] item_emb total change after epoch {epoch}: {weight_change:.6f}")
        init_item_emb = model.item_emb.weight.data.clone()  # 更新快照供下一epoch

        # 每个 epoch 存一次（逻辑保持原版简化）
        torch.save(model.state_dict(), Path(os.environ.get('TRAIN_CKPT_PATH'), f"epoch_{epoch}.pt"))

        # ========== 训练结束，保存所有物品 embedding ==========
    print("💾 正在构建 re-id 特征字典...")
    # 构建以 re-id 为 key 的完整特征字典（解决 save_item_emb 的 KeyError）
    reid_feat_dict = {}
    for i in range(1, dataset.itemnum + 1):           # i 是 re-id
        if i in dataset.indexer_i_rev:
            orig_id = dataset.indexer_i_rev[i]
            raw_feat = dataset.item_feat_dict.get(orig_id, {})
        else:
            raw_feat = {}
        # 用 fill_missing_feat 补全所有特征（包括多模态向量）
        reid_feat_dict[i] = dataset.fill_missing_feat(raw_feat, i)

    # 生成检索用连续 ID
    all_item_ids = list(range(1, dataset.itemnum + 1))
    retrieval_ids = list(range(len(all_item_ids)))

    feat_list = [reid_feat_dict[item_id] for item_id in all_item_ids]

    save_dir = Path(os.environ['TRAIN_CKPT_PATH']).parent / "item_embeddings"
    os.makedirs(save_dir, exist_ok=True)

    print("💾 正在批量生成物品 embedding...")
    model.save_item_emb(
        item_ids=all_item_ids,
        retrieval_ids=retrieval_ids,
        feat_dict=feat_list,          # 使用修正后的字典
        save_path=save_dir,
        batch_size=1024                    # 可以根据显存调整
    )
    print(f"✅ 物品 embedding 已保存至 {save_dir}/embeddings.npz")
    # ====================================================
    print("✅ Done!")
    log_file.close()