import os
import argparse
import numpy as np
import torch
from pathlib import Path

from qwenrec.sasrec.dataset0 import MyDataset
from qwenrec.sasrec.model0 import BaselineModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='electronic', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mm_emb_id', nargs='+', default=['512'], type=str)
    parser.add_argument('--norm_first', action='store_true')        # 训练时如果没加就不要传这个参数
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的模型路径，例如 ../output/electronic/checkpoints/epoch_3.pt')
    args = parser.parse_args()

    # 路径设置
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    data_dir = os.path.join(project_root, "dataprepare", args.dataset)
    output_dir = os.path.join(project_root, "output", args.dataset)
    save_dir = os.path.join(output_dir, "item_embeddings")
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集（会加载多模态特征、特征统计等）
    dataset = MyDataset(data_dir, args)

    # ---------- 关键：构建 re-id → 完整特征 的字典 ----------
    print("📦 正在构建 re-id 特征字典...")
    reid_feat_dict = {}
    for i in range(1, dataset.itemnum + 1):          # i 就是重新编号的物品 ID（re-id）
        if i in dataset.indexer_i_rev:
            orig_id = dataset.indexer_i_rev[i]        # 原始物品 ID（字符串）
            raw_feat = dataset.item_feat_dict.get(orig_id, {})
        else:
            raw_feat = {}
        # 用 fill_missing_feat 补全所有特征（包括多模态向量）
        reid_feat_dict[i] = dataset.fill_missing_feat(raw_feat, i)
    print(f"✅ 构建完成，共 {len(reid_feat_dict)} 个物品")

    # 初始化模型并加载训练好的权重
    model = BaselineModel(
        dataset.usernum, dataset.itemnum,
        dataset.feat_statistics, dataset.feature_types, args
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()

    # 所有有效物品的 re-id（跳过 0 号 padding）
    all_item_ids = list(range(1, dataset.itemnum + 1))
    retrieval_ids = list(range(len(all_item_ids)))

    # ---------- 修复 KeyError 的关键：转为列表 ----------
    feat_list = [reid_feat_dict[item_id] for item_id in all_item_ids]

    # 生成并保存 embedding
    print("💾 正在批量生成物品 embedding...")
    model.save_item_emb(
        item_ids=all_item_ids,
        retrieval_ids=retrieval_ids,
        feat_dict=feat_list,          # 这里必须传入列表
        save_path=save_dir,
        batch_size=args.batch_size
    )
    print(f"✅ 物品 embedding 已保存至 {save_dir}/embeddings.npz")

if __name__ == '__main__':
    main()

# cd ~/autodl-tmp/qwenrec/sasrec
# python get_item_emb.py --device cuda --checkpoint ../output/electronic/checkpoints/epoch_3.pt