# /root/autodl-tmp/qwenrec/sasrec/sasrec_inference.py

# python /root/autodl-tmp/qwenrec/sasrec/sasrec_inference.py

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/qwenrec/sasrec')
from dataset import MyTestDataset, load_mm_emb_amazon
from model import BaselineModel

# ── 配置 ──────────────────────────────────────────────────────────────
DATA_DIR      = '/root/autodl-tmp/qwenrec/dataprepare/electronic'
CKPT_PATH     = '/root/autodl-tmp/qwenrec/checkpoints/global_step12810.valid_loss=0.6978/model.pt'
ITEM_EMB_PATH = '/root/autodl-tmp/qwenrec/cache/emb/embeddings.npz'
DEVICE        = 'cuda'
BATCH_SIZE    = 512
KS            = [10, 20, 30]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     default=BATCH_SIZE, type=int)
    parser.add_argument('--lr',             default=0.001,      type=float)
    parser.add_argument('--maxlen',         default=50,         type=int)
    parser.add_argument('--hidden_units',   default=64,         type=int)
    parser.add_argument('--num_blocks',     default=2,          type=int)
    parser.add_argument('--num_epochs',     default=3,          type=int)
    parser.add_argument('--num_heads',      default=1,          type=int)
    parser.add_argument('--dropout_rate',   default=0.2,        type=float)
    parser.add_argument('--l2_emb',         default=0.0,        type=float)
    parser.add_argument('--device',         default=DEVICE,     type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path',default=None,       type=str)
    parser.add_argument('--norm_first',     action='store_true')
    parser.add_argument('--mm_emb_id',      nargs='+', default=['87'], type=str)
    return parser.parse_args([])


class LightTestDataset(Dataset):
    def __init__(self, data_dir, args, item_feat_dict, feature_default_value,
                 feature_types, feat_statistics, mm_emb_dict, itemnum):
        self.data_dir              = Path(data_dir)
        self.maxlen                = args.maxlen
        self.item_feat_dict        = item_feat_dict
        self.feature_default_value = feature_default_value
        self.feature_types         = feature_types
        self.feat_statistics       = feat_statistics
        self.mm_emb_dict           = mm_emb_dict
        self.itemnum               = itemnum

        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.offsets = pickle.load(f)
        self.data_file = open(self.data_dir / 'seq.jsonl', 'rb')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, uid):
        self.data_file.seek(self.offsets[uid])
        seq = json.loads(self.data_file.readline())

        # 除最后一条之外作为历史输入，最后一条是 ground truth
        history = seq[:-1]

        arr_seq  = np.zeros([self.maxlen + 1], dtype=np.int32)
        arr_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        arr_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen
        for record in reversed(history):
            u, i, user_feat, item_feat, action_type, ts = record
            if not i or i == 0 or i > self.itemnum:
                continue
            feat = self.item_feat_dict.get(str(i), {})
            feat = self._fill_feat(feat, i)
            arr_seq[idx]  = i
            arr_type[idx] = 1
            arr_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        arr_feat = np.where(arr_feat == None, self.feature_default_value, arr_feat)
        return arr_seq, arr_type, arr_feat, uid

    def _fill_feat(self, feat, item_id):
        filled = dict(feat) if feat else {}
        all_ids = []
        for v in self.feature_types.values():
            all_ids.extend(v)
        for fid in set(all_ids) - set(filled.keys()):
            filled[fid] = self.feature_default_value[fid]
        for fid in self.feature_types.get('item_emb', []):
            if item_id != 0 and fid in self.mm_emb_dict:
                emb = self.mm_emb_dict[fid].get(item_id, None)
                if emb is not None:
                    filled[fid] = emb
        return filled

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_ids = zip(*batch)
        seq        = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        return seq, token_type, list(seq_feat), list(user_ids)


def load_ground_truth(data_dir):
    seq_path     = Path(data_dir) / 'seq.jsonl'
    offsets_path = Path(data_dir) / 'seq_offsets.pkl'

    with open(offsets_path, 'rb') as f:
        offsets = pickle.load(f)

    gt = {}
    with open(seq_path, 'rb') as f:
        for uid, offset in enumerate(tqdm(offsets, desc="加载 ground truth")):
            f.seek(offset)
            seq = json.loads(f.readline())
            for record in reversed(seq):
                item_id = record[1]
                if item_id and item_id != 0:
                    gt[uid] = int(item_id)
                    break
    return gt


def compute_metrics(recommended, ground_truth, ks):
    recalls = {k: [] for k in ks}
    ndcgs   = {k: [] for k in ks}

    for uid, top_items in recommended.items():
        if uid not in ground_truth:
            continue
        target = ground_truth[uid]
        for k in ks:
            topk = top_items[:k]
            hit  = int(target in topk)
            recalls[k].append(hit)
            if hit:
                rank = topk.index(target) + 1
                ndcgs[k].append(1.0 / np.log2(rank + 1))
            else:
                ndcgs[k].append(0.0)

    results = {}
    for k in ks:
        results[f'Recall@{k}'] = np.mean(recalls[k]) if recalls[k] else 0.0
        results[f'NDCG@{k}']   = np.mean(ndcgs[k])   if ndcgs[k]   else 0.0
    return results, {k: len(recalls[k]) for k in ks}


def main():
    args = get_args()

    # ── 1. 加载 item embedding ─────────────────────────────────────────
    print("加载 item embedding...")
    npz = np.load(ITEM_EMB_PATH)
    item_embs_arr = npz['embs']                                          # (159729, 64)
    zero_row = np.zeros((1, item_embs_arr.shape[1]), dtype=np.float32)
    item_embs_1based = np.concatenate([zero_row, item_embs_arr], axis=0) # (159730, 64)
    item_emb_matrix = torch.tensor(item_embs_1based, dtype=torch.float32, device=DEVICE)
    item_emb_matrix = F.normalize(item_emb_matrix, dim=-1)
    item_emb_search = item_emb_matrix[1:]                                # (159729, 64)
    print(f"item embedding 矩阵: {item_emb_search.shape}")

    # ── 2. 加载模型元数据 + 模型 ───────────────────────────────────────
    print("加载模型...")
    _meta = MyTestDataset(DATA_DIR, args)
    model = BaselineModel(
        _meta.usernum, _meta.itemnum,
        _meta.feat_statistics, _meta.feature_types, args
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型加载完成，usernum={_meta.usernum}, itemnum={_meta.itemnum}")

    # ── 3. 加载 ground truth ───────────────────────────────────────────
    ground_truth = load_ground_truth(DATA_DIR)
    print(f"ground truth 用户数: {len(ground_truth)}")

    # ── 4. 构建 Dataset & DataLoader ──────────────────────────────────
    dataset = LightTestDataset(
        DATA_DIR, args,
        item_feat_dict        = _meta.item_feat_dict,
        feature_default_value = _meta.feature_default_value,
        feature_types         = _meta.feature_types,
        feat_statistics       = _meta.feat_statistics,
        mm_emb_dict           = _meta.mm_emb_dict,
        itemnum               = _meta.itemnum,
    )
    del _meta

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = LightTestDataset.collate_fn,
    )

    # ── 5. 推理 ────────────────────────────────────────────────────────
    recommended = {}
    print("开始推理...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理"):
            seq, token_type, seq_feat, user_ids = batch
            seq        = seq.to(DEVICE)
            token_type = token_type.to(DEVICE)

            user_emb = model.predict(seq, seq_feat, token_type)   # (B, 64)
            user_emb = F.normalize(user_emb, dim=-1)

            scores = torch.matmul(user_emb, item_emb_search.T)    # (B, 159729)
            _, topk_idx = torch.topk(scores, k=max(KS), dim=-1)
            topk_idx = topk_idx.cpu().numpy()

            for i, uid_raw in enumerate(user_ids):
                uid = int(uid_raw)
                recommended[uid] = (topk_idx[i] + 1).tolist()    # 0-based → 1-based reid

    # ── 6. 计算指标 ────────────────────────────────────────────────────
    print("计算指标...")
    results, counts = compute_metrics(recommended, ground_truth, KS)

    print("\n" + "=" * 65)
    print("  SASRec Baseline — Amazon Electronics")
    print("=" * 65)
    for k in KS:
        print(f"  Recall@{k:<3} = {results[f'Recall@{k}']:.4f}    "
              f"NDCG@{k:<3} = {results[f'NDCG@{k}']:.4f}    "
              f"(evaluated on {counts[k]} users)")
    print("=" * 65)


if __name__ == '__main__':
    main()

