# dataset.py - 完整改造版，适配 Amazon Electronics 数据集

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id  # 例如 ['87']

        self.item_feat_dict = json.load(
            open(Path(data_dir, "item_feat_dict.json"), 'r')
        )

        # ── 加载图像 Embedding（key 是 int reid）──────────────────
        self.mm_emb_dict = load_mm_emb_amazon(
            Path(data_dir, "item_img_emb_512.pkl"),
            self.mm_emb_ids
        )

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])

        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = \
            self._init_feat_info()

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
        # seq_offsets 是 list，直接用下标索引

    def _load_user_data(self, uid):
        """uid 是行号（0-based），直接用 list 下标"""
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        return json.loads(line)

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        # ── 解析序列 ──────────────────────────────────────────────
        # 格式: [user_reid, item_reid, user_feat, item_feat, action_type, timestamp]
        # Amazon 数据：user_feat={}, item_feat={} 均为空，特征从 item_feat_dict 取
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            # Amazon 无用户侧特征，只处理 item token
            if i and i != 0:
                # 从 item_feat_dict 补充特征
                item_feat = self.item_feat_dict.get(str(i), {})
                ext_user_sequence.append((i, item_feat, 1, action_type))

        if len(ext_user_sequence) < 2:
            # 序列太短，返回全零
            seq = np.zeros([self.maxlen + 1], dtype=np.int32)
            pos = np.zeros([self.maxlen + 1], dtype=np.int32)
            neg = np.zeros([self.maxlen + 1], dtype=np.int32)
            token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            seq_feat = np.where(
                np.empty([self.maxlen + 1], dtype=object) == None,
                self.feature_default_value, 
                np.empty([self.maxlen + 1], dtype=object)
            )
            pos_feat = seq_feat.copy()
            neg_feat = seq_feat.copy()
            return seq, pos, neg, token_type, next_token_type, \
                   next_action_type, seq_feat, pos_feat, neg_feat

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)

            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(
                    self.item_feat_dict.get(str(neg_id), {}), neg_id
                )
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, \
               next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        return len(self.seq_offsets)

    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}

        # ── Amazon Electronics 特征配置 ───────────────────────────
        # 你的 indexer['f'] 只有 '100','101','102','103'
        # 全部是 item 侧 sparse 特征，无用户特征

        feat_types['user_sparse']    = []   # Amazon 无用户特征
        feat_types['user_array']     = []
        feat_types['user_continual'] = []
        feat_types['item_sparse']    = ['100', '101', '102', '103']
        feat_types['item_array']     = []
        feat_types['item_continual'] = []
        feat_types['item_emb']       = self.mm_emb_ids  # ['87'] 或 []

        # sparse 特征：缺省值=0，统计特征值数量
        # '102' 价格桶特征不在 indexer['f'] 中，需要手动指定大小
        MANUAL_FEAT_SIZE = {
            '102': 10   # 价格桶，实际最大值是5，留一些余量
        }

        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id in self.indexer.get('f', {}):
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            elif feat_id in MANUAL_FEAT_SIZE:
                feat_statistics[feat_id] = MANUAL_FEAT_SIZE[feat_id]
            else:
                feat_statistics[feat_id] = 1


        # 图像 Embedding 特征
        EMB_SHAPE_DICT = {'87': 512}
        for feat_id in feat_types['item_emb']:
            shape = EMB_SHAPE_DICT.get(feat_id, 512)
            feat_default_value[feat_id] = np.zeros(shape, dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """填充缺失特征，并注入图像 Embedding"""
        if feat is None:
            feat = {}
        filled_feat = dict(feat)

        # 补全所有缺失的特征字段
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        for feat_id in set(all_feat_ids) - set(filled_feat.keys()):
            filled_feat[feat_id] = self.feature_default_value[feat_id]

        # 注入图像 Embedding（key 是 int reid）
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and feat_id in self.mm_emb_dict:
                emb = self.mm_emb_dict[feat_id].get(item_id, None)
                if emb is not None and isinstance(emb, np.ndarray):
                    filled_feat[feat_id] = emb

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, \
            next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq              = torch.from_numpy(np.array(seq))
        pos              = torch.from_numpy(np.array(pos))
        neg              = torch.from_numpy(np.array(neg))
        token_type       = torch.from_numpy(np.array(token_type))
        next_token_type  = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat  = list(seq_feat)
        pos_feat  = list(pos_feat)
        neg_feat  = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, \
               next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        predict_path = self.data_dir / "predict_seq.jsonl"
        predict_offsets_path = Path(self.data_dir, 'predict_seq_offsets.pkl')
        if predict_path.exists() and predict_offsets_path.exists():
            self.data_file = open(predict_path, 'rb')
            with open(predict_offsets_path, 'rb') as f:
                self.seq_offsets = pickle.load(f)
        else:
            # 没有预测集时回退到训练集
            self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
            with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
                self.seq_offsets = pickle.load(f)

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        user_id = str(uid)  # 默认用行号作为 user_id

        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if isinstance(u, str):
                    user_id = u
            if i and i != 0:
                if i > self.itemnum:
                    i = 0
                item_feat = self.item_feat_dict.get(str(i), {})
                ext_user_sequence.append((i, item_feat, 1))

        seq      = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx]        = i
            token_type[idx] = type_
            seq_feat[idx]   = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        return seq, token_type, seq_feat, user_id

    def __len__(self):
        return len(self.seq_offsets)

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq        = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat   = list(seq_feat)
        return seq, token_type, seq_feat, user_id


# ── 图像 Embedding 加载（Amazon 版本）─────────────────────────────────

def load_mm_emb_amazon(img_emb_path, feat_ids):
    """
    加载 item_img_emb_512.pkl
    原始 key = int(reid)，直接使用，无需转换
    返回格式: {feat_id: {reid(int): np.array(512,)}}
    """
    mm_emb_dict = {}

    if '87' not in feat_ids:
        # 没有请求图像特征，返回空字典
        for feat_id in feat_ids:
            mm_emb_dict[feat_id] = {}
        return mm_emb_dict

    if not Path(img_emb_path).exists():
        print(f"⚠️  图像 Embedding 文件不存在: {img_emb_path}，跳过")
        for feat_id in feat_ids:
            mm_emb_dict[feat_id] = {}
        return mm_emb_dict

    print(f"加载图像 Embedding: {img_emb_path}")
    with open(img_emb_path, 'rb') as f:
        raw_dict = pickle.load(f)  # {int(reid): np.array(512,)}

    # 确保 value 是 float32
    converted = {}
    for k, v in raw_dict.items():
        if isinstance(v, np.ndarray):
            converted[k] = v.astype(np.float32)
        else:
            converted[k] = np.array(v, dtype=np.float32)

    for feat_id in feat_ids:
        mm_emb_dict[feat_id] = converted
    print(f"✅ 图像 Embedding 加载完成，共 {len(converted)} 个item")
    return mm_emb_dict


def save_emb(emb, save_path):
    num_points     = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


# 保留原版接口兼容性
def load_mm_emb(mm_path, feat_ids):
    """原版接口，Amazon 场景下重定向到 load_mm_emb_amazon"""
    # 如果调用了原版接口，返回空字典（Amazon 不用这个路径）
    return {feat_id: {} for feat_id in feat_ids}
