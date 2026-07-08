import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen # 序列最大长度
        self.mm_emb_ids = args.mm_emb_id # 要加载的多模态特征id列表

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r')) # 物品特征字典 key：itemid value：特征
        self.mm_emb_dict = load_mm_emb(Path(data_dir), self.mm_emb_ids) 
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff) # 索引器 原始物品/用户id 到 重拍id（reid）
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        # 反向映射 reid 到 原始物品/用户id
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        # 初始化特征信息 文档4.2
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载文件偏移量
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样
        生成不在用户交互序列（s）中且存在物品特征的随机物品 ID，作为负样本

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户原始序列

        ext_user_sequence = [] # 扩展序列
        # ----user token和item token混排----------
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            # 必须用 is not None，否则空属性 {} 会被原代码直接丢弃
            if u and user_feat is not None:    
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat is not None:    
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 【防崩溃兜底】如果数据极度异常导致序列依然为空，塞一个空动作
        if len(ext_user_sequence) == 0:
            ext_user_sequence.append((0, {}, 1, 0))

        # 下面是你代码里原有的部分，保持不变
        seq = np.zeros([self.maxlen + 1], dtype=np.int32) # 序列id，padding用0填充
        pos = np.zeros([self.maxlen + 1], dtype=np.int32) # 每个位置的下一个真实物品 证样本
        neg = np.zeros([self.maxlen + 1], dtype=np.int32) # 每个位置的负样本
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32) # 区分user/item token
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32) # 下一个token的类型
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        # 序列中每个元素的特征
        seq_feat = np.empty([self.maxlen + 1], dtype=object) 
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set() # 存用户交互过的所有物品id，用于负采样排除
        for record_tuple in ext_user_sequence: 
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i) # 填充缺失特征保证模型不报错
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i # 当前token的id
            token_type[idx] = type_ # 当前token的类型
            next_token_type[idx] = next_type # 下一个token的类型
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0: # 下一个token是物品，next-i为下一个物品的id号，排除0是为了排除填充位只给真实的用户行为生成正负样本
                pos[idx] = next_i # 下一个交互的物品为正样本
                pos_feat[idx] = next_feat 
                neg_id = self._random_neq(1, self.itemnum + 1, ts) # 生成负样本id
                neg[idx] = neg_id # 负样本id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1 # 指针前移，左填充，从后往前填序列
            # 先处理用户最新的行为，放在数组最后面；再处理用户更早的行为放在数组前面
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        # 返回模型需要的输入数据
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息：终极合体版
        解决 102 价格特征不在 indexer 里的报错问题
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        
        # 1. 获取 indexer 里的字段（用于 brand/category 等需要查表的特征）
        actual_indexer_feats = list(self.indexer.get('f', {}).keys())
        print(f"🔍 indexer 字典中包含的字段: {actual_indexer_feats}")

        # 2. 静态与动态结合
        feat_types['user_sparse'] = []
        # 💡 这里我们确定 100, 101, 102, 103 都要，不管是探测到的还是手动算的
        feat_types['item_sparse'] = ['100', '101', '102', '103'] 
        feat_types['item_array'] = []
        feat_types['user_array'] = []
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        # 3. 离散型特征统计 (重点修复区)
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id == '102':
                # 🛡️ 价格分桶是 0-5，直接给大小 6，不查 indexer
                feat_statistics[feat_id] = 6 
            else:
                # 检查一下，如果不在 indexer 里就给个保底 0
                if feat_id in actual_indexer_feats:
                    feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
                else:
                    feat_statistics[feat_id] = 0
                    print(f"⚠️ 警告：特征 {feat_id} 既不是 102，也不在 indexer 中！")
            
        # 4. 多模态向量 (保持你最得意的那个修复逻辑)
        for feat_id in feat_types['item_emb']:
            current_vals = list(self.mm_emb_dict[feat_id].values())
            if len(current_vals) > 0:
                dim = current_vals[0].shape[0]
                print(f"✅ 多模态特征 {feat_id} 维度确认: {dim}")
            else:
                dim = 512
                print(f"⚠️ 警告：多模态特征 {feat_id} 为空，使用保底 512 维")
            
            feat_default_value[feat_id] = np.zeros(dim, dtype=np.float32)
            feat_statistics[feat_id] = dim

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值
        【梦蔚修复版：解决 User ID 和 Item ID 混用导致的 KeyError】
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
            
        for feat_id in self.feature_types['item_emb']:
            # 💡 核心修复：先判断这个 item_id 是否真的在商品映射字典里存在！
            # 防止把二十多万的用户 ID 当成商品 ID 去查字典导致崩溃
            if item_id != 0 and item_id in self.indexer_i_rev:
                orig_id = self.indexer_i_rev[item_id]
                if orig_id in self.mm_emb_dict[feat_id]:
                    if type(self.mm_emb_dict[feat_id][orig_id]) == np.ndarray:
                        filled_feat[feat_id] = self.mm_emb_dict[feat_id][orig_id]

        return filled_feat

    @staticmethod
    def collate_fn(batch):  # 批量处理
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集 只需要用户历史，不需要标签（和训练集的不同）
    做的事情：
    1.加载测试集用户序列。   2.处理冷启动（训练集没见过的用户/物品/特征）
    3.把序列变成模型能吃的固定长度 左填充。4.返回原是user-id，方便最后算指标

    复用大部分逻辑，重写 _load_data_and_offsets，加载测试集的序列文件 predict_seq.jsonl 和偏移量文件
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self): # 加载测试集文件 predict_seq.jsonl
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。  教程4.4
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户原始序列

        ext_user_sequence = [] # 扩展序列
        # ----user token和item token混排----------
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            # 💡 梦蔚修复：强制判断 is not None，防止空字典 {} 被判定为 False 导致行为丢失
            if u and user_feat is not None:    
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type)) 
            if i and item_feat is not None:    
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 🛡️ 终极安全网：如果遇到极端脏数据导致序列依然为空，塞一个兜底动作防崩溃
        if len(ext_user_sequence) == 0:
            ext_user_sequence.append((0, {}, 1, 0))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32) # 序列id，padding用0填充
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
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1 
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        # 返回模型需要的输入数据
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)




# 修小数据集bug版
def load_mm_emb(mm_path, feat_ids):
    SHAPE_DICT = {"512": 512, "81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        if feat_id != '81':
            path = Path(mm_path, f'item_img_emb_{feat_id}.pkl')
            if path.exists():
                with open(path, 'rb') as f:
                    emb_dict = pickle.load(f)
            else:
                print(f"❌ 找不到文件: {path}")
                emb_dict = {}
        else:
            shape = SHAPE_DICT[feat_id]
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        
        # 🌟 核心归一化 + 清洗逻辑
        print(f"🛠️ 正在为 #{feat_id} 特征进行 L2 归一化...")
        nan_count = 0
        for k in emb_dict:
            v = emb_dict[k]
            # 检测 NaN/Inf
            if np.isnan(v).any() or np.isinf(v).any():
                nan_count += 1
                if nan_count <= 5:
                    print(f"⚠️ [{feat_id}] key={k} 包含 NaN/Inf，将替换为0并归一化")
            # 清洗 + 归一化
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                emb_dict[k] = v / norm
            else:
                emb_dict[k] = np.zeros_like(v)
        if nan_count > 0:
            print(f"⚠️ 特征 {feat_id} 中共发现 {nan_count} 个异常向量，已全部安全处理")
        
        # ✅ 最关键的一行：把处理后的特征字典存起来
        mm_emb_dict[feat_id] = emb_dict
        print(f'✅ Loaded & Normalized #{feat_id} mm_emb, count: {len(emb_dict)}')
        
    return mm_emb_dict