from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb

# 多头注意力
# 输入是一排向量（比如 4 个位置，每个位置一个 128 维向量），每个位置去“问”前面的所有位置：“你们有什么信息？”，
# 根据前面的回答，更新自己的表示，输出仍然是一排向量（形状不变）
# Flash:代码里做了数值稳定性优化，防止某些位置的向量变成 NaN。
# 和真正意义上的 Flash Attention 不完全一样，但你只需要知道：它比原版实现更稳定、更省显存。
class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        assert hidden_units % num_heads == 0

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        B, L, _ = query.size()

        Q = self.q_linear(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ===== 关键修复：保护"全 False 行"，避免 softmax(-inf) → NaN =====
        # attn_mask: [B, L, L] (bool, True 表示可见)
        if attn_mask is not None:
            # 给每一行的对角线强行置 True，保证至少有一个可见 token
            # （对应 padding 位置的 query 也能产出有限值，不会污染网络）
            eye = torch.eye(L, dtype=torch.bool, device=attn_mask.device).unsqueeze(0)
            safe_mask = attn_mask | eye  # [B, L, L]
            safe_mask = safe_mask.unsqueeze(1)  # [B, 1, L, L]
        else:
            safe_mask = None

        # 强制使用稳定的手写实现（统一 fp32 softmax，避免 SDPA 在 fp16/bf16 下的边界问题）
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, L, L]

        if safe_mask is not None:
            # 用大负数代替 -inf，进一步降低数值风险
            scores = scores.masked_fill(~safe_mask, -1e4)

        # softmax 在 fp32 下计算，回投到原 dtype
        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_units)
        return self.out_linear(attn_output), None


# 实现“前馈网络”
# 输入：一个位置的 128 维向量.
# 操作：放大重要的维度的值，抑制不重要的;
# 输出：仍然是 128 维向量
# kernel_size=1 的 Conv1d 和全连接层数学等价，但在某些硬件上效率更高。
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.relu = torch.nn.GELU()  # GELU 比 ReLU 在小信号下不易塌缩
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)

    def forward(self, inputs):
        x = inputs.transpose(-1, -2)
        x = self.conv2(self.dropout(self.relu(self.conv1(x))))
        return x.transpose(-1, -2)


# “搭建模型骨架”
# 在正式训练前，先把所有需要的“零件”都造出来，放在一边。
class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        # eps 调到 1e-6，混合精度更稳
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-6)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-6))
            self.attention_layers.append(
                FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-6))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

        # ===== 关键修复：合理初始化，防止向量塌缩 =====
        self._reset_parameters(args.hidden_units)

    # 类似 BERT 的初始化策略：用标准差 0.02 的正态分布初始化，同时强制 padding_idx=0 的行保持全零
    def _reset_parameters(self, hidden_units):
        """统一初始化，避免 sqrt(d) 缩放后方差爆炸/塌缩。"""
        std = 0.02  # 类 BERT 初始化
        for emb in [self.item_emb, self.user_emb, self.pos_emb]:
            torch.nn.init.normal_(emb.weight, mean=0.0, std=std)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)  # padding_idx 保持 0

        for emb in self.sparse_emb.values():
            torch.nn.init.normal_(emb.weight, mean=0.0, std=std)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)

        for lin in [self.userdnn, self.itemdnn]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

        for lin in self.emb_transform.values():
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}

        EMB_SHAPE_DICT = {
            "81": 32, "82": 1024, "83": 3584,
            "84": 4096, "85": 3584, "86": 3584,
            "87": 512,
        }
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    # 把特征字典里的值抽出来，变成整齐的矩阵
    # Dataset 输出的 seq_feat 是一个嵌套列表，每个位置是一个字典，如 {"100":5, "101":23, ...}
    # 输入: [{"100":5}, {"100":3}, {"100":8}];输出: tensor([[5, 3, 8]])
    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)

    # 把一堆不同来源的特征，融合成每个位置一个 128 维向量
    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
            ])

        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue
            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # 连续特征做轻量 clip，防止极端值
                    cont = tensor_feature.float().unsqueeze(2).clamp(-10.0, 10.0)
                    feat_list.append(cont)

        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for i, s in enumerate(feature_array):
                for j, item in enumerate(s):
                    if k in item:
                        batch_emb_data[i, j] = item[k]
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            # 对外部预训练 emb 做 L2 归一化，避免不同模态尺度悬殊
            tensor_feature = F.normalize(tensor_feature, dim=-1)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = F.gelu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.gelu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    # 给融合后的序列加上“上下文
    def log2feats(self, log_seqs, mask, seq_feature):
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]

        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        # 注意：因为我们用了 std=0.02 的初始化，这里的 sqrt(d) 缩放仍然保留以匹配原 SASRec 设计
        seqs = seqs * (self.item_emb.embedding_dim ** 0.5)

        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss = poss * (log_seqs != 0).long()
        seqs = seqs + self.pos_emb(poss)

        # padding 位置整体清零，防止其参与残差污染
        valid = (mask != 0).to(self.dev).unsqueeze(-1).float()
        seqs = seqs * valid

        seqs = self.emb_dropout(seqs)

        L = seqs.shape[1]
        ones_matrix = torch.ones((L, L), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    # 定义训练时的一次前向传播
    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type,
                seq_feature, pos_feature, neg_feature):
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    # 推理
    def predict(self, log_seqs, seq_feature, mask):
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        return log_feats[:, -1, :]

    # 训练完后，导出所有商品的向量
    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = [feat_dict[i] for i in range(start_idx, end_idx)]
            batch_feat = np.array(batch_feat, dtype=object)
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        np.savez(
            Path(save_path, 'embeddings.npz'),
            ids=final_ids.flatten(),
            embs=final_embs,
        )
