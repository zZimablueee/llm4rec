from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from qwenrec.sasrec.dataset0 import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads # d_k
        self.dropout_rate = dropout_rate 

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        # (batch, seq_len, hidden_units) → (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, 
                attn_mask=attn_mask.unsqueeze(1) # 把 mask 为 False 的位置的注意力分数设为负无穷，softmax 之后就变成 0 了，模型就不会关注这些位置了
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5 # 缩放因子，1/sqrt(d_k)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale #Q *K^T / sqrt(d_k)，形状为（batch， num_heads， seq_len， seq_len）

            # 应用注意力掩码，mask掉的位置的分数设为负无穷，使其softmax后权重为0
            # attn_mask在后面baseline model的log2feats函数中构造
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1) # 计算注意力权重，形状为（batch， num_heads， seq_len， seq_len）
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V) # 注意力权重与V相乘得到输出，形状为（batch， num_heads， seq_len， head_dim）

        # reshape回原来的格式 （把多头的结果拼回去，恢复到原来的维度）
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        
        # 用两个1*1 conv1d代替传统两层linear
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs形状为（batch， seq_len， hidden_units），需要转置为（batch， hidden_units， seq_len）才能输入Conv1d
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # 输出再转置回原来的格式（batch， seq_len， hidden_units）
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

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
        batch_size, seq_len, _ = query.size()

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 稳定版注意力计算
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attn_mask is not None:
            # 用 -1e9 替代 -inf，避免产生 NaN
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), -1e9)

        # 减去最大值，防止 softmax 溢出
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores = scores - scores_max

        attn_weights = F.softmax(scores, dim=-1)
        # 兜底：万一有 NaN 就置 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_linear(attn_output)

        return output, None


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
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

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

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

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"512":512, "81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
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
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)   #序列id转入gpu/cpu
        # pre-compute embedding
        # ID Embedding 查表
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            # 存成列表，方便后续拼接其他特征
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        # 处理离散特征
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                # 调用feat2tensor将原始list数据转化为tensor
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    # 离散特征：[batch_size, maxlen] -> [batch_size, maxlen, hidden_units]
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    # 数组特征：
                    # [batch_size, maxlen, max_array_len] -> [batch_size, maxlen, hidden_units]，对数组维度进行sum pooling
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # 连续特征：[batch_size, maxlen] -> [batch_size, maxlen, 1]
                    feat_list.append(tensor_feature.unsqueeze(2))

        # 多模态特征对齐
        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array) #一次处理用户数量
            emb_dim = self.ITEM_EMB_FEAT[k] #多模态特征的原始维度
            seq_len = len(feature_array[0]) #用户行为序列的长度

            # pre-allocate tensor
            # 存放所有多模态向量 （用户数， 序列长度， 多模态原始维度）
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array): #遍历每个用户的行为序列
                for j, item in enumerate(seq): #遍历用户行为序列中的每个item
                    if k in item: #如果这个item有这个多模态特征
                        batch_emb_data[i, j] = item[k] #将多模态特征的原始向量放入预分配的numpy数组中，形状为（用户数，序列长度，多模态原始维度）

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev) #将numpy数组转换为tensor并转入GPU
            item_feat_list.append(self.emb_transform[k](tensor_feature)) #用线性层emb_transform，把【高维原始向量】→ 映射成【模型固定维度hidden_units】

        # merge features
        # 拼接特征 id向量+类别向量+多模态对齐向量
        #all_item_emb = torch.cat(item_feat_list, dim=2)
        # itemdnn压缩维度，relu提纯特征
        #all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        #if include_user:
        #    all_user_emb = torch.cat(user_feat_list, dim=2)
         #   all_user_emb = torch.relu(self.userdnn(all_user_emb))
            # 用户与item的特征融合，为相加！
            # 商品内部不同特征语义完全不同，用拼接；而用户和商品特征语义互补，用相加 不改变向量维度且计算开销更小
          #  seqs_emb = all_item_emb + all_user_emb
        #else: #导出向量给rqvae时走else分之，只需要商品本身的静态向量
         #   seqs_emb = all_item_emb
        # 返回同意维度 融合所有信息的向量seqsemb

        # ========== 修复版:拼接 + 投影 + 归一化 ==========
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        # all_item_emb = F.layer_norm(all_item_emb, all_item_emb.shape[-1:])  # ← 关键

        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            # all_user_emb = F.layer_norm(all_user_emb, all_user_emb.shape[-1:])
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    
    def log2feats(self, log_seqs, mask, seq_feature):
        def stat(name, t):
            if torch.isnan(t).any():
                print(f"❌ {name} has NaN")
            elif torch.isinf(t).any():
                print(f"⚠️ {name} has Inf")
            else:
                print(f"✅ {name} min={t.min().item():.4f} max={t.max().item():.4f}")

        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        # stat("feat2emb output", seqs)                          # ← 检查点1

        seqs *= self.item_emb.embedding_dim ** 0.5
        # stat("after scaling", seqs)                            # ← 检查点2

        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        # stat("after pos_emb", seqs)                            # ← 检查点3

        seqs = self.emb_dropout(seqs)
        # stat("after dropout", seqs)                            # ← 检查点4

        # 构造 attention mask（保持不变）
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                # stat(f"after attn block {i}", seqs)            # ← 检查点5

                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
                # stat(f"after ffn block {i}", seqs)             # ← 检查点6
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                # stat(f"after attn block {i}", seqs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
                # stat(f"after ffn block {i}", seqs)

        log_feats = self.last_layernorm(seqs)
        # stat("final log_feats", log_feats)                     # ← 检查点7

        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        # 1. 计算用户序列的表征
        log_feats = self.log2feats(user_item, mask, seq_feature)

        # 2. 生成loss mask，只有当下一个token是item且动作类型为点击时才计算loss
        loss_mask = (next_mask == 1).to(self.dev)

        # 3. 计算正负样本的表征
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # 4. 计算正负样本的logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # 5. 用loss mask过滤无效位置的损失
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        # 线上推理阶段用，输入用户历史行为序列，输出用户的偏好表征，以和商品embedding匹配生成推荐结果
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        # 同forward函数第一步，计算用户序列的表征，log2feats为先前定义的，通过transformer编码得到每个时间步的用户序列特征，包含用户到当前位置的所有历史行为信息为后续打分基础
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        # 取最后一个时间步的特征作为用户的最终表征，进行推荐打分
        final_feat = log_feats[:, -1, :] # batch维度全取， seq_len取最后一个， hidden_units全取
        # 取最后一个是因为sasrec为自回归序列模型，最后一个时间步的特征已经融合了用户历史行为序列的全部信息，可以看作是用户当前的兴趣表征，直接用来和商品embedding进行匹配打分，生成推荐结果

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            # faiss需要输入uint64类型的ID，所以需要两套ID体系，训练时用原始ID，检索时用re-id
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []
        # 分批遍历所有商品
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids)) #确定当前批次结束位置

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0) # 当前批次的原始商品id转化为gpu张量
            # 拿到当前批次商品的所有特征（类别 图片 多模态向量），构造成feat2emb需要的输入格式（列表套字典），并转化为numpy数组
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)
            # 生成商品静态向量，只调用feat2emb，不用transformer
            # include=False 纯商品向量不带用户信息
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            # 将向量丛gpu转到cpu，存入列表
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        # 保存为npz文件，用于rq-vae部分
        np.savez(
            Path(save_path, 'embeddings.npz'),
            ids=final_ids.flatten(),  # 形状 (N,) 存连续检索ID数组
            embs=final_embs          # 形状 (N, D) 存商品embedding数组
        )
