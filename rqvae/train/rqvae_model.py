import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

class MLPLayers(nn.Module):

    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn # batch norm

        mlp_modules = []
        # 遍历层维度，搭网络
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            # 论文中提到，最后一层不使用BN和激活函数 （因为最后一层是量化层的输入，直接输出特征向量），所以这里加了条件判断
            # 如果use_bn为True且当前层不是倒数第二层（即不是最后一层的前一层），则添加BatchNorm1d层
            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)
        # 所有层打包成序列
        self.mlp_layers = nn.Sequential(*mlp_modules)
        # 初始化权重
        self.apply(self.init_weights)
    # 权重初始化函数，使用xavier_normal_初始化线性层的权重，偏置初始化为0
    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                # bias不需要随机，设成0，模型会自己学
                module.bias.data.fill_(0.0)
    # 前向传播函数，输入特征通过MLP层得到输出
    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):
    # 输入激活函数名称，返回对应的激活函数实例
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

def kmeans(   # 码本初始化工具，输入样本数据，输出聚类中心作为码本向量
    samples,  # 输入样本数据，形状为 (B, dim)，B为样本数量，dim为特征维度
    num_clusters, # 聚类中心数量，即码本向量数量
    num_iters = 10, # k-means算法迭代次数，默认为10
):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers # 形状为 (num_clusters, dim)，num_clusters为码本向量数量，dim为特征维度


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    # 满足1. 每行元素之和为1/B 2. 每列元素之和为1/K的约束条件下，计算距离矩阵的相似度概率矩阵

    # distances[i][j] 样本i到中心j的距离 Q为概率矩阵，Q[i][j]表示样本i分配给中心j的概率
    Q = torch.exp(- distances / epsilon) # 计算相似度概率 形状为 (B, K)，B为样本数量，K为中心数量
    B = Q.shape[0] # number of samples to assign 样本数
    K = Q.shape[1] # how many centroids per block (usually set to 256) 中心数

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q # 先整体归一化，使得所有元素之和为1
    # print(Q.sum())
    # sinkhorn迭代，每次迭代先按列归一化，再按行归一化，交替进行，直到满足行和为1/B，列和为1/K的约束条件
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
           # 乘之Q到行和为1，列和为B/K。 如果不乘B, 每个样本的总概率为1/B，不符合我们对概率分布的直觉
    return Q

class VectorQuantizer(nn.Module):
    # 把编码器输出的连续隐向量，映射到码本（Codebook）里的离散向量，输出：量化后的向量、量化损失、码本索引

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,):
        super().__init__()
        self.n_e = n_e # 码本大小
        self.e_dim = e_dim # 码本向量维度
        self.beta = beta # 承诺损失系数
        self.kmeans_init = kmeans_init # 是否使用k-means初始化码本向量
        self.kmeans_iters = kmeans_iters # k-means算法迭代次数
        self.sk_epsilon = sk_epsilon # sinkhorn算法的温度参数，控制分配概率的平滑程度，值越小分配越稀疏，值越大分配越平滑
        self.sk_iters = sk_iters # sinkhorn算法迭代次数

        self.embedding = nn.Embedding(self.n_e, self.e_dim) # 码本，包含n_e个e_dim维的向量
        # 初始化逻辑
        if not kmeans_init:
            self.initted = True #已初始化
            # 码本向量随机初始化，范围为[-1/e_dim, 1/e_dim]
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False #未初始化，等到第一次前向传播时用k-means算法根据输入数据初始化码本向量
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        # 获取码本所有向量，形状为 (n_e, e_dim)，可导出查看调试
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None): 
        # 输入离散索引，输出索引对应的码本向量
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape) # 将码本向量重塑为输入的形状，方便后续计算损失和反向传播

        return z_q

    def init_emb(self, data):
        # kmeans初始化码本，只在训练开始时执行一次。比随机初始化手链快

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers) # 将聚类得到的中心向量作为码本向量，复制到embedding层的权重中
        self.initted = True # 标记为已初始化，后续前向传播不再执行k-means初始化

    @staticmethod
    def center_distance_for_constraint(distances):
        # sinkhorn算法前的距离归一化，将原始距离缩放到[-1, 1]附近，防止数值爆炸/消失，保证算法稳定性
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input 展平输入向量
        # x形状：[batch_size, e_dim],展平后形状：[总样本数量, e_dim]，适配码本计算
        latent = x.view(-1, self.e_dim)

        # kmeans初始化，仅在训练时执行1次。
        # 如果没初始化+处于训练模式， 调用kmeans初始化码本
        if not self.initted and self.training:
            self.init_emb(latent)

        # 计算 输入向量 和 所有码本向量 的欧氏距离平方 Calculate the L2 Norm between latent and Embedded weights
        # ||x - e||² = ||x||² + ||e||² - 2·x·e
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        # 选择匹配方式：硬匹配/sinkhorn软匹配
        if not use_sk or self.sk_epsilon <= 0:
            # 硬匹配：找到距离最近的码本向量，返回对应索引 （VQ-VAE的原始做法）
            indices = torch.argmin(d, dim=-1)
        else:
            # sinkhorn软匹配
            d = self.center_distance_for_constraint(d) #归一化距离
            d = d.double() #双精度
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1) # 选择概率最大的码本向量作为匹配结果，返回对应索引

        # indices = torch.argmin(d, dim=-1)
        # 根据索引获取码本向量，重塑为输入的形状，得到量化后的向量x_q
        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding 计算VQ核心损失
        # 码本损失： 让码本向量靠近编码器输出，梯度传给码本 更新码本向量
        commitment_loss = F.mse_loss(x_q.detach(), x)
        # 承诺损失： 让编码器输出靠近码本向量，梯度传给编码器 更新编码器参数
        codebook_loss = F.mse_loss(x_q, x.detach())
        # 总损失 = 码本损失 + beta * 承诺损失，beta控制两者权重，通常设置为0.25
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients 通过直通估计器（Straight-Through Estimator）实现量化操作的可微性，梯度直接传递给编码器输出x
        x_q = x + (x_q - x).detach()
        # 输出结果
        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

class ResidualVectorQuantizer(nn.Module):
    # 多层残差量化器，堆叠多个单层的VectorQuantizer，每层量化后计算残差，下一层量化残差，最终输出量化后的向量、平均损失、所有层的码本索引
    # 输入：编码器输出的连续隐向量  输出：总量化向量、平均损失、所有层的码本索引
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, # 每层量化器的码本大小，列表形式
                 e_dim, #向量维度，所有层共用
                 sk_epsilons, # 每层的sinkhorn温度系数，和n_e_list长度一致
                 beta = 0.25, # 承诺损失系数，和单层量化器一致
                 kmeans_init = False, # 是否使用k-means初始化码本向量，和单层量化器一致
                 kmeans_iters = 100, # 码本初始化参数，和单层量化器一致
                 sk_iters=100, # sinkhorn算法迭代次数，和单层量化器一致
                 ):
        super().__init__()
        self.n_e_list = n_e_list # 保存每层码本大小
        self.e_dim = e_dim #统一向量维度
        self.num_quantizers = len(n_e_list) # 量化层数 = 列表长度
        # 共享超参数
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        # 堆叠N个单层vector quantizer
        # zip(n_e_list, sk_epsilons) 一一对应，每层用自己的码本大小+sinkhorn参数
        # 所有层共享 beta/e_dim，但每层有独立的码本（互不干扰）
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        # 收集所有层的码本向量
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook() # 调用单层VQ的get_codebook方法，获取当前层的码本向量，形状为 (n_e, e_dim)
            all_codebook.append(codebook)
        # 堆叠成张量： [num_quantizers层数, n_e码本大小, e_dim向量维度]
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        # 初始化空列表：收集每层的量化损失和每层的索引
        all_losses = []
        all_indices = []

        # 残差量化初始化
        x_q = 0 # 最终的总量化向量（初始为0）
        residual = x # 初始残差 = 输入的原始向量（第一层量化原始向量）
        # 遍历每层的量化器，逐层做残差量化
        for quantizer in self.vq_layers:
            # 对 当前残差 做单层向量量化
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            # 更新残差： 残差 = 上一轮残差 - 本轮量化结果（即剩下没拟合的误差）
            residual = residual - x_res
            # 叠加量化结果： 总量化向量 = 上一轮总量化向量（之前的结果） + 本轮量化结果
            x_q = x_q + x_res

            # 保存本轮的损失和索引
            all_losses.append(loss)
            all_indices.append(indices)

        # 损失处理： 所有层的损失求平均 
        # 每层都有自己的VQ损失，取平均可以使训练更稳定，不会某一层损失过大主导训练
        mean_losses = torch.stack(all_losses).mean()
        # 索引处理： 把每层的索引堆叠在一起 形状为[batch, 序列长度， 层数]
        # 单层 VQ 输出：1 个索引（0-255）；4 层 RQ 输出：4 个索引 如 [123, 45, 67, 89]；即RQ-VAE 的离散编码，比单层索引表达能力强上万倍
        all_indices = torch.stack(all_indices, dim=-1)
        # 输出：总量化向量、平均损失、所有层的索引
        return x_q, mean_losses, all_indices

class RQVAE(nn.Module):
    # RQ-VAE模型，包含一个MLP编码器，一个残差量化器RQ，一个MLP解码器
    def __init__(self,
                 in_dim=768, # 输入维度：商品向量维度，sasrec导出的
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None, # 残差量化每层码本大小
                 e_dim=64, # 量化隐向量维度 压缩后的64维
                 # layers=[512,256,128],
                 layers=None, # mlp编码器/解码器 的隐藏层
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse", # 重构损失类型，mse/l1
                 quant_loss_weight=1.0, # 量化损失权重，控制重构损失和量化损失的相对重要性，通常设置为1.0
                 beta=0.25, # 承诺损失系数，控制码本损失和承诺损失的权重，通常设置为0.25
                 kmeans_init=False, # 是否使用k-means初始化码本向量
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        # 保存所有超参数
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # MLP编码器 高维输入 -> 低维隐向量
        # 编码器层维度：输入维度 -> 隐藏层1 -> 隐藏层2 -> ... -> 量化维度    768 -> 512 -> 256 -> 128 -> 64
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        
        # 残差向量量化器RQ
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        
        # 解码器MLP 低纬隐向量 -> 高维输出
        # 解码器维度（编码器的逆序）: 量化维度 -> 隐藏层2 -> 隐藏层1 -> 输出维度    64 -> 128 -> 256 -> 512 -> 768
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        # 编码 768维商品向量 -> 64维隐向量
        x = self.encoder(x)
        # 残差量化 64维隐向量 -> 64维量化向量 + 量化损失 + 多层索引
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        # 解码 64维量化向量 -> 768维重构向量
        out = self.decoder(x_q)
        # 输出：重构向量、量化损失、所有层的索引
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        # 推理模式 禁用梯度计算
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        # 输出：多层离散索引 即需要的 语义ID/SID
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        # 计算重构损失 让模型输入=原始输入
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        # 总损失 = 重构损失 + 量化损失权重 * 量化损失，量化损失权重控制两者的相对重要性，通常设置为1.0（前面定义过）
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        # 输出：总损失、重构损失
        return loss_total, loss_recon