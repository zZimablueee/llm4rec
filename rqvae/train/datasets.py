# 从.npz文件里读取物品的向量，喂给rqvae训练/推理
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os

import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
import numpy as np
import json

class CustomNpzFile(data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.embeddings = []
        self.item_ids = []

        for file in self.file_list:
            npz_data = np.load(file,allow_pickle=True)

            embeddings = npz_data['embs']
            item_id = npz_data['ids']

            if item_id.ndim == 1:
                item_id = item_id.reshape(-1, 1)

            # 把所有文件的向量拼在一起
            self.embeddings.append(embeddings)
            self.item_ids.append(item_id)
            print(f'Processed {file}')

        self.embeddings = np.vstack(self.embeddings)
        self.item_ids = np.vstack(self.item_ids)
        self.item_ids = self.item_ids.squeeze(axis=1)
        # 把多个文件里的向量 拼成一个巨大的矩阵 [总物品数， 向量维度]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index] # 取第index个物品向量
        item_id = self.item_ids[index]  # 此时为标量值 取对应id
        tensor_emb = torch.FloatTensor(emb) # 转成torch张量
        # print(type(item_id))
        # print(item_id)
        # print(type(tensor_emb))
        # print(tensor_emb)
        return tensor_emb # 返回向量给rqvae

    def __len__(self):  # 数据集长度
        return len(self.embeddings)
