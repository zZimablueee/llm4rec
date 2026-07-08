from torch.utils.data import IterableDataset, Dataset
import copy
import torch
import os
import pandas as pd
import random
import numpy as np
import json


class CustomTrainDataset(IterableDataset):
    def __init__(self, json_path, item2token_dict, tokenizer, data_args):
        """
        流式JSON数据加载类，保持与原Parquet处理相同的接口
        Args:
            json_path: JSON文件路径（支持本地路径或网络路径）
            item2token_dict: item到token的映射字典。 商品id到语义id token的映射 过滤冷启动商品
            tokenizer: 文本tokenizer
            data_args: 包含所有数据参数的命名空间对象
        """
        self.json_path = json_path 
        self.tokenizer = tokenizer
        self.token_depth = data_args.token_depth
        self.tokenizer.padding_side = data_args.padding_side
        self.max_seq_length = data_args.max_seq_length # 主用户点击序列的最大长度
        self.max_imp_seq_length = data_args.max_imp_seq_length # 用户曝光序列的最大长度
        self.max_click_seq_length = data_args.max_click_seq_length # 额外点击序列的最大长度
        self.item2token_dict = item2token_dict
        self.response_flag = data_args.response_flag
        
        # 计算total_seq_length（保持与原逻辑完全一致）。 教程5.3
        self.total_seq_length = self.max_seq_length * self.token_depth + self.max_imp_seq_length * self.token_depth + self.max_click_seq_length * self.token_depth + self.token_depth + 2 * 1
        # ← 新加这两行，预读数据用于计算长度
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)
        # 过滤掉 len < 2 的用户（和 _stream_json_data 里的逻辑一致）
        self._data_len = sum(1 for seq in raw_data.values() if len(seq) >= 2)

    def __len__(self):
        return self._data_len

    def _stream_json_data(self):
        """流式读取JSON文件并生成记录"""
        with open(self.json_path, 'r') as f:
            # 先读取整个JSON（大文件建议使用ijson等流式JSON解析器）
            data = json.load(f)
            for user_id, item_sequence in data.items():
                if len(item_sequence) < 2:  # 至少需要1个历史行为和1个目标，相当于过滤len(item_sequence)<2的用户 见教程5.2
                    continue
                
                # 构造与原Parquet格式兼容的字典 yield迭代器每次只处理一个用户的数据，不会把所有用户数据都加载到内存
                yield {
                    'user_id': user_id,
                    'user_behavior_list': np.array(item_sequence),
                    'main_imp_item_list': np.array([]),
                    'main_click_item_list': np.array([]),
                    'similar_imp_item_list': np.array([]),
                    'similar_click_item_list': np.array([]),
                    'guess_imp_item_list': np.array([]),
                    'guess_click_item_list': np.array([]),
                    'user_imp_item_list': np.array([]),
                    'user_click_item_list': np.array([])
                }

    def _process_data_model_log(self, example):
        user_behavior_list = np.array(example['user_behavior_list'])
        
        target_item = ''
        if len(user_behavior_list) == 0:
            return None
        
        # -----1. 选择目标商品---------
        for i in range(len(user_behavior_list)-1, -1, -1):
            item = str(user_behavior_list[i])
            if item in self.item2token_dict:
                target_item = item
                break
        if not target_item:
            return None
        
        # ------2. 构造历史序列---------
        input_user_behavior_list = []
        for user_behavior in user_behavior_list:
            user_behavior_key = str(user_behavior)
            if user_behavior_key == target_item:
                continue
            if user_behavior_key in self.item2token_dict:
                input_user_behavior_list.append(self.item2token_dict[user_behavior_key])
        input_user_behavior_list = input_user_behavior_list[-self.max_seq_length:]
        
        # ------3. 拼接prompt---------
        target_item_seid = self.item2token_dict[target_item]
        history_behavior_input = '<|hist_clk_start|>' + ''.join(input_user_behavior_list) + '<|hist_clk_end|>'
        full_inputs = history_behavior_input + target_item_seid

        # -----4. tokenize + 标签掩码（修复版）---------
        # 先单独tokenize prompt，拿到准确的prompt长度
        prompt_ids = self.tokenizer(
            history_behavior_input,
            add_special_tokens=False
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # 只tokenize完整序列，不传text_target
        result = self.tokenizer(
            full_inputs,
            padding='max_length',
            max_length=self.total_seq_length,
            truncation=True,
            add_special_tokens=False
        )

        input_ids = result["input_ids"]

        # 构造labels：prompt部分和pad部分设为-100，只保留目标token位置
        labels = copy.deepcopy(input_ids)
        for i in range(len(labels)):
            if i < prompt_len or labels[i] == self.tokenizer.pad_token_id:
                labels[i] = -100
        
        # 临时诊断：检查第一个样本的 labels 是否正确
        if not hasattr(self, '_debug_printed'):
            print("pad_token_id:", self.tokenizer.pad_token_id)
            self._debug_printed = True
            print("\n======= [DEBUG] 第一个训练样本的 labels 验证 =======")
            print("history_behavior_input:", history_behavior_input)
            print("target_item_seid:", target_item_seid)
            print("prompt_len:", prompt_len)
            # 打印 prompt 部分的 token
            prompt_tokens = self.tokenizer.decode(input_ids[:prompt_len])
            print("prompt tokens:", prompt_tokens)
            # 打印目标部分的 token (只取非 pad 部分)
            target_end = prompt_len + 3  # 目标 token 通常3个
            target_tokens = self.tokenizer.decode([tid for tid in input_ids[prompt_len:target_end] if tid != self.tokenizer.pad_token_id])
            print("target tokens (from input_ids):", target_tokens)
            # 打印 labels 中非 -100 的部分对应的 token
            label_ids = [tid for i, tid in enumerate(labels) if tid != -100]
            label_tokens = self.tokenizer.decode(label_ids)
            print("labels' non-masked tokens:", label_tokens)
            # 打印 labels 中 -100 的分布（前几个和后几个）
            print("labels[:20]:", labels[:20])
            print("==========================================\n")
        result['labels'] = labels
        return result


    def __iter__(self): # iterable dataset的迭代器实现
            """流式迭代器实现"""
            for example in self._stream_json_data():
                out_put = self._process_data_model_log(example)
                if out_put:
                    yield out_put


