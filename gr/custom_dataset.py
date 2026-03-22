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
            item2token_dict: item到token的映射字典
            tokenizer: 文本tokenizer
            data_args: 包含所有数据参数的命名空间对象
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.token_depth = data_args.token_depth
        self.tokenizer.padding_side = data_args.padding_side
        self.max_seq_length = data_args.max_seq_length
        self.max_imp_seq_length = data_args.max_imp_seq_length
        self.max_click_seq_length = data_args.max_click_seq_length
        self.item2token_dict = item2token_dict
        self.response_flag = data_args.response_flag
        
        # 计算total_seq_length（保持与原逻辑完全一致）
        self.total_seq_length = self.max_seq_length * self.token_depth + self.max_imp_seq_length * self.token_depth + self.max_click_seq_length * self.token_depth + self.token_depth + 2 * 1

    def _stream_json_data(self):
        """流式读取JSON文件并生成记录"""
        with open(self.json_path, 'r') as f:
            # 先读取整个JSON（大文件建议使用ijson等流式JSON解析器）
            data = json.load(f)
            for user_id, item_sequence in data.items():
                if len(item_sequence) < 2:  # 至少需要1个历史行为和1个目标
                    continue
                
                # 构造与原Parquet格式兼容的字典
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
        """保持与原有完全相同的处理逻辑"""
        user_behavior_list = np.array(example['user_behavior_list'])
        
        target_item = ''
        if len(user_behavior_list) == 0:
            return None
        
        for i in range(len(user_behavior_list)-1, -1, -1):
            item = str(user_behavior_list[i])
            if item in self.item2token_dict:
                target_item = item
                break
        
        if not target_item:
            return None
        
        input_user_behavior_list = []
        for user_behavior in user_behavior_list:
            user_behavior_key = str(user_behavior)
            if user_behavior_key == target_item:
                continue
            if user_behavior_key in self.item2token_dict:
                input_user_behavior_list.append(self.item2token_dict[user_behavior_key])
        
        input_user_behavior_list = input_user_behavior_list[-self.max_seq_length:]
        
        target_item_seid = self.item2token_dict[target_item]
        history_behavior_input = '<|hist_clk_start|>' + ''.join(input_user_behavior_list) + '<|hist_clk_end|>'
        full_inputs = history_behavior_input + target_item_seid
        target = history_behavior_input

        result = self.tokenizer(
            text=full_inputs,
            text_target=target,
            padding='max_length',
            max_length=self.total_seq_length,
            truncation=True
        )

        labels = copy.deepcopy(result["input_ids"])
        labels = [
            -100 if labels[i] == self.tokenizer.pad_token_id or result['labels'][i] != self.tokenizer.pad_token_id 
            else labels[i] 
            for i in range(len(labels))
        ]
        result['labels'] = labels

        return result

    def __iter__(self):
        """流式迭代器实现"""
        for example in self._stream_json_data():
            out_put = self._process_data_model_log(example)
            if out_put:
                yield out_put


