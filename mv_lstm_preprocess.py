#!/usr/bin/env python
# coding: utf-8

# # 多変量LSTM: データ整形
# - nn.Dataset多変量用の入力データを作成
# 
# ### 実装する機能
# - dataframeの統合: OK
# - dataframeの共通部分を抽出: OK
# - torch tensorに変換: OK
# - データの標準化: OK
# - 面倒なので飛んでいる日付に関するデータの補間はまだやっていません

# In[1]:


import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import torch.utils.data as data


# In[2]:


class mv_dataset(data.Dataset):
    def __init__(self, file_paths, seq_len = 50, predict_len = 10, predict_data = [0]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.predict_data = predict_data
        self.data_mean, self.data_std = None, None

        self.raw_dfs = [pd.read_csv(file_path, index_col=0) for file_path in self.file_paths] # list of dataframes
        self.dfs = self.combine_dfs() # dataframe
        self.data = self.dfs.values # ndarray, shape = (self.dfs.shape[0], len(dfs))
        self.normalized_data = self.normalize(self.data, axis=0) # normalized among each column
        self.data_size = self.data.shape[0] - self.seq_len - self.predict_len

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        seq = [self.normalized_data[index + i] for i in range(self.seq_len)]
        label = [self.normalized_data[index + self.seq_len + i, self.predict_data] for i in range(self.predict_len)]
        return torch.tensor(seq, dtype=torch.float32).to(self.device), torch.tensor(label, dtype=torch.float32).to(self.device)

    def normalize(self, x, axis = None):
        self.data_mean = x.mean(axis=axis, keepdims=True)
        self.data_std  = np.std(x, axis=axis, keepdims=True)
        return (x - self.data_mean) / self.data_std

    def combine_dfs(self):
        dfs = self.raw_dfs[0]
        for df in self.raw_dfs[1:]:
            dfs = dfs.join(df, how='inner')
        return dfs
    
    def interpolation(self):
        # 日付が飛んでいる部分を補完
        # 未実装
        pass


# In[3]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'mv_lstm_preprocess.ipynb'])

