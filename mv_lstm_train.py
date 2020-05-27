#!/usr/bin/env python
# coding: utf-8

# # 多変量LSTM: モデルの訓練
# 
# ### 使用するデータ
# - S&P500
# - アメリカ国債
# - 日経平均
# 
#  #### 長続きして、なおかつある程度安定しているデータがいい？
#  - 日経平均はアメリカとあんまり関係ないし、戦後からバブルにかけて差が大きすぎる？
#  - 為替相場は歴史が浅い
#  - アメリカ国内失業率は月ごとのデータなので使い方がわからない
#  - ロンドンと香港とかがあればもっと良さそう
# 
# ### 利点と展望
# - 経済の専門的な知識を必要としない
# - 使用するデータの種類(feature_vector)が多ければ多いほど、人間が気づかない予兆や規則を検出してくれるかもしれない
# - 予測に役立つ
# 
# ### 今後に向けて
# - Attention、Transformerなど、いろいろな機能を追加/変更していければ
# - このフレームワークを使って、昔の似たようなパターンを見つけてくるタスクに関してはやり方がわかっていない

# In[1]:


import sys
import inspect
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from mv_lstm_preprocess import mv_dataset
import mv_lstm_model
importlib.reload(mv_lstm_model)
from mv_lstm_model import mv_lstm, mv_lstm_same_seq


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
print(device)


# ### parameters

# In[3]:


seq_len = 50
predict_len = seq_len
predict_features = [0,1]
batch_size = 50
dropout = 0.2

epochs_num = 50
eps = 0.01
hidden_size = 20
num_layers = 2
input_size = None # 後から
output_size = len(predict_features)


# In[4]:


file_paths = ["data/market_data_rm_inflation.csv", "data/nikkei_data.csv", "data/jgbcm_9.csv"]
dataset = mv_dataset(file_paths, seq_len=seq_len, predict_len=predict_len, predict_data=predict_features)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)


# In[5]:


print(len(train_dataset))
print(len(test_dataset))


# In[6]:


input_size = dataset.dfs.shape[1]

model = mv_lstm_same_seq(input_size, 
                            output_size,
                            hidden_size=hidden_size, 
                            num_layers = num_layers, 
                            batch_size = batch_size, 
                            seq_len = seq_len, 
                            predict_len = predict_len,
                            dropout = dropout)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

train_ls, test_ls = [], []
train_as, test_as = [], []


# ## 訓練
# - ist clusterのGPUの使い方がわかりません
# - 少なくともtest_accuracyの値のスケールは正しくないです
# - cpuだととても遅いです...( hours), 20:40~

# In[7]:


for epoch in range(epochs_num):
    
    train_loss = 0.0
    train_accurate_num = 0
    for i, (batch_seqs, batch_labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(batch_seqs)
        loss = criterion(output, batch_labels).to(torch.float32)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        train_accurate_num += np.sum(np.abs((output.data - batch_labels).numpy()) < eps)

    train_accuracy = train_accurate_num / (train_size * predict_len * output_size)
    
    test_loss = 0.0
    test_accurate_num = 0
    for i, (batch_seqs, batch_labels) in enumerate(test_dataloader):
        output = model(batch_seqs)
        loss = criterion(output, batch_labels).to(torch.float32)
        test_loss += loss.data
        test_accurate_num += np.sum(np.abs((output.data - batch_labels).numpy()) < eps)

    test_accuracy = test_accurate_num / (test_size * predict_len * output_size)

    train_ls.append(train_loss)
    train_as.append(train_accuracy)
    test_ls.append(test_loss)
    test_as.append(test_accuracy)

    print("%d train_loss: %.3f, train_accuracy: %.5f, test_loss: %.3f, test_accuracy: %.5f" %(epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy))


# In[16]:


model_name = "mv_lstm_same_seq"
using_data = "_data_usgd_ussp_nikkei_jgbcm"
torch.save(model.state_dict(), model_name + ".pth")


# In[17]:


condition = model_name + using_data  + "_with_seq" + str(seq_len) + "_pred" + str(predict_len) +  "_ep" + str(epochs_num) + "_batch" + str(batch_size)+ "_hs" + str(hidden_size) + "_eps" + str(eps)
print(condition)


# In[18]:


def plot_results(result, which_result, condition, save_or_not=True):
    plt.clf()
    plt.plot(result)
    plt.title(condition + "_" + which_result)
    plt.savefig(condition + "_" + which_result + ".png") if save_or_not else plt.show()


# In[19]:


plot_results(train_ls, "train_loss", condition)


# In[20]:


plot_results(train_as, "train_accuracy", condition)


# In[21]:


plot_results(test_ls, "test_loss", condition)


# In[22]:


plot_results(test_as, "test_accuracy", condition)


# In[24]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'mv_lstm_train.ipynb'])


# ### Accuracy が1を超えるのはなぜ？
