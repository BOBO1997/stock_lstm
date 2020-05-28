#!/usr/bin/env python
# coding: utf-8

# # 多変量LSTM: モデルの訓練の続き
# 
# ### 使用するデータ
# - S&P500
# - アメリカ国債
# - 日経平均
# - 日本国債
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

# In[20]:


seq_len = 50
predict_len = seq_len
predict_features = [0,1]
batch_size = 50
dropout = 0.2

epochs_num = 50
eps = 0.2
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


# In[18]:


a = [1.3, 1.23, 1.34, 1.66, 1.56, 1.43, 1.34, 1.46, 1.56, 1.76]
b = [1.4, 1.33, 1.54, 1.65, 1.56, 1.35, 1.22, 1.32, 1.43, 1.54]
plt.clf()
plt.plot(a)
plt.plot(b)
plt.show()
print(np.sum(np.linalg.norm(np.array(a) - np.array(b), ord=2)))
print(50 * 0.08 ** 2)


# In[21]:


input_size = dataset.dfs.shape[1]

model = mv_lstm_same_seq(input_size, 
                            output_size,
                            hidden_size=hidden_size, 
                            num_layers = num_layers, 
                            batch_size = batch_size, 
                            seq_len = seq_len, 
                            predict_len = predict_len,
                            dropout = dropout)
model.load_state_dict(torch.load("mv_lstm_same_seq.pth"))
model.eval()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

train_ls, test_ls = [], []
train_as1, test_as1, train_as2, test_as2 = [], [], [], []


# ## 訓練
# - ist clusterのGPUの使い方がわかりません
# - 少なくともtest_accuracyの値のスケールは正しくない
# - 大したデータ数でもないのにcpuでやると待ち時間が長すぎる...

# In[22]:


for epoch in range(epochs_num - 30):
    
    train_loss = 0.0
    train_accurate_num1, train_accurate_num2 = 0, 0
    for i, (batch_seqs, batch_labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(batch_seqs)
        loss = criterion(output, batch_labels).to(torch.float32)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        # print(np.linalg.norm((output.data[:, :, 0] - batch_labels[:, :, 0]).numpy(), ord=2, axis=1))
        train_accurate_num1 += np.sum(np.linalg.norm((output.data[:, :, 0] - batch_labels[:, :, 0]).numpy(), ord=2, axis=1) < eps)
        train_accurate_num2 += np.sum(np.linalg.norm((output.data[:, :, 1] - batch_labels[:, :, 1]).numpy(), ord=2, axis=1) < eps)

    train_accuracy1 = train_accurate_num1 / train_size
    train_accuracy2 = train_accurate_num2 / train_size
    
    test_loss = 0.0
    test_accurate_num1, test_accurate_num2 = 0, 0
    for i, (batch_seqs, batch_labels) in enumerate(test_dataloader):
        output = model(batch_seqs)
        loss = criterion(output, batch_labels).to(torch.float32)
        test_loss += loss.data

        test_accurate_num1 += np.sum(np.linalg.norm((output.data[:, :, 0] - batch_labels[:, :, 0]).numpy(), ord=2, axis=1) < eps)
        test_accurate_num2 += np.sum(np.linalg.norm((output.data[:, :, 1] - batch_labels[:, :, 1]).numpy(), ord=2, axis=1) < eps)
        
    test_accuracy1 = test_accurate_num1 / test_size
    test_accuracy2 = test_accurate_num2 / test_size

    train_ls.append(train_loss)
    train_as1.append(train_accuracy1)
    train_as2.append(train_accuracy2)
    test_ls.append(test_loss)
    test_as1.append(test_accuracy1)
    test_as2.append(test_accuracy2)

    print("epoch %d --- [train] loss: %.3f, accuracy1: %.3f, accuracy2: %.3f --- [test] loss: %.3f, accuracy1: %.3f, accuracy2: %.3f" %(epoch + 1, train_loss, train_accuracy1, train_accuracy2, test_loss, test_accuracy1, test_accuracy2))
    


# In[23]:


img_path = "images/"
model_name = "mv_lstm_same_seq_train2"
using_data = "_data_usgd_ussp_nikkei_jgbcm"
torch.save(model.state_dict(), model_name + ".pth")


# In[24]:


condition = model_name + using_data  + "_with_seq" + str(seq_len) + "_pred" + str(predict_len) +  "_ep" + str(epochs_num) + "_batch" + str(batch_size)+ "_hs" + str(hidden_size) + "_eps" + str(eps)
print(condition)


# In[25]:


def plot_results(result, which_result, path, condition, save_or_not=True):
    plt.clf()
    plt.plot(result)
    plt.title(path + condition + "_" + which_result)
    plt.savefig(path + condition + "_" + which_result + ".png") if save_or_not else plt.show()


# In[26]:


plot_results(train_ls, "train_loss", img_path, condition)


# In[27]:


plot_results(train_as1, "train_accuracy1", img_path, condition)


# In[28]:


plot_results(train_as2, "train_accuracy2", img_path, condition)


# In[29]:


plot_results(test_ls, "test_loss", img_path, condition)


# In[30]:


plot_results(test_as1, "test_accuracy1", img_path, condition)


# In[31]:


plot_results(test_as2, "test_accuracy2", img_path, condition)


# In[29]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'mv_lstm_train2.ipynb'])

