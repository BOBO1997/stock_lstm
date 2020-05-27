#!/usr/bin/env python
# coding: utf-8

# # 多変量LSTM: モデルの定義
# 
# 

# In[1]:


import sys
import inspect
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


class mv_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_size, seq_len, predict_len, dropout = 0):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len # 可変長にできるならいらないかも
        self.predict_len = predict_len # 可変長にできるならいらないかも

        self.lstm = nn.LSTM(input_size = input_size,
                             hidden_size = hidden_size,
                             num_layers = num_layers,
                             batch_first = True,
                             dropout = dropout) # definition of lstm rnn layer
        self.linear = nn.Linear(self.seq_len * self.hidden_size, self.predict_len * self.output_size) # definition of linear layer

    def forward(self, lstm_input, hidden=None, training=True):
        lstm_output, (hidden, cell) = self.lstm(lstm_input, hidden) # lstm rnn layer
        # reshape the lstm layer output tensor for input of linear layer
        linear_input = lstm_output.reshape(self.batch_size, self.seq_len * self.hidden_size) if training else lstm_output.reshape(1, self.seq_len * self.hidden_size)
        linear_output = self.linear(linear_input) # linear layer
        # reshape the linear layer output for computing loss
        output = linear_output.view(self.batch_size, self.predict_len, self.output_size) if training else linear_output.view(1, self.predict_len, self.output_size)
        return output


# In[3]:


class mv_lstm_same_seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_size, seq_len, predict_len, dropout = 0):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len # 可変長にできるならいらないかも
        self.predict_len = predict_len # 可変長にできるならいらないかも

        self.lstm = nn.LSTM(input_size = input_size,
                             hidden_size = hidden_size,
                             num_layers = num_layers,
                             batch_first = True,
                             dropout = dropout) # definition of lstm rnn layer
        self.linear = nn.Linear(self.hidden_size, self.output_size) # definition of linear layer

    def forward(self, lstm_input, hidden=None):
        lstm_output, (hidden, cell) = self.lstm(lstm_input, hidden) # lstm rnn layer
        linear_output = self.linear(lstm_output) # linear layer, this is ok if seq_len == predict_len
        return linear_output


# In[4]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'mv_lstm_model.ipynb'])

