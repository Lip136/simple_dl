# encoding:utf-8
'''
功能：位子网络层(FeedForward, Norm)
参数：d_model
     
'''

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()
        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
                / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm

