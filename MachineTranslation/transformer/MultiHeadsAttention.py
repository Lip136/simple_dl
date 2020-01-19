# encoding:utf-8
'''
功能：Multi-heads的self-attention
参数：heads
     d_model
     dropout：default=0.1
'''
import torch
import torch.nn as nn
import math
class MultiHeadsAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadsAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model//heads

        self.heads = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_szie = q.size(0)

        k = self.k_linear(k).view(batch_szie, -1, self.heads, self.d_k)
        q = self.q_linear(q).view(batch_szie, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(batch_szie, -1, self.heads, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(batch_szie, -1, self.d_model)
        output = self.out(concat)
        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

