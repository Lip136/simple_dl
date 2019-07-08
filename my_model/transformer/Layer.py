# encoding:utf-8
'''
功能：一个transformer的结构
参数：d_model

'''
import torch
import torch.nn as nn
from SubLayer import Norm, FeedForward
from MultiHeadsAttention import MultiHeadsAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadsAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x1, x1, x1, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadsAttention(heads, d_model)
        self.attn_2 = MultiHeadsAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x1, x1, x1, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x3 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x3))
        return x


