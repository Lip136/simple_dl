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
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        # 参数
        self.emb_dim = config["emb_dim"]
        self.heads = config["heads"]
        self.dropout = config["dropout"]
        # 模型
        self.norm_1 = Norm(self.emb_dim)
        self.norm_2 = Norm(self.emb_dim)
        self.attn = MultiHeadsAttention(self.heads, self.emb_dim)
        self.ff = FeedForward(self.emb_dim)
        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x1, x1, x1, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        # 参数
        self.emb_dim = config["emb_dim"]
        self.heads = config["heads"]
        self.dropout = config["dropout"]
        # 模型
        self.norm_1 = Norm(self.emb_dim)
        self.norm_2 = Norm(self.emb_dim)
        self.norm_3 = Norm(self.emb_dim)

        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)
        self.dropout_3 = nn.Dropout(self.dropout)

        self.attn_1 = MultiHeadsAttention(self.heads, self.emb_dim)
        self.attn_2 = MultiHeadsAttention(self.heads, self.emb_dim)
        self.ff = FeedForward(self.emb_dim).to("cuda")

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x1, x1, x1, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x3 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x3))
        return x


