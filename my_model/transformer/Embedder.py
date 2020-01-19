# encoding:utf-8
'''
功能：nlp中的Embedding
参数：vocab_size
     d_model
     x是one-hot编码后的词向量
'''

import torch.nn as nn
import math
import torch

class EmbedBase(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbedBase, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # 从pe的维度就可以看出输入的词向量维度
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*(i+1))/d_model)))

        pe = pe.unsqueeze(0) #由于词向量输入的时候有batch_size，所以这里加入一个维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 扩大x的数值，减小pe的影响
        x = x*math.sqrt(self.d_model)

        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        if x.is_cuda:
            pe = pe.cuda()
        x = x + pe
        return x
