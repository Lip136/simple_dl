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
import codecs

class EmbedBase(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbedBase, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


def LoadEmbed(word2id, wordVec_path):
    embedding_pre = []
    word2vec = {}
    with codecs.open(wordVec_path, 'r', 'utf-8') as input_data:
        for line in input_data.readlines()[1:]:
            word2vec[line.split()[0]] = list(map(eval, line.split()[1:]))
    unknow_pre = []
    unknow_pre.extend([1] * 100)
    embedding_pre.append(unknow_pre)  # wordvec id 0
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    embedding_pre = torch.tensor(embedding_pre)

    return embedding_pre
# self.embed = nn.Embedding.from_pretrained(embedding_pre, freeze=True)

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

