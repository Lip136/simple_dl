# encoding:utf-8
'''
功能：经典的Encoder结构
结构：双向GRU充当编码器
后续：
输入：input_seq:一批句子,shape=(max_length, batch_size)
     input_lengths:每个批次中每个句子的长度列表,shape=(batch_size)
     hidden:隐层状态,shape=(n_layers * num_directions, batch_size, hidden_size)
输出：outputs:双向GRU最后一个隐藏层的双向输出特征之和,shape=(max_length, batch_Size, hidden_size)
     hidden:从GRU更新隐藏状态,shape=(n_layers*num_directions, batch_size, hidden_size)
'''

import torch.nn as nn
import torch
class EncoderGRU(nn.Module):
    def __init__(self, config, embedding):
        super(EncoderGRU, self).__init__()
        self.n_layers = config["encoder_n_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.embedding_dim = config["emb_dim"]
        self.dropout = config["dropout"]
        self.embedding = embedding

        # 初始化GRU,由于input_size通过embedding会和hidden_size相同
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)

        # 去掉padding
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.gru(packed, hidden)
        # 加上padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 计算双向GRU输出的和,原本outputs.shape=(input_lengths, batch_size, 2*hidden_size)
        outputs = sum(outputs.split([self.hidden_dim, self.hidden_dim], dim=2))
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

