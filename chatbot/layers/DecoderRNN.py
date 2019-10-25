# encoding:utf-8
'''
功能：经典的Decoder结构
结构：单向GRU充当解码器
后续：
输入：input_setp:输入序列批次的一步（一个字）,shape=(1, batch_size)
     last_hidden:GRU的最终隐层,shape=(n_layers * num_directions, batch_size, hidden_size)
     encoder_outputs:编码器模型的输出,shape=(max_length, batch_size, hidden_size)
输出：outputs:softmax归一化张量，给出每个单词在解码序列中正确的下一个单词的概率,shape=(batch_Size, voc.num_words)
     hidden:GRU最终隐藏状态,shape=(n_layers*num_directions, batch_size, hidden_size)
'''
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from .AttentionLayer import Attn
class DecoderGRU(nn.Module):
    def __init__(self, config, embedding, output_Size):
        super(DecoderGRU, self).__init__()

        # Keep for reference
        self.attn_model = config["attn_model"]
        self.hidden_dim = config["hidden_dim"]
        self.embedding_dim = config["emb_dim"]
        self.dropout = config["dropout"]
        self.output_size = output_Size
        self.n_layers = config["decoder_n_layers"]

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.n_layers, dropout=(0 if self.n_layers==1 else self.dropout))
        self.concat = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, output_Size)

        self.attn = Attn(self.attn_model, self.hidden_dim)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 1 * 64 * 128, 2 * 64 * 128 单向GRU
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # 64 * 1 * 10
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 64 * 1 * 128
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = torch.softmax(output, dim=1)
        return output, hidden

