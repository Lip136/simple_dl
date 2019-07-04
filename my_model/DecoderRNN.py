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
from AttentionLayer import Attn
class DecoderGRU(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_Size, n_layers=1, dropout=0.1):
        super(DecoderGRU, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_Size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_Size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = torch.softmax(output, dim=1)
        return output, hidden

