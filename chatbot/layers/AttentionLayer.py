# encoding:utf-8
'''
功能：经典的AttentionLayer结构
结构：三个计算方式
后续：
输入：embedded word tensor:shape=(1, batch_size, hidden_size)
     encoder_output:encoder输出,shape=(1, batch_size, hidden_size)
输出：softmax标准化权重张量,shape=(batch_Size, 1, max_length)
'''
import torch
import torch.nn as nn
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_socre(self, hidden, encoder_output):
        energy = self.attn(torch.cat(hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_socre(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t() # 转置

        return torch.softmax(attn_energies, dim=1).unsqueeze(1)

