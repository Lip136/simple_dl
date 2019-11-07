# encoding: utf-8

import torch
from torch import nn

class ASR_cnn(nn.Module):

    def __init__(self, ch_in, ch_out, label_shape):
        super(ASR_cnn, self).__init__()

        self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1),
                nn.BatchNorm1d(ch_out)
        )

        self.res_block1 = nn.Sequential(
                nn.Conv1d(ch_out, ch_out, kernel_size=7, dilation=1, padding=3),
                nn.BatchNorm1d(ch_out)
        )
        self.res_block2 = nn.Sequential(
                nn.Conv1d(ch_out, ch_out, kernel_size=7, dilation=2, padding=3 * 2),
                nn.BatchNorm1d(ch_out)
        )
        self.res_block4 = nn.Sequential(
                nn.Conv1d(ch_out, ch_out, kernel_size=7, dilation=4, padding=12),
                nn.BatchNorm1d(ch_out)
        )
        self.res_block8 = nn.Sequential(
                nn.Conv1d(ch_out, ch_out, kernel_size=7, dilation=8, padding=24),
                nn.BatchNorm1d(ch_out)
        )
        self.res_block16 = nn.Sequential(
                nn.Conv1d(ch_out, ch_out, kernel_size=7, dilation=16, padding=3 * 16),
                nn.BatchNorm1d(ch_out)
        )

        self.out = nn.Sequential(
                nn.Conv1d(ch_out, label_shape, kernel_size=1),
                nn.BatchNorm1d(label_shape)
        )

    def res_block(self, inputs, res_block):
        hf = torch.tanh(res_block(inputs))
        hg = torch.sigmoid(res_block(inputs))
        h0 = hf * hg

        ha = torch.tanh(h0)
        return ha + inputs, ha

    def forward(self, x, num_blocks=3):
        # [bs, 13, seq_len] => [bs, 128, seq_len]
        h0 = torch.tanh(self.extra(x))
        shortcut = 0
        for i in range(num_blocks):
            h0, s1 = self.res_block(h0, self.res_block1)
            h0, s2 = self.res_block(h0, self.res_block2)
            h0, s4 = self.res_block(h0, self.res_block4)
            h0, s8 = self.res_block(h0, self.res_block8)
            h0, s16 = self.res_block(h0, self.res_block16)
            shortcut = s1 + s2 + s4 + s8 + s16

        shortcut = torch.relu(shortcut)
        output = torch.log_softmax(self.out(shortcut), dim=1)
        output = output.transpose(1, 0)
        output = output.transpose(2, 0)
        return output

class ASR_gru(nn.Module):

    def __init__(self, mfcc_dim, hidden_dim, label_shape):
        super(ASR_gru, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(mfcc_dim, hidden_dim//2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )

        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, label_shape)

    def forward(self, x):
        # seq_len, bs, 13
        x = self.conv(x)
        x = x.transpose(2, 0)
        x = x.transpose(2, 1)
        output, hidden = self.gru(x)
        output = self.linear(torch.relu(output))

        return torch.log_softmax(output, dim=2)

if __name__ == "__main__":

    model = ASR_cnn(13, 128, 1133)
    x = torch.randn(8, 13, 410)
    pred_y = model(x) # [8, 1133, 410]
    assert pred_y.shape == (410, 8, 1133)

    model_gru = ASR_gru(13, 128, 1133)
    assert model_gru(x).shape == (410, 8, 1133)
