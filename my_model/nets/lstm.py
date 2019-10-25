# encoding:utf-8
'''
功能：双向的LSTM结构
结构：LSTM是RNN的一种，基本结构相同，是横向展开的结构。
后续：dropout可以直接加载lstm模型中。
'''
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, conf_dict):
        super(LSTM, self).__init__()

        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.lstm_dim = conf_dict["net"]["lstm_dim"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.dropout = conf_dict["net"]["dropout"]


        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.lstm_dim,
                            dropout=self.dropout, bidirectional=True
        )

        self.fc = nn.Linear(self.lstm_dim*2, self.hidden_dim)

    def forward(self, X):
        """
        input.shape = (batch_size, seq_length, emb_dim)

        output.shape = (seq_length, batch_size, hidden_dim)
        h.shape = (2, batch_size, lstm_dim)
        c.shape = (2, batch_size, lstm_dim)
        """

        X = X.transpose(0, 1)

        outputs, (hidden_state, cell_state) = self.lstm(X, (hidden_state, cell_state))

        outputs = torch.relu(self.fc(outputs))
        print("输出维度{}, (seq_length, batch_size, hidden_dim)".format(outputs.shape))

        return outputs, hidden_state, cell_state

