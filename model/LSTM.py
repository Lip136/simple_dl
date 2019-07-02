# encoding:utf-8
'''
功能：经典的LSTM结构
结构：LSTM是RNN的一种，基本结构相同，是横向展开的结构。
后续：dropout可以直接加载lstm模型中。
'''
import torch
import torch.nn as nn
n_class = 7 # nlp任务中：代表词表里面一共有多少个词
hidden_size = 5 # 每一步的隐层单元
n_step = 2 # 一共多少步
batch_size = 3
#TODO：在使用LSTM时，需要先想好上述四个参数。
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=hidden_size)
        # 由于默认的batch_first=False,所以没有设定为True，就要求输入的X:(n_step, batch_size, n_class)
        self.W = nn.Parameter(torch.randn([hidden_size, n_class]))
        self.b = nn.Parameter(torch.randn([n_class]))

    def forward(self, X):
        # 默认我们的X在输入时是:(batch_size, n_step, n_class)
        X = X.transpose(0, 1)
        hidden_state = torch.zeros(1, batch_size, hidden_size)
        # 其中1是num_layers * num_directions = 1*1=1
        cell_state = torch.zeros(1, batch_size, hidden_size)
        outputs, (_, _) = self.lstm(X, (hidden_state, cell_state))
        # 我们只要最后一个step的输出
        output = outputs[-1]
        model = torch.mm(output, self.W) + self.b
        return model

