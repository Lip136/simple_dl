# encoding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

sentences = ["i like dog", "i love cat", "i hate milk"]

word_list = list(set(" ".join(sentences).split()))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)
# parameter
batch_size = len(sentences)
n_step = 2
n_hidden = 5

def make_batch(sentences):
    x_batch = []
    y_batch = []

    for sen in sentences:
        words = sen.split()
        x = [word_dict[n] for n in words[:-1]]
        target = word_dict[words[-1]]

        x_batch.append(np.eye(n_class)[x])
        y_batch.append(target)
    return torch.Tensor(x_batch), torch.tensor(y_batch)

input_batch, target_batch = make_batch(sentences)

# model
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]))
        self.b = nn.Parameter(torch.randn([n_class]))

    def forward(self, X):
        X = X.transpose(0, 1) # X:n_step, batch_size, n_class
        hidden_state = torch.zeros(1, batch_size, n_hidden)
        cell_state = torch.zeros(1, batch_size, n_hidden)
        outputs, _ = self.lstm(X, (hidden_state, cell_state))

        outputs = outputs[-1]
        model = torch.mm(outputs, self.W) + self.b
        return model

model = TextLSTM()
# 返回可被学习的参数（权重）列表和值
# params = list(model.parameters())
# print(len(params), params[0].size())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5000):
    optimizer.zero_grad()

    output = model(input_batch)

    loss = criterion(output, target_batch)

    if (epoch + 1)%1000 == 0:
        print('Epoch:%04d cost=%.6f'%(epoch+1, loss))

    loss.backward()
    optimizer.step()

