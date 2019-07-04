# encoding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sentences = [ "i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

# TextRNN Parameter
batch_size = len(sentences)
# n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch
# batch_size = 3

# to Torch.Tensor
input_batch, target_batch = make_batch(sentences)
input_batch = torch.Tensor(input_batch)
target_batch = torch.LongTensor(target_batch)

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]))
        self.b = nn.Parameter(torch.randn([n_class]))

    def forward(self, X, hidden_state=None):
        # X.shape():3, 2, 7
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        # hidden_state = torch.zeros(1, batch_size, n_hidden)
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # cell_state = Variable(torch.zeros(1, batch_size, n_hidden)) # LSTM
        outputs, _ = self.rnn(X, (hidden_state))
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = torch.mm(outputs, self.W) + self.b # model : [batch_size, n_class]
        return model

model = TextRNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1000):
    optimizer.zero_grad()
    # input_batch : [batch_size, n_step, n_class]
    output = model(input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

input = [sen.split()[:2] for sen in sentences]

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])