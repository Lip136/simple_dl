# encoding:utf-8
'''
功能：经典的CNN结构
结构：两层卷积+激活函数+max pooling，两层全连接层。
后续：dropout？softmax？
'''
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # (input_channel, output_channel, 5*5)
        # 如：mnist数据集是28*28*1的图片，输入通道是1
        self.conv1 = nn.Conv2d(1, 6, 5) # 通过1层卷积变成：(28-5)+1=24,通过max_pool:(24-2)/2+1= 12
        self.conv2 = nn.Conv2d(6, 16, 5) # 通过1层卷积变成：(12-5)+1=8,通过max_pool:(8-2)/2+1= 4
        # (2*2, stride=2)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, X):
        X = self.max_pool(torch.relu(self.conv1(X)))
        X = self.max_pool(torch.relu(self.conv2(X)))
        X = X.view(-1, 16*4*4)
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)
        return torch.softmax(X, dim=1)

