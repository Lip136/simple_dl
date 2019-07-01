# encoding:utf-8

import torchsnooper
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(32)
batch_size = 64
epochs = 20

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root='/media/ptface02/H1/dataSet/standardDate/',
                                           train=True,
                                           transform=transform,
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root='/media/ptface02/H1/dataSet/standardDate/',
                                           train=False,
                                           transform=transform,
                                           download=False)

load_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
load_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)



# 搭建模型
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 24, pool:12
        self.conv2 = nn.Conv2d(6, 16, 5) # 8, pool:4
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 10)


    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.max_pool(torch.relu(self.conv1(x)))
        x = self.max_pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        # return torch.log_softmax(out, dim=1)
        return out

# 损失函数和优化器
model = NN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# train
def train(model, train_dataset, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataset):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 300 == 0:
            print("Epoch:{}\t[{}/{} ({:.0f}%)]\tLoss:{:.4f}".format(
                epoch,
                (batch_idx+1)*len(data), len(train_dataset.dataset), 100.*batch_idx/len(train_dataset),
                loss.item()))

# test
def eval_model(model, test_dataset, epoch):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_dataset:
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            accuracy += pred.eq(target.view_as(pred)).sum().item() #eq返回两个tensor相等的数, view_as返回由下标组合的数

    test_loss /= len(test_dataset.dataset)
    print("Epoch:{}\tLoss:{:.4f}\tAccuracy:{}/{} ({:.2f}%)".format(
        epoch,
        test_loss,
        accuracy, len(test_dataset.dataset), 100.*accuracy / len(test_dataset.dataset))
    )

if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(model, load_train, epoch)
        eval_model(model, load_test, epoch)
