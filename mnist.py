# encoding:utf-8
import torchsnooper
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
batch_size = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS=20

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3801, ))])

trainDataSet = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=False)

testDataSet = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         transform=transforms.ToTensor(),
                                         download=False)

trainLoad = torch.utils.data.DataLoader(dataset=trainDataSet,
                                        shuffle=True,
                                        batch_size=64,
                                        num_workers=2)

testLoad = torch.utils.data.DataLoader(dataset=testDataSet,
                                       shuffle=False,
                                       batch_size=64,
                                       num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    # @torchsnooper.snoop()
    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 16*4*4)  # (64, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# train
def train(model, device, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 200 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100.*correct/len(test_loader.dataset)
        ))
if __name__ == "__main__":
    for epoch in range(1, EPOCHS+1):
        train(net, DEVICE, trainLoad, epoch)
        test(net, DEVICE, testLoad)