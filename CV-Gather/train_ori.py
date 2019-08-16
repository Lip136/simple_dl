# encoding:utf-8
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visdom import Visdom
from utils import Flatten

from Lenet5_ori import Lenet5
from Resnet_ori import ResNet18
from torchvision.models import resnet18


batch_size = 32
lr = 1e-3
epochs = 10

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
viz = Visdom(env="Picture CLS")

# 验证模型acc
def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    # total = len(loader)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def main():

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    cifar_val = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_val = DataLoader(cifar_val, batch_size=batch_size, shuffle=True)


    # net = Lenet5().to(device)
    """
    epoch:0, train_loss:0.054862088589668276
    epoch:0, test accuracy:0.4418
    epoch:1, train_loss:0.046294315960407256
    epoch:1, test accuracy:0.4867
    """
    net = ResNet18().to(device)
    """
    epoch:0, train_loss:0.04010723890185356
    epoch:0, test accuracy:0.6452
    epoch:1, train_loss:0.030493649019002915
    epoch:1, test accuracy:0.6585
    """
    # train_model = resnet18(pretrained=True)
    # net = nn.Sequential(*list(train_model.children())[:-1],
    #                     Flatten(),
    #                     nn.Linear(512, 10)).to(device)
    # best accuracy:{0.79}\tbest epoch{4}
    print(net)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_acc = 0
    best_epoch = 0
    global_step = 0
    net.train()
    for epoch in range(epochs):
        total_loss = 0
        total_l_num = 0
        for batch_idx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)


            logits = net(x)
            loss = criterion(logits, label)
            total_loss += loss.item()
            total_l_num += x.size(0)
            # backprop
            net.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win="loss", update="append")
            global_step += 1

        train_loss = total_loss / total_l_num
        print("epoch:{}, train_loss:{}".format(epoch, train_loss))
        # val
        if epoch % 1 == 0:
            net.eval()
            val_accuracy = evaluate(net, cifar_val)
            print("epoch:{}, val accuracy:{}".format(epoch, val_accuracy))
            viz.line([val_accuracy], [global_step], win="val_acc", update="append")

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_epoch = epoch

                torch.save(net.state_dict(), "best_Pic_model.bin")

    print("best accuracy:{}\tbest epoch{}".format(best_acc, best_epoch))

    # test
    # net.load_state_dict(torch.load("best_Pic_model.bin"))
    # print("loaded from ckpt!")
    # test_acc = evaluate(net, test_loader)
    # print("test accuracy:{}".format(test_acc))

if __name__ == '__main__':
    main()