# encoding:utf-8
import torch
from torch import nn
from torchvision.models import resnet18

class ResBlk(nn.Module):
    """
    一个残差网络
    """
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        self.b_norm1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.b_norm2 = nn.BatchNorm2d(ch_out)

        # self.extra = nn.Sequential()
        # if ch_in != ch_out:
        self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = torch.relu(self.b_norm1(self.conv1(x)))
        out = self.b_norm2(self.conv2(out))
        print("x", x.shape, "out-", out.shape)
        out = self.extra(x) + out
        print("x", self.extra(x).shape)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )
        self.blk1 = ResBlk(16, 32)
        self.blk2 = ResBlk(32, 64)
        self.blk3 = ResBlk(64, 128)
        self.blk4 = ResBlk(128, 256)

        self.outlayer = nn.Linear(256*1*1, 10)

    def forward(self, x):
        # x:[b, 3, 32, 32] => [b, 64, 32, 32]
        x = torch.relu(self.conv1(x))
        # [b, 64, 32, 32] => [b, 1024, 32/2^4, 32/2^4]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)


        # print(x.shape)
        # [b, 256, 2, 2] => [b, 256, 1, 1]
        x = torch._adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    tmp = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(tmp)
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print(torch.argmax(out, dim=1), "参数量:{}".format(p))

if __name__ == '__main__':
    main()

