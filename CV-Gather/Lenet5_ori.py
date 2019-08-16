# encoding:utf-8
import torch
from torch import nn

class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x:[b, 3, 32, 32] => [b, 6, 14, 14]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 6, 14, 14] => [b, 16, 5, 5]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        # flatten
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        """
        
        :param x:[b, 3, 32, 32] 
        :return: 
        """
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size, 16*5*5)
        logits = self.fc_unit(x)

        return logits

def main():

    # 可以验证自己计算的维度对不对
    tmp = torch.randn(2, 3, 32, 32)
    model = Lenet5()
    out = model(tmp)
    print("conv out:", out.shape)


if __name__ == '__main__':
    main()