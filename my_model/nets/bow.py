# encoding:utf-8
'''
功能：特征提取
结构：池化+激活函数+全连接层+激活函数。
后续：
'''
import torch
import torch.nn as nn
class BOW(nn.Module):
    def __init__(self, conf_dict):
        super(BOW, self).__init__()
        """
        input.shape = (batch_size, seq_length, emb_dim)
        output.shape = (batch_size, seq_length//2, bow_dim)
        """
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]


        # (input_size - kernel_size + padding * 2)/stride + 1
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.softsign = nn.Softsign()

        self.fc = nn.Linear(self.emb_dim//2, self.bow_dim)

    def forward(self, X):
        batch_size = X.shape[0]


        X = self.softsign(self.pool(X))
        X = X.view(batch_size, -1, self.emb_dim//2)

        X = torch.relu(self.fc(X))
        print(X.shape)

        return X

def main():

    import json
    conf_dict = json.loads(open("../config/bow.json", 'r').read())

    model = BOW(conf_dict)
    X = torch.randn(32, 20, 128)
    model(X)

if __name__ == '__main__':
    main()