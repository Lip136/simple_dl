# encoding:utf-8
'''
功能：特征提取
结构：bag of words
'''
import torch
import torch.nn as nn
class BOW(nn.Module):
    def __init__(self, conf_dict):
        super(BOW, self).__init__()

        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]

        # (batch_size, seq_length, emb_dim) => (batch_size, seq_length//2, emb_dim//2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.softsign = nn.Softsign()
        # (batch_size, seq_length//2, emb_dim//2) => (batch_size, seq_length//2, bow_dim)
        self.fc = nn.Linear(self.emb_dim//2, self.bow_dim)

    def forward(self, X):

        X = self.softsign(self.pool(X))
        output = torch.relu(self.fc(X))
        return output


if __name__ == '__main__':
    import json

    conf_dict = json.loads(open("../config/bow.json", 'r').read())

    model = BOW(conf_dict)
    X = torch.randn(32, 20, 128)
    print(model(X).shape)