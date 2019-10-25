# encoding:utf-8
'''
功能：特征提取
结构：卷积+BN+激活函数+全连接层+激活函数。
后续：dropout？softmax？max pooling?
'''
import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, conf_dict):
        super(CNN, self).__init__()
        """
        input.shape = (batch_size, seq_length, emb_dim)
        output.shape = (batch_size, seq_length, hidden_dim)
        """
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.filter_size = conf_dict["net"]["filter_size"]
        self.num_filters = conf_dict["net"]["num_filters"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]


        # (input_size - kernel_size + padding * 2)/stride + 1
        self.padding = (self.filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.num_filters, kernel_size=self.filter_size, stride=1, padding=self.padding),
            nn.BatchNorm2d(self.num_filters)
        )

        self.fc = nn.Linear(self.num_filters*self.emb_dim, self.hidden_dim)

    def forward(self, X):
        batch_size = X.shape[0]
        X = X.unsqueeze(dim=1)
        X = torch.relu(self.conv(X))
        X = X.view(batch_size, -1, self.num_filters*self.emb_dim)

        X = torch.relu(self.fc(X))
        print(X.shape)

        return X

def main():

    import json
    conf_dict = json.loads(open("../config/cnn.json", 'r').read())

    model = CNN(conf_dict)
    X = torch.randn(32, 20, 128)
    model(X)

if __name__ == '__main__':
    main()