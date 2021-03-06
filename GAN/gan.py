# encoding:utf-8
"""
gan会出现不稳定性，也就是当两个分布完全不相交的时候，鉴别器的能力很强
但是出现的效果也还可以，思考：为什么？
"""
import torch
from torch import nn, optim
import numpy as np
import random
import visdom


h_dim = 400
batchsz = 512
viz = visdom.Visdom()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # [b, 2] => [b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # [b, 2] => [b, 1]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = self.net(z)
        return output.view(-1)
# 生成数据
def data_generator():
    """
    8-gaussian minxture models
    :return: 
    """
    scale = 2
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            # N(0, 1) + center_x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset

def main():
    torch.manual_seed(32)
    np.random.seed(32)

    data_iter = data_generator()
    x = next(data_iter)
    # print(x.shape)

    G = Generator().cuda()
    D = Discriminator().cuda()
    # print(G, D)

    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))
    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    for epoch in range(1000):
        # 1. train Discriminator firstly
        for _ in range(5):
            # 1.1 train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr).cuda()
            # [b, 2] => [b, 1]
            predr = D(xr)
            # max predr
            lossr = -predr.mean()

            # 1.2 train on fake data
            z = torch.randn(batchsz, 2).cuda()
            xf = G(z).detach() # tf.stop_gradient()
            predf = D(xf)
            lossf = predf.mean()
            # aggregate loss
            loss_D = lossr + lossf

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        predf = D(xf)
        loss_G = -predf.mean()
        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update="append")
            print("loss: D{}\tG{}".format(loss_D.item(), loss_G.item()))

if __name__ == '__main__':
    main()