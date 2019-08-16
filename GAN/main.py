from ae import AE
from vae import VAE
from torchvision import datasets, transforms
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import visdom

def main():

    torch.manual_seed(32)
    batch_size = 32
    mnist_train = datasets.MNIST('/media/ptface02/H1/dataSet/standardDate/', True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               ]),
                                 download=False)
    mnist_test = datasets.MNIST('/media/ptface02/H1/dataSet/standardDate/', False,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               ]),
                                 download=False)
    mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    viz = visdom.Visdom()
    device = torch.device("cuda")

    # model = AE().to(device)
    model = VAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):

        for idx, (x, _) in enumerate(mnist_train):
            x = x.to(device)
            x_hat, kld = model(x)
            loss = criterion(x_hat, x)

            if kld is not None:
                elbo = -loss - 1.0 *kld
                loss = - elbo
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:{}\tlast loss:{}\tkld:{}".format(epoch, loss.item(), kld.item()))

        x, _ = next(iter(mnist_test))
        x = x.to(device)
        with torch.no_grad():
            x_hat, kld = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))



if __name__ == '__main__':
    main()