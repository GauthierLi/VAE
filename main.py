import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from AE import AE
from VAE import VAE

import visdom

def mainAE():
    mnist_train = datasets.MNIST('mninst', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('mninst', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    x, _ = iter(mnist_train).next()
    print('x:', x.shape)

    device = torch.device('cuda')
    model = AE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    viz = visdom.Visdom()

    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            # [b,1,28,28]
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)

            # bp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, "loss: ", loss.item())
        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x)

        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


def mainVAE():
    mnist_train = datasets.MNIST('mninst', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('mninst', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    x, _ = iter(mnist_train).next()
    print('x:', x.shape)

    device = torch.device('cuda')
    model = VAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    viz = visdom.Visdom()

    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            # [b,1,28,28]
            x = x.to(device)
            x_hat, kld = model(x)
            loss = criterion(x_hat, x)

            # bp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, "loss: ", loss.item())
        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat, kld = model(x)

        if kld is not None:
            elbo = -loss - 1.0 * kld
            loss = -elbo
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    mainVAE()
