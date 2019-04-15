import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from fmnist_loader import get_fmnist_loader

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.l1 = nn.Linear(100, 784)
        self.l2 = nn.Linear(784, 784)
        self.l3 = nn.Linear(784, 784)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.l1 = nn.Linear(784, 784)
        self.l2 = nn.Linear(784, 784)
        self.l3 = nn.Linear(784, 100)
        self.out = nn.Linear(100, 1)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return torch.sigmoid(self.out(x))

def train(G, D, lr, data_loader, epochs):
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    optimizer_D = optim.Adam(D.parameters(), lr = lr)
    loss_G = list()
    loss_D = list()
    for e in range(epochs):
        print(e)
        for _imgs, _ in data_loader:
            G.zero_grad()
            D.zero_grad()
            imgs = _imgs.cuda().view(-1, 28 * 28)
            seeds = torch.tensor(np.random.random((imgs.size()[0], 100))/100.0, dtype=torch.float).cuda()
            generated = G(seeds)
            loss = - torch.sum(torch.log(D(imgs))) - torch.sum(torch.log(1.0 - D(generated)))
            loss = loss / imgs.size()[0]
            loss.backward()
            optimizer_D.step()
            loss_D.append(loss.item())
            #
            G.zero_grad()
            D.zero_grad()
            seeds = torch.tensor(np.random.random((imgs.size()[0], 100))/100.0, dtype=torch.float).cuda()
            generated = G(seeds)
            loss = - torch.sum(torch.log(D(generated))) / imgs.size()[0]
            loss.backward()
            optimizer_G.step()
            loss_G.append(loss.item())
    return (loss_D, loss_G)

if __name__ == "__main__":
    g = G().cuda()
    d = D().cuda()
    data_loader = get_fmnist_loader(30)
    loss_D, loss_G = train(g, d, 0.0001, data_loader, 10)
    plt.plot(loss_D)
    plt.plot(loss_G)
    plt.show()

