import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5), std=(0.5))
])

def get_fmnist_loader(batch_size):
    fmnist = torchvision.datasets.FashionMNIST(
        root="./", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=fmnist, batch_size=batch_size, shuffle=True)
    return data_loader

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.l1 = nn.Linear(100, 784)
        self.l2 = nn.Linear(784, 784)
        self.l3 = nn.Linear(784, 784)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.l1 = nn.Linear(784, 784)
        self.l2 = nn.Linear(784, 784)
        self.l3 = nn.Linear(784, 256)
        self.out = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return torch.sigmoid(self.out(x))

def train(G, D, lr, data_loader, epochs):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    optimizer_D = optim.Adam(D.parameters(), lr = lr)
    loss_G = list()
    loss_D = list()
    for e in range(epochs):
        print(e)
        for _imgs, _ in data_loader:
            imgs = (_imgs.cuda().view(-1, 28 * 28) - 0.5) / 0.5
            labels_real = torch.ones(imgs.size(0)).cuda()
            seeds = torch.randn(imgs.size(0), 100).cuda()
            generated = G(seeds)
            labels_fake = torch.zeros(imgs.size(0)).cuda()
            # train D
            D.zero_grad()
            out_real = D(imgs)
            loss_real = criterion(out_real, labels_real)
            out_fake = D(generated)
            loss_fake = criterion(out_fake, labels_fake)
            loss = loss_real + loss_fake
            loss.backward()
            optimizer_D.step()
            loss_D.append(loss.item())
            #
            seeds = torch.randn(imgs.size(0), 100).cuda()
            generated = G(seeds)
            # train G
            G.zero_grad()
            loss = criterion(D(generated), labels_real)
            loss.backward()
            optimizer_G.step()
            loss_G.append(loss.item())
    return (loss_D, loss_G)

g = G().cuda()
d = D().cuda()

seeds = torch.randn(20, 100).cuda()
generated = g(seeds).cpu()
plt.imshow(generated[0].detach().numpy().reshape((28,28)) * 0.5 + 0.5, cmap='gray')
plt.show()

data_loader = get_fmnist_loader(100)
loss_D, loss_G = train(g, d, 0.0002, data_loader, 180)
plt.plot(loss_D)
plt.show()
plt.plot(loss_G)
plt.show()

seeds = torch.randn(20, 100).cuda()
generated = g(seeds).cpu()
plt.imshow(generated[0].detach().numpy().reshape((28,28)) * 0.5 + 0.5, cmap='gray')
plt.show()

