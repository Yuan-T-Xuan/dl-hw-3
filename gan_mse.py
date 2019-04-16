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

def train(G, lr, data_loader, epochs):
    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    loss_G = list()
    for e in range(epochs):
        if e % 3 == 0:
            print(e)
            seeds = torch.randn(1, 100).cuda()
            generated = g(seeds).cpu()
            plt.imshow(generated[0].detach().numpy().reshape((28,28)) * 0.5 + 0.5, cmap='gray')
            plt.show()
        for _imgs, _ in data_loader:
            imgs = (_imgs.cuda().view(-1, 28 * 28) - 0.5) / 0.5
            seeds = torch.randn(imgs.size(0), 100).cuda()
            generated = G(seeds)
            loss = criterion(generated, imgs)
            loss.backward()
            optimizer_G.step()
            loss_G.append(loss.item())
    return loss_G

g = G().cuda()
data_loader = get_fmnist_loader(32)
loss_G = train(g, 0.00001, data_loader, 50)
plt.plot(loss_G)
plt.show()

