import copy
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

def GANLoss_D(real_imgs, generated_imgs, dis):
    part1 = - torch.mean(torch.log(dis(real_imgs)))
    part2 = - torch.mean(torch.log(1.0 - dis(generated_imgs)))
    return part1 + part2

def GANLoss_G(generated_imgs, dis):
    return - torch.mean(torch.log(dis(generated_imgs)))

def train(G, D, lr, data_loader, data_loader_another, epochs):
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    optimizer_D = optim.Adam(D.parameters(), lr = lr)
    loss_G = list()
    loss_D = list()
    for e in range(epochs):
        if e % 5 == 0:
            print("Epoch: " + str(e))
            seeds = torch.randn(1, 100).cuda()
            generated = g(seeds).cpu()
            plt.imshow(generated[0].detach().numpy().reshape((28,28)) * 0.5 + 0.5, cmap='gray')
            plt.show()
            
        for _imgs, _ in data_loader:
            imgs = (_imgs.cuda().view(-1, 28 * 28) - 0.5) / 0.5
            seeds = torch.randn(imgs.size(0), 100).cuda()
            generated = G(seeds)
            # train D
            D.zero_grad()
            loss = GANLoss_D(imgs, generated, D)
            loss.backward()
            optimizer_D.step()
            loss_D.append(loss.item())
            # train D for unrolled steps
            D0 = copy.deepcopy(D)
            optimizer_D0 = optim.Adam(D0.parameters(), lr = lr)
            _cc = 0
            for _imgs_another, _ in data_loader_another:
                imgs = (_imgs_another.cuda().view(-1, 28 * 28) - 0.5) / 0.5
                seeds = torch.randn(imgs.size(0), 100).cuda()
                generated = G(seeds)
                D0.zero_grad()
                loss = GANLoss_D(imgs, generated, D0)
                loss.backward()
                optimizer_D0.step()
                _cc += 1
                if _cc > 5:
                    break
            #
            seeds = torch.randn(imgs.size(0), 100).cuda()
            generated = G(seeds)
            # train G
            G.zero_grad()
            loss = GANLoss_G(generated, D0)
            loss.backward()
            optimizer_G.step()
            loss_G.append(loss.item())
    return (loss_D, loss_G)

g = G().cuda()
d = D().cuda()

data_loader = get_fmnist_loader(32)
data_loader_another = get_fmnist_loader(32)
loss_D, loss_G = train(g, d, 0.0001, data_loader, data_loader_another, 50)
plt.plot(loss_D)
plt.show()
plt.plot(loss_G)
plt.show()

seeds = torch.randn(20, 100).cuda()
generated = g(seeds).cpu()
for i in range(20):
    plt.imshow(generated[i].detach().numpy().reshape((28,28)) * 0.5 + 0.5, cmap='gray')
    plt.show()

