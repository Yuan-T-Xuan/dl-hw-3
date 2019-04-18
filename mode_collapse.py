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

def get_fmnist_loader_test(batch_size):
    fmnist = torchvision.datasets.FashionMNIST(
        root="./", train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=fmnist, batch_size=batch_size, shuffle=True)
    return data_loader

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.l1 = nn.Linear(784, 784)
        self.l2 = nn.Linear(784, 784)
        self.l3 = nn.Linear(784, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.out(x)

def train_D(D, lr, data_loader, data_loader_test, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(D.parameters(), lr = lr)
    for e in range(epochs):
        if e % 10 == 0:
            total = 0
            correct = 0
            for _imgs, _labels in data_loader_test:
                imgs = (_imgs.cuda().view(-1, 28 * 28) - 0.5) / 0.5
                labels = _labels.cuda()
                outputs = D(imgs)
                _, indices = torch.max(outputs, 1)
                total += indices.size()[0]
                correct += torch.sum(indices == labels).tolist()
            print((total, correct, correct/total))
        for _imgs, _labels in data_loader:
            #print(_labels)
            D.zero_grad()
            imgs = (_imgs.cuda().view(-1, 28 * 28) - 0.5) / 0.5
            labels = _labels.cuda()
            predicted = D(imgs)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()

"""d = D().cuda()
train_D(d, 0.0001, get_fmnist_loader(200), get_fmnist_loader_test(200), 51)"""
d = torch.load('D.pt', map_location=torch.device('cuda'))

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

g = torch.load('vanillaGan_G_unrolled.pt', map_location=torch.device('cuda'))
a = list()
for i in range(15):
    seeds = torch.randn(200, 100).cuda()
    generated = g(seeds)
    outputs = d(generated)
    _, indices = torch.max(outputs, 1)
    indices = indices.cpu().tolist()
    a += indices

plt.hist(a)
plt.show()

