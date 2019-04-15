import torch
import torchvision
import torchvision.transforms as transforms

def get_fmnist_loader(batch_size):
    fmnist = torchvision.datasets.FashionMNIST(
        root="./", train=True, download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset=fmnist, batch_size=batch_size, shuffle=True)
    return data_loader

