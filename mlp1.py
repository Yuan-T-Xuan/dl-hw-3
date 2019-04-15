import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
from sentiment_preprocess import *
import pickle
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_size = 400
        self.l1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 2)
    
    def forward(self, input):
        output = self.l1(input)
        output = self.l2(output)
        return self.out(output)

def train_model(model, num_epochs, train_loader, test_loader, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            model.zero_grad()
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('[%5d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
    if test_loader:
        total = 0
        correct = 0
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, indices = torch.max(outputs, 1)
            total += indices.size()[0]
            correct += torch.sum(indices == labels).tolist()
        print((total, correct, correct/total))


if __name__ == "__main__":
    model = MLP().cuda()
    f = open("data/vocab.pickle", 'rb')
    vocab = pickle.load(f)
    f.close()
    #
    m_pos = sentences2matrix("data/train_pos_merged.txt", vocab)
    m_neg = sentences2matrix("data/train_neg_merged.txt", vocab)
    m = np.array(np.vstack((m_pos, m_neg)), dtype="float32")
    labels = np.array([1]*1500 + [0]*1500, dtype=int)
    train_data = TensorDataset(torch.from_numpy(m).cuda(), torch.from_numpy(labels).cuda())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=100)
    #
    m_pos = sentences2matrix("data/test_pos_merged.txt", vocab)
    m_neg = sentences2matrix("data/test_neg_merged.txt", vocab)
    m_test = np.array(np.vstack((m_pos, m_neg)), dtype="float32")
    labels_test = np.array([1]*1500 + [0]*1500, dtype=int)
    test_data = TensorDataset(torch.from_numpy(m_test).cuda(), torch.from_numpy(labels_test).cuda())
    test_loader = DataLoader(test_data, shuffle=True, batch_size=100)
    #
    train_model(model, 1000, train_loader, test_loader, 0.0001)

