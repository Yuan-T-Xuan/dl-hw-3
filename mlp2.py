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
        self.num_embeddings = 52876
        self.embedding_dim = 10
        self.embedding = nn.Embedding(
            num_embeddings = self.num_embeddings, 
            embedding_dim = self.embedding_dim
        )
        self.l1 = nn.Linear(1000, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 500)
        self.out = nn.Linear(500, 2)
    
    def forward(self, input):
        output = self.embedding(input)
        output = output.view(-1, 1000)
        output = self.l1(output)
        output = self.l2(output)
        output = self.l3(output)
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
    m_pos = sentences2matrix("data/train_pos_merged.txt", vocab, maxlen=100)
    m_neg = sentences2matrix("data/train_neg_merged.txt", vocab, maxlen=100)
    m = np.vstack((m_pos, m_neg))
    labels = np.array([1]*1500 + [0]*1500, dtype=int)
    train_data = TensorDataset(torch.from_numpy(m).cuda(), torch.from_numpy(labels).cuda())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=100)
    #
    m_pos = sentences2matrix("data/test_pos_merged.txt", vocab, maxlen=100)
    m_neg = sentences2matrix("data/test_neg_merged.txt", vocab, maxlen=100)
    m_test = np.vstack((m_pos, m_neg))
    labels_test = np.array([1]*1500 + [0]*1500, dtype=int)
    test_data = TensorDataset(torch.from_numpy(m_test).cuda(), torch.from_numpy(labels_test).cuda())
    test_loader = DataLoader(test_data, shuffle=True, batch_size=100)
    #
    train_model(model, 200, train_loader, test_loader, 0.0001)

