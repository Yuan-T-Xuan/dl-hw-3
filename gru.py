import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
from sentiment_preprocess import *
import pickle
import numpy as np

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.num_embeddings = 52876
        self.embedding_dim = 50
        self.gru_hidden_size = 100
        self.gru_layers = 2

        self.embedding = nn.Embedding(
            num_embeddings = self.num_embeddings, 
            embedding_dim = self.embedding_dim
        )
        self.gru = nn.GRU(
            batch_first = True,
            input_size = self.embedding_dim,
            hidden_size = self.gru_hidden_size,
            num_layers = self.gru_layers
        )
        self.linear = nn.Linear(in_features=self.gru_hidden_size, out_features=2)
    
    def initHidden(self, batch_size):
        return torch.zeros(self.gru_layers, batch_size, self.gru_hidden_size)
    
    def forward(self, input, hidden=None):
        if not hidden:
            hidden = self.initHidden(input.size()[0])
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        output = output[:, -1, :].view(-1, self.gru_hidden_size)
        output = self.linear(output)
        return output, hidden

def train_model(model, num_epochs, train_loader, test_loader, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            model.zero_grad()
            inputs, labels = data
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('[%5d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
    # TBD ...


if __name__ == "__main__":
    model = GRU()
    f = open("data/vocab.pickle", 'rb')
    vocab = pickle.load(f)
    f.close()
    m_pos = sentences2matrix("data/train_pos_merged.txt", vocab)
    m_neg = sentences2matrix("data/train_neg_merged.txt", vocab)
    m = np.vstack((m_pos, m_neg))
    labels = np.array([1]*1500 + [0]*1500, dtype=int)
    train_data = TensorDataset(torch.from_numpy(m), torch.from_numpy(labels))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=40)
    train_model(model, 5, train_loader, None, 0.0001)

