import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_setup import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_features import encodeFeature

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(7, 50)
        self.l2 = nn.Linear(50, 50)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(50, 4)
    
    def forward(self, input):
        output = F.relu(self.l1(input))
        output = self.dropout(F.relu(self.l2(output)))
        return torch.tanh(self.out(output))

def train(epsilon, alpha, gamma, model=None, epochs=50000):
    # model takes in observations, and outputs 4 Q scores for 4 actions
    if not model:
        model = MLP()
    optimizer = optim.Adam(model.parameters(), lr = alpha)
    MDP = loadMDP()
    for epoch in tqdm(range(epochs)):
        playing = True
        currState = 3
        currVector = torch.from_numpy(encodeFeature(currState)).view(1, -1)
        while playing:
            if np.random.random() < epsilon:
                currAction = np.random.randint(4)
            else:
                currAction = model(currVector).argmax().item()
            playing, newState, rt = processAction(MDP, currState, currAction)
            newVector = torch.from_numpy(encodeFeature(newState)).view(1, -1)
            #
            Qplus = rt + gamma * model(newVector).max()
            Q = model(currVector)[0, currAction]
            L = 0.5 * (Q - Qplus) * (Q - Qplus)
            L.backward()
            optimizer.step()
            #
            currState = newState
            currVector = newVector
    return model

def play(model):
    model.eval()
    MDP = loadMDP()
    steps = [3]
    playing = True
    currState = 3
    currVector = torch.from_numpy(encodeFeature(currState)).view(1, -1)
    while playing:
        currAction = model(currVector).argmax().item()
        playing, currState, rt = processAction(MDP, currState, currAction)
        steps.append(currState)
        if playing:
            currVector = torch.from_numpy(encodeFeature(currState)).view(1, -1)
    return (steps, rt)

if __name__ == "__main__":
    model = torch.load("model.pt")
    model = train(0.2, 0.005, 0.9, model, epochs=5000)
    torch.save(model, "model7000.pt")

