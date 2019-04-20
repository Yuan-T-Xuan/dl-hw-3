from rl_setup import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

adict = {
    3  : np.array([0, 0, 1, 0, 8, 4, 1], dtype=float),
    11 : np.array([0, 0, 2, 1, 7, 5, 1], dtype=float),
    12 : np.array([1, 1, 0, 0, 7, 4, 1], dtype=float),
    15 : np.array([0, 0, 1, 2, 7, 1, 1], dtype=float),
    16 : np.array([0, 1, 0, 1, 7, 0, 1], dtype=float),
    17 : np.array([0, 2, 2, 0, 7,-1, 1], dtype=float),
    20 : np.array([1, 0, 1, 0, 6, 5, 1], dtype=float),
    22 : np.array([0, 0, 1, 2, 6, 3, 1], dtype=float),
    23 : np.array([0, 1, 0, 1, 6, 2, 1], dtype=float),
    24 : np.array([1, 2, 0, 0, 6, 1, 1], dtype=float),
    26 : np.array([1, 0, 1, 0, 6,-1, 1], dtype=float),
    29 : np.array([2, 0, 0, 2, 5, 5, 1], dtype=float),
    30 : np.array([0, 1, 4, 1, 5, 4, 1], dtype=float),
    31 : np.array([1, 2, 0, 0, 5, 3, 1], dtype=float),
    34 : np.array([0, 0, 5, 1, 5, 0, 1], dtype=float),
    35 : np.array([2, 1, 0, 0, 5,-1, 1], dtype=float),
    39 : np.array([1, 0, 3, 0, 4, 4, 1], dtype=float),
    43 : np.array([1, 0, 4, 0, 4, 0, 1], dtype=float),
    48 : np.array([2, 0, 2, 0, 3, 4, 1], dtype=float),
    52 : np.array([2, 0, 3, 1, 3, 0, 1], dtype=float),
    53 : np.array([0, 1, 2, 0, 3,-1, 1], dtype=float),
    56 : np.array([0, 0, 0, 6, 2, 5, 1], dtype=float),
    57 : np.array([3, 1, 1, 5, 2, 4, 1], dtype=float),
    58 : np.array([0, 2, 0, 4, 2, 3, 1], dtype=float),
    59 : np.array([0, 3, 0, 3, 2, 2, 1], dtype=float),
    60 : np.array([0, 4, 0, 2, 2, 1, 1], dtype=float),
    61 : np.array([3, 5, 2, 1, 2, 0, 1], dtype=float),
    62 : np.array([1, 6, 1, 0, 2,-1, 1], dtype=float),
    66 : np.array([4, 0, 0, 0, 1, 4, 1], dtype=float),
    70 : np.array([4, 0, 1, 1, 1, 0, 1], dtype=float),
    71 : np.array([2, 1, 0, 0, 1,-1, 1], dtype=float),
    ######
    47 : np.array([0, 0, 1, 1, 3, 5, 1], dtype=float),
    49 : np.array([0, 1, 1, 0, 3, 3, 1], dtype=float),
    51 : np.array([0, 0, 1, 2, 3, 1, 1], dtype=float),
    65 : np.array([1, 0, 0, 1, 1, 5, 1], dtype=float),
    67 : np.array([1, 1, 0, 0, 1, 3, 1], dtype=float),
    69 : np.array([1, 0, 0, 2, 1, 1, 1], dtype=float),
    79 : np.array([5, 0, 0, 0, 0, 0, 1], dtype=float)
}

def encodeFeature(num):
    global adict
    return adict[num]

def train(num_actions, epsilon, alpha, gamma, W=None, epochs=50000):
    # W is a matrix with size (#actions, 7)
    for_plot = list()
    if not W:
        W = np.random.random((num_actions, 7))
    MDP = loadMDP()
    prevW = None
    for epoch in tqdm(range(epochs)):
        prevW = copy.deepcopy(W)
        playing = True
        currState = 3
        currVector = encodeFeature(currState)
        while playing:
            if np.random.random() < epsilon:
                currAction = np.random.randint(4)
            else:
                currAction = np.dot(W, currVector).argmax()
            playing, newState, rt = processAction(MDP, currState, currAction)
            newVector = encodeFeature(newState)
            #
            Qplus = rt + gamma * np.dot(W, newVector).max()
            Q = np.dot(W[currAction], currVector)
            dL = (Q - Qplus) * currVector
            W[currAction] -= alpha * dL
            #
            currState = newState
            currVector = newVector
        for_plot.append(np.sum(np.square(prevW - W)))
    #
    plt.plot(for_plot)
    plt.show()
    #
    return W

def play(W):
    MDP = loadMDP()
    steps = [3]
    playing = True
    currState = 3
    currVector = encodeFeature(currState)
    while playing:
        currAction = np.dot(W, currVector).argmax()
        playing, currState, rt = processAction(MDP, currState, currAction)
        steps.append(currState)
        if playing:
            currVector = encodeFeature(currState)
    return (steps, rt)

if __name__ == "__main__":
    W = train(4, 0.3, 0.01, 0.9, epochs=5000)
    np.save("trainedW_features.npy", W)

