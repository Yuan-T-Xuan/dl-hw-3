from rl_setup import *
import numpy as np
from tqdm import tqdm

def encodeOneHot(num, max_state):
    encoded = np.zeros(max_state + 1)
    encoded[num] = 1.0
    encoded[0] = 0.01
    return encoded

def train(num_actions, num_states, epsilon, alpha, gamma, W=None, epochs=50000):
    # W is a matrix with size (#actions, #states + 1)
    if not W:
        W = np.random.random((num_actions, num_states + 1))
    MDP = loadMDP()
    for epoch in tqdm(range(epochs)):
        playing = True
        currState = 3
        currVector = encodeOneHot(currState, num_states)
        while playing:
            if np.random.random() < epsilon:
                currAction = np.random.randint(4)
            else:
                currAction = np.dot(W, currVector).argmax()
            playing, newState, rt = processAction(MDP, currState, currAction)
            newVector = encodeOneHot(newState, num_states)
            #
            Qplus = rt + gamma * np.dot(W, newVector).max()
            Q = np.dot(W[currAction], currVector)
            dL = (Q - Qplus) * currVector
            W[currAction] -= alpha * dL
            #
            currState = newState
            currVector = newVector
    return W

def play(W):
    MDP = loadMDP()
    steps = [3]
    playing = True
    currState = 3
    currVector = encodeOneHot(currState, max_state=81)
    while playing:
        currAction = np.dot(W, currVector).argmax()
        playing, currState, rt = processAction(MDP, currState, currAction)
        steps.append(currState)
        currVector = encodeOneHot(currState, max_state=81)
    return (steps, rt)

if __name__ == "__main__":
    W = train(4, 81, 0.3, 0.01, 0.9, epochs=200000)
    np.save("trainedW_onehot.npy", W)

