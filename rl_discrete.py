from rl_setup import *
import numpy as np
from tqdm import tqdm

def train(num_actions, num_states, epsilon, alpha, gamma, Q=None, epochs=50000):
    # Q is a matrix with size (#actions, #states + 1)
    # Q[:, 0] should never be used
    if not Q:
        Q = np.random.random((num_actions, num_states + 1))
    MDP = loadMDP()
    for epoch in tqdm(range(epochs)):
        playing = True
        currState = 3
        while playing:
            if np.random.random() < epsilon:
                currAction = np.random.randint(4)
            else:
                currAction = Q[:, currState].argmax()
            playing, newState, rt = processAction(MDP, currState, currAction)
            Q[currAction, currState] = (1 - alpha) * Q[currAction, currState]
            Q[currAction, currState] += alpha * (rt + gamma * Q[:, newState].max())
            currState = newState
    return Q

def play(Q):
    MDP = loadMDP()
    steps = [3]
    playing = True
    currState = 3
    while playing:
        currAction = Q[:, currState].argmax()
        playing, currState, rt = processAction(MDP, currState, currAction)
        steps.append(currState)
    return (steps, rt)

if __name__ == "__main__":
    Q = train(4, 81, 0.3, 0.1, 0.9, epochs=50000)
    np.save("trainedQ.npy", Q)

