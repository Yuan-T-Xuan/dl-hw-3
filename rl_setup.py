import numpy as np

def loadMDP():
    next_states = [dict(), dict(), dict(), dict()]
    probabilities = [dict(), dict(), dict(), dict()]
    for i in range(4):
        f = open("rl-files/prob-a" + str(i+1) + ".txt")
        lines = f.readlines()
        lines = [l.split() for l in lines]
        f.close()
        for line in lines:
            _curr = int(line[0])
            _next = int(line[1])
            _prob = float(line[2])
            if _curr not in next_states[i]:
                next_states[i][_curr] = list()
                probabilities[i][_curr] = list()
            next_states[i][_curr].append(_next)
            probabilities[i][_curr].append(_prob)
    f = open("rl-files/rewards.txt")
    rewards = f.readlines()
    f.close()
    rewards = [float(l) for l in rewards]
    rewards = [.0] + rewards    # rewards[0] should never be used
    return (next_states, probabilities, rewards)

def processAction(MDP, curr_state, curr_action):
    # states start from 1
    # actions start from 0
    next_states, probabilities, rewards = MDP
    next_state = np.random.choice(next_states[curr_action][curr_state], p=probabilities[curr_action][curr_state])
    reward = rewards[next_state]
    return (reward == 0, next_state, reward)

