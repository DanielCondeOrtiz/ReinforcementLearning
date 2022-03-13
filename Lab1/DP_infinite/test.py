
# author@litingyi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_Y, WORLD_X = 3, 6
# left,up,right,down,stay
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

#left, up, right, down
ACTIONS_police = [np.array([0, -1]),
                  np.array([-1, 0]),
                  np.array([0, 1]),
                  np.array([1, 0])]


def robber_step(state, action):
    flag = True
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:
        # out of boundary
        flag = False
        next_state = state.tolist()

    return next_state, flag


def police_chase(state, action):
    state = np.array(state)
    next_state = (state + action).tolist()

    return next_state


def index(argument):
    if argument[0] == 0 and argument[1] == -1:
        return 0
    if argument[0] == -1 and argument[1] == 0:
        return 1
    if argument[0] == 0 and argument[1] == 1:
        return 2
    if argument[0] == 1 and argument[1] == 0:
        return 3
    if argument[0] == 0 and argument[1] == 0:
        return 4

    return -1
'''
#left, up, right, down
ACTIONS_police = [np.array([0, -1]),
                  np.array([-1, 0]),
                  np.array([0, 1]),
                  np.array([1, 0])]
'''
# left,up,right,down,stay
def police_dir(robber, police):
    x, y = robber
    m, n = police
    action = []
    if x < m:
        if y < n:
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action
        elif y > n:
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action
        else:#y==n
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m > 0:
                action.append(ACTIONS_police[1])#up
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            return action
    elif x > m:
        if y < n:
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            return action
        elif y > n:
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            return action
        else:
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            return action
    else:#x == m
        if y < n:
            if n > 0:
                action.append(ACTIONS_police[0])#left
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action
        else:
            if n < WORLD_X - 1:
                action.append(ACTIONS_police[2])#right
            if m < WORLD_Y - 1:
                action.append(ACTIONS_police[3])#down
            if m > 0:
                action.append(ACTIONS_police[1])#up
            return action


def value_iteration_inf(LAMBDA=0.2):
    # state value
    #reward = np.full((WORLD_Y, WORLD_X), 0)
    reward_bank = 10 # at the bank
    reward_caught = -50 # death

    robber_init = (0, 0)
    police_init = (1, 2)
    bank_pos = ((0, 0), (0, 5), (2, 0), (2, 5))

    state_value = np.zeros((WORLD_Y, WORLD_X, WORLD_Y, WORLD_X))# me position & minotaur position
    policy = np.full((WORLD_Y, WORLD_X, WORLD_Y, WORLD_X), -1)
    value = np.zeros(state_value.shape)
    # value iteration
    #a = Reward_calculate()
    iteration = 0

    while True:
        #new_state_value = np.copy(state_value)
        for m in range(0, WORLD_Y):
            for n in range(0, WORLD_X):
                # all possible positions of police
                for x in range(0, WORLD_Y):
                    for y in range(0, WORLD_X):
                        # each position of robber
                        if (x == m) and (y == n):
                            continue
                        #if in the same cell, always to initialized states
                        action_returns = []
                        act_returns = []
                        for action in ACTIONS:
                            (next_x, next_y),flag = robber_step([x, y], action)
                            if (flag):
                                action_value = []
                                possible_action = police_dir([x, y], [m, n])
                                count = len(possible_action)
                                    # count of the num of police to go in each direction
                                prob_state = 1/count
                                for act in possible_action:
                                    (next_m, next_n) = police_chase([m, n], act)
                                    if (next_m == next_x) and (next_n == next_y): # end up in the same cell
                                        action_value.append(np.round(prob_state * (reward_caught + LAMBDA * state_value[robber_init[0],robber_init[1],police_init[0],police_init[1]]), 4))
                                    elif (next_x, next_y) in bank_pos:# rob
                                        action_value.append(np.round(prob_state * (reward_bank + LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))
                                    else:
                                        action_value.append(np.round(prob_state * (LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))

                                action_returns.append(np.sum(action_value))
                                act_returns.append(action)


                        new_value = np.max(action_returns)
                        state_value[x][y][m][n] = new_value
                        argument = act_returns[np.argmax(action_returns)]
                        policy[x][y][m][n] = index(argument)

        #state_value = new_state_value
        if np.sum(np.square(state_value - value)) < 1e-200:
            break
        else:
            value = state_value.copy()
            iteration += 1

    return state_value, policy


if __name__ == '__main__':
    f = value_iteration_inf()
