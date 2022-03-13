
# author@litingyi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_Y, WORLD_X = 5, 6
# left,up,right,down,stay
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

ACTIONS_MIN = [np.array([0, -1]),
               np.array([-1, 0]),
               np.array([0, 1]),
               np.array([1, 0])]

LAMBDA = 0.9

def step(state, action):
    x_, y_ = state
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    flag = True
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:# out of boundary
        next_state = state.tolist()
        flag = False
    if 0 <= x <= 2:
        if (y_== 1 and y == 2) or (y_ == 2 and y == 1):
            next_state = state.tolist()
            flag = False
        if 1 <= x <= 2:
            if (y_== 3 and y == 4) or (y_ == 4 and y == 3):
                next_state = state.tolist()
                flag = False
    if 4 <= y <= 5:
        if (x_== 1 and x == 2) or (x_ == 2 and x == 1):
            next_state = state.tolist()
            flag = False
    if 1 <= y <= 4:
        if (x_== 3 and x == 4) or (x_ == 4 and x == 3):
            next_state = state.tolist()
            flag = False
    if x == 4:
        if (y_ == 3 and y == 4) or (y_ == 4 and y == 3):
            next_state = state.tolist()
            flag = False

    return next_state, flag


def minotaur_step(state, action):
    x_, y_ = state
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:# out of boundary
        next_state = state.tolist()

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

def value_iteration_inf():
    # state value
    #reward = np.full((WORLD_Y, WORLD_X), 0)
    reward_win = 1 # win
    reward_death = -1 # death

    state_value = np.zeros((WORLD_Y, WORLD_X, WORLD_Y, WORLD_X))# me position & minotaur position
    policy = np.full((WORLD_Y, WORLD_X, WORLD_Y, WORLD_X), -1)
    value = np.zeros(state_value.shape)
    # value iteration
    #a = Reward_calculate()
    iteration = 0

    while True:
        new_state_value = np.copy(state_value)
        for m in range(0, WORLD_Y):
            for n in range(0, WORLD_X):
                # all possible positions of minotaur
                for x in range(0, WORLD_Y):
                    for y in range(0, WORLD_X):
                        # each position of me
                        if (m == x and n == y):
                        # minotaur not in the (x,y)
                            continue

                        action_returns = []
                        act_returns = []
                        for action in ACTIONS:
                            (next_x, next_y), flag = step([x, y], action)
                            if (flag):
                                action_value = []
                                count = 4
                                # count of the num of min to go in each dir
                                for act in ACTIONS_MIN:
                                    (next_m, next_n) = minotaur_step([m, n], act)
                                    if (next_m == m and next_n == n):
                                        count -= 1 # this action is not applicable

                                prob_state = 1/count # prob of min going each dir

                                for act in ACTIONS_MIN:
                                    (next_m, next_n) = minotaur_step([m, n], act)
                                    if (next_m != m or next_n != n):
                                        if (next_m == next_x) and (next_n == next_y): # end up in the same cell
                                            action_value.append(np.round(prob_state * (reward_death + LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))
                                                                #1/30 * (reward_death + LAMBDA * state_value[next_x, next_y, next_m, next_n])
                                        elif next_x == 4 and next_y == 4:# win
                                            action_value.append(np.round(prob_state * (reward_win + LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))
                                                                #1/30 * (reward_death + LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))
                                        else:
                                            action_value.append(np.round(prob_state * (LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))
                                                                #1/30 * (reward_death + LAMBDA * state_value[next_x, next_y, next_m, next_n]), 4))

                                action_returns.append(LAMBDA*np.sum(action_value))
                                act_returns.append(action)


                        new_value = np.max(action_returns)
                        new_state_value[x][y][m][n] = new_value
                        argument = act_returns[np.argmax(action_returns)]
                        policy[x][y][m][n] = index(argument)

        state_value = new_state_value
        iteration += 1

        if np.sum(np.square(state_value - value)) < 1e-200:
            break
        else:
            value = state_value.copy()


    print(iteration)

    return policy


if __name__ == '__main__':
    f, time = value_iteration_inf()
    print(time)
