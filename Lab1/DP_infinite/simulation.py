
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import collections  as mc
import random
import os

#from calculation import value_iteration
import test
from test import value_iteration_inf

WORLD_X=6
WORLD_Y=3
bank_pos = ((0, 0), (5, 0), (0, 2), (5, 2))
# left,up,right,down,stay
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

ACTIONS_police = [np.array([0, -1]),
                  np.array([-1, 0]),
                  np.array([0, 1]),
                  np.array([1, 0])]

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

#For plotting the example execution
def draw_image(policy, name):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    ncols = WORLD_X
    nrows = WORLD_Y

    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for i in range(WORLD_Y):
        for j in range(WORLD_X):
            color = 'white'

            #bank
            if (j, i) in bank_pos:
                color = '#9bb4db'

            if policy[i][j][1][2] == -1:
                text = "Police"
            elif policy[i][j][1][2] == 0:
                text= "Left"
            elif policy[i][j][1][2] == 1:
                text= "Up"
            elif policy[i][j][1][2] == 2:
                text= "Right"
            elif policy[i][j][1][2] == 3:
                text= "Down"
            elif policy[i][j][1][2] == 4:
                text= "Stay"

            tb.add_cell(i, j, width, height, text=text, loc='center', edgecolor='#63b1f2', facecolor=color)
    # Row Labels...

    for i, label in enumerate(range(WORLD_Y)):
        tb.add_cell(i, -1, width, height, text=label+1, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(range(WORLD_X)):
        tb.add_cell(-1, j, width, height/2, text=label+1, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)

    #Limits

    borders = [[(0, 0), (1, 0)], [(0, 0), (0, 1)], [(1, 1), (1, 0)], [(1, 1), (0, 1)]]
    lc = mc.LineCollection(borders, colors='k', linewidths=4)

    ax.add_collection(lc)




    plt.savefig('lambda' + name + '.png')
    plt.close()



#Moving the police toward robber
def police_move(position, actions):
    pos = np.array(position)
    act = random.choice(actions)
    next_state = (pos + act).tolist()
    return next_state

def robber_step(position, actions):
    pos = np.array(position)
    next_state = (pos + actions).tolist()
    '''
    if x < 0 or x >= WORLD_Y or y < 0 or y >= WORLD_X:
        # out of boundary
        next_state = pos.tolist()
        flag = False
    '''
    return next_state

#Simulating with inf time and action grid given
def simulate_inf(policy):
    #initialzed positions
    police_path=[[1,2]]
    robber_path = [[0,0]]

    #Checking if we have won or not
    caught = False
    reward = 0

    while True:
        #Where each one is
        police_pos = police_path[-1]
        robber_pos = robber_path[-1]
        #print(police_pos, " ", robber_pos)
        a = policy[robber_pos[0]][robber_pos[1]][police_pos[0]][police_pos[1]]
        #Moving the player
        new_robber_pos = robber_step(robber_pos, ACTIONS[a])
        actions = police_dir(robber_pos, police_pos)
        #print(actions)
        new_police_pos = police_move(police_pos, actions)
        robber_path.append(new_robber_pos)
        police_path.append(new_police_pos)
        #print(reward)

        #If caught
        if new_police_pos[0] == new_robber_pos[0] and new_police_pos[1] == new_robber_pos[1]:
            caught = True
            reward -= 50
            break
        elif (new_robber_pos[0], new_robber_pos[1]) in bank_pos:
            reward += 10
        else:
            pass

    return robber_path, police_path, reward


if __name__ == '__main__':

    lambdas = [0.5,0.5,0.7,0.9]

    for lamb in lambdas:
        state_value, policy = value_iteration_inf(lamb)
        draw_image(policy, str(lamb))


    result = []
    episodes = []
    for LAMBDA in range(100):
        print(LAMBDA)
        state_value,police = value_iteration_inf(LAMBDA/100)
        result.append(state_value[0,0,1,2])
        #print(result)
        episodes.append(LAMBDA/100)


    plt.plot(episodes, result)
    plt.xlabel('Lambda')
    plt.ylabel('Initial value function')
    plt.savefig('./try.png')
    plt.close()


    time = []
    wins = []

'''
    print("start!")
    state_value, policy_inf = value_iteration_inf(0.8)
    print("policy Done!")


    win_counter = 0
    total_simulations = 10000

    for i in range(total_simulations):
        print(i)
        _, _, win = simulate_inf(policy_inf)

        if win:
            win_counter += 1


    print("Total wins:" + str(win_counter))
    print("Out of:" + str(total_simulations))

    print("Done")
'''
