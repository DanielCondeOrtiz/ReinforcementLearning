
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib import collections  as mc
import random
import os

from calculation import value_iteration
from calculation_inf import value_iteration_inf

WORLD_X=6
WORLD_Y=5

T=15

ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0]),
           np.array([0, 0])]

#Moving the minotaur randomly
def min_move(position):
    moves = []
    if position[0] > 0:
        moves.append(np.array([-1, 0]))

    if position[0] < WORLD_Y -1:
        moves.append(np.array([1, 0]))

    if position[1] > 0:
        moves.append(np.array([0, -1]))

    if position[1] < WORLD_X -1:
        moves.append(np.array([0, 1]))

    return position + random.choice(moves)



#For plotting the example execution
def draw_image(player_path, min_path, name):
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

            #Finish
            if i==4 and j==4:
                tb.add_cell(i, j, width, height,
                            loc='center', edgecolor='#63b1f2', facecolor='#9bb4db')
            #Start
            elif i==0 and j==0:
                tb.add_cell(i, j, width, height,
                            loc='center', edgecolor='#63b1f2', facecolor='#9fe592')
            else:
                tb.add_cell(i, j, width, height,
                            loc='center', edgecolor='#63b1f2', facecolor=color)
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

    walls = [[(2/6, 1), (2/6, 2/5)], [(4/6, 4/5), (4/6, 2/5)], [(4/6, 3/5), (1, 3/5)], [(1/6, 1/5), (5/6, 1/5)], [(4/6, 1/5), (4/6, 0)]]
    lc = mc.LineCollection(walls, colors='k', linewidths=2)

    ax.add_collection(lc)


    #Paths
    #May need improvements
    last = ((player_path[0][1] + 0.5)/6,(4.5-player_path[0][0])/5)
    path=[]
    for k in range(len(player_path)):
        path.append([last,((player_path[k][1] + 0.5)/6,(4.5-player_path[k][0])/5)])
        last = ((player_path[k][1] + 0.5)/6,(4.5-player_path[k][0])/5)

    lc = mc.LineCollection(path, colors='r', linewidths=2)

    ax.add_collection(lc)


    last = ((min_path[0][1] + 0.35)/6,(4.65-min_path[0][0])/5)
    path=[]
    for k in range(len(min_path)):
        path.append([last,((min_path[k][1] + 0.35)/6,(4.65-min_path[k][0])/5)])
        last = ((min_path[k][1] + 0.35)/6,(4.65-min_path[k][0])/5)

    lc = mc.LineCollection(path, colors='b', linewidths=2)

    ax.add_collection(lc)

    plt.savefig(name)
    plt.close()


#Simulating with maximum time T and action grid given
def simulate(policy, T):
    #Starting positions
    min_path=[[4,4]]
    player_path = [[0,0]]

    #Checking if we have won or not
    win = False


    for t in range(T):
        #Where each one is
        pos_min = min_path[-1]
        pos_player = player_path[-1]

        #Moving the player
        new_pos_player = pos_player + ACTIONS[policy[pos_player[0]][pos_player[1]][pos_min[0]][pos_min[1]]]



        player_path.append(new_pos_player)
        min_path.append(min_move(pos_min))

        #If won
        if new_pos_player[0] == 4 and new_pos_player[1] == 4:
            win = True
            break
        #If eaten by minotaur
        elif new_pos_player[0] == min_path[-1][0] and new_pos_player[1] == min_path[-1][1]:
            break


    return player_path, min_path, win


#Simulating with inf time and action grid given
def simulate_inf(policy):
    #Starting positions
    min_path=[[4,4]]
    player_path = [[0,0]]

    #Checking if we have won or not
    win = False


    while True:
        #Where each one is
        pos_min = min_path[-1]
        pos_player = player_path[-1]

        #Moving the player
        new_pos_player = pos_player + ACTIONS[policy[pos_player[0]][pos_player[1]][pos_min[0]][pos_min[1]]]
        #if np.random.binomial(1, 1/30) == 1:
            #break

        player_path.append(new_pos_player)
        min_path.append(min_move(pos_min))

        #If won
        if new_pos_player[0] == 4 and new_pos_player[1] == 4:
            win = True
            break
        #If eaten by minotaur
        elif new_pos_player[0] == min_path[-1][0] and new_pos_player[1] == min_path[-1][1]:
            break


    return player_path, min_path, win


if __name__ == '__main__':


    #Example for drawing
    #policy = value_iteration_inf()
    '''
    policy = value_iteration(15)
    #
    #
    player_path, min_path, _ = simulate(policy, 15)
    #
    draw_image(player_path, min_path,'./example.png')
    #
    #
    time = []
    wins = []


    #Simulations
    for t in range(0,120):
        win_counter = 0
        total_simulations = 10000
        policy = value_iteration(t)

        for i in range(total_simulations):
            _, _, win = simulate(policy, t)

            if win:
                win_counter += 1

        print("T:" + str(t))
        print("Total wins:" + str(win_counter))
        print("Out of:" + str(total_simulations))
        print("-------------------------")
        time.append(t)
        wins.append(float(win_counter)/(total_simulations/100))

    plt.plot(np.asarray(time),np.asarray(wins))
    plt.xlabel("T")
    plt.ylabel("Win %")
    plt.savefig('./graph.png')
    plt.close()

    '''


    #For 1_c
    print("start!")
    policy_inf = value_iteration_inf()
    print("policy Done!")

    player_path, min_path, _ = simulate_inf(policy_inf)
    print("simulate done!")

    #draw_image(player_path, min_path,'./example_inf.png')


    win_counter = 0
    total_simulations = 10000

    for i in range(total_simulations):
        _, _, win = simulate_inf(policy_inf)

        if win:
            win_counter += 1


    print("Total wins:" + str(win_counter))
    print("Out of:" + str(total_simulations))

    print("Done")
