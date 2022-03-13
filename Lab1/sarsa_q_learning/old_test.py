#Copied (and modified) from:
#https://github.com/michaeltinsley/Gridworld-with-Q-Learning-Reinforcement-Learning-/blob/master/Gridworld.ipynb
import numpy as np
import operator
import matplotlib.pyplot as plt
import random

class GridWorld:
    ## Initialise starting data
    def __init__(self):
        # Set information about the gridworld
        self.height = 4
        self.width = 4
        self.grid = np.zeros(( self.height, self.width))

        # Set start location for the agent
        self.current_location = ( 0,0)


        self.police_location = (3,3)
        self.bank_location = (1,1)

        # Set grid rewards for special cells
        self.grid[self.bank_location[0], self.bank_location[1]] = 1
        self.police_reward = -10

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


    ## Put methods here:

    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions

    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[new_location[0], new_location[1]]

    #Moving the police randomly
    def move_police(self):
        moves = []
        position = self.police_location
        if position[0] > 0:
            moves.append(np.array([-1, 0]))

        if position[0] < self.width -1:
            moves.append(np.array([1, 0]))

        if position[1] > 0:
            moves.append(np.array([0, -1]))

        if position[1] < self.height -1:
            moves.append(np.array([0, 1]))

        self.police_location = position + random.choice(moves)


# make the Q_Agent move to a specific direction
    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location

        # UP
        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)

        # STAY
        elif action == 'STAY':
            # Stay still, collect reward
            reward = self.get_reward(last_location)

        self.move_police()

        if self.police_location[0] == self.current_location[0] and self.police_location[1] == self.current_location[1]:
            reward = self.police_reward
            self.current_location = (0,0)
            self.police_location = (3,3)


        return reward
'''
class RandomAgent():
    # Choose a random action
    def choose_action(self, available_actions):
        """Returns a random choice of the available actions"""
        return np.random.choice(available_actions)
'''

class Q_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.1, alpha=0.1, gamma=0.8):
        self.environment = environment
        self.q_table = dict() # Store all Q-values in dictionary of dictionaries
        self.q_count = dict()
        for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(environment.width):
                self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0, 'STAY':0}
                 # Populate sub-dictionary with zero values for possible moves
                self.q_count[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0, 'STAY':0}
                #count n(s,a)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, available_actions):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        return action

    def sarsa_learn(self, old_state, reward, new_state, action, new_action):
        """Updates the Q-value table using sarsa-learning"""
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = self.q_table[new_state][new_action]
        current_q_value = self.q_table[old_state][action]

        n_count = self.q_count[old_state][action]
        if (n_count!=0):
            step_size = 1/((n_count)**(2/3))
        else:
            step_size = 1/(0.1**(2/3))

        self.q_table[old_state][action] = (1 - step_size) * current_q_value + step_size * (reward + self.gamma * max_q_value_in_new_state)
        self.q_count[old_state][action] += 1

    def q_learn(self, old_state, reward, new_state, action):
        """Updates the q table using q-learning"""
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]
        # update step size
        n_count = self.q_count[old_state][action]
        if (n_count!=0):
            step_size = 1/((n_count)**(2/3))
        else:
            step_size = 1/(0.1**(2/3))

        self.q_table[old_state][action] = (1 - step_size) * current_q_value + step_size * (reward + self.gamma * max_q_value_in_new_state)
        self.q_count[old_state][action] += 1


def play(environment, agent, trials=1000, max_steps_per_episode=1000):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log

    for trial in range(trials): # Run trials
        print(trial)
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        #q learning
        while step < max_steps_per_episode: # Run until max steps or until game is finished
            old_state = environment.current_location
            action = agent.choose_action(environment.actions)
            reward = environment.make_step(action)
            new_state = environment.current_location

            agent.q_learn(old_state, reward, new_state, action)

            action = action_next
            old_state = new_state

            cumulative_reward += reward
            step += 1

        #sarsa learning
        old_state = environment.current_location
        action = agent.choose_action(environment.actions)
        while step < max_steps_per_episode: # Run until max steps or until game is finished
            reward = environment.make_step(action)
            new_state = environment.current_location
            action_next = agent.choose_action(environment.actions)


            # agent.q_learn(old_state, reward, new_state, action)
            agent.sarsa_learn(old_state, reward, new_state, action, action_next)

            action = action_next
            old_state = new_state

            cumulative_reward += reward
            step += 1
        '''
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log

    return reward_per_episode # Return performance log


environment = GridWorld()
agentQ = Q_Agent(environment)

# Note the learn=True argument!
reward_per_episode = play(environment, agentQ)

# Simple learning curve
plt.plot(reward_per_episode)

plt.savefig('./plot.png')
plt.close()
