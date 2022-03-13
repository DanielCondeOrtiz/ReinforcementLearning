import numpy as np
import copy as cp
import sys
import random
import matplotlib.pyplot as plt
import math
import itertools

# environment variables
random.seed()

DISCOUNT_FACTOR = 0.8
DEBUG = False

class ENV:
	def __init__(self):
		self.RowSize = 4
		self.ColSize = 4
		# self.robber = [0, 0]
		# self.police = [3, 3]
		self.bank = [1, 1]
		self.MAP_SIZE = self.RowSize * self.ColSize
		self.NUM_STATUS = self.MAP_SIZE ** 2

		self.NUM_ACTION = 5			# robber's action
		self.ActionList = [0, 1, 2, 3, 4]
		self.ActionName = ['stay', 'up', 'right', 'down', 'left']
		self.Q_table = self.NUM_STATUS * self.NUM_ACTION

		self.Status = self.coord_to_index([0, 0], [3, 3])

	def coord_to_index(self, coord_r, coord_p):
		t1 = coord_r[0] * self.ColSize + coord_r[1]
		t2 = coord_p[0] * self.ColSize + coord_p[1]
		idx = t1 + t2 * self.MAP_SIZE
		return idx

	def index_to_coord(self, idx=None):
		if idx is None:
			idx = self.Status
		r = idx % self.MAP_SIZE
		d = idx // self.MAP_SIZE
		coord_r = [r // self.ColSize, r % self.ColSize]
		coord_p = [d // self.ColSize, d % self.ColSize]
		return [coord_r, coord_p]

	def check_move(self, target, action, idx=None):
		# target: 0 - robber, 1 - police
		# if move is valid, return target coord
		# otherwise return None
		if idx is None:
			idx = self.Status
		coord = self.index_to_coord(idx)[target]
		if action == 0:
			if target == 1:
				return None
		if action == 1:
			if coord[0] <= 0:
				return None
			coord[0] -= 1
		if action == 2:
			if coord[1] >= self.ColSize-1:
				return None
			coord[1] += 1
		if action == 3:
			if coord[0] >= self.RowSize-1:
				return None
			coord[0] += 1
		if action == 4:
			if coord[1] <= 0:
				return None
			coord[1] -= 1
		return coord

	def possible_action(self, target, idx=None):
		if idx is None:
			idx = self.Status
		coord_pair = self.index_to_coord(idx)
		coord = coord_pair[target]
		action_list = []

		for action in self.ActionList:
			tmp = self.check_move(target, action, idx)
			if tmp is not None:
				action_list.append(action)
		return action_list

	# def possible_next_status(self, idx=None):
	# 	# return the next_status list, action_r list, action_p list
	# 	if idx is None:
	# 		idx = self.Status
	# 	[coord_r, coord_p] = self.index_to_coord(idx)
	# 	next_status = []
	# 	action_r_list = []			# for return
	# 	action_p_list = []			# for return
	#
	# 	coord_r_new_list = []
	# 	action_r_new_list = []
	# 	coord_p_new_list = []
	# 	action_p_new_list = []
	# 	for action_r in self.ActionList:
	# 		tmp = self.check_move(0, action_r, idx)
	# 		if tmp is not None:
	# 			coord_r_new_list.append(tmp)
	# 			action_r_new_list.append(action_r)
	# 	for action_p in self.ActionList:
	# 		tmp = self.check_move(1, action_p, idx)
	# 		if tmp is not None:
	# 			coord_p_new_list.append(tmp)
	# 			action_p_new_list.append(action_p)
	# 	combine_idx = itertools.product(range(len(coord_r_new_list)), range(len(coord_p_new_list)))
	# 	for (i, j) in combine_idx:
	# 		idx_new = self.coord_to_index(coord_r_new_list[i], coord_p_new_list[j])
	# 		next_status.append(idx_new)
	# 		action_r_list.append(action_r_new_list[i])
	# 		action_p_list.append(action_p_new_list[j])
	# 	return next_status, action_r_list, action_p_list

	def perform_action(self, action_r, action_p):
		# update the self.Status
		coord_r = self.check_move(0, action_r)
		coord_p = self.check_move(1, action_p)
		assert (coord_r is not None) and (coord_p is not None)
		new_idx = self.coord_to_index(coord_r, coord_p)
		self.Status = new_idx
		return new_idx

	def simulate_action(self, action_r, action_p):
		# return new status index
		coord_r = self.check_move(0, action_r)
		coord_p = self.check_move(1, action_p)
		assert (coord_r is not None) and (coord_p is not None)
		return self.coord_to_index(coord_r, coord_p)

	def init_reward_status(self):
		for idx in range(self.NUM_STATUS):
			reward = 0
			[coord_r, coord_p] = self.index_to_coord(idx)
			if coord_r == coord_p:
				reward -= 10
			if coord_r == self.bank:
				reward += 1
			self.reward_status[idx] = reward

	def draw(self, idx=None):
		[coord_r, coord_p] = self.index_to_coord(idx)
		for r in range(self.RowSize):
			for c in range(self.ColSize):
				if [r, c] == self.bank:
					print(' \u0332', end='')
				else:
					print(' ', end='')
				if [r, c] == coord_r:
					if [r, c] == coord_p:
						print('X ', end='')
					else:
						print('R ', end='')
				elif [r, c] == coord_p:
					print('P ', end='')
				else:
					print('. ', end='')
			print()


def greedy_choose_action(game, Q, GREEDY_FACTOR):
	# choose an action_r
	action_possible_list = game.possible_action(0)
	rand0 = random.uniform(0, 1)
	'''choose by random'''
	if rand0 <= GREEDY_FACTOR:
		action_idx = random.randint(0, len(action_possible_list)-1)
		return action_possible_list[action_idx]
	'''choose by greedy'''
	if DEBUG:
		print('Choose greedy')
	action_tmp = []
	action_possible_list_Q_value = [Q[game.Status, i] for i in action_possible_list]
	'''get the max-Q-val action(s)'''
	max_Q = -1e10
	action_idx_list_with_max_Q = []
	for i, Q_val_tmp in enumerate(action_possible_list_Q_value):
		if Q_val_tmp > max_Q:
			action_idx_list_with_max_Q.clear()
			action_idx_list_with_max_Q.append(i)
			max_Q = Q_val_tmp
		elif Q_val_tmp == max_Q:
			action_idx_list_with_max_Q.append(i)

	'''pick an action'''
	rand = random.randint(0, len(action_idx_list_with_max_Q)-1)

	return action_possible_list[action_idx_list_with_max_Q[rand]]


def simulation(Q, length=100):
	curr_game = ENV()
	curr_game.reward_status = np.zeros(curr_game.NUM_STATUS)
	curr_game.init_reward_status()
	curr_game.draw()
	total_reward = 0
	for t in range(length):
		curr_idx = curr_game.Status
		action_r = greedy_choose_action(curr_game, Q, 0)
		action_p_list = curr_game.possible_action(1)
		action_p = action_p_list[random.randint(0, len(action_p_list) - 1)]
		next_idx = curr_game.perform_action(action_r, action_p)
		curr_game.draw()
		total_reward = curr_game.reward_status[next_idx] + total_reward * DISCOUNT_FACTOR
		print('reward', total_reward)


def q_learning(MAX_ITER=10000000, GREEDY_FACTOR=1):
	game = ENV()
	Q = np.zeros([game.NUM_STATUS, game.NUM_ACTION])
	Q_update_count = np.ones([game.NUM_STATUS, game.NUM_ACTION])
	game.reward_status = np.zeros(game.NUM_STATUS)
	game.init_reward_status()
	my_iter = 0
	ep_cnt = 0
	V_record = []

	# loop for each episode
	while my_iter < MAX_ITER:
		curr_game = ENV()
		total_reward = 0
		if DEBUG:
			print('Init state')
			curr_game.draw()
		# loop for each step of episode:
		while True:
			my_iter += 1
			curr_idx = curr_game.Status
			action_r = greedy_choose_action(curr_game, Q, GREEDY_FACTOR)
			action_p_list = curr_game.possible_action(1)
			action_p = action_p_list[random.randint(0, len(action_p_list) - 1)]
			next_idx = curr_game.perform_action(action_r, action_p)
			if DEBUG:
				print('reward', game.reward_status[curr_idx], ' action:', action_r, action_p)
				curr_game.draw()
			max_Q = max(Q[next_idx, :])
			expectation = game.reward_status[curr_idx] + DISCOUNT_FACTOR * max_Q

			# expectation_list = []
			# max_Q = -1e10
			# for action_p in action_p_list:
			#
			# 	new_idx = curr_game.simulate_action(action_r, action_p)
			# 	max_Q_tmp = max(Q[new_idx, :])
			# 	if max_Q_tmp > max_Q:
			# 		max_Q = max_Q_tmp
			#
			# expectation = game.reward_status[curr_game.Status] + DISCOUNT_FACTOR * max_Q

			# '''get expectation of different police actions'''
			# for action_p in action_p_list:
			# 	new_idx = curr_game.simulate_action(action_r, action_p)
			# 	if DEBUG:
			# 		print('possible: ')
			# 		curr_game.draw(new_idx)
			# 	max_Q = max(Q[new_idx, :])
			# 	if DEBUG:
			# 		print('Q:', Q[new_idx, :])
			# 	val_tmp = game.reward_status[new_idx] + DISCOUNT_FACTOR * max_Q
			# 	if DEBUG:
			# 		print('val', val_tmp)
			# 	expectation_list.append(val_tmp)
			# expectation = sum(expectation_list) / len(expectation_list)

			# update Q value
			if DEBUG:
				print('related Q:', Q[next_idx, :])
				print('Q before update:', Q[curr_idx, :])
			step_size = Q_update_count[curr_idx, action_r] ** (-2/3)
			Q[curr_idx, action_r] = (1-step_size) * Q[curr_idx, action_r] + step_size*expectation
			Q_update_count[curr_idx, action_r] += 1
			if DEBUG:
				print('Q after update :', Q[curr_idx, :])
				print('============================')
				curr_game.draw()
			V_record.append(max(Q[game.Status, :]))
			if (my_iter+1)%10000 == 0:
				print('my_iter(%)', int((my_iter+1)/MAX_ITER*100), V_record[-1])
			# total_reward = game.reward_status[next_idx] + total_reward * DISCOUNT_FACTOR
			total_reward += game.reward_status[next_idx]
			if game.reward_status[next_idx] < 0:
				break

	np.save('V_record', V_record)
	np.save('Q', Q)

	return Q, V_record


def SARSA(MAX_ITER=10000000, GREEDY_FACTOR=0.1):
	game = ENV()
	Q = np.zeros([game.NUM_STATUS, game.NUM_ACTION])
	Q_update_count = np.ones([game.NUM_STATUS, game.NUM_ACTION])
	game.reward_status = np.zeros(game.NUM_STATUS)
	game.init_reward_status()
	my_iter = 0
	ep_cnt = 0
	V_record = []

	# loop for each episode
	while my_iter < MAX_ITER:
		curr_game = ENV()
		total_reward = 0
		if DEBUG:
			print('Init state')
			curr_game.draw()
		# loop for each step of episode:
		next_action_r = greedy_choose_action(curr_game, Q, GREEDY_FACTOR)
		while True:
			my_iter += 1
			curr_idx = curr_game.Status
			action_r = next_action_r
			action_p_list = curr_game.possible_action(1)
			action_p = action_p_list[random.randint(0, len(action_p_list) - 1)]
			next_idx = curr_game.perform_action(action_r, action_p)
			next_action_r = greedy_choose_action(curr_game, Q, GREEDY_FACTOR)

			if DEBUG:
				print('reward', game.reward_status[curr_idx], ' action:', action_r, action_p)
				curr_game.draw()
			expectation = game.reward_status[curr_idx] + DISCOUNT_FACTOR * Q[next_idx, next_action_r]

			# update Q value
			if DEBUG:
				print('related Q:', Q[next_idx, :])
				print('Q before update:', Q[curr_idx, :])
			step_size = Q_update_count[curr_idx, action_r] ** (-2 / 3)
			Q[curr_idx, action_r] = (1 - step_size) * Q[curr_idx, action_r] + step_size * expectation
			Q_update_count[curr_idx, action_r] += 1
			if DEBUG:
				print('Q after update :', Q[curr_idx, :])
				print('============================')
				curr_game.draw()
			V_record.append(max(Q[game.Status, :]))
			if (my_iter + 1) % 10000 == 0:
				print('my_iter(%)', int((my_iter + 1) / MAX_ITER * 100), V_record[-1])
			# total_reward = game.reward_status[next_idx] + total_reward * DISCOUNT_FACTOR
			total_reward += game.reward_status[next_idx]
			if game.reward_status[next_idx] < 0:
				break

	np.save('V_record', V_record)
	np.save('Q', Q)

	return Q, V_record


if __name__ == '__main__':

	LOAD_DATA = False
	# Q, V_record = q_learning(GREEDY_FACTOR=1)
	if LOAD_DATA:
		V_record = np.load('V_record.npy')
		Q = np.load('Q.npy')
	else:
		Q, V_record = SARSA(GREEDY_FACTOR=0.01)

	print(V_record[-1])
	x = range(1, len(V_record)+1)
	plt.plot(x[1000:], V_record[1000:])
	plt.xscale('log')
	plt.show()

	simulation(Q)