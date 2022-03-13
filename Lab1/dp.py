import numpy as np
from math import *
import copy as cp
import sys
import random
import matplotlib.pyplot as plt

random.seed()


class MDP:
	def __init__(self, T=15, allow_minotaur_stay=False, status=None, p_death=1/30):
		# type = 0: minotaur cannot stand still

		self.T = T
		self.rowSize = 5			# max row
		self.colSize = 6			# max col

		# self.reward = 0

		self.allow_minotaur_stay = allow_minotaur_stay
		self.p_death = p_death
		self.alive = True

		self.goal = [4, 4]
		self.MAP_SIZE = self.colSize * self.rowSize
		self.NUM_STATUS = self.MAP_SIZE ** 2

		if status is None:
			self.player = [0, 0]				# player coord
			self.minotaur = [4, 4]				# minotaur coord
			self.status_idx = self.status_to_index(self.player, self.minotaur)
		else:
			self.status_idx = status
			[self.player, self.minotaur] = self.index_to_status(status)
		# self.STATUS = []
		# self.status_init()
		# self.dead = False

		self.actionDict = {'stay': 0, 'up': 1, 'right': 2, 'down': 3, 'left': 4}
		self.actionList = [0, 1, 2, 3, 4]
		self.actionName = ['stay', 'up', 'right', 'down', 'left']

		# wall   x-y location + forbidden actions   e.g, {(0,1), [2]}
		self.wall = {}
		self.define_wall()

		self.SUCCESS_STATUS_IDX = []
		for r_m in range(self.rowSize):
			for c_m in range(self.colSize):
				coord_p = self.goal
				coord_m = [r_m, c_m]
				if coord_p != coord_m:
					self.SUCCESS_STATUS_IDX.append(self.status_to_index(coord_p, coord_m))

	def define_wall(self):
		self.wall[(0, 1)] = [self.actionDict['right']]
		self.wall[(1, 1)] = [self.actionDict['right']]
		self.wall[(2, 1)] = [self.actionDict['right']]
		self.wall[(3, 1)] = [self.actionDict['down']]
		self.wall[(4, 1)] = [self.actionDict['up']]

		self.wall[(0, 2)] = [self.actionDict['left']]
		self.wall[(1, 2)] = [self.actionDict['left']]
		self.wall[(2, 2)] = [self.actionDict['left']]
		self.wall[(3, 2)] = [self.actionDict['down']]
		self.wall[(4, 2)] = [self.actionDict['up']]

		self.wall[(1, 3)] = [self.actionDict['right']]
		self.wall[(2, 3)] = [self.actionDict['right']]
		self.wall[(3, 3)] = [self.actionDict['down']]
		self.wall[(4, 3)] = [self.actionDict['up'], self.actionDict['right']]

		self.wall[(1, 4)] = [self.actionDict['left'], self.actionDict['down']]
		self.wall[(2, 4)] = [self.actionDict['left'], self.actionDict['up']]
		self.wall[(3, 4)] = [self.actionDict['down']]
		self.wall[(4, 4)] = [self.actionDict['up'], self.actionDict['left']]

		self.wall[(1, 5)] = [self.actionDict['down']]
		self.wall[(2, 5)] = [self.actionDict['up']]

	def show_map(self):
		for r in range(self.rowSize):
			hWalls = []
			vWalls = []
			for c in range(self.colSize):
				if self.player == [r, c]:
					print(' O', end='')
				elif self.minotaur == [r, c]:
					print(' X', end='')
				else:
					print(' .', end='')
				# sys.stdout.flush()
				walls = self.wall.get((r, c))
				if walls is not None:
					if 2 in walls:
						print('|', end='')
						vWalls.append(True)
					else:
						print(' ', end='')
						vWalls.append(False)
					if 3 in walls:
						hWalls.append(True)
					else:
						hWalls.append(False)
				else:
					print(' ', end='')
					vWalls.append(False)
					hWalls.append(False)
					# sys.stdout.flush()
			print()
			for i in range(6):
				if hWalls[i] and vWalls[i]:
					print('--|', end='')
				elif hWalls[i] and not vWalls[i]:
					print('---', end='')
				elif not hWalls[i] and vWalls[i]:
					print('  |', end='')
				elif not hWalls[i] and not vWalls[i]:
					print('   ', end='')
				# sys.stdout.flush()
			print()

	def show_map_small(self):
		for r in range(self.rowSize):
			for c in range(self.colSize):
				walls = self.wall.get((r, c))
				if walls:
					if 3 in walls:
						if self.player == [r, c]:
							print('\u0332O', end='')
						elif self.minotaur == [r, c]:
							print('\u0332X', end='')
						else:
							print('\u0332.', end='')
					else:
						if self.player == [r, c]:
							print('O', end='')
						elif self.minotaur == [r, c]:
							print('X', end='')
						else:
							print('.', end='')
					if 2 in walls:
						print('|', end='')
					else:
						print(' ', end='')
				else:
					if self.player == [r, c]:
						print('O ', end='')
					elif self.minotaur == [r, c]:
						print('X ', end='')
					else:
						print('. ', end='')
					sys.stdout.flush()
			print()

	def check_move(self, action_p, action_m, coord_p=None, coord_m=None, status=None):
		if status is None:
			if coord_p is None:
				coord_p = cp.copy(self.player)
			if coord_m is None:
				coord_m = cp.copy(self.minotaur)
		else:
			[coord_p, coord_m] = self.index_to_status(status)

		# for minotaur:
		if action_m == 0:
			if not self.allow_minotaur_stay:
				return False
		elif action_m == 1:
			if coord_m[0] <= 0:
				return False
		elif action_m == 2:
			if coord_m[1] >= self.colSize - 1:
				return False
		elif action_m == 3:
			if coord_m[0] >= self.rowSize - 1:
				return False
		elif action_m == 4:
			if coord_m[1] <= 0:
				return False

		# for player:
		if action_p == 0:
			pass
		elif action_p == 1:
			if coord_p[0] <= 0:
				return False
		elif action_p == 2:
			if coord_p[1] >= self.colSize - 1:
				return False
		elif action_p == 3:
			if coord_p[0] >= self.rowSize - 1:
				return False
		elif action_p == 4:
			if coord_p[1] <= 0:
				return False

		# wall detect
		wall_check_coord = (coord_p[0], coord_p[1])
		action_blocked = self.wall.get(wall_check_coord)
		if action_blocked is not None and action_p in action_blocked:
			return False

		return True

	# def move(self, target, action, verbose):
	# 	ss = ''
	# 	if target == 0:
	# 		ss += 'P'
	# 		[r, c] = self.player
	# 	else:
	# 		ss += 'm'
	# 		[r, c] = self.minotaur
	# 	ss += '(' + str(r) + ', ' + str(c) + ') '
	# 	if action == 0:
	# 		ss += ' stand still'
	# 	elif action == 1:
	# 		r -= 1
	# 		ss += ' move up'
	# 	elif action == 2:
	# 		c += 1
	# 		ss += ' move right'
	# 	elif action == 3:
	# 		r += 1
	# 		ss += ' move down'
	# 	elif action == 4:
	# 		c -= 1
	# 		ss += ' move left'
	# 	if target == 0:
	# 		self.player = [r, c]
	# 	else:
	# 		self.minotaur = [r, c]

	def perform_move(self, action_p, action_m):
		# return [coord_p, coord_m, idx]
		coord_p = cp.copy(self.player)
		coord_m = cp.copy(self.minotaur)
		if action_p == 0:
			pass
		elif action_p == 1:
			coord_p[0] -= 1
		elif action_p == 2:
			coord_p[1] += 1
		elif action_p == 3:
			coord_p[0] += 1
		elif action_p == 4:
			coord_p[1] -= 1

		if action_m == 0:
			pass
		elif action_m == 1:
			coord_m[0] -= 1
		elif action_m == 2:
			coord_m[1] += 1
		elif action_m == 3:
			coord_m[0] += 1
		elif action_m == 4:
			coord_m[1] -= 1

		idx = self.status_to_index(coord_p, coord_m)
		return [coord_p, coord_m, idx]

	def update_status(self, status_idx):
		self.status_idx = status_idx
		[self.player, self.minotaur] = self.index_to_status(status_idx)

	def Bellman_eq(self, u, policy, t):
		# t chosen from T-1 ~ 0
		# update u[t, st] and chosen_action[t, st]

		u_best = -1e10
		action_best = 0
		for action_p in range(len(self.actionList)):		# player action
			prob = 0
			status_next = []
			reward = 0
			if (self.player == self.minotaur) or (self.player == self.goal):  # already fail or success
				prob = 1
				status_next.append(self.status_idx)
				reward = 0
			else:
				for action_m in range(len(self.actionList)):		# minotaur action
					if self.check_move(action_p, action_m):
						[_, _, idx] = self.perform_move(action_p, action_m)
						# if idx in self.SUCCESS_STATUS_IDX:
						# 	reward = 1
						status_next.append(idx)
				if len(status_next) == 0:
					prob = 0
				else:
					prob = (1-self.p_death)/len(status_next)
				# if p_death is not 0, then it has p_death probability to fail.
				# as ut(st) = max[ rt(st, a) + Sum(pt(j|st, a)*u_t+1(s_t+1)),
				# where u_t+1 = 0 if play dead, so the p(dead)*u_t+1 will not
				# affect the value of ut
			u_value = reward
			for st in status_next:
				u_value += prob*u[t+1, st]
			if u_value > u_best:
				u_best = u_value
				action_best = action_p
		u[t, self.status_idx] = u_best
		policy[t, self.status_idx] = action_best

	# def transition_probability_mat(self, policy, t=None, coord_p=None, coord_m=None, status=None):
	# 	# Judge the transition probability with status and time
	# 	# Return the transition matrix (5*5), row presents p_action, col presents m_action
	# 	# res_mat[i, j] means player takes i action, minotaur takes j action
	#
	# 	if t is None:
	# 		t = self.T
	# 	if status is None:
	# 		if coord_p is None:
	# 			coord_p = self.player
	# 		if coord_m is None:
	# 			coord_m = self.minotaur
	# 		idx = self.status_to_index(coord_p, coord_m)
	# 	else:
	# 		idx = self.status_idx
	#
	# 	res_mat = np.zeros([len(self.actionList), len(self.actionList)])
	# 	if self.player == self.minotaur:
	# 		return
	# 	if self.player == self.goal:
	# 		return
	#
	# 	if self.type == 0:
	# 		m_prob = 0.25 * np.ones(5)
	# 		m_prob[0] = 0
	# 	else:
	# 		m_prob = 0.2 * np.ones(5)
	#
	# 	p_prob = policy[t, idx, :]
	#
	# 	for r in range(len(self.actionList)):			# row
	# 		for c in range(len(self.actionList)):		# col
	# 			res_mat[r, c] = p_prob[r] * m_prob[c]
	# 			'''i, j order???'''
	# 	return res_mat

	def status_to_index(self, coord_p=None, coord_m=None):
		if coord_p is None:
			coord_p = self.player
		if coord_m is None:
			coord_m = self.minotaur
		t1 = coord_p[0] * self.colSize + coord_p[1]
		t2 = coord_m[0] * self.colSize + coord_m[1]
		idx = t1 + t2 * self.MAP_SIZE
		''' debug '''
		return idx

	def index_to_status(self, idx):
		if idx == -1:
			print('Dead status')
			return [[], []]
		r = idx % self.MAP_SIZE
		d = idx // self.MAP_SIZE
		coord_m = [d//self.colSize, d%self.colSize]
		coord_p = [r//self.colSize, r%self.colSize]
		return [coord_p, coord_m]

	def simulation(self, policy, show=False):
		record = [self.status_idx]
		if show:
			self.show_map_small()
		for t in range(self.T):
			action_p = int(policy[t, self.status_idx])
			if self.player == self.minotaur:  # fail
				break
			elif not self.alive:			# dead fail
				break
			elif self.player == self.goal:  # success
				break
			else:
				if show:
					print('Round ' + str(t) + ' Player choose ' + self.actionName[action_p])
				possible_status = []
				for action_m in range(len(self.actionList)):		# minotaur action
					if self.check_move(action_p, action_m):
						[_, _, idx] = self.perform_move(action_p, action_m)
						possible_status.append(idx)
				if self.p_death > 0:
					prob = (1-self.p_death)/len(possible_status)
					rand = random.uniform(0, 1)
					p_sum = 0
					for pp in range(len(possible_status)):
						p_sum += prob
						if p_sum >= rand:
							self.update_status(possible_status[pp])
							break
					if p_sum < rand:
						self.alive = False
						if show:
							print('Round ' + str(t) + ' Player dead')
						record.append(-1)
						break
				else:
					rand = random.randint(0, len(possible_status)-1)
					self.update_status(possible_status[rand])
				record.append(self.status_idx)
				if show:
					self.show_map_small()
		return record


if __name__ == "__main__":
	# game = MDP()
	# MDP().show_map()
	# game.show_map_small()
	read_policy = False

	if read_policy:
		policy = np.load('policy.npy')
	else:
		record = []
		for T in range(1, 15 + 1):  # different ending time cases   T = 1 ~ 15
			game = MDP(T=T)
			policy = np.zeros([game.T + 1, game.NUM_STATUS])
			u = np.zeros([game.T+1, game.NUM_STATUS])

			# # init success status
			for idx in game.SUCCESS_STATUS_IDX:
				u[T, idx] = 1

			# start with the status before final status (t=14) and end with
			#  init status (t=1)
			for t in range(game.T-1, -1, -1):
				for s_idx in range(game.NUM_STATUS):
					new_game = MDP(status=s_idx)
					new_game.Bellman_eq(u, policy, t)

					'''   end here  11/24-5:10  '''

			idx = MDP().status_idx
			record.append(u[0, idx])
			print('T = '+str(T)+' end')

		np.save('policy', policy)

	# Q 1.2
	tmp = MDP()
	status_record = tmp.simulation(policy, show=True)

	# Q 1.3
	# count = 0
	# for i in range(10000):
	# 	tmp = MDP()
	# 	if (i + 1) % 1000 == 0:
	# 		status_record = tmp.simulation(policy, show=True)
	# 	else:
	# 		status_record = tmp.simulation(policy, show=False)
	# 	if not tmp.alive:
	# 		# print('Fail')
	# 		pass
	# 	elif status_record[-1] in tmp.SUCCESS_STATUS_IDX:
	# 		count += 1
	# 		# print('Success')
	# 	if (i + 1) % 1000 == 0:
	# 		print('---- Simulation ' + str(i+1) + ', success prob ' + str(count/(i+1)))
	#
	# print('success prob ' + str(count/10000))

	print(record)
	x = [i + 1 for i in range(len(record))]
	plt.plot(x, record)
	plt.show()