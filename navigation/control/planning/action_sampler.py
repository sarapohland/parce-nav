import os
import random
import numpy as np
import matplotlib.pyplot as plt


class ActionSampler():
	def __init__(self, H, N, u_low, u_high, turn_in_place=False, perturbation=False, Sigma=None, beta=0.25):
		self.H = H
		self.N = N
		self.u_low = u_low
		self.u_high = u_high
		self.turn = turn_in_place
		self.perturbation = perturbation
		self.Sigma = Sigma if Sigma is not None else np.eye(2)
		self.beta = beta

		self.C = 10
		self.cmds = []
		self.labels = []

		self.v_mid = (u_high[0] + u_low[0]) / 2
		self.w_mid = (u_high[1] + u_low[1]) / 2

	def go_straight(self):
		v = np.random.uniform(self.u_low[0], self.u_high[0])
		u = np.array([v, 0])
		cmds = np.array([u] * self.H)
		if self.perturbation:
			cmds = self.add_perturbation(cmds)
		if v < self.v_mid:
			self.labels.append(0)
		else:
			self.labels.append(1)
		return cmds

	def concave_turn(self):
		u = np.random.uniform(self.u_low, self.u_high)
		cmds = np.array([u] * self.H)
		if self.perturbation:
			cmds = self.add_perturbation(cmds)
		if (u[0] < self.v_mid) and (u[1] < self.w_mid):
			self.labels.append(2)
		elif (u[0] >= self.v_mid) and (u[1] < self.w_mid):
			self.labels.append(3)
		elif (u[0] < self.v_mid) and (u[1] >= self.w_mid):
			self.labels.append(4)
		elif (u[0] >= self.v_mid) and (u[1] >= self.w_mid):
			self.labels.append(5)
		return cmds 

	def convex_turn(self, cmd_rate):
		t = int((0.785 / self.u_high[1]) * cmd_rate)
		# t = int(2 * np.pi / self.u_high[1] / 0.727)
		# t = int(np.random.uniform(self.H / 2, self.H))
		if np.random.uniform() > 0.5:
			u1 = np.array([np.array([0, self.u_high[1]])] * t)
			v2 = np.random.uniform(self.u_low[0], self.u_high[0])
			w2 = np.random.uniform(self.u_low[1], 0)
			u2 = np.array([np.array([v2, w2])] * (self.H - t))
			if self.perturbation:
				u2 = self.add_perturbation(u2)
			if v2 < self.v_mid:
				self.labels.append(6)
			else:
				self.labels.append(7)
		else:
			u1 = np.array([np.array([0, self.u_low[1]])] * t)
			v2 = np.random.uniform(self.u_low[0], self.u_high[0])
			w2 = np.random.uniform(0, self.u_high[1])
			u2 = np.array([np.array([v2, w2])] * (self.H - t))
			if self.perturbation:
				u2 = self.add_perturbation(u2)
			if v2 < self.v_mid:
				self.labels.append(8)
			else:
				self.labels.append(9)
		return np.vstack((u1, u2))

	def turn_in_place(self):
		w = np.random.uniform(self.u_low[1], self.u_high[1])
		u = np.array([0, w])
		return np.array([u] * self.H)

	def sample_action_seq(self, cmd_rate):
		maxint = 4 if self.turn else 3
		self.cmds, self.labels = [], []
		for i in range(self.N):
			rand = random.randint(1,maxint)
			if rand == 1:
				self.cmds.append(self.go_straight())
			elif rand == 2:
				self.cmds.append(self.concave_turn())
			elif rand == 3:
				self.cmds.append(self.convex_turn(cmd_rate))
			elif rand == 4:
				self.cmds.append(self.turn_in_place())
		return np.array(self.cmds)

	def add_perturbation(self, cmds):
		eps = np.dot(np.random.normal(size=np.shape(cmds)), self.Sigma)
		eps [:, 0] *= self.beta
		for i in range(1, np.shape(cmds)[1]):
			eps[:, i] = self.beta * eps[:, i] + (1 - self.beta) * eps[:, i-1]
		cmds += eps

		if self.u_low is not None and self.u_high is not None:
			cmds = np.clip(cmds, self.u_low, self.u_high)

		return cmds

	def sample_rand_action_seq(self, cmd_rate, lin_accel, ang_accel):
		# Sample sequence of actions
		dt = 1 / cmd_rate
		all_cmds = [np.random.uniform(self.u_low, self.u_high, (self.N, 2))]
		for i in range(1, self.H):
			all_cmds.append(np.random.normal(loc=all_cmds[-1], scale=[lin_accel*dt, ang_accel*dt]))
		all_cmds = np.stack(all_cmds, axis=1)
		
		# Clip actions to control bounds
		all_cmds = np.clip(all_cmds, self.u_low, self.u_high)

		return np.array(all_cmds)

def plot_trajectories(trajs, to_save):
	if os.path.isdir(to_save):
		for i, traj in enumerate(trajs):
			plt.figure()
			xs = [pos[0] for pos in traj]
			ys = [pos[1] for pos in traj]
			thetas = [pos[2] for pos in traj]
			plt.scatter(xs, ys, c='black', marker='.')
			for x, y, theta in zip(xs, ys, thetas):
				plt.arrow(x, y, np.cos(theta)/25, np.sin(theta)/25, color='b')
			plt.xlim([-0.1,1.2])
			plt.ylim([-0.5,0.5])
			plt.xlabel('X Position')
			plt.ylabel('Y Position')
			plt.title('Sample Trajectory {}'.format(i))
			plt.savefig(to_save + 'traj{}.png'.format(i))

	else:
		plt.figure()
		for i, traj in enumerate(trajs):
			xs = [pos[0] for pos in traj]
			ys = [pos[1] for pos in traj]
			plt.plot(xs, ys)
			plt.xlim([-0.1,1.4])
			plt.ylim([-0.6,0.6])
			plt.xlabel('X Position')
			plt.ylabel('Y Position')
			plt.title('Sample Trajectories')
			plt.savefig(to_save)
