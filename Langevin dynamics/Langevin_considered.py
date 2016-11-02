"""Langevin thermostat adds correction terms: one is damping constant gamma * velocity and 
the other with sqrt(2 * gamma * kb * Temp) * R(t) where R(t) is stationary Gaussian process.
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

class L:
# Implement double-well:
	def __init__(self, dt, total, T, initial):
		self.dt = dt
		self.total = total
		self.T = T
		self.old = initial
		self.range, self.velocity, self.v_anyother, self.t_total, self.t_anyother, self.r = Langevin(self.dt, self.total, self.T, self.old)

	def do_langevin_hist(self):
		# get data from Langevin algorithm
		# get statistical parameter from dataset
		self.avg, self.sigma = np.mean(self.r), math.sqrt(np.var(self.r))
		# plot histogram
		self.n, self.bins, self.patches = plt.hist(self.r, 20, normed = True)
		# get a approximated normal curve
		self.y = mlab.normpdf(self.bins, self.avg, self.sigma)
		# put that curve on the graph
		#l = plt.plot(self.bins, self.y, 'r--', linewidth=1)
		# labels and stuff
		U = lambda x: np.exp(-U_1D(x)/self.T)
		K = lambda x: np.exp(-x ** 2 / 2 / self.T)
		partition = np.sqrt(2*np.pi*self.T) * quad(K, -np.inf, np.inf)[0] * quad(U, -np.inf, np.inf)[0]
		diff = (max(self.r) - min(self.r)) / 20
		l = list(range(0, 50))

		b = [min(self.r) - 15 * diff + i * diff for i in l]
		numerator = [np.sqrt(2*np.pi*self.T) * quad(K, -np.inf, np.inf)[0] * quad(U, i, i + 1)[0] for i in b]
		prob = [i / partition for i in numerator]
		plt.xlabel('Position')
		plt.ylabel('Probability')
		plt.title(r'$\mathrm{Histogram\ of\ Position:}\ \mu=self.avg,\ \sigma=self.sigma$')
		ax = plt.gca()
		ax.set_ylim([0, max(max(prob), max(self.n))])
		plt.grid(True)
		bn = [i + 0.5 for i in b]
		plt.plot(bn, prob)
		plt.show()

	def do_q_vs_t(self):
		plt.plot(self.t_total, self.r)
		# labels and stuff
		plt.xlabel('Time')
		plt.ylabel('Position')
		ax = plt.gca()
		ax.set_xlim([0, 200])
		plt.show()

	def do_q_hist(self):
		self.n, self.bins, self.patches = plt.hist(self.r, 50, normed = True)
		x = [i for i in self.bins]
		y1 = [i for i in self.n]
		x.pop()
		plt.plot(x, y1)
		plt.xlabel('Position')
		plt.ylabel('Probability')
		ax = plt.gca()
		ax.set_ylim([0, max(self.n)])
		y2 = [np.exp(-U_1D(i)/self.T) for i in x]
		plt.plot(x, y2)
		plt.show()
			

	def do_langevin_v_to_lnp(self):
		# take log of each pdf at each velocity datapoint in approximation above
		self.lny = np.log(self.n)
		plt.plot(self.bins - 1, self.lny, 'r--', linewidth=1)
		plt.show()
		

U_1D = lambda x: (x + 1) ** 2 * (x - 1) ** 2
F_1D = lambda x: -4 * x ** 3 + 4 * 1 * x

def Langevin(dt, total, T, old, m = 1, r0 = 0, gamma = 1):
	"""
	We need to assign damping(friction) constant gamma, current temperature T, total cycle time T, 
	change of time dt, initial position r0, and mass m
	histogram of v proportioanl to Boltzmann dsn (v vs P)
	"""
	i, kb = 0, 1 # eV
	v_s, r_s, t_s, v_anyother, t_anyother = [], [], [], [], [] 
	E = [0]# Et, Ep, Ek, t_ram, r_ram, v_ram 
	a = (1 - gamma * dt / 2 / m) / (1 + gamma * dt / 2 / m)
	b = 1 / (1 + gamma * dt / 2 / m)
	while i < total:
		# set two independent random forces
		R_new = np.random.normal(scale = math.sqrt(gamma * T * kb))
		R_curr = np.random.normal(scale = math.sqrt(gamma * T * kb))
		# store current position
		memory = r0
		# getting new position from old one via Markov process
		r0 = 2 * b * r0 - a * old + b / m * F_1D(r0) * dt ** 2 + b * dt / 2 / m * (R_curr + R_new)
		# getting current velocity by approximating old and new positions
		v = a / (2 * dt * b ** 2) * (r0 - old) - a * b * (dt ** 3) / (2 * m ** 2) * F_1D(memory) \
		    + b / 2 / m * dt * (a * R_new - R_curr)
		# increasing the loop and replace the old position
		i += 1
		old = memory
		# find any other velocities and corresponding time
		if i % 2 == 0:
			v_anyother += [v]
			t_anyother += [i * dt]
		# gathering data: velocity, position, time, energy, etc
		v_s += [v]
		r_s += [memory]
		t_s += [i * dt]
		E += [m * v ** 2 / 2 + U_1D(memory)]
	# check the final flow of velocities
	return [min(r_s), max(r_s)], v_s, [v_anyother], t_s, t_anyother, r_s

"""Make variance vs. dt plot more "quadratic" aka add more data points"""

"""Construct probability density function of v created by Langevin"""
# Not always arranges to normal
"""algorithm (v vs. P), along with v vs. ln(p)"""

"""Try every v or any other v"""

# Compute <var(v)>
