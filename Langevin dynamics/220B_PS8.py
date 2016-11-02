import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.mlab as mla
import math as m

def w(q):
	return (q + 1) ** 2 * (q - 1) ** 2 
def harmonic(q):
	return 1 - 4 * q ** 2
def erfc(x):
	f = lambda x: np.exp(-x**2)
	return 1 - 1 / np.sqrt(np.pi) * quad(f, x, np.inf)[0]
	
def Botlzmann_factor(x, q, kbT = 1):
	return np.exp(x(q)/kbT)

def PhiB_Onsager(q0, kbT = 0.1):
	a = lambda x: Botlzmann_factor(lambda x: w(x), x, kbT)
	return quad(a, -1, q0)[0] / quad(a, -1, 1)[0]

def PhiB_Harmonic(q0, kbT = 0.1):
	a = lambda x: Botlzmann_factor(lambda x: harmonic(x), x, kbT)
	return quad(a, -1, q0)[0] / quad(a, -1, 1)[0]

def result_Onsager(kbT = 0.1):
	l = list(range(1, 1000))
	x = [- k / 1000 for k in l] + [i / 1000 for i in l] + [0]
	x = sorted(x)
	y1 = [PhiB_Onsager(j, 0.1) for j in x]
	y2 = [PhiB_Onsager(j, 0.2) for j in x]
	y4 = [PhiB_Onsager(j, 0.4) for j in x]
	p1, = plt.plot(x, y1, label = "kBT = 0.1")
	p2, = plt.plot(x, y2, label = "kBT = 0.2")
	p3, = plt.plot(x, y4, label = "kBT = 0.4")
	plt.legend(handles = [p1, p2, p3])
	plt.xlabel('q')
	plt.ylabel('Splitting Probability')
	plt.show()

def result_trajection(kbT = 0.1):
	l = list(range(1, 1000))
	x = [- k / 1000 for k in l] + [i / 1000 for i in l] + [0]
	q0 = sorted(x)
	p_collection = []
	for i in q0:
		p_collection += [langevin(i, kbT)]
	plt.plot(q0, p_collection)
	plt.xlabel('q')
	plt.ylabel('Splitting Probability Traj')
	plt.show()

U_1D = lambda x: (x + 1) ** 2 * (x - 1) ** 2
F_1D = lambda x: -4 * x ** 3 + 4 * 1 * x

# def langevin(r, T, dt = 0.01, total = 1, m = 1, gamma = 1):
# 	"""
# 	We need to assign damping(friction) constant gamma, current temperature T, total cycle time T, 
# 	change of time dt, initial position r0, and mass m
# 	histogram of v proportioanl to Boltzmann dsn (v vs P)
# 	"""
# 	r0 = r
# 	t = []
# 	q = []
# 	i, kb = 0, 1 # eV
# 	qa, qb = 0, 0 #number of time reaching qA and qB
# 	old = -0.001
# 	a = (1 - gamma * dt / 2 / m) / (1 + gamma * dt / 2 / m)
# 	b = 1 / (1 + gamma * dt / 2 / m)
# 	while i < total:
# 		while r0 < 0.75 and r0 > -0.75:
# 			# set two independent random forces
# 			R_new = np.random.normal(scale = np.sqrt(gamma * T * kb))
# 			R_curr = np.random.normal(scale = np.sqrt(gamma * T * kb))
# 			# store current position
# 			memory = r0
# 			# getting new position from old one via Markov process
# 			r0 = 2 * b * r0 - a * old + b / m * F_1D(r0) * dt ** 2 + b * dt / 2 / m * (R_curr + R_new)
# 			# getting current velocity by approximating old and new positions
# 			v = a / (2 * dt * b ** 2) * (r0 - old) - a * b * (dt ** 3) / (2 * m ** 2) * F_1D(memory) \
# 			    + b / 2 / m * dt * (a * R_new - R_curr)
# 			# increasing the loop and replace the old position
# 			old = memory
# 			q += [r0]
# 			print(r0)
# 		# if r0 >= 0.75:
# 		# 	qb += 1
# 		# else:
# 		# 	qa += 1
# 		# i += 1
# 		# r0 = r
# 	# check the final flow of velocities
# 	return q
def langevin(r, T, dt = 0.01, total = 100, m = 1, gamma = 1):
	"""
	We need to assign damping(friction) constant gamma, current temperature T, total cycle time T, 
	change of time dt, initial position r0, and mass m
	histogram of v proportioanl to Boltzmann dsn (v vs P)
	"""
	r0 = r
	t = []
	q = []
	i, kb = 0, 1 # eV
	qa, qb = 0, 0 #number of time reaching qA and qB
	old = -0.001
	while i < total:
		while r0 < 0.75 and r0 > -0.75:
			# set two independent random forces
			R_new = np.random.normal(scale = np.sqrt(gamma * T * kb))
			R_curr = np.random.normal(scale = np.sqrt(gamma * T * kb))
			# store current position
			memory = r0
			# getting new position from old one via Markov process
			r0 = r0 + dt/gamma * F_1D(r0) + R_new*np.sqrt(dt)
			# getting current velocity by approximating old and new positions
			# increasing the loop and replace the old position
			old = memory
		if r0 >= 0.75:
			qb += 1
		else:
			qa += 1
		i += 1
		r0 = r
	return qb/(qa+qb)
	
		

def result_Onsager_vs_harmonic(kbT = 0.1):
	l = list(range(1, 1000))
	x = [- k / 1000 for k in l] + [i / 1000 for i in l] + [0]
	x = sorted(x)
	y1 = [PhiB_Onsager(j, kbT) for j in x]
	y2 = [PhiB_Harmonic(j, kbT) for j in x]
	p1, = plt.plot(x, y1, label = "Onsager")
	p2, = plt.plot(x, y2, label = "Harmonic")
	plt.legend(handles = [p1, p2])
	plt.xlabel('q')
	plt.ylabel('Splitting Probability')
	plt.show()
