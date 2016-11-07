import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed
import multiprocessing


"""
Some of the important factors are below: time dependent potential V_rt,
one-dimensional net force F_1D, random Gaussian number generator Xi, and
timestep rescaling factor rescaling_c.
"""
V_rt = lambda r, t, omega, epsilon: r ** 4 / 4 - 9 / 2 * r ** 2 + epsilon * np.sin(omega * t) * r
F_1D = lambda r, t, omega, epsilon: - r ** 3 + 9 * r - epsilon * np.sin(omega * t)
Xi = lambda gamma, beta, m, dt: np.random.normal(scale = np.sqrt(2 * m / beta * (1 - np.exp(- gamma * dt))))
rescaling_c = lambda dt, gamma: np.sqrt(2 / gamma / dt * np.tanh(gamma * dt / 2))

def waiting_time(i, total = 10, dt = 0.01, r0 = - 3, m = 1, gamma = 1, epsilon = 2, beta = 1):
	"""
	Parameters required for this simulation: time difference dt, initial position
	r0, mass of particle m, damping constant gamma, time-dependent part Parameter
	epsilon and omega, beta = 1/kBT.

	Steps for this simulation: define storage for the crossing time and waiting
	time; forming a loop for each omega; within the loop, update momentum every
	half dt, and position every dt; count the time it crosses the barrier until
	10 times, finding the average time scale;
	"""
	t_continue, t_count, t_collection, crossing_collection, r_collection, left = 0, 0, [0], [], [r0], True
	p_half, p, c = 0, 0, rescaling_c(dt, gamma)
	indicator_count, p_upper, p_lower, loop_count = 0, 0.05, -0.05, 1
	while len(crossing_collection) < total:
		one_fourth_random = Xi(gamma, beta, m, dt)
		three_fourth_random = Xi(gamma, beta, m, dt)
		p_half = p * np.exp(- gamma * dt / 2) + one_fourth_random + F_1D(r0, t_continue, i, epsilon) * c * dt / 2
		r0 = r0 + p_half * c * dt / m
		t_continue += dt
		t_count += dt
		p = (p_half + F_1D(r0, t_continue, i, epsilon) * c * dt / 2) * np.exp(- gamma * dt / 2) + three_fourth_random
		if (left and r0 > 0):
			left = not left
		if (not left and r0 < 0):
			left = not left
			crossing_collection += [t_count]
			t_count = 0
		if r0 >= p_lower and r0 <= p_upper:
			indicator_count += 1
		loop_count += 1
	k = total / np.mean(crossing_collection) / 2
	probability = indicator_count / loop_count / dt
	return k, probability


def plottingk(x, y):
	plt.plot(x, y[0])
	plt.xlabel('Omega')
	plt.ylabel('rate')
	ax = plt.gca()
	ax.set_ylim([min(y)*1.1, max(y)*1.1])
	pp = PdfPages('rate.pdf')
	pp.savefig()
	pp.close()

def plottingp(x, y):
	plt.plot(x, y[1])
	plt.xlabel('Omega')
	plt.ylabel('rate')
	ax = plt.gca()
	ax.set_ylim([min(y)*1.1, max(y)*1.1])
	pp = PdfPages('probability.pdf')
	pp.savefig()
	pp.close()

omega = list(range(1, 75))
o = [-i/10 for i in omega] + [i / 10 for i in omega]
omega = sorted(o)

def process_waiting_time(omega):
	return waiting_time(omega)

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs = num_cores)(delayed(process_waiting_time)(i) for i in omega)

np.savetxt("rate_accurate.txt", results)
plottingk(omega, result[0])
plottingp(omega, result[1])




