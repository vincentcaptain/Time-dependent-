import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed
import multiprocessing
import random



"""
Some of the important factors are below: time dependent potential V_rt,
one-dimensional net force F_1D, random Gaussian number generator Xi, and
timestep rescaling factor rescaling_c.
"""
V_rt = lambda r, t, omega, epsilon: r ** 4 / 4 - 9 / 2 * r ** 2 + epsilon * np.sin(omega * t) * r
F_1D = lambda r, t, omega, epsilon: - r ** 3 + 9 * r - epsilon * np.sin(omega * t)
Xi = lambda gamma, beta, m, dt: np.random.normal(scale = np.sqrt(m / beta * (1 - np.exp(- gamma * dt))))
rescaling_c = lambda dt, gamma: np.sqrt(2 / gamma / dt * np.tanh(gamma * dt / 2))

def waiting_time(i, total = 10, dt = 0.01, r0 = - 4, m = 1, gamma = 1, epsilon = 2, beta = 1):
	"""
	Parameters required for this simulation: time difference dt, initial position
	r0, mass of particle m, damping constant gamma, time-dependent part Parameter
	epsilon and omega, beta = 1/kBT.

	Steps for this simulation: define storage for the crossing time and waiting
	time; forming a loop for each omega; within the loop, update momentum every
	half dt, and position every dt; count the time it crosses the barrier until
	10 times, finding the average time scale;
	"""
	t_continue, t_collection, left = 0, [0], True
	p_half, c = 0, rescaling_c(dt, gamma)
	p = np.random.normal(scale = np.sqrt(m/beta))
	indicator_count, p_upper, p_lower, cross_count = 0, 0.05, -0.05, 0
	while cross_count < total:
		one_fourth_random = Xi(gamma, beta, m, dt)
		three_fourth_random = Xi(gamma, beta, m, dt)
		p_half = p * np.exp(- gamma * dt / 2) + one_fourth_random + F_1D(r0, t_continue, i, epsilon) * c * dt / 2
		r0 = r0 + p_half * c * dt / m
		t_continue += dt
		p = (p_half + F_1D(r0, t_continue, i, epsilon) * c * dt / 2) * np.exp(- gamma * dt / 2) + three_fourth_random
		if r0 >= p_lower and r0 <= p_upper:
			indicator_count += 1
		if (left and r0 > 0):
			left = not left
			cross_count += 1
		if (not left and r0 < 0):
			left = not left
			cross_count += 1
	k = total / t_continue
	probability = indicator_count / t_continue
	return k, probability

def initial_flux(i, From, To, limit = 0.5, dt = 0.01, m = 1, gamma = 1, epsilon = 2, beta = 1):
	"""
	Initial flux is calculated by # of reach / t_obs, where # of reach should not be too far away.
	The same position and velocity propagation and update methods as waiting_time.
	"""
	r0, t_obs = From, 0
	p_half, p, c = 0, np.random.normal(scale = np.sqrt(m/beta)), rescaling_c(dt, gamma)
	while r0 < To:
		one_fourth_random = Xi(gamma, beta, m, dt)
		three_fourth_random = Xi(gamma, beta, m, dt)
		p_half = p * np.exp(- gamma * dt / 2) + one_fourth_random + F_1D(r0, t_obs, i, epsilon) * c * dt / 2
		r0 = r0 + p_half * c * dt / m
		t_obs += dt
		p = (p_half + F_1D(r0, t_obs, i, epsilon) * c * dt / 2) * np.exp(- gamma * dt / 2) + three_fourth_random
	if r0 > To + limit:
		return t_obs, False
	else:
		return t_obs, True

def initial_prob(i, From, To, sample_size, dt = 0.01, m = 1, gamma = 1, epsilon = 2, beta = 1):
	"""
	Initial probability is calculated by # of reach / total # of attempts.
	"""
	r0, j, t_obs = From, dt, 0
	p_half, c = 0, rescaling_c(dt, gamma)
	p = np.random.normal(scale = np.sqrt(m/beta))
	total, p_re, p_collection, t_collection = 0, [], [], []
	while total < sample_size:
		while r0 < To:
			one_fourth_random = Xi(gamma, beta, m, dt)
			three_fourth_random = Xi(gamma, beta, m, dt)
			p_half = p * np.exp(- gamma * dt / 2) + one_fourth_random + F_1D(r0, t_obs, i, epsilon) * c * dt / 2
			r0 = r0 + p_half * c * dt / m
			t_obs += dt
			p = (p_half + F_1D(r0, t_obs, i, epsilon) * c * dt / 2) * np.exp(- gamma * dt / 2) + three_fourth_random
			j += dt
		p_re += [1 / j]
		p_collection += [p]
		t_collection += [t_obs]
		total += 1
		r0, j, t_obs = From, 1, 0
		p_half, c = 0, rescaling_c(dt, gamma)
		p = np.random.normal(scale = np.sqrt(m/beta))
	return np.mean(p_re), p_collection, t_collection

def transition_prob(i, From, To, sample_size, start, p_collection, t_collection, limit = 0.05, dt = 0.01, m = 1, gamma = 1, epsilon = 2, beta = 1):
	forward, backward, r0, t_obs = 0, 0, From, random.choice(t_collection)
	p_half, c = 0, rescaling_c(dt, gamma)
	p = p_collection[t_collection.index(t_obs)]
	p_collection1, t_collection1 = [], []
	while forward < sample_size:
		while From <= r0 <= To:
			one_fourth_random = Xi(gamma, beta, m, dt)
			three_fourth_random = Xi(gamma, beta, m, dt)
			p_half = p * np.exp(- gamma * dt / 2) + one_fourth_random + F_1D(r0, t_obs, i, epsilon) * c * dt / 2
			r0 = r0 + p_half * c * dt / m
			t_obs += dt
			p = (p_half + F_1D(r0, t_obs, i, epsilon) * c * dt / 2) * np.exp(- gamma * dt / 2) + three_fourth_random
		if r0 >= To:
			forward += 1
			t_collection1 += [t_obs]
			if p_half > p:
				p_collection1 += [p_half]
			else:
				p_collection1 += [p]
		else:
			backward += 1
		r0 = From
		t_obs = random.choice(t_collection)
		p_half, p = 0, p_collection[t_collection.index(t_obs)]
	result = forward / (forward + backward)
	return result, p_collection1, t_collection1



	

# def plottingk(x, y):
# 	plt.plot(x, y[0])
# 	plt.xlabel('Omega')
# 	plt.ylabel('rate')
# 	ax = plt.gca()
# 	ax.set_ylim([min(y)*1.1, max(y)*1.1])
# 	pp = PdfPages('rate.pdf')
# 	pp.savefig()
# 	pp.close()

# def plottingp(x, y):
# 	plt.plot(x, y[1])
# 	plt.xlabel('Omega')
# 	plt.ylabel('rate')
# 	ax = plt.gca()
# 	ax.set_ylim([min(y)*1.1, max(y)*1.1])
# 	pp = PdfPages('probability.pdf')
# 	pp.savefig()
# 	pp.close()

def process_waiting_time(omega):
	return waiting_time(omega)

def process_initial_flux(omega, num_cross, From, To):
	i, t_obs = 0, 0
	while i < num_cross:
		t = initial_flux(omega, From, To)
		if t[1]:
			t_obs += t[0]
			i += 1
	return num_cross / t_obs 


def total_prob(omega, sample_size, interval, starting):
	"""
	interval describes the range of calculation for transition_prob,
	Steps describes number of transition_prob we calculated, and 
	starting is the starting position of the particle.
	"""
	From = starting
	To = interval[0]
	pi = initial_prob(omega, From, To, sample_size)
	pi_zero = pi[0]
	p_collection = pi[1]
	t_collection = pi[2]
	final_p, p_track = pi_zero, [pi_zero]
	i = 0
	starting = interval[0]
	while i < len(interval) - 1:
		From = To
		i += 1
		To = interval[i]
		trans = transition_prob(omega, From, To, sample_size, starting, p_collection, t_collection)
		trans_p = trans[0]
		p_collection = trans[1]
		t_collection = trans[2]
		final_p *= trans_p
		p_track += [trans_p]
	return final_p * 10 ** 10

def accurate_k(omega, sample_size, interval, starting, final_p):
	flux = process_initial_flux(omega, sample_size, starting, interval[0])
	return flux * final_p
		
omega = list(range(1, 75))
o = [-i/10 for i in omega] + [i/10 for i in omega]
omega = sorted(o)
num_cores = multiprocessing.cpu_count()
sample_size = 2000
starting = -3
r = [-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.5, 1, 1.5, 2]
logfinal = total_prob(0, sample_size, r, starting)
# r = [-2, -1.8]
# result = Parallel(n_jobs = num_cores)(delayed(process_waiting_time)(i) for i in omega)
# logfinal_p1 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p2 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p3 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p4 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p5 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p6 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p7 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p8 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p9 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# logfinal_p10 = Parallel(n_jobs = num_cores)(delayed(total_prob)(i, sample_size, r, starting) for i in omega)
# flux = Parallel(n_jobs = num_cores)(delayed(process_initial_flux)(i, sample_size, r, starting for i in omega)
np.savetxt("logfinal_p.txt", [logfinal])
# np.savetxt("flux.txt", flux)
# np.savetxt("rate_accurate.txt", result)




