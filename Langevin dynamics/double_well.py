"""Pre-determined constants in global frame, use hydrogen as an example"""

def L_J_potential(r, sigma = 0.3345, epsilon = 0.0661):
	"""
	Use Lennard-Jones potential as standard calculation of potential between any two particles
	in the system.
	>>> L_J_potential(1)
	0
	>>> L_J_potential(2)
	- 0.0625
	"""
	ratio = sigma / r
	return 4 * epsilon * (ratio ** 12 - ratio ** 6)

# 1-D double-well is quite simple as follow:
import matplotlib.pyplot as plt
import numpy as np

def verlet_1D(dt, mass = 1, adj = 2, repeats = 100000, p0 = 0):
	"""
	From initial position, we can keep track of x as t changes by dt each time, until reach repeats limit.
	>>> dt = 
	"""
	i, old, t, x, E, v = 0, - dt, [0], [p0], [0], [0]
	double_well_1D = lambda x: x * x * (x + adj) * (x - adj)
	force_1D = lambda x: -(4 * x ** 3 - adj * adj * 2 * x)
	Ep = [double_well_1D(old)]
	while i < repeats:
		memory = p0
		p0 = 2 * p0 - old + force_1D(p0) * dt * dt / mass
		x += [memory]
		t += [i * dt]
		v += [(old - p0) / dt]
		old = memory
		Ep += [double_well_1D(p0)]
		E += [mass / 2 * ((old - p0) / dt) ** 2 + double_well_1D(p0)]
		i += 1
	return np.var(v)

def t(intersect):
	return [j * intersect for j in range(1,51)]

def data_gathering(dt):
	"""
	dt is a list of dt that we want to keep as x coordinate. It returns a list of y and 
	plot it out.
	"""
	v = [verlet_1D(i) for i in dt]
	plt.plot(dt, v)
	plt.show()
	return
	
		

class particle():
	"""
	Attributes of a single particle: mass, position, momentum (velocity), where
	both position and momentum are lists with same length.
	"""
	def __init__(self, mass, potential, coordinate = [0, 0, 0]):
		self.mass = mass
		self.coordinate = coordinate
		self.potential = potential

	# def force(self, interval):
	# 	"""
	# 	Find force acting on the particle based on its potential
	# 	"""
	# 	current_potential = self.potential(self.coordinate)
	# 	r = sum([x * x for x in self.coordinate])
	# 	fx = current_potential * (-2) / 
	# 	return 


class system():
	"""A system class where each desired particle is assigned"""
	def __init__(self, ):
		super(system, self).__init__()
		self.arg = arg
		
	
		