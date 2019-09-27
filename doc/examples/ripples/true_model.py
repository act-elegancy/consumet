import numpy as np

def simulate(x):
	'''
	This function defines a function z = simulate(x), where x is an ND variable and
	z is a 1D variable. In this example, simulate defines sin[sqrt(x·x)] function.
	'''
	return [ np.sin(np.sqrt(np.dot(x,x))) ]
