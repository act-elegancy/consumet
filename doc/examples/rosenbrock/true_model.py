import numpy as np

def simulate(x):
	'''
	This function defines a function z = simulate(x), where x is a 2D variable and
	z is a 1D variable. In this example, simulate defines the Rosenbrock function.
	'''
	return [ (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 ]
