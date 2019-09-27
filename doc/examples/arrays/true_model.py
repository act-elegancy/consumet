import numpy as np

def simulate(x):
	'''
	This function defines a function z = simulate(x), where x is an 2D variable and
	z is a 2D variable. This file combines the `rosenbrock` and `ripples` examples.
	'''
	z = []
	z.append( (1-x[0])**2+100*(x[1]-x[0]**2)**2 )
	z.append( np.sin(np.sqrt(np.dot(x,x))) )
	return z
