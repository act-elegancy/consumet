import numpy as np

def simulate(x):
	'''
	This function defines a function z = simulate(x), where x is a 1D variable and
	z is an 2D variable. In this example, the elements of z are simply the Chebyshev
	polynomials T_n, resulting in a simple way to test the surrogate modeling tool.
	'''

	# Calculate the 10 first Chebyshev polynomials
	T = [1, x[0]]
	for n in range(8):
		T.append(2*x[0]*T[-1] - T[-2])
	
	# Generate the output z based on these
	z = []
	z.append(T[0])
	z.append(1*T[1] + 3*T[3] + 5*T[5])
	z.append(2*T[2] + 4*T[4] + 6*T[6])

	# Return this list of polynomials
	return z
