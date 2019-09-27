'''
This file contains the actual surrogate model function. In other words:
the class below defines the basis functions used for model construction,
as well as helper methods for evaluating the surrogate model afterwards.

TODO:
 * If anyone has ideas for how to send a slice x[:,i] from
   Pyomo instead of explicitly operating on 2D arrays below, 
   feel free to implement it. (I couldn't get that to work.)
'''

import numpy as np
import scipy.special
import itertools
import functools
import operator

class Surrogate:
	'''
	This class is a general representation of a surrogate model.
	When you instantiate the object, you have to provide a dict
	filled with configuration options (usually read from file).
	The constructor than figures out the dimension of the input
	space and what basis functions to use for model construction.
	The result is a callable object, which evaluates a surrogate
	model of the given class when provided with values for the
	relevant regression parameters (p) and variable values (x).
	
	The surrogate model is constructed based on 1-dimensional
	basis functions bₙ(u). The simplest example is the monomial
	basis bₙ(u) = uⁿ. When modeling e.g. a 3-dimensional input
	space, and using a basis of order 2, we could then use the
	following products as our 3-dimensional basis functions:
	  1, x, y, z, x², y², z², xy, xz, yz.
	In the more general notation, these can be written:
	  b₀(x) b₀(y) b₀(z),
	  b₁(x) b₀(y) b₀(z),    b₀(x) b₁(y) b₀(z),    b₀(x) b₀(y) b₁(z),
	  b₂(x) b₀(y) b₀(z),    b₀(x) b₂(y) b₀(z),    b₀(x) b₀(y) b₂(z),
	  b₁(x) b₁(y) b₀(z),    b₁(x) b₀(y) b₁(z),    b₀(x) b₁(y) b₁(z).
	In other words, we construct products bₙ(x) bₘ(y) bₖ(z) such
	that the subscripts n+m+k ≤ t, where t=2 was the total order
	used above. This strategy is easily generalized to any dimension,
	using any set of basis functions {bₙ}, and any model order t.
	
	Note that the 3-dimensional basis functions bₙ(x) bₘ(y) bₖ(z)
	above can be classified by their indices (n,m,k). All allowed
	such index tuples are stored in the member variable self.index.
	
	Note also that the basis functions bₙ(u) available are defined
	near the end of this class, and new ones can easily be added.
	'''
	
	def __init__(self, conf):
		'''
		Construct a surrogate model. All parameters necessary for
		model creation should be provided via the dictionary `conf`.
		'''
		
		# Save all config options
		self.conf = conf
		
		# Dimension of the input variable
		self.dim = conf['input_dim']
		
		# How many basis functions to use
		self.order = conf['model_order']
		
		# Used to save regression parameters
		self.p = None
		
		# Used for constraints data file
		self.data = None
		
		# Select what class of basis functions to use
		key = 'model_class'
		val = conf[key]
		try:
			self.basis = getattr(self, 'basis_' + val)
		except:
			raise ValueError('"{}" cannot be set to "{}".'.format(key, val))
		
		# Calculate all acceptable basis index tuples. This is done by:
		#  * Generating a list of acceptable indices in one dimension,
		#    which can be directly found from the variable self.order;
		#  * Taking the Cartesian product of one such list per dimension;
		#  * Constraining the sum of indices in order to get the acceptable
		#    subset of basis function indices in higher dimensions.
		possible = itertools.product(range(self.order + 1), repeat=self.dim)
		
		# All combinations of dim-tuples with one basis index from each range
		self.index = [k for k in possible if sum(k) <= self.order]
		
		# Number of regression coefficients
		self.terms = len(self.index)
		
		# Save the variable bounds (used for standardizing variables)
		self.lower_bound = conf['input_lb']
		self.upper_bound = conf['input_ub']
		self.xs = [np.max([np.abs(b[0]), np.abs(b[1])]) for b
		           in zip(self.lower_bound, self.upper_bound)]
	
	def __call__(self, x, p=None, pos=None):
		'''
		Evaluate the surrogate model using regression parameters
		given by `p` and input variables given by `x`. If the
		parameters p are not specified, the model will attempt
		to look for regression parameters saved in the object.
		
		Arguments:
			x:
				Variables  used in the model (x).
			p:
				Parameters used in the model (θ).
			pos:
				If this argument is set, the function is called
				from Pyomo, and this is an additonal array index.
		'''
		
		# Check whether regression parameters have been supplied
		if p is None: p = self.p
		
		# For technical reasons, the code below requires 2D arrays to work
		# with the Pyomo model. This converts a 1D array to a 2D array if
		# such an array is provided from NumPy/NOMAD instead of Pyomo.
		if pos is None:
			pos = 0
			x = np.asmatrix(x).transpose()
		
		# Actual model defintion
		return sum(p[j] * self.product(x, j, pos) for j in range(len(p)))
		
	def standard(self, x):
		'''
		Standardize variables x based on their known bounds.
		'''
		return [(x[i] - self.lower_bound[i]) / (self.upper_bound[i] - self.lower_bound[i]) for i in range(self.dim)]
	
	def restore(self, x):
		'''
		Restore the true values of standardized variables x.
		'''
		return [self.lower_bound[i] + x[i] * (self.upper_bound[i] - self.lower_bound[i]) for i in range(self.dim)]
	
	def product(self, x, n, m):
		'''
		This function constructs the n'th basis function in any dimension from
		the known basis functions in one dimension. The result is evaluated at x.
		'''
		
		# Evaluate basis function number self.index[n][k] at point x[k,m]
		factors = (self.basis(x[k, m], self.index[n][k]) for k in range(self.dim))
		
		# Multiply all the one-dimensional results to get the net result
		return functools.reduce(operator.mul, factors, 1)
	
	#################################################################
	# BASIS FUNCTIONS
	#################################################################
	# Define the basis functions available for model construction.
	# All basis functions should be static methods, and their names
	# should start with `basis_`. They will then be made available
	# automatically: if we e.g. set the option `model_class`
	# to `taylor` in the config file, the program automatically
	# searches for a function named `basis_taylor` below.
	#
	# The basis function itself should take in a variable x∊[0,1] and
	# integer n≥0, and then return the value of the n'th basis function
	# evaluated at x. Note that you only need to define one-dimensional
	# basis functions, since the higher-dimensional basis functions are
	# automatically constructed from the products of these functions.
	#################################################################
	
	@staticmethod
	def basis_taylor(x, n):
		'''
		Monomial basis x**n. Using this as a basis yields a Taylor expansion
		around the lower-bound corner of the data set (i.e. the point x=0).
		'''
		return x**n
	
	@staticmethod
	def basis_legendre(x, n):
		'''
		Legendre polynomial P_n(x). These are rescaled from having a domain
		of [0,1] to [-1,1], since that's where they form an orthogonal basis.
		'''
		return scipy.special.eval_legendre(n, 2*x - 1)
	
	@staticmethod
	def basis_chebyshev(x, n):
		'''
		Chebyshev polynomial T_n(x). These are rescaled from having a domain
		of [0,1] to [-1,1], since that's where they form an orthogonal basis.
		'''
		return scipy.special.eval_chebyt(n, 2*x - 1)
	
	@staticmethod
	def basis_fourier(x, n):
		'''
		Fourier sine series. If n=0, this function simply returns 1, corresponding
		to a constant term. If n>1, it returns sin(πnx), which alternate between
		being even and odd functions with respect to the centerpoint of [0,1].
		'''
		return np.sin(np.pi*n*x) if n > 0 else 1
