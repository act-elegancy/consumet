'''
This file contains the regression model used to fit surrogate models
to sampled data. More specifically, the class below defines both the
Pyomo model used for surrogate fitting, the information criteria used
to perform model selection, and helper methods to interact with these.
Note that this is closely related to the file `surrogate.py`, which
defines the Surrogate class used to instantiate Regression objects.
'''
from pyomo.environ import *
from pyomo.opt import *
import numpy as np
import sys

class Regression:
	'''
	This object is used to:
	 * Generate penalized regression models for surrogate construction;
	 * Fit these regression models to previously obtained sample data;
	 * Automatically select the penalty based on information criteria.
	Note that the information criteria used for penalty selection are
	defined at the end of this class, and new ones can easily be added.
	'''
	
	def __init__(self, surrogate):
		'''
		Argument:
			Surrogate model object containing all the model details.
		
		Returns:
			Abstract Pyomo model which can be used for regression.
		'''
		
		# Create an abstract optimization model (concrete data can be loaded later)
		self.model = AbstractModel()
		
		# Declare the sets used for variable indices etc.
		self.model.SampleSet    = Set(dimen=1, ordered=True)
		self.model.IndvarSet    = RangeSet(0, surrogate.dim        - 1, 1)
		self.model.RegressorSet = RangeSet(0, len(surrogate.index) - 1, 1)
		
		# Set without first element (representing constant term θ₀)
		self.model.ConstraintSet = RangeSet(1, len(surrogate.index) - 1, 1)
		
		# Scaled versions of the independent variables
		self.model.x = Param(self.model.IndvarSet, self.model.SampleSet)
		
		# Actual output model/plant output
		self.model.z = Param(self.model.SampleSet)
		
		# LASSO regularization parameter λ (Lagrange multiplier)
		self.model.penalty = Param(domain=Reals, mutable=True)
		
		# Regression parameters (currently initialized to 0 with bounds [-100,100])
		self.model.theta = Var(self.model.RegressorSet, domain=Reals, initialize=0, bounds=(None,None))
		
		# Reformulate L1 terms as constraints and auxiliary variables
		# (The reformulation is |θ| = |θ+| + |θ-| s.t. θ = θ+ - θ-)
		self.model.theta_p    = Var(self.model.ConstraintSet, domain=NonNegativeReals)
		self.model.theta_m    = Var(self.model.ConstraintSet, domain=NonNegativeReals)
		self.model.constraint = \
			Constraint(self.model.ConstraintSet, rule=lambda m, i:
			           m.theta_p[i] - m.theta_m[i] == m.theta[i])
		
		# Save a copy of the surrogate object inside the model
		self.surrogate = surrogate
		
		# Define the objective function used for LASSO regression
		# (Note that for penalty=0, we get ordinary least squares.)
		self.model.objective = \
			Objective(sense=minimize, rule=lambda m:
			         (sum((m.z[i] - self.surrogate(m.x, m.theta, pos=i))**2 for i in m.SampleSet) \
			          + m.penalty * sum(m.theta_p[j] + m.theta_m[j] for j in m.ConstraintSet)))
		
		# Check what information criterion to use to set the penalty
		key = 'regpen_crit'
		val = surrogate.conf[key]
		try:
			self.criterion = getattr(self, 'criterion_' + val)
		except:
			raise ValueError('"{}" cannot be set to "{}".'.format(key, val))
		
		# Load miscellaneous config options
		self.penalty_lim = surrogate.conf['regpen_lim']
		self.penalty_num = surrogate.conf['regpen_num']
		self.penalty_lb  = surrogate.conf['regpen_lb']
		self.penalty_ub  = surrogate.conf['regpen_ub']
	
	def fit(self, coords, results, penalty):
		'''
		This function performs a regression to a Pyomo model using provided
		sampling coordinates and sampling results for a given penalty value.
		
		Arguments:
			coords:
				2D Numpy array containing all sampling coordinates.
			results:
				Values of the true function at those coordinates.
			penalty:
				Size of the regression penalty factor.
		'''
		
		# Pack sampling data into a Pyomo-compatible format
		data = {}
		data['penalty']   = {None: penalty}
		data['SampleSet'] = {None: [i for i, _ in enumerate(coords)]} 
		data['x'] = {(i,j) : v for (j,i), v in np.ndenumerate(coords)}
		data['z'] = { i    : v for i,     v in np.ndenumerate(results)}
		
		# Instantiate the Pyomo model using this data
		instance = self.model.create_instance({None: data})
		
		# Perform the regression using ipopt
		ipopt  = SolverFactory('ipopt', warmstart=True)
		output = ipopt.solve(instance, load_solutions=False)
		
		# Check if the regression succeeded
		if output.solver.termination_condition == TerminationCondition.optimal:
			# Extract the successful results
			instance.solutions.load_from(output)
			
			# Extract the regression parameters
			self.theta = [value(instance.theta[i]) for i in instance.RegressorSet]
			
			# Eliminate parameters below cutoff
			pmax = max(np.abs(self.theta))
			for i, p in enumerate(self.theta):
				if np.abs(p/pmax) < self.penalty_lim:
					self.theta[i] = 0
			
			# Calculate the model error. The infinitesimal constant added
			# at the end prevents crashes when calculating information criteria
			# for models that are *exactly* correct (which can happen for trivial
			# test functions such as f(x)=1-x, but rarely happens in real life).
			self.error = sum((results[i] - self.surrogate(xi, self.theta))**2
			             for i, xi in enumerate(coords))/len(results) + 1e-256
		else:
			# Use fallback values in case of failure
			self.theta = [np.nan for i in instance.RegressorSet]
			self.error = np.nan
	
	
	def autofit(self, coords, results):
		'''
		This function tries to find an optimal penalized regression
		model using an information criterion to compare penalties. 
		In other words, while Regression.fit requires the user to
		manually set a regression penalty, Regression.autofit will
		autodetect an optimal penalty using an information criterion.
		What information criterion is used (AICC, AIC, BIC, etc.)
		can be specified by the user at runtime using `config.ini`.
		
		TODO:
		 * In order to speed up the implementation, we might want to look into
		   whether we can reuse one Pyomo instance instead of recreating it.
		   (This might be possible if we mark all parameters as 'Mutable'.)
		 * We might also want to change the logspace-search to a more efficient
		   optimization algorithm if determining λ is a performance bottleneck.
		'''
		
		# Define the penalty search space
		domain = np.logspace(np.log10(self.penalty_lb),
		                     np.log10(self.penalty_ub),
		                     self.penalty_num)
		
		# Storage for information criteria
		values = np.zeros(np.size(domain))
		
		# Perform regressions for each penalty value
		print('Penalty \tCriterion')
		for i, penalty in enumerate(domain):
			# Perform a new regression for this penalty
			self.fit(coords, results, penalty)
			
			# Count the number of finite parameters
			params = len(np.nonzero(self.theta)[0])
			
			# Count the total sample size
			samples = len(results)
			
			# Evaluate information criterion
			values[i] = self.criterion(samples, params, self.error)
			
			# Print status information
			if np.isnan(values[i]):
				# Failed results
				values[i] = np.inf
				print('% 8.4e\t [diverged]' % penalty)
			else:
				# Valid results
				print('% 8.4e\t% 8.4e' % (penalty, values[i]))
		
		# Find the penalty that minimizes the information criterion
		penalty = domain[np.argmin(values)]
		
		# Perform one last fit using that result
		self.fit(coords, results, penalty)
		
		# Confirm that the final results are usable
		if np.isnan(self.error) or any(np.isnan(self.theta)):
			print('Regression did not converge for any value of the penalty parameter.')
			sys.exit(1)
		
		# Print status report and return
		print(f'\nChoosing regression penalty:\n  λ = {penalty:.2e}')
	
	
	#################################################################
	# INFORMATION CRITERIA
	#################################################################
	# Define the information criteria available for model evaluation.
	# All such functions should be static methods, and their names
	# should start with `criterion_`. They will then be made available
	# automatically: if we e.g. set the option `regpen_crit`
	# to `aicc` in the config file, the program automatically searches
	# for a function named `criterion_aicc` defined below. Each such
	# criterion should take as inputs (i) number of sample points, 
	# (ii) number of nonzero parameters, and (iii) mean square error.
	#################################################################
	
	@staticmethod
	def criterion_aic(samples, params, error):
		'''Akaike information criterion.'''
		return samples*np.log(error) + 2*params
	
	@staticmethod
	def criterion_bic(samples, params, error):
		'''Bayesian information criterion.'''
		return samples*np.log(error) + np.log(samples)*params
	
	@staticmethod
	def criterion_hqic(samples, params, error):
		'''Hannan-Quinn information criterion.'''
		return samples*np.log(error) + 2*np.log(np.log(samples))*params
	
	@staticmethod
	def criterion_aicc(samples, params, error):
		'''Akaike information criterion with low-sample corrections.'''
		return (samples-params-2)*np.log(error) + 2*params
	
	@staticmethod
	def criterion_bicc(samples, params, error):
		'''Bayesian information criterion with low-sample corrections.'''
		return (samples-params-2)*np.log(error) + np.log(samples)*params
	
	@staticmethod
	def criterion_hqicc(samples, params, error):
		'''Hannan-Quinn information criterion with low-sample corrections.'''
		return (samples-params-2)*np.log(error) + 2*np.log(np.log(samples))*params
