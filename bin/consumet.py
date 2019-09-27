#!/usr/bin/env python3

'''
This is the main program of the project, and is used to construct
surrogate models for a user-provided simulator via a mixture of:
 * Batch sampling (via DoE methods or loading presampled data);
 * Adaptive sampling (error-maximization with anti-clustering);
 * Constrained sampling (either box or inequality constraints);
 * Penalized regression (used for automatic variable selection);
 * Information criteria (used to choose the regression penalty).
For more information on the theoretical background, installation,
and actual usage of this program, we refer to the documentation.
'''

# Import system libraries
import numpy as np
import pickle
import math
import time
import sys
import os

# Add project libraries to path
sys.path.append(os.path.normpath(os.path.join(__file__, '..', '..', 'lib')))

# Import the project libraries
import regression
import nomad
from conf import conf, confs
from surrogate import Surrogate 
from regression import Regression 
from doe import DoECase, DoEDef

try:
	# Import true_model from runtime directory
	sys.path.append(os.getcwd())
	from true_model import simulate
except:
	print('Error importing function `simulate` from `true_model.py`. Please ensure that ' + os.path.join(os.getcwd(), 'true_model.py') + ' exists and contains a function `simulate`.')
	sys.exit(1)


############################################################
# PART I: Initialization procedure.
############################################################
# This includes loading existing data from a user-provided
# file, and setting up the surrogate and regression models.
############################################################

# Start a timer used for benchmarks
timer = time.time()/60

# Instantiate the surrogate models. The array `surrogates` contains all the
# constructed surrogate models, while `surrogate` is a nickname used for the
# first element of that array. The latter should only be used to access
# attributes that are common for all instances of the Surrogate class.
surrogates = np.array([Surrogate(conf) for conf in confs], dtype = object) 
surrogate  = surrogates[0] 

# Instantiate the regression objects used to fit each surrogate model
regressions = np.array([Regression(surrogate) for surrogate in surrogates], dtype = object)

# Instantiate the arrays that will be used to save regression parameters
thetas = [[] for _ in range(conf['output_dim'])]

# List of messages to user, which will be printed after the simulation
log = []

# Simple function used to print headlines
def hline(message, character):
	print('\n' + message + '\n' + character*len(message))

# If a batch file is available, load existing samples from it
if conf['batch_file']:
	coords_un = np.genfromtxt(conf['batch_file'], delimiter=',')[:, 1:1+conf['input_dim'] ]
	coords    = np.array([surrogate.standard(x) for x in coords_un])
	results   = np.genfromtxt(conf['batch_file'], delimiter=',')[:,   1+conf['input_dim']:]
else:
	coords  = np.zeros([0, conf['input_dim' ]])
	results = np.zeros([0, conf['output_dim']])



############################################################
# PART II: Batch sampling based on a DoE method.
############################################################
# If the user provides a batch file with a sufficient amount
# of data, this part is skipped. If not, the program
# samples either (i) a user-defined number of batch samples,
# or (ii) the minimum number of samples needed to fit an
# initial surrogate model, whichever of these is largest.
#
# NOTE:
# Currently, many potential parameters are not included.
# They are not necessary at this stage, but can easily be
# included as additional configuration parameters if needed.
############################################################

# Print a status message
hline('Batch sampling', '=')
print(' * Selecting points via the ' + conf['batch_doe'] + ' method.')

# Design-of-experiment case definition
doe_case = DoECase()
doe_case.n_samp = conf['batch_num']
doe_case.method = conf['batch_doe']
doe_case.ind = np.arange(conf['input_dim'])
doe_case.corn_points = bool(conf['batch_corn'])
doe_case.lb_m_ub = np.array([surrogate.lower_bound, (surrogate.lower_bound+surrogate.upper_bound)/2, surrogate.upper_bound])

# The number of initial sampling points should at least be as large as
# the number of regression parameters to avoid having an underdetermined
# regression problem. Note that for regular grids, doe_case.n_samp is the
# number of sampling points in *each* dimension of input space, while for
# all other sampling methods it refers to the *total* number of points.

# Largest number of regression coefficients among surrogate models
terms = np.max([s.terms for s in surrogates]) 

# Note that we subtract `len(results)` when calculating `minimum`, to
# account for the data from the optional user-provided `batch_file`.
# We prevent this from becoming negative to avoid problems below.
diff = terms - len(results) + 3
if diff < 0:
	diff = 0

# Calculate the minimum number of sampling points required based on `diff`
if doe_case.method == 'RegularGrid':
	minimum = math.ceil(diff**(1/len(doe_case.ind)))
	if doe_case.n_samp < minimum:
		log.append('Updated number of batch sampling points ({} → {}) to fit {} coefficients.'
		          . format(doe_case.n_samp**len(doe_case.ind), minimum**len(doe_case.ind), terms))
		doe_case.n_samp = minimum
else:
	minimum = diff
	if doe_case.n_samp + 2**conf['input_dim'] * doe_case.corn_points < minimum:
		log.append('Updated number of batch sampling points ({} → {}) to fit {} coefficients.'
		          . format(doe_case.n_samp, minimum, terms))
		doe_case.n_samp = minimum - 2**conf['input_dim'] * doe_case.corn_points

# Perform DoE construction if batch sampling of a finite number of points is required
if minimum < 1:
	print(' * No batch sampling seems to be required.')
else:
	# Calculation of polytope based on the provided data if provided
	if conf['input_file']:
		# Read data from CSV, assuming a header line
		data = np.genfromtxt(conf['input_file'], delimiter=',')
		
		# Assumption 1: The 3 different constraint types are hard coded into the
		# problem combining all flows. This can be adjusted later if desired.
		tmp_1 = np.shape(data)[1]
		doe_case.ind_flow = data[0, np.isfinite(data[0,:])].astype(int)
		data_ind_flow 	  = tuple(np.arange(data.shape[1])[np.isfinite(data[0,:])])
		data_ind_ratio    = tuple((data_ind_flow[k], data_ind_flow[i]) for i in range(tmp_1) for k in range(i+1, tmp_1))
		data_ind_sum      = tuple((data_ind_flow[k], data_ind_flow[i]) for i in range(tmp_1) for k in range(i+1, tmp_1))
		data = data[2:, :]
		data_dict = {'Domain':    data,
		             'Ind_flow':  data_ind_flow,
		             'Ind_ratio': data_ind_ratio,
		             'Ind_sum':   data_ind_sum}
		doe_def = DoEDef(doe_case, Data_dict=data_dict)
		
		# Adjustment of the lower and upper bounds of the problem
		for surrogate in surrogates:
			surrogate.lower_bound[doe_case.ind_flow] = doe_def.poly.Bounds[0, 0:tmp_1]
			surrogate.upper_bound[doe_case.ind_flow] = doe_def.poly.Bounds[1, 0:tmp_1]
		
		ineq_A = doe_def.poly.InEq_A[len(data_ind_flow)*2:, :]
		ineq_b = doe_def.poly.InEq_b[len(data_ind_flow)*2:]
		ineq_A = doe_def.poly.InEq_A[len(data_ind_flow)*2:len(data_ind_flow)*2+5, :]
		ineq_b = doe_def.poly.InEq_b[len(data_ind_flow)*2:len(data_ind_flow)*2+5]
		
		# Extraction and save of the inequality constraints and the bounds update
		with open('InEq_constraints.pkl', 'wb') as f:
			pickle.dump([ineq_A, ineq_b, doe_case.ind_flow], f)
		
		# Parameter specifying that data has been used
		surrogates[0].data = True 
		
	else:
		doe_def = DoEDef(doe_case)
		surrogates[0].data = False
	
	# Perform simulations at each of these sample coordinates,
	# while keeping any previously loaded coords and results.
	coords_un = doe_def.DoE(doe_case)
	print(' * Sampling the simulator at {} points.'.format(len(coords_un)))
	coords    = np.append(coords,  np.array([surrogate.standard(x) for x in coords_un]), axis=0)
	results   = np.append(results, np.array([simulate(x)           for x in coords_un]), axis=0)

# Write the initial results to output files
with open('samples.csv', 'w') as f:
	samples = np.array([[i, *v[0], *v[1]] for i, v in enumerate(zip(coords, results))])
	np.savetxt(f, samples, fmt='%i'+',%e'*(len(samples[0])-1))



############################################################
# PART IIIA: Adaptive sequential sampling.
############################################################
# Adaptive sampling via error-maximization sampling. In this
# version of the code, we attempt to create one surrogate
# per output variable, by selecting one model at a time and
# sampling points in regions where that model is deficient.
# When all surrogates have converged, all the gathered data
# is used to refit one final surrogate per output variable.
############################################################

if conf['adapt_type'] == 'seq' and conf['adapt_num'] > 0:
	# Write status information to stdout and to the log
	hline('\nAdaptive sampling', '=')
	log.append('Chose the sequential strategy for adaptive sampling.')
	
	# Loop over output dimensions and adaptive sampling iterations
	for k in range(conf['output_dim']):
		for i in range(conf['adapt_num']):
			# Fit surrogate model k via penalized regression
			hline(f'Regression #{i+1} for model #{k+1}', '-')
			regressions[k].autofit(coords, results[:, k])
			thetas[k] = regressions[k].theta
			
			# Perform adaptive sampling of submodel k
			hline(f'Adaptive sampling #{i+1} for model #{k+1}', '-')
			error = nomad.sample(surrogates, thetas, [k]).max()
			
			# Read the sampled data from files
			samples = np.loadtxt('samples.csv', delimiter=',')
			coords  = samples[:, 1:1+conf['input_dim'] ]
			results = samples[:,   1+conf['input_dim']:]
			
			# Terminate if the error is within tolerance
			print(f'\nMaximal error:\n  ε = {error:.2e}')
			if error < conf['adapt_tol']:
				break
		
		# Log the observed error
		log.append(f'Maximum relative error discovered in model #{k+1} during sampling: ε = {error:.2e}.')
		if error > conf['adapt_tol']:
			log.append(f'This error exceeds tolerance; consider changing adapt_num, model_class, or model_order.')



############################################################
# PART IIIB: Adaptive simultaneous sampling.
############################################################
# Adaptive sampling via error-maximization sampling. In this
# version of the code, we attempt to create one surrogate
# model per output variable, by sampling points in regions
# where the average of the model errors is maximized. In
# other words, we simultaneously improve all surrogates
# by gathering data in regions all models are deficient.
############################################################

elif conf['adapt_type'] == 'sim' and conf['adapt_num'] > 0:
	# Write status information to stdout and to the log
	hline('\nAdaptive sampling', '=')
	log.append('Chose the simultaneous strategy for adaptive sampling.')
	
	# Loop over adaptive sampling iterations
	for i in range(conf['adapt_num']):
		# Fit each surrogate model via penalized regression
		for k in range(conf['output_dim']):
			hline(f'Regression #{i+1} for model #{k+1}', '-')
			regressions[k].autofit(coords, results[:, k])
			thetas[k] = regressions[k].theta
		
		# Perform simultaneous adaptive sampling of all models
		hline(f'Adaptive sampling #{i+1} for all models', '-')
		error = nomad.sample(surrogates, thetas, np.arange(conf['output_dim'])).max()
		
		# Read the sampled data from files
		samples = np.loadtxt('samples.csv', delimiter=',')
		coords  = samples[:, 1:1+conf['input_dim'] ]
		results = samples[:,   1+conf['input_dim']:]
		
		# Terminate if the error is within tolerance
		print(f'\nMaximal error:\n  ε = {error:.2e}')
		if error < conf['adapt_tol']:
			break
	
	# Record the final error to log
	log.append(f'Maximum relative error discovered in models during sampling: ε = {error:.2e}.')
	if error > conf['adapt_tol']:
		log.append(f'This error exceeds tolerance; consider changing adapt_num, model_class, or model_order.')



############################################################
# PART IIIC: No adaptive sampling required.
############################################################

else:
	# Write status information to the log
	log.append('Chose to not perform any adaptive sampling (probably due to adapt_num=0).')



############################################################
# PART IV: Finalization
############################################################
# In this block, we write results to file and stdout.
############################################################

# Perform final regressions using all sampled data. This is
# necessary since (1) sampling is done *before* regressions
# in Part III, and (2) sequential sampling of new submodels
# results in data that can be used to improve all submodels.
hline('\nSampling complete', '=')
for k in range(conf['output_dim']):
	hline(f'Final regression for model #{k+1}', '-')
	regressions[k].autofit(coords, results[:, k])
	thetas[k] = regressions[k].theta   

# Write the final results (unscaled) to output files
with open('samples.csv', 'w') as f:
    samples = np.array([[i, *v[0], *v[1]] for i, v in enumerate(zip((surrogate.restore(x) for x in coords), results))])
    np.savetxt(f, samples, fmt='%i'+',%e'*(len(samples[0])-1))

# Calculate the elapsed time
timer = time.time()/60 - timer
log.append(f'Sampling and surrogate fitting required in total {timer:.1f} min.')

# Remove saved constraint data
if surrogate.data:
	os.remove('InEq_constraints.pkl')

# Extract the final regression coefficients
coefficients = [(i, *surrogates[i].index[j], v) 
                for i, theta in enumerate(thetas)
                for j, v     in enumerate(theta)]

# Save the regression coefficients to file
log.append('Saved surrogate model coefficients to regression.csv.')
np.savetxt('regression.csv', coefficients, fmt='%i,'*(1+len(surrogates[0].index[0]))+'%e')

# Save the surrogate models to file
log.append('Saved binary copies of surrogate models to surrogate.pkl.')
with open('surrogate.pkl', 'wb') as f:
	pickle.dump(surrogates, f)

# Display final results
hline('\nSurrogate generation complete', '=')
print('\nModels:')
for i, conf in enumerate(confs):
	print(' * Model #{}: {} series of order {}, with coefficients θ({}; *).'.format(i+1, conf['model_class'].capitalize(), conf['model_order'], i))
print('\nCoefficients:')
for c in coefficients:
	print('  θ{} = {}{:<g}'.format(c[:-1], '-' if c[-1]<0 else ' ', np.abs(c[-1])).replace(',', ';', 1))

# Playback message log
if log:
	print('\n * '.join(['\nNOTES:', *log]))
