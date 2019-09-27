'''
This file contains an interface for running the derivative-free blackbox
optimization program NOMAD. This is used to perform adaptive sampling.
'''

import numpy as np
import subprocess
import pickle
import sys
import os

from conf import conf

def sample(surrogates, params, indices):
	'''
	This function writes out a NOMAD-compatible config file, and then
	executes the derivative-free optimizer NOMAD. Previous sample data
	in `samples.csv` is automatically fed to NOMAD during initialization
	(via `x0` option below combined with memoization in `error_max.py`),
	and both box and inequality constraints are handled automatically.
	The new points sampled by NOMAD are finally stored in `samples.csv`.
	'''
	
	# Save parameters inside surrogate models
	for i in indices:
		surrogates[i].p = params[i]
	
	# Load of the inequality constraints and the bounds update
	# [Assumption: all surrogates have the same input bounds]
	if surrogates[0].data: 
		with open('InEq_constraints.pkl', 'rb') as f:
			ineq_A, ineq_b, ind_flow = pickle.load(f)
		num_constraints = len(ineq_b)
	else:
		num_constraints = 0
	
	# Load previous sample points from comma-separated file
	x0 = np.genfromtxt('samples.csv', delimiter=',')[:,1:1+conf['input_dim']]
	
	# Save previous sample points in a nomad-compatible file
	np.savetxt('nomad.dat', x0)
	
	# Initialize file used to record errors
	np.savetxt('nomad.err', [0])
	
	# Set default values for NOMAD options
	options = \
	{
		'bb_exe':             '"$python {}"'.format(os.path.relpath(os.path.join(sys.path[0], conf['nomad_exe']))),
		'bb_output_type':     'obj' + ' eb'* num_constraints,
		'dimension':          conf['input_dim'],
		'max_bb_eval':        conf['nomad_num'] + len(x0),
		'stat_sum_target':    conf['nomad_tol'],
		'lower_bound':        '* 0.0',
		'upper_bound':        '* 1.0',
		'display_degree':     1,
		'stop_if_feasible':   False,
		'vns_search':         True,
		'display_all_eval':   False,
		'display_stats':      'bbe sol obj',
		'stats_file':         'nomad.log bbe sol %.2e obj',
		'x0':                 'nomad.dat'
	}
	
	# Write the NOMAD options to a file
	filename = 'nomad.conf'
	with open(filename, 'w') as f:
		for k in options:
			if type(options[k]) is not bool:
				v = str(options[k])
			else:
				v = 'yes' if options[k] else 'no'
			f.write(k + ' ' + v + '\n')
	
	# Save the surrogate model as pkl
	with open('surrogate.pkl', 'wb') as f:
		pickle.dump([[surrogates, indices]], f)
	
	# Make sure the output file is empty
	with open('nomad.csv', 'w') as f:
		f.write('')
	
	# Call NOMAD as a Python subprocess
	process = subprocess.Popen(['nomad','nomad.conf'])
	process.communicate()
	
	# Raise an exception if it did not succeed
	if process.returncode:
		raise RuntimeError('NOMAD exited with errors. Aborting...')
	
	# Count the size of the existing sample data
	with open('samples.csv', 'r') as old:
		count = sum(1 for i in old.readlines())
	
	# Merge new sample data into old sample data
	with open('nomad.csv', 'r') as new:
		with open('samples.csv', 'a+') as old:
			for line in new.readlines():
				old.write(str(count) + ',' + line)
				count += 1
	
	# Load error estimate from file
	error = np.loadtxt('nomad.err')
	
	# Remove unneccesary output files
	os.remove('surrogate.pkl')
	os.remove('nomad.0.log')
	os.remove('nomad.csv')
	os.remove('nomad.err')
	os.remove('nomad.dat')
	os.remove('nomad.conf')
	
	return error
