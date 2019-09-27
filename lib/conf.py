'''
This file contains a config file parser, which is used to read
key-value options for the model from a standard ini file.  The
variable `conf` that is exported by this file is a Python dict,
and can e.g. be passed as kwargs into functions.
'''

import configparser as cp
import numpy as np
import ast
import sys
import os
import re



############################################################
# Declaration of input files, default values, and data types
############################################################

# Declare filename used as config file
filename = 'config.ini'

# Check that the config file exists
if not os.path.isfile(filename):
	print('Error loading configuration file `config.ini`. Please ensure that ' + os.path.join(os.getcwd(), 'config.ini') + ' exists and is a valid configuration file.')
	sys.exit(1)

# Declare default values for options not in the config file
defaults = \
{
	'model_class': 'taylor',
	'model_order': '4',
	'input_dim':   '1',
	'input_lb':    '[]',
	'input_ub':    '[]',
	'input_file':  '',
	'output_dim':  '1',
	'batch_doe':   'LHS',
	'batch_corn':  '0',
	'batch_num':   '0',
	'batch_file':  '',
	'adapt_num':   '5',
	'adapt_tol':   '1e-3',
	'adapt_pen':   '0.50',
	'adapt_rad':   '0.03',
	'adapt_type':  'seq',
	'nomad_exe':   'error_max.py',
	'nomad_num':   '30',
	'nomad_tol':   '1e-3',
	'regpen_crit': 'aicc',
	'regpen_lim':  '1e-4',
	'regpen_num':  '49',
	'regpen_lb':   '1e-8',
	'regpen_ub':   '1e8'
}

# Declare which values should be considered integers
ints = \
[
	'output_dim',
	'input_dim',
	'batch_num',
	'batch_corn',
	'adapt_num',
	'nomad_num',
	'regpen_num'
]

# Declare which values should be considered floats
floats = \
[
	'adapt_tol',
	'adapt_pen',
	'adapt_rad',
	'nomad_tol',
	'regpen_lb',
	'regpen_ub',
	'regpen_lim'
]

# Declare which values should be considered arrays
arrays = \
[
	'input_lb',
	'input_ub',
]

# These values are `distributable`. By this, we mean
# that they are arrays where value number i will be
# distributed to surrogate model number i. If this is
# set to a scalar, it is converted automatically to
# an array of identical entries with the right size.
dists = \
[
	'model_class',
	'model_order'
]



############################################################
# Loading and parsing of the config file
############################################################

# Read and parse the config file. Note that this is 
# a workaround for parsing section-less ini-files. If
# we were to use ini-files *with* sections, the code
# could be simplified to conf.read_file(filename).
with open(filename, 'r') as f:
	# Create a config parser object
	conf = cp.RawConfigParser()

	# Read default values from a dict
	conf.read_dict({'default': defaults})

	# Read user-set values from file
	try:
		# Automatically add section name [default]
		conf.read_string('[default]\n' + f.read())
	except cp.DuplicateSectionError:
		# Don't do that if [default] already exists
		f.seek(0)
		conf.read_string(f.read())

	# Extract section [default]
	conf = conf['default']

	# Convert the results to a dict
	conf = {k : conf[k] for k in conf}


# The `model_class` option requires special care to ensure
# that it can be used without explicit quotes around strings
conf['model_class'] = re.sub(r'(\w+)', r'"\1"', conf['model_class'].strip())

# Cast values to the correct data types
for k in conf:
	try:
		if k in ints:
			conf[k] = int(conf[k])
		elif k in floats:
			conf[k] = float(conf[k])
		elif k in arrays or k in dists:
			conf[k] = np.array(ast.literal_eval(conf[k]))
		# else:
		#	conf[k] is a string and no conversion is needed
	except ValueError:
		raise RuntimeError('"{}" is not a valid value for "{}".'.format(conf[k],k))

# Some values can be specified as either arrays or scalars, and
# should be automatically converted to arrays in the latter case
for k in dists:
	if conf[k].shape == ():
		conf[k] = np.repeat(conf[k], conf['output_dim'])

# Create a list of configuration dicts
confs = [conf.copy() for n in range(conf['output_dim'])]

# Modify each to have a suitable model_class and model_order
for i, conf in enumerate(confs):
	conf.update({k: conf[k][i] for k in dists})



############################################################
# Miscellaneous checks for acceptable parameters
############################################################

# Manual checks for permitted values
if conf['input_dim']  < 1:
	raise RuntimeError('You need to set input_dim ≥ 1.')
if conf['output_dim'] < 1:
	raise RuntimeError('You need to set output_dim ≥ 1.')
if conf['adapt_type'] not in ['seq', 'sim']:
	raise RuntimeError(f'adapt_type is set to an invalid value "{conf["adapt_type"]}".')
if conf['batch_doe'] not in ['RegularGrid', 'LHS', 'Sobol', 'MonteCarlo']:
	raise RuntimeError(f'batch_doe is set to an invalid value "{conf["batch_doe"]}".')
if not conf['input_file']:
	if len(conf['input_lb']) == 0:
		raise RuntimeError('You need to specify input_lb.')
	if len(conf['input_ub']) == 0:
		raise RuntimeError('You need to specify input_ub.')
	if len(conf['input_lb']) != conf['input_dim']:
		raise RuntimeError('input_lb has the wrong length.')
	if len(conf['input_ub']) != conf['input_dim']:
		raise RuntimeError('input_ub has the wrong length.')
