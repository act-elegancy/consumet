#!/usr/bin/env python

'''
This file demonstrates how to import surrgate 
models using the binary `surrogate.pkl` method.
'''

########################################
# Load surrogate models
########################################

# Add Consumet libraries to path
from sys import path
path.append('../../../lib')

# Load surrogates from a binary file
import pickle
import surrogate

with open('surrogate.pkl', 'rb') as f:
    surrogates = pickle.load(f)

# We only made one surrogate model (#0)
# let us just call that one 'rosenbrock'.
rosenbrock = surrogates[0]


########################################
# Evaluate surrogate models
########################################

# Make a 200×200 coordinate grid
import numpy as np
xs = np.linspace(0, 1, 200)
ys = np.linspace(0, 1, 200)

# Evaluate the function on grid
zs = np.zeros((200, 200))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        zs[j,i] = rosenbrock([x, y])


########################################
# Plot surrogate models
########################################

# Load plotting library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Convert coordinates into mesh
xs, ys = np.meshgrid(xs, ys)

# Make a new figure with 3d axes
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(45, -125)

# Plot the results
ax.plot_surface(xs, ys, zs)
plt.show()
