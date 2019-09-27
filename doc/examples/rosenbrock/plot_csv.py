#!/usr/bin/env python

'''
This file demonstrates how to import surrgate 
models by loading the `regression.csv` file.
'''

########################################
# Recreate the surrogate model
########################################

# Load regression coefficients from file
import numpy as np
data = np.genfromtxt('regression.csv', delimiter=',')

# Reimplement the polynomial surrogate model
def rosenbrock(x, y):
    return sum([θ * x**n * y**m for _, n, m, θ in data])


########################################
# Evaluate surrogate models
########################################

# Make a 200×200 coordinate grid
xs = np.linspace(0, 1, 200)
ys = np.linspace(0, 1, 200)

# Evaluate the function on grid
zs = np.zeros((200, 200))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        zs[j,i] = rosenbrock(x, y)


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
