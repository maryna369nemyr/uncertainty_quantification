import numpy as np
from matplotlib.pyplot import *

# Lagrange's cardinal polynomial
def lagrange_poly(eval_point, i, grid):
    return NotImplementedError

# Lagrange interpolation; be aware to take into consideration the evaluation of the 
# interpolating polynomial at the grid points
def lagrange_interp(eval_point, grid, func_eval):
    return NotImplementedError

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid):
    return NotImplementedError

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, func_eval):
    return NotImplementedError

if __name__ == '__main__':
    # define test's function

    # define uniform interpolation's grid size

    # define x axis for plotting: x \in [-1, 1]

    # perform interpolation using Lagrange's and barycentric polynomials over different number of nodes
    for nothing in [None]:

        # generate the unform grid, the function's evaluations and perform the interpolation

        # plot the test function

        # plot the interpolating function
        raise NotImplementedError