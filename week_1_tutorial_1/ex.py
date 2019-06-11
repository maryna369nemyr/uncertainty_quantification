import numpy as np
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# Assignment 2: Discretize the oscillator using system of first order ode and explicit Euler
# You can use python lists or numpy lists 
def discretize_oscillator_sys(t, dt, params):
    raise NotImplementedError


# Assignment 3: Create function callable to be used in odeint
# We need to use system of two first order ode here
def model(init_cond, t, params):
    raise NotImplementedError


# Assignment 3: Solve oscillator using the odeint function
# Call scipy.integrate.odeint on function callable
def discretize_oscillator_odeint(model, init_cond, t, params, atol, rtol):
    raise NotImplementedError

if __name__ == '__main__':
	# parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    w   = 1.05
    y0  = 0.5
    y1  = 0.

    # time domain setup
    t_max   = 20.
    dt_vec  = [0.5, 0.1, 0.05, 0.01] #0.5

    # arguments setup for calling the three discretization functions
    params              = c, k, f, w, y0, y1
    init_cond           = y0, y1
    params_odeint       = c, k, f, w

    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    print "Hello World"
    #main dummy
    #initialize parameters
    #solve system 
    #plot result
