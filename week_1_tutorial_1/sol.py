import numpy as np
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# Assignment 2: Discretize the oscillator using system of first order ode and explicit Euler
# you can use python lists or numpy lists 
def discretize_oscillator_sys(t, dt, params):
    c, k, f, w, y0, y1 = params

    # initalize the solution vector with zeros
    z0 = np.zeros(len(t))
    z1 = np.zeros(len(t))

    # initalize the solution vector with zeros
    #z0 = [0. for i in range(len(t))]
    #z1 = [0. for i in range(len(t))]

    z0[0] = y0
    z1[0] = y1

    # implement the obtained Euler scheme
    for i in range(0, len(t) - 1):
        z1[i + 1] = z1[i] + dt*(-k*z0[i] - c*z1[i]+f*np.cos(w*t[i]))
        z0[i + 1] = z0[i] + dt*z1[i]

    return z0

# Assignment 3: Create function callable to be used in odeint
# we need to use system of two first order ode here
def model(init_cond, t, params):
    z0, z1      = init_cond
    c, k, f, w  = params

    f = [z1, f*np.cos(w*t) - k*z0 - c*z1]

    return f

# Assignment 3: Discretize the oscillator using the odeint function
# call scipy odeint on function callable
def discretize_oscillator_odeint(model, init_cond, t, args, atol, rtol):
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

    return sol[:, 0]

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

    # get all three solutions and plot them
    for dt in dt_vec:
        grid_size = int(t_max/dt) + 1
        t 		  = [i*dt for i in range(grid_size)]

        start = time.time()
        sol_sys  = discretize_oscillator_sys(t, dt, params)
        time_sys = time.time() - start
        

        start = time.time()
        sol_odeint  = discretize_oscillator_odeint(model, init_cond, t, params_odeint, atol, rtol)
        time_odeint = time.time() - start

        print( "dt:", dt)
        print("time_sys:", time_sys)
        #print "time_odeint:", time_odeint
        print( "----------------------------")
        figure()
        plot(t, sol_sys, '--',color = 'b', label='dt = ' + str(dt) + ', sol_sys', linewidth=2.0)
        plot(t, sol_odeint, '--r', label='dt = ' + str(dt) + ', odeint', linewidth=2.0)
        legend(loc='best', fontsize=15)

    show()

