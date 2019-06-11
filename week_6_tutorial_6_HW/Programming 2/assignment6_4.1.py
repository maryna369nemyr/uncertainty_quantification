import numpy as np
import chaospy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import time


# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(init_cond, t, params):
    z0, z1 = init_cond
    c, k, f, w = params
    f = [z1, f * np.cos(w * t) - k * z0 - c * z1]
    return f

def discretize_oscillator_odeint(model, init_cond, t, args, atol, rtol):
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    return sol[:, 0] # only y_0


if __name__ == '__main__':
    c,k ,f, w, y0, y1 = 0.5, 2.0, 0.5, 1.0, 0.5, 0.

    # time domain setup
    t_max, dt  = 10., 0.01

    # arguments setup for calling the three discretization functions
    params = c, k, f, w, y0, y1
    init_cond = y0, y1
    params_odeint = c, k, f, w

    # relative and absolute tolerances for the ode int solver
    atol , rtol= 1e-10, 1e-10

    grid_len = int(t_max / dt) + 1
    x_axis = np.linspace(0.0, t_max, num = grid_len,  endpoint=True)
    sol_odeint_true = discretize_oscillator_odeint(model, init_cond, x_axis, params_odeint, atol, rtol)

    N = [2,4,6]
    K = [1,2,3]

    mu, V = np.zeros(len(N)), np.zeros(len(N))



    for i, n in enumerate(M):
        distr = cp.Uniform(0.95, 1.05)
        w_generated = distr.sample(size=n)
        outputs_MC_y= [] #y0

        print("Calculating ... ", n)

        output_interpol =  []
        for w_value in w_generated:
            # direct MC sampling
            params_odeint_new = c, k, f, w_value
            sol_odeint = discretize_oscillator_odeint(model, init_cond, x_axis, params_odeint_new, atol, rtol)
            outputs_MC_y.append(sol_odeint[-1])


        mu[i] = np.mean(np.array(outputs_MC_y))
        V[i] = np.var(np.array(outputs_MC_y), ddof  =1)


    print("Generated mean and variance:")
    for i in range(mu.shape[0]):
        print("M = %6d" % M[i], "mean :", "%.3f" % (mu[i]))
        print("\t\t\t var :%.6f" % (V[i]))


    mu_ref = [-0.43893703]
    V_ref  = [0.00019678]

    #cols -sampling M
    rel_err_mu = np.abs(1 - mu / mu_ref)
    rel_err_V = np.abs(1 - V / V_ref)
    #cols - grid points
    #rows - sampling
    rel_err_mu_interpol = np.abs(1 - mu_interpol / mu_ref).T
    rel_err_V_interpol = np.abs(1 - V_interpol/ V_ref).T