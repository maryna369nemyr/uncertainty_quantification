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

def compute_barycentric_weights(grid):
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, func_eval, weights):
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G

    return res


def interpolation(cheb_grids, weights_grid, func_evals, w_eval):
   # return an array with columns - grid_size
    output_interpolation = np.zeros(len(cheb_grids))
    i= 0
    for cheb_grid, weights, func_eval in zip(cheb_grids, weights_grid, func_evals):
        output_interpolation[i] = barycentric_interp(w_eval, cheb_grid, func_eval, weights)
        i+=1
    return output_interpolation


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

    # define uniform interpolation's grid size
    N = [5, 10, 20]
    grid_N = [6, 11, 21] # points should on one more than N - degree of the polynomial

    grid_len = int(t_max / dt) + 1
    x_axis = np.linspace(0.0, t_max, num=grid_len,  endpoint=True)

    M = [10, 100, 1000]#, 10000]
    mu, V = np.zeros(len(M)), np.zeros(len(M))
    mu_interpol, V_interpol = np.zeros(((len(M)), len(grid_N))), np.zeros((len(M),len(grid_N)))
    #columns - number of interpolation points
    #rows - number of sampled points


    begin = 0.95
    end = 1.05


    ### Precalculation for interpolation method

    # grid from 0 to t_max, minus - from 0 to t_max not from t_max to 0
    cheb_grids = [np.array(
        [0.5 * (end + begin) - 0.5 * (end - begin) * np.cos((2 * j - 1) / (2. * grid_size) * np.pi) for j in range(1, grid_size + 1)])
                  for grid_size in grid_N]  # len(grid_N)
    weights_grid = [compute_barycentric_weights(grid) for grid in cheb_grids]
    func_evals =[]

    for grid in cheb_grids:
        func_result_eval = np.array(
            [discretize_oscillator_odeint(model, init_cond, x_axis, (c, k, f, grid_point), atol, rtol)[-1]
             for grid_point in grid])
        func_evals.append(func_result_eval)
    # The end of precalculation

    for i, n in enumerate(M):
        distr = cp.Uniform(0.95, 1.05)
        w_generated = distr.sample(size=n)
        outputs_MC_y= [] #y0

        print("Calculating ... ", n)

        now_mc = time.time()
        for w_value in w_generated:
            params_odeint_new = c, k, f, w_value
            # direct MC sampling
            sol_odeint = discretize_oscillator_odeint(model, init_cond, x_axis, params_odeint_new, atol, rtol)
            outputs_MC_y.append(sol_odeint[-1])
        print(f'Time for {n} generated values using MC sampling {time.time() - now_mc}')

        now_interpol = time.time()
        output_interpol = []
        for w_value in w_generated:
            # using_interpolation
            output_interpol.append(interpolation(cheb_grids, weights_grid, func_evals, w_value))
        print(f'Time for {n} generated values using interpolation {time.time() - now_interpol}')
        output_interpol = np.array(output_interpol)


        mu_interpol[i] = np.mean(output_interpol, axis = 0)
        V_interpol[i]= np.var(output_interpol,axis =0 , ddof = 1)
        print("mean_interpol:", (mu_interpol[i]))
        print("var_interpol: ",(V_interpol[i]))

        mu[i] = np.mean(np.array(outputs_MC_y))
        V[i] = np.var(np.array(outputs_MC_y), ddof  =1)


    print("Mean and variance obtained from the direct MC sampling:")
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


    plt.figure("Mean")
    plt.loglog(M, rel_err_mu, 'r-', label='MC mean')
    plt.loglog(M, rel_err_mu_interpol[0], 'gx', label='grid 6')
    plt.loglog(M, rel_err_mu_interpol[1], 'bx', label='grid 11')
    plt.loglog(M, rel_err_mu_interpol[2], 'yx', label='grid 21')
    plt.legend(loc='best', fontsize=8)
    plt.ylabel('Error values')
    plt.xlabel('Number of samples (loglog)')



    plt.figure("Variance")
    plt.loglog(M, rel_err_V, 'r-', label='MC variance')
    plt.loglog(M, rel_err_V_interpol[0], 'gx', label='grid 6')
    plt.loglog(M, rel_err_V_interpol[1], 'bx', label='grid 11')
    plt.loglog(M, rel_err_V_interpol[2], 'yx', label='grid 21')
    plt.legend(loc='best', fontsize=8)
    plt.ylabel('Error values')
    plt.xlabel('Number of samples (loglog)')
    plt.show()


