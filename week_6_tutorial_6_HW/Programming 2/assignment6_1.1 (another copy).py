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

def barycentric_interp3(eval_point, grid, func_eval):
    weights = compute_barycentric_weights(grid)
    L_G = np.product(eval_point - grid)
    result =  0
    for i, x in enumerate(grid):
        result += func_eval[i] * weights[i]/ (eval_point - x )
    return result*L_G

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
def barycentric_interp(eval_point, grid, func_eval):
    weights     = compute_barycentric_weights(grid)
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

def interpolation_over_nodes(sol_odeint_true, grid_N, t_max, x_axis, model, init_cond, params_odeint, atol, rtol):
    # perform interpolation using barycentric polynomials over different number of nodes
    output_interpolation = []
    #list_interpolations = []
    interpolated_value = {}
    for i, grid_size in enumerate(grid_N):
        # grid from 0 to t_max, minus - from 0 to t_max not from t_max to 0
        cheb_grid = np.array([0.5 * t_max - 0.5*t_max* np.cos((2 * i - 1) / (2. * grid_size) * np.pi) for i in range(1, grid_size + 1)])

        func_eval =  discretize_oscillator_odeint(model, init_cond, cheb_grid, params_odeint, atol, rtol)
        #plt.figure()
        #plt.plot(cheb_grid, func_eval, 'g-', label='func_grid')

        output_interpolation = np.array([barycentric_interp(x_eval, cheb_grid, func_eval) for x_eval in x_axis])
        #list_interpolations.append(output_interpolation)
        interpolated_value[str(grid_size)] = output_interpolation[-1]

        #print("Grid:\n", cheb_grid)
        #plt.figure(str(grid_size)+ "_" + str(random.random()))
        #plt.plot(x_axis, sol_odeint_true, 'b-', label='y0')
        #plt.plot(x_axis, output_interpolation, 'r--', label='Bary')
        #plt.legend(loc='best', fontsize=8)
        #plt.ylabel('function')
        #plt.xlabel('t')
    return interpolated_value



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


    #interpolation
    func = lambda x: np.sin(10 * x)

    # define uniform interpolation's grid size
    N = [5, 10, 20]
    grid_N = [6, 11, 21] # points should on one more than N - degree of the polynomial

    grid_len = int(t_max / dt) + 1
    x_axis = np.linspace(0.0, t_max, num=grid_len,  endpoint=True)
    sol_odeint_true = discretize_oscillator_odeint(model, init_cond, x_axis, params_odeint, atol, rtol)
    #y_10_determ = sol_odeint_true[len(sol_odeint_true)-1]
    #print("Deterministic solution:", y_10_determ)

    plt.figure("True oscillator")
    plt.plot(x_axis, sol_odeint_true, '--r', label='y0')  # y0
    plt.ylabel('y(t)')
    plt.xlabel('time')
    plt.legend(loc='best', fontsize=12)


    M = [10, 100]#, 1000]#, 10000]
    mu, V = np.zeros(len(M)), np.zeros(len(M))
    mu_interpol, V_interpol = np.zeros(((len(M)), len(grid_N))), np.zeros((len(M),len(grid_N)))

    np.random.seed(100)
    lists_of_dict = []

    for i, n in enumerate(M):
        distr = cp.Uniform(0.95, 1.05)
        w_generated = distr.sample(size=n)
        outputs_MC_y= [] #y0

        print("Calculating ... ", n)

        #keys = [str(item) for item in grid_N]
        #results = dict.fromkeys(keys, None)
        results = {}
        for w_value in w_generated:
            # direct MC sampling
            params_odeint_new = c, k, f, w_value
            sol_odeint = discretize_oscillator_odeint(model, init_cond, x_axis, params_odeint_new, atol, rtol)
            outputs_MC_y.append(sol_odeint[-1])

            # using_interpolation
            dict_interpol = interpolation_over_nodes(sol_odeint,grid_N, t_max, x_axis, model, init_cond, params_odeint_new, atol, rtol)

            for key, value in dict_interpol.items():
                if key in results.keys():
                    results[key].append(value)
                else:
                    results[key] = []
                    results[key].append(value)

        print(f'M = {n}')
        for  k, (key, values) in enumerate(results.items()):
            print(f'key = {key}')
            print(f'len {len(values)}, {values}')

            mu_interpol[i][k] = np.mean(np.array(values))
            V_interpol[i][k]= np.var(np.array(values), ddof = 1)
            print("mean_interpol:", (mu_interpol[i]))
            print("var_interpol: ",(V_interpol[i]))

        lists_of_dict.append(results)


        mu[i] = np.mean(np.array(outputs_MC_y))
        V[i] = np.var(np.array(outputs_MC_y), ddof  =1)


    print("Generated mean and variance:")
    for i in range(mu.shape[0]):
        print("M = %6d" % M[i], "mean :", "%.3f" % (mu[i]))
        print("\t\t\t var :%.6f" % (V[i]))


    mu_ref = [-0.43893703]
    V_ref  = [0.00019678]

    #rel_err_mu = np.abs(1 -  mu/ mu_ref).T
    #rel_err_V = np.abs(1 - V / V_ref).T

    #rel_err_mu_quasi = np.abs(1 - mu_quasi / mu_ref).T
    #rel_err_V_quasi = np.abs(1 - V_quasi / V_ref).T

    #plotting relative errors
    plt.show()

