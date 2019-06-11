import numpy as np
import chaospy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
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



if __name__ == '__main__':
    c,k ,f, w, y0, y1 = 0.5, 2.0, 0.5, 1.0, 0.5, 0.

    # time domain setup
    t_max, dt  = 20., 0.01

    # arguments setup for calling the three discretization functions
    params = c, k, f, w, y0, y1
    init_cond = y0, y1
    params_odeint = c, k, f, w

    # relative and absolute tolerances for the ode int solver
    atol , rtol= 1e-10, 1e-10


    # ploting
    x_axis = int(t_max / dt) + 1
    t = np.linspace(0, t_max, x_axis, endpoint=True)
    t_10 = int(10 / dt) + 1
    print(t_10)

    sol_odeint = discretize_oscillator_odeint(model, init_cond, t, params_odeint, atol, rtol)
    y_10_determ = sol_odeint[t_10]
    print("Deterministic solution:", y_10_determ)

    #interpolation
    func = lambda x: np.sin(10 * x)

    # define uniform interpolation's grid size
    N = [4, 8, 12, 16]
    grid_N = [5, 9, 13, 17] # points should on one more than N - degree of the polynomial
    # define x axis for plotting: x \in [-1, 1]
    x_axis = np.linspace(-1.0, 1.0, num=50)

    output_interpolation = []

    # perform interpolation using Lagrange's and barycentric polynomials over different number of nodes
    for i, grid_size in enumerate(grid_N):
        # generate the unform grid, the function's evaluations and perform the interpolation
        generated_grid = np.linspace(-1.0, 1.0, num=grid_size )
        cheb_grid = np.array([np.cos((2 * i - 1) / (2. * grid_size) * np.pi) for i in range(1, grid_size + 1)])
        # plot the test function #  nis a grid_size

        # plot the test function
        func_eval = func(cheb_grid)


        output_interpolation = np.array([barycentric_interp3(x_eval, cheb_grid, func_eval) for x_eval in x_axis])
        #print(output_interpolation)
        print("Grid:\n", cheb_grid)
        plt.figure(str(grid_size))
        plt.plot(x_axis, func(x_axis), 'b-', label='sin(10x)')
        plt.plot(x_axis, output_interpolation, 'r--', label='Bary')


        plt.legend(loc='best', fontsize=8)
        plt.ylabel('function')
        plt.xlabel('x')

    ##### Generating different trajectories

    #N = [10, 100, 1000, 10000]
    #mu, V = np.zeros((len(N), 2)), np.zeros((len(N), 2))
    #mu_quasi, V_quasi = np.zeros((len(N), 2)), np.zeros((len(N), 2))



    mu_ref = [-0.43893703]
    V_ref  = [0.00019678]

    #rel_err_mu = np.abs(1 -  mu/ mu_ref).T
    #rel_err_V = np.abs(1 - V / V_ref).T

    #rel_err_mu_quasi = np.abs(1 - mu_quasi / mu_ref).T
    #rel_err_V_quasi = np.abs(1 - V_quasi / V_ref).T

    #plotting relative errors
    plt.figure("True oscillator")
    plt.plot(t, sol_odeint, '--r', label='y0')  # y0
    plt.ylabel('y(t)')
    plt.xlabel('time')
    plt.legend(loc='best', fontsize=12)
    plt.show()

