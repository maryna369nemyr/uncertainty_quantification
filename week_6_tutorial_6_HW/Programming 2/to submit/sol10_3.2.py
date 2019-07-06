import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint


# Discretize the oscillator using system of first order ode and explicit Euler
# you can use python lists or numpy lists
def discretize_oscillator_sys(t, dt, params, f):
    c, k, w, y0, y1 = params

    # initalize the solution vector with zeros
    z0 = np.zeros(len(t))
    z1 = np.zeros(len(t))

    # initalize the solution vector with zeros
    # z0 = [0. for i in xrange(len(t))]
    # z1 = [0. for i in xrange(len(t))]

    z0[0] = y0
    z1[0] = y1

    # implement the obtained Euler scheme
    for i in range(0, len(t) - 1):
        z1[i + 1] = z1[i] + dt * (-k * z0[i] - c * z1[i] + f[i] * np.cos(w * t[i]))
        z0[i + 1] = z0[i] + dt * z1[i]

    return z0


# standard definition of the Wiener process
# W_0 = 0, W_{t + dt} = W_t + zeta_t. zeta_t ~ N(0, dt)
def WP_std_def(t, f_mean):
    n = len(t)
    W = np.zeros(n)
    dW = np.zeros(n)
    W[0] = f_mean

    for i in range(1, n):
        dW[i] = np.sqrt(t[i] - t[i - 1]) * np.random.randn()
        W[i] = W[i - 1] + dW[i]

    return W


# THE KL approximation of the Wiener process
def WP_KL_approx(t, f_mean, KL_dim):
    n = len(t)
    W = np.zeros(n) + f_mean
    dW = np.zeros(n)

    zeta = np.random.normal(0, 1, KL_dim)

    for i in range(KL_dim):
        W += np.sqrt(2. / t[-1]) * zeta[i] * (t[-1] ** 2) * np.sin(((i + 1 + 0.5) * np.pi * t) / t[-1]) / (
                    (i + 1 + 0.5) * np.pi)

    return W


if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c = 0.5
    k = 2.0
    y0 = 0.5
    y1 = 0.
    w = 1.0
    f_mean = 0.5

    # time domain setup
    t_max = 10.
    dt = 0.01
    grid_size = int(t_max / dt) + 1
    t = np.array([i * dt for i in range(grid_size)])
    t_interest = -1
    n_timesteps = grid_size

    # the two sizes mentioned in the worksheet
    mc_terms = 100

    M = [10, 100, 1000]

    ode_wiener = np.zeros((mc_terms, n_timesteps))
    print(ode_wiener.shape)

    lw = 4
    figure()
    for i in range(mc_terms):
        # Sample Wiener path
        W_std_def = WP_std_def(t, f_mean)
        params = c, k, w, y0, y1
        f = W_std_def
        ode_wiener[i, :] = discretize_oscillator_sys(t, dt, params, f)
        plot(t, ode_wiener[i, :])

    mean_mc = np.mean(ode_wiener, axis=0)
    var_mc = np.var(ode_wiener, axis=0, ddof=1)
    print("y(10) (mean, var) =  ", mean_mc[-1], var_mc[-1])
    plot(t, mean_mc, linewidth=lw, color='k')
    plot(t, mean_mc + np.sqrt(var_mc), linewidth=lw, color='r')
    plot(t, mean_mc - np.sqrt(var_mc), linewidth=lw, color='r')

    # Sample Wiener path
    for j in range(0, len(M)):
        ode_kl = np.zeros((mc_terms, n_timesteps))
        figure()
        for i in range(mc_terms):
            W_KL = WP_KL_approx(t, f_mean, M[j])
            params = c, k, w, y0, y1
            f = W_KL
            ode_kl[i, :] = discretize_oscillator_sys(t, dt, params, f)
            plot(t, ode_kl[i, :])

        mean_kl = np.mean(ode_kl, axis=0)
        var_kl = np.var(ode_kl, axis=0, ddof=1)
        print("y(10) (mean, var) =  ", mean_kl[-1], var_kl[-1])
        plot(t, mean_kl, linewidth=lw, color='k')
        plot(t, mean_kl + np.sqrt(var_kl), linewidth=lw, color='r')
        plot(t, mean_kl - np.sqrt(var_kl), linewidth=lw, color='r')

    show()