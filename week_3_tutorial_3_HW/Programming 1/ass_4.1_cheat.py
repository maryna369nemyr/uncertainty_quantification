import numpy as np
from scipy.stats import beta
from matplotlib.pyplot import *

# function to compute relative error
def get_rel_err(approx, ref):
    return abs(1. - approx/ref)

# standard monte carlo samplig
def std_mcs(func, n_samples):
    samples = np.random.uniform(0, 1, n_samples)
    I_hat_f = np.mean(func(samples))

    return I_hat_f

# control variate with control variates; we assume that we know the integral of the control variate
def control_variate(func, cv, integral_cv_eval, n_samples):
    samples = np.random.uniform(0, 1, n_samples)

    integral_diff_approx = np.mean(func(samples) - cv(samples))

    I_hat_f = integral_cv_eval + integral_diff_approx

    return I_hat_f

# importance sampling using a beta distribution
def importance_sampling(func, a, b, n_samples):
    # generate samples from the beta distribution
    samples = np.random.beta(a=a, b=b, size=n_samples)

    # ration between f and g_X
    h = lambda x: func(x)/beta.pdf(x, a, b)

    I_hat_f = np.mean(h(samples))

    return I_hat_f

if __name__ == '__main__':
    # declare function to integrate via Monte Carlo sampling
    func = lambda x: np.exp(x)
    # compute integreal_0^1 func(x)dx
    ref_sol = func(1.) - func(0.)

    # declare vector with number of samples
    N = [10, 100, 1000, 10000, 100000, 1000000]

    # declare the control variates and compute their integral
    cv              = [lambda x: x, lambda x: 1. + x]
    integral_cv     = [0.5, 1.5]

    # declare values for the a and b parameters for the beta distribution
    a = [5, 0.5]
    b = [1, 0.5]

    # vector to put all relative errors
    rel_err_mcs   = np.zeros(len(N))
    rel_err_cv    = np.zeros((len(N), len(integral_cv)))
    rel_err_ip    = np.zeros((len(N), len(a)))

    # for each N, perform Monte Carlo integration
    for i, n in enumerate(N):
        approx_mcs      = std_mcs(func, n)
        rel_err_mcs[i]  = get_rel_err(approx_mcs, ref_sol)

        for j in range(len(a)):
            approx_cv  = control_variate(func, cv[j], integral_cv[j], n)
            approx_ip  = importance_sampling(func, a[j], b[j], n)
            print("N = ", n)
            print('CV x, x+1 {1}', approx_cv)
            print('IS a = 5, b = 1, IS a = 0.5, b= 0.5 ', approx_ip)
            rel_err_cv[i, j] = get_rel_err(approx_cv, ref_sol)
            rel_err_ip[i, j] = get_rel_err(approx_ip, ref_sol)

    # plot results
    loglog(N, rel_err_mcs, 'r', label='relative error standard MCS')
    loglog(N, rel_err_cv[:, 0], 'b', label='relative error when control variate = x')
    loglog(N, rel_err_cv[:, 1], 'm', label='relative error when control variate = 1 + x')
    loglog(N, rel_err_ip[:, 0], 'g', label='relative error importance sampling, a = 5, b = 1')
    loglog(N, rel_err_ip[:, 1], 'c', label='relative error importance sampling, a = 0.5, b = 0.5')

    legend(loc='best')
    show()

    print("Integral:", ref_sol)
    print('MC', approx_mcs)

