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

def plot_rel_errors(N, rel_error1, rel_error2, label_name1, label_name2, loglog_bool =True):
    plt.figure()
    if(loglog_bool):
        plt.loglog(N, rel_error1, 'rx', label=label_name1)
        plt.loglog(N, rel_error2, 'gx', label=label_name2)
        plt.xlabel('Number of samples (loglog)')
    else:
        plt.plot(N, rel_error1, 'rx', label=label_name1)
        plt.plot(N, rel_error2, 'gx', label=label_name2)
        plt.xlabel('The degree of polynomial N')
    plt.legend(loc='best', fontsize=8)
    plt.ylabel('Error values')

def plot_rel_error(N, rel_error, label_name, xlabel_name, line_type, loglog_bool =True):
    plt.figure()
    if(loglog_bool):
        plt.loglog(N, rel_error, line_type, label=label_name)
        plt.xlabel('Number of samples (loglog)')
    else:
        plt.plot(N, rel_error, line_type, label=label_name)
        plt.xlabel(xlabel_name)
    plt.legend(loc='best', fontsize=8)
    plt.ylabel('Error values')

def get_quantity_of_interest(expansion_coefficients):
    mean = expansion_coefficients[0]
    var = 0
    for i in range(1, len(expansion_coefficients)):
            var+= expansion_coefficients[i] **2
    return(mean, var)

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

    K = [2,4,6]#, 8, 10] # for approximation
    N = [1,2,3]#, 4, 5] # for expansion coefficients

    #K = [1,2,3]
    #N = [1,2,3]

    mu, V = np.zeros(len(N)), np.zeros(len(N))
    mu_expansion, V_expansion= np.zeros(len(N)), np.zeros(len(N))

    distr_unif_w = cp.Uniform(0.95, 1.05)
    orth_polies = []
    for i, n in enumerate(N):
        # generate K Gaussian nodes and weights based on normal distr (we need to appr with quadratures)
        nodes, weights = cp.generate_quadrature(K[i], distr_unif_w, rule = "G") #nodes [[1,2]]

        # approximating with gaussian polynomials
        orth_poly = cp.orth_ttr(N[i], distr_unif_w, normed = True)

        #evaluate f(x) at all quadrature nodes and take the y(10), i.e [-1]
        y_out = [discretize_oscillator_odeint(model, init_cond, x_axis, (c, k, f, node), atol, rtol)[-1] for node in nodes[0]]

        #print(nodes, weights)
        #print(orth_poly[1](10))

        #SUPER STRANGE BUG
        #for j in range(len(orth_poly)):
        #    for k in range(len(nodes[0])):
        #       continue


        # find generalized Polynomial chaos and expansion coefficients
        gPC_m, expansion_coeff = cp.fit_quadrature(orth_poly, nodes, weights, y_out, retall = True)
        #gPC_m = cp.fit_quadrature(orth_poly, nodes, weights, y_out)

        # gPC_m is the polynomial that approximates the most
        print(f'Expansion coeff chaospy: {expansion_coeff}')
        print(f'The best polynomial of degree {n} that approximates f(x): {cp.around(gPC_m, 1)}')
        #print(f'Expansion coeff [0] = {expansion_coeff[0]}')#, expect_weights: {expect_y}')

        mu[i] = cp.E(gPC_m, distr_unif_w)
        V[i]= cp.Var(gPC_m, distr_unif_w)

        print("mu = %.8f,V = %.8f" %(mu[i], V[i]))

# manual calculation of the expansion coefficients
#Note if you do it in the same loop, the mean results changing only due to the fact that we do the loop over K[i] without any action
    print("____________Manual expansion coefficients__________")

    for i, n in enumerate(N):
        # generate K Gaussian nodes and weights based on normal distr (we need to appr with quadratures)
        nodes, weights = cp.generate_quadrature(K[i], distr_unif_w, rule="G")  # nodes [[1,2]]

        # appr with gaussian polynomials
        orth_poly = cp.orth_ttr(N[i], distr_unif_w, normed=True)

        # evaluate f(x) at all quadrature nodes and take the y(10), i.e [-1]
        y_out = [discretize_oscillator_odeint(model, init_cond, x_axis, (c, k, f, node), atol, rtol)[-1] for node in
                 nodes[0]]

        #print(nodes, weights)

        expansion_coeff_manual = np.zeros(len(orth_poly))
        # manual expansion coefficients
        for j in range(len(orth_poly)):
            for k in range(len(nodes[0])):
                #value =  y_out[k] * orth_poly[j](nodes[0][k])* weights[k]
                #print(f'{(j,k)} - {value}')
                expansion_coeff_manual[j] += y_out[k] * orth_poly[j](nodes[0][k])* weights[k]

        mu_expansion[i], V_expansion[i] = get_quantity_of_interest(np.array(expansion_coeff_manual))
        print(f'Expansion coeff for N = {n}:\n {expansion_coeff_manual}')
        print("mu_exp = %.8f,V_exp = %.8f" %( mu_expansion[i], V_expansion[i]))

    mu_ref = [-0.43893703]
    V_ref  = [0.00019678]

    #cols -sampling M
    rel_err_mu = np.abs(1 - mu / mu_ref)
    rel_err_V = np.abs(1 - V / V_ref)
    rel_err_mu_exp = np.abs(1 - mu_expansion / mu_ref)
    rel_err_V_exp = np.abs(1 - V_expansion/ V_ref)

    #print(rel_err_V, rel_err_mu)

    plot_rel_error(N,rel_err_mu, "mean", "The degree of the polynomials N", 'rx',  loglog_bool = False)
    plot_rel_error(N,rel_err_V, "var", "The degree of the polynomials N", 'gx',  loglog_bool = False)
    #plot_rel_errors(N, rel_err_mu, rel_err_mu_exp, "mean chaospy", "mean exp coeff", False)
    #plot_rel_errors(N, rel_err_V, rel_err_V_exp, "var chaospy", "var exp coeff", False)
    #plt.show()
