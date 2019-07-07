import numpy as np
import chaospy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p

	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]

	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[:, 0]

def plot_sobol_indices(total_sobol_indices, title_name, show = True):
    sobol_indices_x = [1, 2, 3, 4, 5]
    labels = ['c', 'k', 'f', r"$y_0$", r"$y_1$"]

    plt.figure()
    barlist = plt.bar(sobol_indices_x, total_sobol_indices, align='center')
    barlist[0].set_color('r')
    barlist[1].set_color('b')
    barlist[2].set_color('g')
    barlist[3].set_color('m')
    barlist[4].set_color('k')
    plt.xticks(sobol_indices_x, labels)
    plt.title(title_name)
    plt.tight_layout()
    if show:
        plt.show()

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol, rtol = 1e-10, 1e-10

    # w is deterministic
    w = 1.00

    # create uniform distribution object
    distr_c     = cp.Uniform(0.08, 0.12)
    distr_k     = cp.Uniform(0.03, 0.04)
    distr_f     = cp.Uniform(0.08, 0.12)
    distr_y0    = cp.Uniform(0.45, 0.55)
    distr_y1    = cp.Uniform(-0.05,  0.05)

    # create the multivariate distribution

    # quad deg 1D
    quad_deg_1D = 4
    poly_deg_1D = 4

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = len(t)/2


    # create the orthogonal polynomials

    #################### full grid computations #####################

    joint_distr = cp.J(distr_c, distr_k,  distr_f, distr_y0, distr_y1)
    # get the non-sparse quadrature nodes and weight
    nodes, weights = cp.generate_quadrature(quad_deg_1D, joint_distr, rule='G', sparse=True)
    c,k,f,y0,y1 = nodes
   #print(c.shape)
    poly = cp.orth_ttr(poly_deg_1D, joint_distr, normed=True)

    y_out = [discretize_oscillator_odeint(model, atol, rtol, (y0_,y1_), (c_,k_,f_,w), t)[-1] for y0_,y1_,c_,k_,f_ in zip(c,k,f,y0,y1)]

    # find generalized Polynomial chaos and expansion coefficients
    gPC_m, expansion_coeff = cp.fit_quadrature(poly, nodes, weights, y_out, retall=True)
    print(f'The best polynomial of degree {poly_deg_1D} that approximates f(x): {cp.around(gPC_m, 1)}')
    # gPC_m is the polynomial that approximates the most
    print(f'Expansion coeff [0] = {expansion_coeff[0]}')  # , expect_weights: {expect_y}')

    mu = cp.E(gPC_m, joint_distr)
    print(f'Mean value from gPCE: {mu}')

    #Sobol indices

    first_order_Sobol_ind_sparse = cp.Sens_m(gPC_m, joint_distr)
    total_Sobol_ind_sparse = cp.Sens_t(gPC_m, joint_distr)

    print("the number of quadrature nodes, i.e. model evaluation, for the sparse grid is", len(nodes.T))
    print("the first order Sobol' indices are", first_order_Sobol_ind_sparse)
    print("the total Sobol' indices are", total_Sobol_ind_sparse)

    plot_sobol_indices(total_Sobol_ind_sparse, "Total Sobol indices on sparse grid", False)
    plot_sobol_indices(first_order_Sobol_ind_sparse, "First order Sobol indices on sparse grid", True)
