import numpy as np
import chaospy as cp
from scipy.integrate import odeint

# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p

	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]

	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[t_interest, 0]

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # the parameters are no longer deterministic
    c_left      = 0.08
    c_right     = 0.12
    k_left      = 0.03
    k_right     = 0.04
    f_left      = 0.08
    f_right     = 0.12
    y0_left     = 0.45
    y0_right    = 0.55
    y1_left     = -0.05
    y1_right    = 0.05

    # w is deterministic
    w = 1.00

    # create uniform distribution object
    distr_c     = cp.Uniform(c_left, c_right)
    distr_k     = cp.Uniform(k_left, k_right)
    distr_f     = cp.Uniform(f_left, f_right)
    distr_y0    = cp.Uniform(y0_left, y0_right)
    distr_y1    = cp.Uniform(y1_left, y1_right)

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
    # get the non-sparse quadrature nodes and weight

    # create vector to save the solution

    # perform sparse pseudo-spectral approximation

    # obtain the gpc approximation

    # compute first order and total Sobol' indices
    ##################################################################

    #################### full grid computations #####################
    # get the sparse quadrature nodes and weight

    # create vector to save the solution

    # perform sparse pseudo-spectral approximation

    # obtain the gpc approximation

    # compute first order and total Sobol' indices
    ##################################################################

    # print or plot results