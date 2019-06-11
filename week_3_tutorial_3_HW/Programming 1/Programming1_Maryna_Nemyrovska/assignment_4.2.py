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


# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, init_cond, t, args, atol, rtol):
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

    return sol


c = 0.5
k = 2.0
f = 0.5
w = 1.0
y0 = 0.5
y1 = 0.

# time domain setup
t_max = 20.
dt = 0.01

# arguments setup for calling the three discretization functions
params = c, k, f, w, y0, y1
init_cond = y0, y1
params_odeint = c, k, f, w

# relative and absolute tolerances for the ode int solver
atol = 1e-10
rtol = 1e-10

# ploting
grid_size = int(t_max / dt) + 1
t = np.linspace(0, t_max, grid_size, endpoint=True)
t_10 = int(10 / dt) + 1
print(t_10)

sol_odeint = discretize_oscillator_odeint(model, init_cond, t, params_odeint, atol, rtol)
y_10_determ = sol_odeint[t_10]
print("Deterministic solution:", y_10_determ)


##### Generating different trajectories

N = [10, 100, 1000, 10000]
mu, V = np.zeros((len(N), 2)), np.zeros((len(N), 2))
mu_quasi, V_quasi = np.zeros((len(N), 2)), np.zeros((len(N), 2))

np.random.seed(120)
for i, n in enumerate(N):
    distr = cp.Uniform(0.95, 1.05)
    w_generated, w_Halton = distr.sample(size=n), distr.sample(size=n, rule ="H")
    outputs_y, outputs_y_Halton = [], []
    print("Calculating ... ", n)
    for (w_value, w_quasi) in zip(w_generated, w_Halton):
        params_odeint = c, k, f, w_value
        params_odeint_Halton = c, k, f, w_quasi
        sol_odeint = discretize_oscillator_odeint(model, init_cond, t, params_odeint, atol, rtol)
        sol_odeint_Halton = discretize_oscillator_odeint(model, init_cond, t, params_odeint_Halton, atol, rtol)
        outputs_y.append(sol_odeint[t_10])
        outputs_y_Halton.append(sol_odeint_Halton[t_10])
    mu[i] = np.mean(np.array(outputs_y), axis = 0)
    V[i] = np.var(np.array(outputs_y), axis = 0)
    mu_quasi[i] = np.mean(np.array(outputs_y_Halton), axis=0)
    V_quasi[i] = np.var(np.array(outputs_y_Halton), axis=0)

print("Generated mean and variance:")
for i in range(mu.shape[0]):
    print("N = %6d" %N[i],"mean :","%.3f\t%.3f" %(mu[i][0], mu[i][1]))
    print("\t\t\tvar :%.6f\t%.6f" %(V[i][0], V[i][1]))

print("Generated mean and variance via Halton sequences:")
for i in range(mu.shape[0]):
    print("N = %6d" % N[i], "mean :", "%.3f\t%.3f" % (mu[i][0], mu[i][1]))
    print("\t\t\tvar :%.6f\t%.6f" % (V[i][0], V[i][1]))


mu_ref = [-0.43893703, 0.04293818]
V_ref  = [0.00019678, 0.01336294]

rel_err_mu = np.abs(1 -  mu/ mu_ref).T
rel_err_V = np.abs(1 - V / V_ref).T

rel_err_mu_quasi = np.abs(1 - mu_quasi / mu_ref).T
rel_err_V_quasi = np.abs(1 - V_quasi / V_ref).T

#plotting relative errors
plt.figure("Relative error, mean")
plt.loglog(N, rel_err_mu[0], 'r--', label='rel err (mu), y0')
plt.loglog(N, rel_err_mu[1], 'b--', label='rel err (mu), y1')
plt.loglog(N, rel_err_mu_quasi[0], 'r-', label='rel err quasi (mu), y0')
plt.loglog(N, rel_err_mu_quasi[1], 'b-', label='rel err quasi (mu), y1')
plt.legend(loc='best', fontsize=8)
plt.ylabel('Error values')
plt.xlabel('Number of samples (loglog)')


plt.figure("Relative error, variance")
plt.loglog(N, rel_err_V[0], 'r--', label='rel err (Var), y0')
plt.loglog(N, rel_err_V[1], 'b--', label='rel err (Var), y1')
plt.loglog(N, rel_err_V_quasi[0], 'r-', label='rel err quasi (Var), y0')
plt.loglog(N, rel_err_V_quasi[1], 'b-', label='rel err quasi (Var), y1')
plt.legend(loc='best', fontsize=8)
plt.ylabel('Error values')
plt.xlabel('Number of samples (loglog)')
plt.show()

