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

def plot_trajectory(sol_odeint, idx):
	string = "51"  + str(idx + 1)
	plt.subplot(int(string))
	plt.plot(t, sol_odeint[:, 0], '--r', label='y0')  # y0
	plt.plot(t, sol_odeint[:, 1], '--b', label='y1')  # y1
	plt.ylabel('y(t)')
	plt.xlabel('time')
	plt.legend(loc='best', fontsize=12)


def plot_trajectory2(sol_odeint):
	plt.figure()
	plt.plot(t, sol_odeint[:, 0], '--r', label='y0')  # y0
	plt.plot(t, sol_odeint[:, 1], '--b', label='y1')  # y1
	plt.ylabel('y(t)')
	plt.xlabel('time')
	plt.legend(loc='best', fontsize=12)
	plt.show()

if __name__ == '__main__':
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

	plt.figure()
	plt.plot(t, sol_odeint[:, 0], '--r', label='dt = ' + str(dt) + ', odeint', linewidth=2.0)  # y0
	plt.plot(t, sol_odeint[:, 1], '--b', label='dt = ' + str(dt) + ', odeint', linewidth=2.0)  # y1
	plt.ylabel('y(t)')
	plt.xlabel('time')
	plt.legend(loc='best', fontsize=12)
	plt.show()
	##### Generating different trajectories

	N = [10, 100, 1000, 10000]
	mu, V = np.zeros((len(N), 2)), np.zeros((len(N), 2))

	solution_trajectories = []
	num_i = 0

	for i, n in enumerate(N):
		distr = cp.Uniform(0.95, 1.05)
		w_generated = distr.sample(size=n)
		outputs_y = []
		print("Calculating ... ", n)

		for w_value in w_generated:
			params_odeint = c, k, f, w_value
			sol_odeint = discretize_oscillator_odeint(model, init_cond, t, params_odeint, atol, rtol)
			if num_i < 5:
				solution_trajectories.append(sol_odeint)
			num_i = num_i + 1
			outputs_y.append(sol_odeint[t_10])
		mu[i] = np.mean(np.array(outputs_y), axis = 0)
		V[i] = np.var(np.array(outputs_y), axis = 0)

	print("Generated mean and variance:")
	for i in range(mu.shape[0]):
		print("N = %6d" %N[i],"mean :","%.3f\t%.3f" %(mu[i][0], mu[i][1]))
		print("\t\t\tvar :%.6f\t%.6f" %(V[i][0], V[i][1]))


	mu_ref = [-0.43893703, 0.04293818]
	V_ref = [0.00019678, 0.01336294]

	rel_err_true= np.abs(1 -  mu/ y_10_determ).T
	rel_err_appr = np.abs(1 -  mu/ mu_ref).T

	#plotting relative errors
	plt.figure()
	plt.loglog(N, rel_err_true[0], 'bx', label='rel err (true), y0')
	plt.loglog(N, rel_err_true[1], 'b+', label='rel err (true), y1')
	plt.loglog(N, rel_err_appr[0], 'rx', label='rel err (appr), y0')
	plt.loglog(N, rel_err_appr[1], 'r+', label='rel err (appr), y1')
	plt.legend(loc='best', fontsize=8)
	plt.ylabel('Error values')
	plt.xlabel('Number of samples (loglog)')
	plt.show()

	#printing out 5 random solutions
	plt.figure()
	for (i,item) in enumerate(solution_trajectories):
		plot_trajectory(item, i)
	plt.show()