import numpy as np
from matplotlib.pyplot import *

# standard definition of the Wiener process
# W_0 = 0, W_{t + dt} = W_t + zeta_t. zeta_t ~ N(0, dt)
def WP_std_def(zeta, t):
	n = len(t)
	W 	= np.zeros(n)
	dW 	= np.zeros(n)

	for i in range(1, n):
		dW[i] 	= np.sqrt(dt)*zeta[i - 1]
		W[i] 	= W[i - 1] + dW[i]
	return W

# THE KL approximation of the Wiener process
def WP_KL_approx(zeta, t, M):
	n = len(t)
	W 	= np.zeros(n)

	for i in range(1, M):
		W += np.sqrt(2)*zeta[i - 1]*np.sin((i + 0.5)*np.pi*t)/((i + 0.5)*np.pi)

	return W


if __name__ == '__main__':
	# the two sizes mentioned in the worksheet
	N = 1000
	M = [10, 100, 1000]

	dt = 1./N
	t 	= np.arange(0, 1+dt, dt)

	# first, use the standard defition to generate a realization with N samples
	# generate random variables
	zeta = np.random.normal(0, 1, N)

	figure()

	W_std_def = WP_std_def(zeta, t)
	plot(t, W_std_def, '--', label='std. approach with N = ' + str(N), linewidth=2.0)
	xlabel('time', fontsize=20)
	title('Wiener process approximated via the standard definition', fontsize=20)
	legend(loc='best', fontsize=20)

	# use the KL expansion to approximation the Wiener process
	# use the same random variables for all M
	zeta = np.random.normal(0, 1, M[-1])

	figure()
	for m in M:
		W_KL = WP_KL_approx(zeta, t, m)

		plot(t, W_KL, '--', label='KL approx with M = ' + str(m), linewidth=2.0)
		xlabel('time', fontsize=20)
		title('Wiener process approximated via an M-term KL expansion', fontsize=20)

	legend(loc='best', fontsize=20)

	show()