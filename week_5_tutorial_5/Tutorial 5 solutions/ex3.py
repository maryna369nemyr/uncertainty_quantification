import chaospy as cp
import numpy as np

if __name__ == '__main__':
	# define test functions
	f = lambda x: x
	g = lambda x: x**2

	# note that if you have a random variable X that is standard normally distributed, i.e. X ~ N(0, 1)
	# the expectation E[X] = int_R xf_X(x)dx and Var[X] = E[X^2] - E[X]^2 = int_R x^2f_X(x)dx - (int_R xf_X(x)dx)^2
	# since X ~ N(0, 1), E[X] = 0 and Var[X] = 1, therefore int_R xf_X(x)dx = 0
	# int_R x^2f_X(x)dx - (int_R xf_X(x)dx)^2 = int_R x^2f_X(x)dx = 1

	# number of nodes and weights; note that for a given M, chaospy generate M + 1 nodes and weights
	N = [0, 1, 2, 3, 4]

	quad_f = np.zeros(len(N))
	quad_g = np.zeros(len(N))
	for idx, N_ in enumerate(N):
		# the computations are performed with respect to the normal distribution
		distr = cp.Normal()
		
		# generate nodes and weights	
		nodes, weights = cp.generate_quadrature(N_, distr, rule='G')

		# approximate the integrals of f and g with respect to the normal distribution
		quad_f[idx] = np.sum([f(n)*w for n, w in zip(nodes[0], weights)])
		quad_g[idx] = np.sum([g(n)*w for n, w in zip(nodes[0], weights)])

	# print result
	print "N |\t int_R f(x)w(x)dx \t| int_R g(x)w(x)dx"
	print "----------------------------------------------------"
	for idx, N_ in enumerate(N):
		if N_ < 2:
			print N_+1, '|\t\t', quad_f[idx], '\t\t|\t', quad_g[idx]
		else:
			print N_+1, '|\t', quad_f[idx], '\t|\t', quad_g[idx]