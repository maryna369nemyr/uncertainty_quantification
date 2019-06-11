
import chaospy as cp
import numpy as np

func = lambda x: np.power(x, 7) + np.power(x, 6) + np.power(x, 5) + np.power(x, 4) + np.power(x, 3) + np.power(x, 2) + np.power(x, 1) + 1
N = [2, 3, 4]
Integ =1/8 + 1/7 + 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1

for i,n in  enumerate(N):

	distr = cp.Uniform(0, 1) #!!!
	nodes, weights = cp.generate_quadrature(n , distr, rule = "G")

	integral_Q =np.sum(func(nodes) * weights)
	print(integral_Q,Integ)