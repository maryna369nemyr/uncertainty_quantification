import chaospy as cp
import numpy as np

f = lambda x: x
g = lambda x: x**2
N = [1, 2, 3, 4]
Integ_f =  0
Integ_g =  1

for i,n in  enumerate(N):

	distr = cp.Normal(0, 1)
	nodes, weights = cp.generate_quadrature(n , distr, rule = "G")

	integral_f =np.sum(f(nodes) * weights)
	integral_g =np.sum(g(nodes) * weights)
	print(integral_f,Integ_f)
	print(integral_g,Integ_g)