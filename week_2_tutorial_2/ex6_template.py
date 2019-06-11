# import numpy
import numpy as np
# import matplotlib related packages
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# given mean and variance
mu_given  = [0.1, 0.5]
V_given   = [[1.0,0.2],[0.2, 1.0]]

# standard mean and variance
mu_std  = [0,0]
V_std   = [[1,0],[0,1]]

# no of samples to generate
no_samples = 10000

# variant 1: sample from multivariate normal, no transformation

sampled_normal  = np.random.multivariate_normal(mu_given, V_given,no_samples)
# variant 2: mu_givenltivariate normal, with transformation

L = np.linalg.cholesky(V_given)
sampled_transfromation =[]
for i in range(no_samples):
    n_std = np.random.multivariate_normal(mu_std, V_std, 1)
    my_n = mu_given + np.matmul(L, n_std)
    sampled_transfromation.append(my_n)
# Plot results


