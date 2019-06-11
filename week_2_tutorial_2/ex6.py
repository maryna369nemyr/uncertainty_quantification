# import numpy
import numpy as np
# import matplotlib related packages
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# given mean and variance
mu_given  = np.array([0.1, 0.5])
V_given   = np.array([[1.0, 0.2], [0.2, 1.0]])

# standard mean and variance
mu_std  = np.array([0.0, 0.0])
V_std   = np.array([[1.0, 0.0], [0.0, 1.0]])

# no of samples to generate
no_samples = 10000

# variant 1: sample from multivariate normal, no transformation
X_no_trans, Y_no_trans = np.random.multivariate_normal(mu_given, V_given, no_samples).T

# variant 2: mu_givenltivariate normal, with transformation
# if mu_given is the mean vector and V the covariance matrix, Ng = mu_given + EN, where EE^T = V and N is a standard normal random variable

# Cholesky of Covariance matrix
E = np.linalg.cholesky(V_given)

X_with_trans, Y_with_trans = np.zeros(no_samples), np.zeros(no_samples)
for i in xrange(no_samples):
    N   = np.array(np.random.multivariate_normal(mu_std, V_std, 1)).T
    Ng  = mu_given + np.dot(E, N).T[0]

    X_with_trans[i] = Ng[0]
    Y_with_trans[i] = Ng[1]

plt.plot(X_no_trans, Y_no_trans, 'gx', label='bivariate normal: no transformation')
plt.plot(X_with_trans, Y_with_trans, 'r*', label='bivariate normal: with transformation')
plt.legend(loc='best', fontsize=20)

# plot the contour and surface of the given 2D Gaussian
# create a 2D meshgrid
delta   = 0.025
left_b  = -5.
right_b = 5.
x       = np.arange(left_b, right_b, delta)
y       = np.arange(left_b, right_b, delta)
X, Y    = np.meshgrid(x, y)

# create the bivariate normal using the mlab's bivariate_normal function
# the sintax is bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0)
Z = mlab.bivariate_normal(X, Y, V_given[0, 0], V_given[1, 1], mu_given[0], mu_given[1], V_given[1, 0])

# plot the contour
fig = plt.figure()
plt.contour(X, Y, Z)
plt.grid()

# plot the surface of the bivariate_normal
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_xlabel('x', fontsize=30)
ax.set_ylabel('y', fontsize=30)
ax.set_zlabel('bivariate normal distribution', fontsize=20)

plt.show()
