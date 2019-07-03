import numpy as np
from matplotlib.pyplot import *

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # mesh size in both directions (NxN)
    mesh_size_x = 40
    mesh_size_y = 40

    # the coordinates that we are interested in are in the middle of each Cartesian cell
    x_coord = np.arange(1. / (2 * mesh_size_x), 1., 1. / mesh_size_x)
    y_coord = np.arange(1. / (2 * mesh_size_y), 1., 1. / mesh_size_y)

    # to make the evaluation of the covariance function easier
    mesh_coord = []
    for i in range(mesh_size_x):
        for j in range(mesh_size_y):
            mesh_coord.append((x_coord[i], y_coord[j]))

   # mesh_coord = mesh_coord.values()

    # the given mean function
    mean_fct = 0.1

    # the given covariance function
    l = 1
    norm = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    norm_sq = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    cov_exp = lambda x, y: np.exp(-(norm(x, y)) / (l))
    cov_expsq = lambda x, y: np.exp(-(norm_sq(x, y)) / (2 * l ** 2))

    # the size of the mean function and covariance function depends on N^2
    no_rows_x = mesh_size_x * mesh_size_x
    no_rows_y = mesh_size_y * mesh_size_y
    cov_hat_exp = np.zeros((no_rows_x, no_rows_y))
    cov_hat_expsq = np.zeros((no_rows_x, no_rows_y))

    # create the discrete mean function
    mean_hat = mean_fct * np.ones(no_rows_x)

    # create the discrete covariance function
    for i in range(no_rows_x):
        for j in range(no_rows_y):
            cov_hat_exp[i, j] = cov_exp(mesh_coord[i], mesh_coord[j])
            cov_hat_expsq[i, j] = cov_expsq(mesh_coord[i], mesh_coord[j])
            if i == j:
                cov_hat_expsq[i, j] += 0.000000001

    print("Condition Exp: ", np.linalg.cond(cov_hat_exp))
    print("Condition SqExp: ", np.linalg.cond(cov_hat_expsq))

    print("Cov Exp: ", (cov_hat_exp))
    print("Cov SqExp: ", (cov_hat_expsq))

    # perform a Cholesky decomposition of the covariance matrix
    E_exp = np.linalg.cholesky(cov_hat_exp)
    E_expsq = np.linalg.cholesky(cov_hat_expsq)

    # we need the standard mean and variance to sample for a standard multivariate Gaussian
    mean_std = np.zeros(no_rows_x)
    cov_std = np.identity(no_rows_x)

    # plot the sample
    no_figures = 3
    for i in range(no_figures):
        figure()

        # take one standard multivariate Gaussian sample
        normal_sample = np.array(np.random.multivariate_normal(mean_std, cov_std, 1)).T

        # shift and rotate the sample sample to a sample from the given random field
        sample = mean_hat + np.dot(E_exp, normal_sample).T[0]

        # transform the sample vector of zie N^2 to a matrix of size N
        sample = sample.reshape(mesh_size_x, mesh_size_y)

        # plot the results
        imshow(sample, cmap='coolwarm')

    for i in range(no_figures):
        figure()

        # take one standard multivariate Gaussian sample
        normal_sample = np.array(np.random.multivariate_normal(mean_std, cov_std, 1)).T

        # shift and rotate the sample sample to a sample from the given random field
        sample = mean_hat + np.dot(E_expsq, normal_sample).T[0]

        # transform the sample vector of zie N^2 to a matrix of size N
        sample = sample.reshape(mesh_size_x, mesh_size_y)

        # plot the results
        imshow(sample, cmap='coolwarm')

    show()