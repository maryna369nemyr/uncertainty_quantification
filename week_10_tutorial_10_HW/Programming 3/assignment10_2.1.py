import numpy as np
import matplotlib.pyplot as plt
import random

def create_grid_dict(N):
    # mesh size in both directions (NxN)
    mesh_size_x = N
    mesh_size_y = N

    # the coordinates that we are interested in are in the middle of each Cartesian cell
    x_coord = np.arange(1. / (2 * mesh_size_x), 1., 1. / mesh_size_x)
    y_coord = np.arange(1. / (2 * mesh_size_y), 1., 1. / mesh_size_y)

    # to make the evaluation of the covariance function easier
    mesh_coord = {}
    for i in range(mesh_size_x):
        for j in range(mesh_size_y):
            mesh_coord[i * mesh_size_x + j] = x_coord[i], y_coord[j]

    mesh_coord = mesh_coord.values()
    return mesh_coord
def create_grid(N_x, N_y):

    x_coord, y_coord = np.arange(1./(2*N_x), 1.0, 1.0/N_x),  np.arange(1./(2*N_y), 1.0, 1.0/N_y)
    x_coord, y_coord = np.round(x_coord, 5), np.round(y_coord, 5)
    mesh_coord = []
    for i in range(N_x):
        for j in range(N_y):
            mesh_coord.append(np.array([x_coord[i], y_coord[j]]))
    return mesh_coord

def print_dict(dict):
    print("The length of the grid points = ", len(dict))
    for key, value in dict.items():
        print(f"dict [{key}] = {value}")

def print_list(list):
    print("The length of the grid points = ", len(list))
    for i, item in enumerate(list):
        print(f"list[{i}] = {item}")

def exp_kernel(points, l):
    cov = np.zeros((len(points), len(points)))
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
         cov[i][j]  = np.exp(- np.linalg.norm( p1 - p2 , ord=None) /l)  # 2-norm
    return cov

def squared_exp_kernel(points, l):
    cov = np.zeros((len(points), len(points)))
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            cov[i][j]  = np.exp(-(np.linalg.norm(p1 - p2 , ord=None)**2) /(2*l**2))  # 2-norm
            if(i == j):
                cov[i][j] = cov[i][j] + 0.00000001 # to make matrix psd for sure
    return cov

def generate_heat_map(N, mean_vec, L, samples, figure_name, axis):

    gaussian_RF =  mean_vec + np.matmul(L, samples.T).T
    # gives (N**2, 1) where all columns are identical

    gaussian_RF = np.reshape(gaussian_RF, (N,N))
    #shape (N, N) what we are interested in
    #plt.figure(figure_name)

    axis.imshow(gaussian_RF, cmap = "coolwarm")
    axis.set_title(figure_name)
    #plt.savefig(figure_name + '.png')
    #plt.savefig('./output_pictures/' + figure_name + '.png')

if __name__== '__main__':
    N =  10
    l = 1.00 #0.05 and mean 0.5
    mean_func = np.array([0.1]*N**2)
    points  = create_grid(N, N)
    print(f"We have generated {len(points)} number of points")
    #print_list(points)

    print(f"Calculating covariance matrices...")
    cov_exp =exp_kernel(points,l)
    cov_sq_exp = squared_exp_kernel(points, l)

    print(f"Calculating Cholesky decomposition...")
    L_cov_exp = np.linalg.cholesky(cov_exp) #shape N**2, N**2
    L_cov_sq_exp  = np.linalg.cholesky(cov_sq_exp)
    print("Generating heatmaps for Gaussian random fields...")

    fig, axes = plt.subplots(2, 3)
    #plt.plot()

    for i in range(3):
        mean_std = np.zeros(N ** 2)
        cov_std = np.identity(N ** 2)
        samples = np.random.multivariate_normal(mean_std, cov_std, 1)
        # samples should be one vector in R^100
        # samples.shape = (1, N**2)

        generate_heat_map(N, mean_func, L_cov_exp, samples,  "Exp kernel_" + str(i), axes[0,i])
        generate_heat_map(N, mean_func, L_cov_sq_exp, samples, "Sq exp kernel_"+ str(i), axes[1,i])
    plt.savefig('result_gaussian_random_fields.png')
    plt.show()

