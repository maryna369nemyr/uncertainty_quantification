import numpy as np
import chaospy as cp

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
    return cov

if __name__== '__main__':
    N =  40
    l = 1.0
    points  = create_grid(N, N)
    print_list(points)
    cov_exp =exp_kernel(points,l)
    cov_sq_exp = squared_exp_kernel(points, l)
    print(cov_exp[1:10,1:10])


