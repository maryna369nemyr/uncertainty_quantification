import numpy as np
import chaospy as cp

if __name__ == '__main__':
    # define target function
    f = None
    # create the multivariate distribution
    distr_5D = None

    # quad deg 1D
    quad_deg_1D_vec = [2, 3, 4]

    # vector to hold number of quad nodes
    quad_full_len       = np.zeros(len(quad_deg_1D_vec), dtype=int)
    quad_sparse_len     = np.zeros(len(quad_deg_1D_vec), dtype=int)
    
    # vector to hold results
    integral_full   = np.zeros(len(quad_deg_1D_vec))
    integral_sparse = np.zeros(len(quad_deg_1D_vec))

    # for different number of quadrature degree compute full and sparse grid quadratue
    for i, quad_deg_1D in enumerate(quad_deg_1D_vec): 

        # get the non-sparse quadrature nodes and weight
        
        # get the sparse quadrature nodes and weight
        
        # compute integral for both

    # visualize result using print or plot 