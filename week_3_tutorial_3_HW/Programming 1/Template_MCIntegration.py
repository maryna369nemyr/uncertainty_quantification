import numpy as np
from matplotlib.pyplot import *

if __name__ == '__main__':
    # declare function to integrate via Monte Carlo sampling
    func = lambda x: np.sin(x)

    # declare vector with number of samples
    N = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

    I_hat       = np.zeros(len(N)) #Approximate Monte Carlo integral result
    I           = None # TODO: Put the exact integral result here
    est_std_dev = np.zeros(len(N)) #Monte Carlo standard error
    rms = np.zeros(len(N)) #Exact error

    # TODO: for each N, perform Monte Carlo integration, compute I_hat, est_std_dev and absolute error

    # TODO: plot results
