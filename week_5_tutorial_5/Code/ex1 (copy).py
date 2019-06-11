import numpy as np
import  matplotlib.pyplot as plt

# Lagrange's cardinal polynomial
def lagrange_poly(eval_point, i, grid):
    result = 1
    for j in range(len(grid)):
        if(j != i):
            result *= (eval_point - grid[j]) / (grid[i] - grid[j])
        else:
            continue
    return result

# Lagrange interpolation; be aware to take into consideration the evaluation of the 
# interpolating polynomial at the grid points
def lagrange_interp(eval_point, grid, func_eval):
    result =  0
    for i, x in enumerate(grid):
        result += func_eval(x) *lagrange_poly(eval_point, i, grid)
    return result

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid, i): #should be without i
    ' you have to precompute weights'
    result = 1
    for j in range(len(grid)):
        if(j != i):
            result *=  (grid[i] - grid[j])
        else:
            continue
    return (1/result)

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, func_eval):

    L_G = np.product(eval_point - grid)
    result =  0
    for i, x in enumerate(grid):
        result += func_eval(x) *L_G *  compute_barycentric_weights(grid, i) / (eval_point - x )
    return result


if __name__ == '__main__':
    # define test's function
    func = lambda x: np.sin(10 * x)
    # define uniform interpolation's grid size
    N = [4, 8, 12, 16]
    # define x axis for plotting: x \in [-1, 1]
    x_axis = np.linspace(-1.0, 1.0, num = 50)
    output_interpolation = []
    output_interpolation2 = []
    # perform interpolation using Lagrange's and barycentric polynomials over different number of nodes
    for i, n  in enumerate(N):

        # generate the unform grid, the function's evaluations and perform the interpolation
        generated_grid = np.linspace(-1.0, 1.0, num=n + 1)


        output_interpolation = np.array([lagrange_interp(x_eval, generated_grid, func) for  x_eval in x_axis ])
        output_interpolation2 = np.array([barycentric_interp(x_eval, generated_grid, func) for x_eval in x_axis])
        print(output_interpolation)
        print("Grid:\n", generated_grid)
        plt.figure(str(n))
        plt.plot(x_axis, func(x_axis), 'b-', label='sin(10x)')

        plt.plot(x_axis, output_interpolation, 'r--', label='Lagr')
        plt.plot(x_axis, output_interpolation2, 'g--', label='Bary')


        plt.legend(loc='best', fontsize=8)
        plt.ylabel('function')
        plt.xlabel('x')

    plt.show()