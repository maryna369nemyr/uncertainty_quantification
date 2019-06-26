import chaospy as cp
import numpy as np


def func_test(x1,x2,x3,x4,x5):
    out = x1*x2*x3 + np.sin(x2+ x4) - np.exp(-x5)*x1 + x5 - x1
    return out

if __name__ == '__main__':
    K = [3,4,5]
    normal_distr1 = cp.Normal(0,1)
    normal_distr2 = cp.Normal(0, 1)
    normal_distr3 = cp.Normal(0, 1)
    normal_distr4 = cp.Normal(0, 1)
    normal_distr5= cp.Normal(0, 1)
    normal_distr_multivariate = cp.J(normal_distr1, normal_distr2, normal_distr3, normal_distr4, normal_distr5)
    print("With spearse grids:")
    print("_____________________")
    for i, k in enumerate(K):
        # we always generate Gaussian quadratures
        # for normal -  hermit
        #for uniform - Legendre ort polynomials
        nodes, weights = cp.generate_quadrature(k, normal_distr_multivariate, rule = 'G', sparse = True)
        #for Gaussian quadratures we have k =N
        # since we are in 5 dim case we need to generate k points in 5
        # here it is automatically
        # nodes = (k+1)**d
        print("Dimensions of nodes")
        print(nodes.shape)
        x1,x2,x3,x4,x5 = nodes


        poly = cp.orth_ttr(k, normal_distr_multivariate, normed = True)

        y_out = func_test(x1,x2,x3,x4,x5)

        # find generalized Polynomial chaos and expansion coefficients
        gPC_m, expansion_coeff = cp.fit_quadrature(poly, nodes, weights, y_out, retall = True)


        # gPC_m is the polynomial that approximates the most
        print(f'Expansion coeff [0] = {expansion_coeff[0]}')#, expect_weights: {expect_y}')

        mu = cp.E(gPC_m, normal_distr_multivariate)
        print(f'Mean value from gPCE: {mu}')


        integral_Q = np.sum(func_test(x1,x2,x3,x4,x5) * weights)
        print("Integral result =", integral_Q)
    print("_____________________")
    print("Without sparse grids")
    for i, k in enumerate(K):
        # we always generate Gaussian quadratures
        # for normal -  hermit
        #for uniform - Legendre ort polynomials
        nodes, weights = cp.generate_quadrature(k, normal_distr_multivariate, rule = 'G', sparse = False)
        #for Gaussian quadratures we have k =N
        # since we are in 5 dim case we need to generate k points in 5
        # here it is automatically
        # nodes = (k+1)**d
        print("Dimensions of nodes")
        print(nodes.shape)
        x1,x2,x3,x4,x5 = nodes


        poly = cp.orth_ttr(k, normal_distr_multivariate, normed = True)

        y_out = func_test(x1,x2,x3,x4,x5)

        # find generalized Polynomial chaos and expansion coefficients
        gPC_m, expansion_coeff = cp.fit_quadrature(poly, nodes, weights, y_out, retall = True)


        # gPC_m is the polynomial that approximates the most
        print(f'Expansion coeff [0] = {expansion_coeff[0]}')#, expect_weights: {expect_y}')

        mu = cp.E(gPC_m, normal_distr_multivariate)
        print(f'Mean value from gPCE: {mu}')


        integral_Q = np.sum(func_test(x1,x2,x3,x4,x5) * weights)
        print("Integral result =", integral_Q)