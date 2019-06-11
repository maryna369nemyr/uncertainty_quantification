import numpy as np
import chaospy as cp

def check_orthonormal(orth_poly, distr, eps = 0.001):
    bool_orth = True
    if(eps):
        signs_after_comma = int(np.ceil(np.abs(np.log10(eps))))
    else:
        signs_after_comma = 8
    matrix_result= np.zeros((len(orth_poly), len(orth_poly)))
    for i in range(len(orth_poly)):
        for j in range(i+1):
            expect = cp.E(orth_poly[i] * orth_poly[j], distr)
            matrix_result[i,j] = np.around(expect, signs_after_comma)
            if(i!= j and np.abs(expect)>eps):
                print(f'Not orthonormal {i,j} according to precision {eps}: {expect}')
                bool_orth = False

    print(matrix_result)
    return bool_orth


if __name__ == '__main__':
    distr_uniform = cp.Uniform(lower= -1, upper =1)
    distr_normal  =cp.Normal(mu = 10, sigma = 1)

    N = [2,5,8]
    # 2 polynomials but in fact we get 3 polynomials where p0 = 1.0 fro all of them due to Stjiltjes formula
    eps = 0.0001
    for i, n in enumerate(N):
        orth_poly_unif = cp.orth_ttr(n, distr_uniform, normed = True)
        ort_poly_normal =  cp.orth_ttr(n, distr_normal, normed = True)

        print("Printing out the polynomials ... ")
        print("Unif ", cp.around(orth_poly_unif, 1))
        print("Norm", cp.around(ort_poly_normal, 1))

        print("______________________________")
        print(f'Orthonormal polynomials wrt UNIFORM distr with N = {n}: {check_orthonormal(orth_poly_unif, distr_uniform, eps)}')
        print("______________________________")
        print(f'Orthonormal polynomials wrt NORMAL distr with N = {n}: {check_orthonormal(ort_poly_normal, distr_normal, eps)}')
        print("______________________________")

