import numpy as np
import chaospy as cp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p
	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]
	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
	return sol[:, 0]

def generate_matrices_MC(joint_distr, n_MC):
    samples_A=joint_distr.sample(size=n_MC).T
    samples_B= joint_distr.sample(size=n_MC).T
    return (samples_A, samples_B)

def generate_matrices(joint_distr, n_MC):
    samples_Sobol_A=joint_distr.sample(size=n_MC, rule="S").T
    samples_Sobol_B =joint_distr.sample(size=n_MC, rule="S").T
    return (samples_Sobol_A, samples_Sobol_B)


def matrix_column_substitute(A, B_from, col):
    #column - random variable
    #rows - samples
    A_output = A.copy()
    A_output[:,col] = B_from[:, col]
    return A_output

def generate_matrices_from_A_B(A, B):
    A_B = []
    for i in range(B.shape[1]):
        A_B.append(matrix_column_substitute(A,B,i))
    return A_B
def evaluate_model(list_all_matrices, params_known):
    # A, B, A_B(0), ..., A_B(4)
    #cols - random vars
    #rows - samples

    atol,rtol,w = params_known
    f_eval = []

    for matrix in list_all_matrices:
        y_out = np.zeros(matrix.shape[0])
        for row_sample_idx in range(matrix.shape[0]):
            c_, k_, f_, y0_, y1_ = matrix[row_sample_idx,:]
            y_out[row_sample_idx] = discretize_oscillator_odeint(model, atol, rtol, (y0_, y1_), (c_, k_, f_, w), t)[-1]
        f_eval.append(y_out)
    return f_eval

def plot_sobol_indices(total_sobol_indices, title_name, show = True):
    sobol_indices_x = [1, 2, 3, 4, 5]
    labels = ['c', 'k', 'f', r"$y_0$", r"$y_1$"]

    plt.figure()
    barlist = plt.bar(sobol_indices_x, total_sobol_indices, align='center')
    barlist[0].set_color('r')
    barlist[1].set_color('b')
    barlist[2].set_color('g')
    barlist[3].set_color('m')
    barlist[4].set_color('k')
    plt.xticks(sobol_indices_x, labels)
    plt.title(title_name)
    plt.tight_layout()
    plt.savefig(title_name + ".png")
    if show:
        plt.show()

def estimate_sobol(f_eval_list, n_mc):
    f_eval_A, f_eval_B, f_eval_rest = f_eval_list[0], f_eval_list[1], f_eval_list[2:len(f_eval_list)]
    sobol_total =np.zeros(len(f_eval_rest))
    sobol_first = np.zeros(len(f_eval_rest))
    for i, f_eval_A_B_i in enumerate(f_eval_rest):

        sobol_first[i]  = np.sum(f_eval_B *(f_eval_A_B_i - f_eval_A))/n_mc
        sobol_total[i]  = np.sum((f_eval_A_B_i - f_eval_A)**2)/(2*n_mc)

    return sobol_first, sobol_total
def show_sobol(sobol_first, sobol_total, title_names):
    print(f'MC. The first order Sobol indices are \n {sobol_first}')
    print(f"MC. The total Sobol' indices are \n {sobol_total}")

    plot_sobol_indices(sobol_first, title_names[0], False)
    plot_sobol_indices(sobol_total, title_names[1], False)




if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    np.random.seed(10)
    atol, rtol = 1e-10, 1e-10

    # w is deterministic
    w = 1.00

    # create uniform distribution object
    distr_c     = cp.Uniform(0.08, 0.12)
    distr_k     = cp.Uniform(0.03, 0.04)
    distr_f     = cp.Uniform(0.08, 0.12)
    distr_y0    = cp.Uniform(0.45, 0.55)
    distr_y1    = cp.Uniform(-0.05,  0.05)

    # create the multivariate distribution

    # quad deg 1D
    quad_deg_1D = 4
    poly_deg_1D = 4

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])

    N = [3,4]
    K = [3,4]

    # create the orthogonal polynomials

    #################### full grid computations #####################

    joint_distr = cp.J(distr_c, distr_k,  distr_f, distr_y0, distr_y1)
    n_MC = 1000
    title_names  =["First order Sobol indices, MC method", "Total Sobol indices, MC method"]

    #A,B = generate_matrices(joint_distr, n_MC) # quasi monte carlo gives the same result for twp matrices
    now_mc = time.time()
    A_mc, B_mc = generate_matrices_MC(joint_distr, n_MC)
    list_A_B = generate_matrices_from_A_B(A_mc,B_mc)
    all_matrices = [A_mc]+ [B_mc] + list_A_B
    f_eval_list = evaluate_model(all_matrices, (atol,rtol,w))

    sobol_first, sobol_total = estimate_sobol(f_eval_list, n_MC)
    show_sobol(sobol_first, sobol_total, title_names)
    print(f" >>> Time for MC method n_mc = {n_MC} (Sobol indices): {time.time() - now_mc}")
    plt.show()

    print("Done.")