import numpy as np
import matplotlib.pyplot as plt


def discretize_oscillator_euler(t, dt, params):
    c, k, f, w, y0, y1 = params

    # initalize the solution vector with zeros
    z0 = np.zeros(len(t))
    z1 = np.zeros(len(t))

    z0[0], z1[0] = y0, y1
    # implement the obtained Euler scheme
    for i in range(1, len(t)- 1):
        z1[i + 1] = z1[i] + dt*(-k*z0[i] - c*z1[i]+f*np.cos(w*t[i]))
        z0[i + 1] = z0[i] + dt*z1[i]

    return z0



def wiener_process(t_vector, f_mean):
    #when we have deviationg in dt liek 0.01 etc
    t_vector_from = t_vector
    t_vector_from = np.insert(t_vector, len(t_vector_from), 0)
    t_vector_what = np.insert(t_vector, 0, 0)
    dt = t_vector_from - t_vector_what
    dt = dt[1:-1]

    dW = np.sqrt(dt) * np.random.normal(0,1, len(t_vector) - 1)
    W = np.cumsum(dW)
    W = np.insert(W, 0, f_mean)
    return W

def wiener_process_dt(dt, f_mean, N):
    dW = np.sqrt(dt) * np.random.normal(0,1, N - 1)
    W = np.cumsum(dW)
    W = np.insert(W, 0, f_mean)
    return W


def plot_wiener(wiener, t_axis, show = True):
    plt.figure("Wiener process std")
    plt.plot(t_axis, wiener, '-k', label='Wiener')
    plt.ylabel('W(t)')
    plt.xlabel('t')
    plt.legend(loc='best', fontsize=12)
    if show:
        plt.show()

def karhunen_loeve_expansion(generated_samples, m, t_point, T_max):
    samples = generated_samples[0:m]
    idx_n =  np.linspace(1, m, m)
    W_t = np.sum(
                np.sqrt(eigen_values(idx_n, T_max))*eigen_vectors(idx_n, t_point, T_max)* samples)

    return W_t

def plot_processes_via_expnsions(M, t, output_processes, show = True):
    plt.figure("Wiener_process")
    colors = ['r', 'g','b']
    #in reverse order so that the worse approximation is on the top of a better approximation
    for i in range(len(M)):
        plt.plot(t, output_processes[len(M) - i-1], color=colors[i], linestyle='-', label='M = ' + str(M[len(M)-i-1]))
    plt.ylabel('W(t)')
    plt.xlabel('t')
    plt.legend(loc='best', fontsize=12)
    plt.savefig("Wiener_expansions" + '.png')
    if show:
        plt.show()

if __name__ == '__main__':
    np.random.seed(120)
    #parameters setup
    c = 0.5
    k = 2.0
    y0 = 0.5
    y1 = 0.
    w = 1.0
    f_mean = 0.5

    #Eigen values and eigen vectors for Wiener process
    eigen_values  = lambda x, T_max:  T_max**2 / (((x + 0.5)**2) * (np.pi**2))
    eigen_vectors = lambda x, t, T_max: np.sqrt(2/T_max)* np.sin((x+ 0.5)*np.pi * t/ T_max)

    #time domain
    t_max  = 10
    dt  = 0.01
    num_t  = t_max // dt +1
    t = np.round(np.linspace(start = 0, stop  = t_max, num = num_t), int(np.abs(np.log10(dt))) + 1)
    N_t_axis = len(t)
    print("here=", t, N_t_axis)

    wiener_f = wiener_process(t, f_mean)
    plot_wiener(wiener_f, t) #gives the same result with a fixed seed

    wiener_2f = wiener_process_dt(dt, f_mean, N_t_axis)
    plot_wiener(wiener_2f, t) # gives the same result with a fixe sid


    n_mc = 1000


    for i in range(n_mc):
        # for each monte carlo sampling
        params =c, k, f_mean, w, y0, y1

        discretize_oscillator_euler(t, dt, params)

    # Karhunen Loeve expansion
    M = [5, 10, 100]

    generated_samples = np.random.normal(0, 1, max(M))

    output_processes = np.zeros((len(M), len(t)))
    # rows M=10, ... M= 100
    # columns time [0 .. 1]

    for i, m in enumerate(M):
        for j, t_point in enumerate(t):
            output_processes[i][j] = karhunen_loeve_expansion(generated_samples, m, t_point, t_max)
    plot_processes_via_expnsions(M,t,output_processes)
