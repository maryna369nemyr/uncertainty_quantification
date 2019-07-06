import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt



def wiener_process(generated_samples, dt):
    dW = np.sqrt(dt) * generated_samples # np.random.normal(0, dt, N) # times sqrt (t-s) when we generate via standard Normal
    return np.cumsum(dW)

def generate_plot_wiener(generated_samples,N, show = True):
    dt = 1 / N  # t_max / N
    wiener = wiener_process(generated_samples, dt)
    x_axis = np.linspace(0, 1, N, endpoint=True)
    plt.figure("Wiener_process")
    plt.plot(x_axis, wiener, '-k', label='Wiener')

    #plt.savefig(figure_name + '.png')
    if show:
        plt.show()

def plot_processes_via_expnsions(M, t, output_processes, show = True):
    plt.figure("Wiener_process")
    colors = ['r', 'g','b']
    #in reverse order so that the worse approximation is on the top of a better approximation
    for i in range(len(M)):
        plt.plot(t, output_processes[len(M) - i-1], color=colors[i], linestyle='dashed', label='M = ' + str(M[len(M)-i-1]))
    plt.ylabel('W(t)')
    plt.xlabel('t')
    plt.legend(loc='best', fontsize=12)
    #plt.ylabel('W(t)')
    #plt.xlabel('t')
    #plt.legend(loc='best', fontsize=12)
    #plt.savefig(figure_name + '.png')
    if show:
        plt.show()


def karhunen_loeve_expansion(generated_samples, m, t_point):
    samples = generated_samples[0:m]
    idx_n =  np.linspace(1, m, m)
    #print(len(np.sqrt(eigen_values(idx_n))*eigen_vectors(idx_n, t_point)*samples))
    W_t = np.sum(
                np.sqrt(eigen_values(idx_n))*eigen_vectors(idx_n, t_point)*samples)
    return W_t


if __name__ == '__main__':
    N = 1000
    M_lambda = 1000
    M = [10,100,1000]
    np.random.seed(120)
    generated_samples  = np.random.normal(0, 1, max(M))

    generate_plot_wiener(generated_samples, N, False)

    #Eigen values for Wiener process
    eigen_values  = lambda x:  1./ (((x + 0.5)**2) * (np.pi**2))
    eigen_vectors = lambda x, t: np.sqrt(2)* np.sin((x+ 0.5)*np.pi * t)
    #test = lambda x, t: (x - 0.5) * np.pi * t

    vector_lambda = np.linspace(1, M_lambda,M_lambda)  # n in the sum from 1 to M
    #print(eigen_values(vector_lambda))
    ##
    eigen_values_wiener = eigen_values(vector_lambda)
    plt.figure("Eigen values of Wiener process")
    plt.plot(vector_lambda, eigen_values(vector_lambda), '--r', label = "eigen values")
    #plt.show()
    ##
    # Karhunen Loeve expansion

    #print(len(generated_samples[0:M[0]]))
    t = np.linspace(0, 1, N, endpoint=True)
    print(t)

    #print(eigen_vectors(np.array(vector_lambda), 0.333))
    output_processes = np.zeros((len(M), len(t)))
    # rows M=10, ... M= 100
    # columns time [0 .. 1]
    for i, m in enumerate(M):
        for j, t_point in enumerate(t):
            output_processes[i][j] = karhunen_loeve_expansion(generated_samples, m, t_point)

    #M [::-1] reverses the list for
    plot_processes_via_expnsions(M,t,output_processes)



