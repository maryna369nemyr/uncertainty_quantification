import numpy as np
import matplotlib.pyplot as plt


def discretize_oscillator_euler(t, dt, params, f_vector):
    c, k, w, y0, y1 = params

    # initalize the solution vector with zeros
    z0 = np.zeros(len(t))
    z1 = np.zeros(len(t))

    z0[0], z1[0] = y0, y1
    # implement the obtained Euler scheme
    for i in range(0, len(t)- 1):
        z1[i + 1] = z1[i] + dt*(-k*z0[i] - c*z1[i]+f_vector[i]*np.cos(w*t[i]))
        z0[i + 1] = z0[i] + dt*z1[i]

    return z0


def wiener_process_dt(dt, f_mean, N):
    #this is not the same as wiener_process starting from mean or generating rv with mean
    dW = np.sqrt(dt) * np.random.normal(0,1, N - 1)
    W = np.cumsum(dW)
    #W = np.insert(W, 0, f_mean)
    #return W
    W = np.insert(W, 0, 0)
    return W + f_mean



def plot_wiener(wiener, t_axis, show = True):
    #plt.figure("Wiener process std")
    plt.figure("Expansion_f M = 1000")
    #plt.plot(t_axis, wiener, '-k', label='Wiener')
    plt.plot(t_axis, wiener, '-k')
    plt.ylabel('W(t)')
    plt.xlabel('t')
    #plt.legend(loc='best', fontsize=12)
    if show:
        plt.show()

def karhunen_loeve_expansion(generated_samples, m, t_point, T_max, f_mean):
    samples = generated_samples[0:m]
    idx_n =  np.linspace(1, m, m)
    W_t = np.sum(
                np.sqrt(eigen_values(idx_n, T_max))*eigen_vectors(idx_n, t_point, T_max)* samples)

    return W_t + f_mean

def karhunen_loeve_expansion_vector(generated_samples, m, t_vector, T_max,f_mean):
    f_appr_generated  = np.zeros(len(t_vector))
    for i, t_point in enumerate(t_vector):
        f_appr_generated[i] = karhunen_loeve_expansion(generated_samples, m, t_point, T_max, f_mean)
    return f_appr_generated

def plot_processes_via_expansions(M, t, output_processes, show = True):

    #in reverse order so that the worse approximation is on the top of a better approximation
    for i in range(len(M)):
        plt.figure("Expansion_f M = " + str(M[len(M)-i-1]))
        plt.plot(t, output_processes[len(M) - i-1], linestyle='-')
    plt.ylabel('W(t)')
    plt.xlabel('t')
    #plt.legend(loc='best', fontsize=12)
    plt.savefig("Expansion_f M = " + str(M[len(M)-i-1]) + '.png')
    if show:
        plt.show()

if __name__ == '__main__':
    np.random.seed(130)
    #parameters setup
    c = 0.5
    k = 2.0
    y0 = 0.5
    y1 = 0.
    w = 1.0
    f_mean = 0.5

    #Eigen values and eigen vectors for Wiener process
    eigen_values  = lambda x, T_max:  T_max**2 / (((x - 0.5)**2) * (np.pi**2))
    eigen_vectors = lambda x, t, T_max: np.sqrt(2/T_max)* np.sin((x- 0.5)*np.pi * t/ T_max)

    #time domain
    t_max  = 10
    dt  = 0.01
    num_t  = t_max // dt +1
    t = np.round(np.linspace(start = 0, stop  = t_max, num = num_t), int(np.abs(np.log10(dt))) + 1)
    N_t_axis = len(t)

    #wiener_f = wiener_process(t, f_mean)
    #plot_wiener(wiener_f, t) #gives the same result with a fixed seed
    wiener_f = wiener_process_dt(dt, f_mean, N_t_axis)
    #plot_wiener(wiener_f, t, False) # gives the same result with a fixe sid


    n_mc = 100
    params = c, k, w, y0, y1

    ode_wiener = []
    ode_wiener_10=[]

    #from Wiener
    print("Wiener process definition...")
    plt.figure("With wiener def")
    for i in range(n_mc):
        # for each monte carlo sampling
        f_generated = wiener_process_dt(dt, f_mean, N_t_axis)
        output = discretize_oscillator_euler(t, dt, params, f_generated)
        ode_wiener.append(output)
        ode_wiener_10.append(output[-1])
        plt.plot(t, output)

    ode_wiener_10 = np.array(ode_wiener_10)
    ode_wiener = np.array(ode_wiener)

    mu = np.mean(ode_wiener_10)
    V =np.var(ode_wiener_10, ddof  = 1)

    mean_mc = np.mean(ode_wiener, axis=0)
    # returns mean columnwise
    # if axis  = 1 the it return mean rowwise
    var_mc = np.var(ode_wiener, axis=0, ddof=1)
    plt.plot(t, mean_mc, linewidth=3, color='k',linestyle='--')
    plt.plot(t, mean_mc + np.sqrt(var_mc), linewidth=3, color='k', linestyle='--')
    plt.plot(t, mean_mc - np.sqrt(var_mc), linewidth=3, color='k', linestyle='--')
    plt.savefig("With wiener def.png")
    #plt.show()
    ##########

    print("Karhunen Loeve expansion...")
    # Karhunen Loeve expansion
    M = [5, 100, 1000]
    f_appr_generated = np.zeros((len(M), len(t)))
    ode_wiener_KL = np.zeros((n_mc, len(M), len(t)))
    ode_wiener_KL_10= np.zeros((n_mc, len(M)))

    for l in range(n_mc):
        generated_samples =  np.random.normal(0, 1, max(M))

        for m_idx, m in enumerate(M):
            plt.figure("Karhuren-Loeve M = "+ str(m))
            f_appr_generated[m_idx] = karhunen_loeve_expansion_vector(generated_samples, m, t, t_max, f_mean)
            output = discretize_oscillator_euler(t,dt,params, f_appr_generated[m_idx,:])


            ode_wiener_KL[l][m_idx] = output
            ode_wiener_KL_10[l][m_idx] = output[-1]
            plt.plot(t, output)
        plot_processes_via_expansions(M, t, f_appr_generated, False)
    plot_wiener(wiener_f, t, False)
    plt.savefig("Expansion_f M = 1000 with std Wiener.png")

    print("Calculating mean and variance for KL expansion...")
    mean_mc = np.mean(ode_wiener_KL, axis = 0)
    var_mc = np.var(ode_wiener_KL, axis=0, ddof  = 1)


    mu_ = np.mean(ode_wiener_KL_10, axis  = 0)
    V_ =np.var(ode_wiener_KL_10,  axis  = 0, ddof  = 1)



    print("Table (approximation via Karhunen-Loeve expansion and Wiener process def)")
    print("%10s  %10s  %10s " %("appr_m", "mean", "var"))
    for i in range(len(M)):
        print("%10d" % M[i], "   %.8f  %.8f" % ( mu_[i], V_[i]))
    print("wiener_def    %.8f  %.8f" % (mu,V))


    for m_idx, m in enumerate(M):
        plt.figure("Karhuren-Loeve M = " + str(m))
        plt.plot(t, mean_mc[m_idx], linewidth=3, color='k', linestyle='--')
        plt.plot(t, mean_mc[m_idx] + np.sqrt(var_mc[m_idx]), linewidth=3, color='k', linestyle='--')
        plt.plot(t, mean_mc[m_idx] - np.sqrt(var_mc[m_idx]), linewidth=3, color='k',linestyle='--')
        plt.savefig("Karhuren-Loeve M = " + str(m)+ ".png")
    plt.show()
