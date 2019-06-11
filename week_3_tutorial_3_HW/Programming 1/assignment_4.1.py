import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


a, b = 0, 1
func = lambda x: np.exp(x)
phi1 = lambda x: x
phi2 = lambda x: x + 1
# ration between f and beta_pdf function
h_beta1 = lambda x: func(x)/beta.pdf(x, 5,1)
h_beta2 = lambda x: func(x)/beta.pdf(x, 0.5, 0.5)

# declare vector with number of samples
N = [10, 100, 1000, 10000, 100000, 1000000]

I_hat = np.zeros(len(N))
I_ref = np.exp(b) - np.exp(a) # TODO: Put the exact integral result here

I_MC, I_cv1, I_cv2, I_is1, I_is2 =np.zeros(len(N)),np.zeros(len(N)),np.zeros(len(N)),np.zeros(len(N)),np.zeros(len(N))
mu_phi1 = 0.5 #theoretical mean of the function phi1 on [0,1]
mu_phi2  =1.5 # theoretical mean of the function phi2 on the [0, 1]
var_phi = 1/12 # theoretical variance is the same for both funcs (phi1 and phi2

phi1_mean, phi2_mean = np.zeros(len(N)), np.zeros(len(N)) #empirical means based on generated points

for i, n in enumerate(N):

    points = np.random.uniform(a, b, n)
    #standard Monte-Carlo
    I_MC[i] = (np.mean(func(points)) * (b - a))
    #control variates

    # points shouldn't necesseraly coincide with the one that were generated and used for f
    samples1 = np.random.uniform(a, b, n)
    samples2 =np.random.uniform(a, b, n)
    phi1_mean[i] =(np.mean(phi1(samples1)) * (b - a))
    phi2_mean[i] =(np.mean(phi2(samples2)) * (b - a))

    #importance sampling
    points_beta_1 = np.random.beta(5,1, n)
    points_beta_2 = np.random.beta(0.5, 0.5, n)
    I_is1[i]= np.mean(h_beta1(points_beta_1)) #  out p here is unifrom pdf on [0, 1] so it is 1
    I_is2[i] = np.mean(h_beta2(points_beta_2))


### this is the oprimal alpha
#
#alpha = r_fh(sigma_f/sigma_h) = cov(f, h)/sigma_h**2, h - is additional function#
#we can put alpha equals 1
#I_cv = f_m + alpha(Eh(x) - h_m)

phi1_mean, phi2_mean = np.array(phi1_mean), np.array(phi2_mean)

# we put alpha is equal to 1
I_cv1 = I_MC + mu_phi1 - phi1_mean
I_cv2 = I_MC + mu_phi2 - phi2_mean


rel_err_MC = np.abs(1 - I_MC/I_ref)
rel_err_cv1 = np.abs(1 - I_cv1/I_ref)
rel_err_cv2 = np.abs(1 - I_cv2/I_ref)
rel_err_is1 = np.abs(1 - I_is1/I_ref)
rel_err_is2 = np.abs(1 - I_is2/I_ref)

plt.figure("Comparison of methods")
plt.loglog(N, rel_err_MC, 'r-', label='MC')
plt.loglog(N, rel_err_cv1, 'b-', label='CV x')
plt.loglog(N, rel_err_cv2, 'm-', label='CV x + 1')
plt.loglog(N, rel_err_is1, 'c-', label='IS a = 5, b = 1')
plt.loglog(N, rel_err_is2, 'g-', label='IS a = 0.5, b= 0.5')

plt.legend(loc='best', fontsize=8)
plt.ylabel('Relative error')
plt.xlabel('Number of samples (loglog)')

# TODO: plot results

print("Integral:",I_ref)
print('MC', I_MC)
print('CV x',I_cv1 )
print('CV x + 1', I_cv2)
print('IS a = 5, b = 1', I_is1)
print('IS a = 0.5, b= 0.5', I_is2)

plt.show()