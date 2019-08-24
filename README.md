# Uncertainty Quantification (SS 2019)
Themain goal of the course was to learn how to propagate the uncertainty of the stochastic model to our output, 
having stochastic and deterministic input varianles. This kind of task is called **_Forward UQ_**

Most of the code was written using the prepared library `chaospy` in python.

1. Sampling methods
 1.1 Monte Carlo Sampling
 
 1.2 Quasi Mpnte Cralo Sampling (Halton Sequences)
 
 1.3 Advanced Sapling (Control variates, Importance sampling, Antithetic sampling)
 
2. Generalized Polynomial Chaos Expansion
  2.1 gPCE + pseudo spectral approach
  
  2.2 gPCE + Stochastic Galerkin
  
3. Sparse grids for high dimensional problems
4. Sensitivity analysis (Sobol indices - variance based).
   How one random input variable is significtn wrt to the contribution of the uncertainty into the output variable.
5.Karhunen Loeve expansion - propagation for srochastic processes.
