import sys
import argparse
sys.path.append('./tools/')
from forwardmodels import *
from forwardmodels import prosail_4d_L8 as PROSAIL

from autograd.misc.flatten import flatten_func
import autograd.numpy as np
from autograd import grad
from autograd.numpy.linalg import solve

from hmc import HMC
from scipy.optimize import fmin_l_bfgs_b

np.random.seed(0)

def diff(f, c0, delta = 1e-9):

    c0 = c0.flatten();    
    d = len(c0)
    derivs = None

    for i in range(d):

        step = np.zeros(d)
        step[i] = delta

        xa = c0 - step
        xb = c0 + step

        value = (f(xb) - f(xa)) /(2 * delta) 

        if derivs is None:
            derivs = value
        else:
            derivs = np.vstack((derivs, value))

    return derivs

# The model that generates effects from causes

def RTM(c):
    if len(c.shape) == 1:
        return PROSAIL(c.reshape(1,3))
    else:
        return PROSAIL(c)

# The model that generates effects from causes

global D, P
D = 4  # cause dimension
P = 9  # effect dimension

def RTM(c):
    if len(c.shape) == 1:
        return PROSAIL(c.reshape(1,D))
    else:
        return PROSAIL(c)

# Actual prior parameters
true_prior_mean = np.array([6., 6., 6., 6.])
true_prior_cov = np.array([[1.0, 0.8, 0.6, 0.4],
                           [0.8, 1.0, 0.4, 0.5],
                           [0.6, 0.4, 1.0, 0.3],
                           [0.4, 0.5, 0.3, 1.0]])


#############################
### Generate trainingdata ###
#############################
n_data = 100

C_true = np.dot(np.random.normal(size = (( n_data, D))), np.linalg.cholesky(true_prior_cov).T) + true_prior_mean
E_true = RTM(C_true)

## Define Likelihood and prior

def logprior(params, c):

    m = params['mean_prior']
    L = np.tril(params['chol_cov_prior'], 0) # DHL this is to enforce that the grdient of the upper triangle is zero for autograd
    L = L - np.diag(np.diag(L)) + np.diag(np.exp(np.diag(L))) # DHL this is to guarantee positive elements in the diagonal
    v = c - m
    upsilon = solve(L, c - m)

    return -1.0 * float(D) / 2 * np.log(2.0  * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(upsilon**2)

def loglikelihood(params, c, e):

#    log_sigmas = params['log_sigmas']
    log_sigmas = np.ones(len(e)) * -15.0
    sigmas = np.exp(log_sigmas)
    diff = e - RTM(c)

    return np.sum(-0.5 * np.log(2.0 * np.pi) - 0.5 * log_sigmas - 0.5 * diff**2 / sigmas)

params_ini = {'mean_prior': np.ones(D) * 4, 'chol_cov_prior': np.eye(D) * 0.0 }
params_true = {'mean_prior': true_prior_mean, 'chol_cov_prior': np.linalg.cholesky(true_prior_cov)}
current_params = params_ini

e = E_true[0,:]
c = C_true[0,:]

# We generate the functions needed for HMC

def log_target_distribution(c, e):
    return logprior(current_params, c) + loglikelihood(current_params, c, e)

def grad_log_target_distribution(c, e):

    def log_target_distribution_spec(c):
        return log_target_distribution(c, e)

    return diff(log_target_distribution_spec, c)

# Finds the MAP solution

def find_C(e, current_params):
    
    def obj(c):
        return -1.0 * log_target_distribution(c, e)

    def gr_obj(c):
        return -1.0 * grad_log_target_distribution(c, e)

    mean = current_params['mean_prior']
    L = current_params['chol_cov_prior']
    L = L - np.diag(np.diag(L)) + np.diag(np.exp(np.diag(L))) 
    covariance = np.dot(L, L.T)

    ini = (np.dot(np.random.normal(size = (( 1, D))), \
        np.linalg.cholesky(covariance).T) + mean).flatten()

    result = fmin_l_bfgs_b(obj, ini, gr_obj)

    best_value = result[ 1 ]
    best_point = result[ 0 ]

    for k in range(10):

        ini = (np.dot(np.random.normal(size = (( 1, D))), \
            np.linalg.cholesky(covariance).T) + mean).flatten()

        result = fmin_l_bfgs_b(obj, ini, gr_obj)

        if result[ 1 ] < best_value:
            best_point = result[ 0 ]
            best_value = result[ 1 ]
    
    return best_point

n_epochs = 50

mt = None
vt = None
t = 0

samples = np.zeros((n_data, D))

from joblib import Parallel, delayed
import multiprocessing

for epoch in range(n_epochs):

    E_true = E_true[ np.random.permutation(E_true.shape[ 0 ]), : ]

    def sample_from_posterior_for_data_instance(i):
        e = E_true[ i , : ]
        c = find_C(e, current_params)

        # We run the chain for 20 steps to generate one independent sample

        def log_target_distribution_HMC(c):
            return log_target_distribution(c, e)

        def grad_log_target_distribution_HMC(c):
            return grad_log_target_distribution(c, e)

        hmc = HMC(c, log_target_distribution_HMC, grad_log_target_distribution_HMC)
        hmc.run_chain(20, 1e-4, 20)

        sys.stdout.write('.')
        sys.stdout.flush()

        return hmc.samples[-1,:].copy()

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(sample_from_posterior_for_data_instance)(i) for i in range(n_data))

    samples = np.asarray(results)

    def Q(params, c, e):
        return logprior(params, c) + loglikelihood(params, c, e)

    value_Q = 0.0

    for i in range(n_data):
        value_Q += Q(current_params, samples[ i, : ], E_true[ i, : ])

    current_params['mean_prior'] = np.mean(samples, 0)
    cov = np.cov(samples.T) * (n_data - 1) / (n_data)
    L = np.linalg.cholesky(cov)
    L = L - np.diag(np.diag(L)) + np.diag(np.log(np.diag(L))) 
    current_params['chol_cov_prior'] = L

    print("\nEpoch:",  epoch, "\n")
    print("\nAvg. Value Q:",  value_Q, "\n")
    print(current_params)
    print("Estimated Mean and covariance matrix")
    print(current_params['mean_prior'])
    L = current_params['chol_cov_prior']
    L = L - np.diag(np.diag(L)) + np.diag(np.exp(np.diag(L))) 
    print(np.dot(L, L.T))


print(current_params)
print(params_true)
print("Estimated Mean and covariance matrix")
print(current_params['mean_prior'])
L = current_params['chol_cov_prior']
L = L - np.diag(np.diag(L)) + np.diag(np.exp(np.diag(L))) 
print(np.dot(L, L.T))
print("True Mean and covariance matrix")
print(true_prior_mean)
print(true_prior_cov)

print("ML mean and covariance matrix")
print(np.mean(C_true,0))
print(np.cov(C_true.T))

import pdb; pdb.set_trace()


