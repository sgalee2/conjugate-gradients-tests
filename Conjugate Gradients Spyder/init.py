import numpy as np
import matplotlib.pyplot as plt

import gpytorch
import torch
import seaborn
import pandas

from scipy.io import loadmat
from scipy.spatial.distance import cdist
from gpytorch.utils import linear_cg, pivoted_cholesky
from sklearn.preprocessing import MinMaxScaler
from linear_conjugate_gradients import *

import os

#RBFKernel
def form_cov(X1, lengthscales, X2 = None):
    if X2 is None:
        X2 = X1
    sq_dist = cdist(X1/lengthscales, X2/lengthscales, metric='sqeuclidean')
    K = np.exp(-0.5*sq_dist)
    return K

#load data sets
def load_protein(tensor=False, info=False):
    data = loadmat('..\MATLAB Files\protein.mat')['data']
    if info:
        print("--------------------")
        print("Protein data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("--------------------")
    np.random.shuffle(data)
    if tensor:
        data = torch.Tensor(data)
    return data
    
def load_elevator(tensor=False, info=False):
    data = loadmat('..\MATLAB Files\elevators.mat')['data']
    if info:
        print("--------------------")
        print("Elevator data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("--------------------")
    np.random.shuffle(data)
    if tensor:
        data = torch.Tensor(data)
    return data

def load_bike(tensor=False, info=False):
    data = loadmat('../MATLAB Files/bike.mat')['data']
    if info:
        print("--------------------")
        print("Bike data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("--------------------")
    np.random.shuffle(data)
    if tensor:
        data = torch.Tensor(data)
    return data

def load_road(tensor=False, info=False):
    data = loadmat(r'..\MATLAB Files\3droad.mat')['data']
    if info:
        print("--------------------")
        print("3D Road data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("--------------------")
    np.random.shuffle(data)
    if tensor:
        data = torch.Tensor(data)
    return data

def load_audio(tensor=False, info=False):
    data = loadmat(r'..\MATLAB Files\audio_data.mat')
    data = np.hstack([data['xfull'], data['yfull']])
    if info:
        print("--------------------")
        print("Audio data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("--------------------")
    np.random.shuffle(data)
    if tensor:
        data = torch.Tensor(data)
    return data

def load_toy(tensor=False, info=True, lengthscale=np.random.uniform(0.1, 10.), noise=np.random.normal(2.0,1.0), n=1000):
    x = np.vstack( np.linspace(0.,100.,n) )
    l = lengthscale
    sq_dist = cdist(x/l, x/l)
    K = np.exp(-0.5*sq_dist) + noise*np.eye(n)
    f = np.random.multivariate_normal([0 for i in range(n)], K)
    data = np.hstack([x, np.vstack(f)])
    if tensor:
        data = torch.Tensor(data)
    if info:
        print("--------------------")
        print("Toy data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("True lengthscale: ",l)
        print("True noise: ",noise)
        print("--------------------")
    return data

def load_2dtoy(lengthscale = None, noise = None, ARD = False, tensor=False, info=True, n=1000):
    x = np.random.uniform(0.,100.,size=[n,2])
    if ARD:
        if lengthscale is None:
            l = np.random.uniform(0.1, 10.0, size=2)
            if noise is None:
                noise = np.random.normal(2.0,1.0)
        sq_dist = cdist(x/l, x/l)
        K = np.exp(-0.5*sq_dist) + noise*np.eye(n)
        f = np.random.multivariate_normal([0 for i in range(n)], K)
        data = np.hstack([x, np.vstack(f)])
    else:
        if lengthscale is None:
            l = np.random.uniform(0.1, 10.0)
            if noise is None:
                noise = np.random.normal(2.0,1.0)
        sq_dist = cdist(x/l, x/l)
        K = np.exp(-0.5*sq_dist) + noise*np.eye(n)
        f = np.random.multivariate_normal([0 for i in range(n)], K)
        data = np.hstack([x, np.vstack(f)])
    if tensor:
        data = torch.Tensor(data)
    if info:
        print("--------------------")
        print("2D Toy data-set info...\n")
        [N, d] = data.shape
        print("Num inputs: ",N)
        print("Num features: ",d-1)
        print("True lengthscale: ",l)
        print("True noise: ",noise)
        print("--------------------")
    return data

def cg_(data_call, lengthscale_range, noise_range, kernel, ARD=False, n=1000, p_strat=None, rank=None):
    #load data
    data = data_call(info=1)
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    dims = len(train_x[0])
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    if ARD:
        if dims == 1:
            raise ValueError("ARD unavailable for univariate data")
        #initiate covariance function
        print("Initialising ARD kernel...\n")
        
        #store lengthscales in array
        lengthscales = np.zeros((num_trials, dims))
        print("Beginning tests...\n")
        for i in range(num_trials):
            if i < num_trials-1:
                dist = lengthscale_range[i+1] - lengthscale_range[i]
            
            #compute lengthscales for current batch
            lengthscales[i] = abs( np.random.normal(lengthscale_range[i], dist, [1,dims]) )
            ave = np.average(lengthscales[i])
            
            #compute cov
            K_XX = kernel(train_x, lengthscales[i])
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                if p_strat is not None:
                    pmvm = np.linalg.inv(p_strat(K_XX, rank) + sigma ** 2 * np.eye(n))
                else:
                    pmvm = None
                #run cg
                cg_run = cg_test(p_cg, K_hat, train_y, sol=sol, pmvm=pmvm)
                #store results
                results = [[np.log10(ave), np.log10(sigma), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        print("Tests complete...\n")
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        return res
    else:
        print("Initialising kernel...\n")
        
        print("Beginning tests...\n")
        for i in range(num_trials):
            
            #compute cov
            K_XX = kernel(train_x, lengthscale_range[i])
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(p_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[np.log10(lengthscale_range[i]), np.log10(sigma), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res

def heatplot(data0, data1, title, ls_, sig_, vmax):
    f, ax = plt.subplots(1, 3, figsize=[18,9], gridspec_kw={'width_ratios':[1,1,0.08]})
    f.suptitle(title)
    ax[0].set_title("No ARD")
    ax[1].set_title("ARD")
    h1 = seaborn.heatmap(np.log10(data0), ax = ax[0], cbar=False, cmap = 'Reds', vmin=0, vmax=vmax, )
    h2 = seaborn.heatmap(np.log10(data1), ax = ax[1], cmap = 'Reds', vmin=0, vmax=vmax, cbar_ax=ax[2])
    ax[1].set_yticks([])
    ax[1].set_ylabel('')

#scaler for input features
def max_min_scale(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

#extract information from a cg run
def cg_test(routine, A, b, sol, pmvm=None):
    if pmvm is not None:
        cg_run = routine(A, b, pmvm=pmvm)
    else:
        cg_run = routine(A, b)
    x_m, info, m = cg_run
    error = np.linalg.norm(x_m - sol)
    return m, error
    
def optimal_precon(C, rank):
    
    if rank is None:
        #if no rank is given we assume rank = sqrt(N)
        N = C.shape[0]
        rank = int(np.sqrt(N))
    
    #perforns low rank approx of C
    vals, vecs = np.linalg.eig(C)
    basis = vecs[:,0:rank]
    diag = np.diag(vals[0:rank])
    
    return basis @ diag @ basis.T
