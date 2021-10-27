import numpy as np
import matplotlib.pyplot as plt

import gpytorch
import torch

from scipy.io import loadmat
from scipy.spatial.distance import cdist
from gpytorch.utils import linear_cg_k, pivoted_cholesky
from sklearn.preprocessing import MinMaxScaler

import os

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
    data = loadmat('..\MATLAB Files\bike.mat')['data']
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
    data = loadmat('..\MATLAB Files\3droad.mat')['data']
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
    data = loadmat('..\MATLAB Files\audio_data.mat')
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

def max_min_scale(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)
