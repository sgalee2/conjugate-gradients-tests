import numpy as np
import torch
import gpytorch
import matplotlib

from matplotlib import pyplot as plt

from gpytorch.utils import linear_cg_k
from gpytorch.kernels import RBFKernel
from gpytorch.utils import pivoted_cholesky
from scipy.spatial.distance import cdist
from time import time
from linear_conjugate_gradients import *
from init import *

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

#np.random.seed(123)

#set up data
n_min = 10
n_max = 10000
n_range = np.linspace(n_min,n_max, 101)
x = np.vstack(np.linspace(0.0,100.0,n_max))
y_true = 0.1 * x * np.exp(-np.sin(x))
y_noise = y_true + np.random.normal(loc=0.0, scale=2.0, size=[n_max,1])

#shuffle data
data = np.hstack([x, y_noise])
np.random.shuffle(data)
x, y = data[:, :-1], data[:, -1:]

#form full covariance matrix
K_full = form_cov(x, 1.0) + np.eye(n_max)
t_pcg = []

for i in n_range:
    i_ = int(i)
    print("Setting n=",i_)
    t_1 = time()
    lin_cg(K_full[0:i_,0:i_], y[0:i_])
    t_2 = time()
    t = t_2 - t_1
    t_pcg.append(t)
    print("Completed in:",t,"seconds.\n")

t_ = np.array(t_pcg)
np.save('pcg_time', t_)