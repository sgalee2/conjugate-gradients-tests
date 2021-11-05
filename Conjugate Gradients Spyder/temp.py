import numpy as np
import torch
import gpytorch

from gpytorch.kernels import RBFKernel
from gpytorch.utils import pivoted_cholesky
from scipy.spatial.distance import cdist
from time import time
from linear_conjugate_gradients import *
from init import *

#np.random.seed(123)

data = load_elevator()

n_train = 4000
train_x, train_y = max_min_scale(data[0:n_train, :-1]), data[0:n_train, -1:]

l_1 = [0.6 for i in range(len(train_x[0]))]
sigma_1 = 1.0

K = form_cov(train_x, l_1)
K_hat = K + sigma_1 ** 2 * np.eye(n_train)

cg_1 = p_cg(K_hat, train_y)
sol_1, its_1 = cg_1[0], cg_1[2]

l_2 = np.random.uniform(0.2,5.0,size=[len(train_x[0])])
sigma_2 = np.random.normal(2.0, 0.2)

K = form_cov(train_x, l_2)
K_hat = K + sigma_2 ** 2 * np.eye(n_train)

cg_2 = p_cg(K_hat, train_y, guess=sol_1)
sol_2, its_2 = cg_2[0], cg_2[2]

print("Iterations for run 1:", its_1)
print("Iterations for run 2:", p_cg(K_hat, train_y)[2])
print("Iterations for run 2 with GUESS = sol_1:", its_2)