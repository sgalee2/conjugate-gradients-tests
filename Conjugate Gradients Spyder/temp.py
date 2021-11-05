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

n_train = 1000
train_x, train_y = max_min_scale(data[0:n_train, :-1]), data[0:n_train, -1:]

l = [0.6 for i in range(len(train_x[0]))]
sigma = 1.0

K = form_cov(train_x, l)
K_hat = K + sigma ** 2 * np.eye(n_train)
vals, vecs = np.linalg.eig(K)

n_comps = [i+1 for i in range(int(np.sqrt(n_train) + 1))]
fro_norm = []

cg_ = p_cg(K_hat, train_y)
its = cg_[2]
pcg_its = []

for i in n_comps:
    D, Sig = np.diag( vals[0:i] ), vecs[:, 0:i]
    Approx = Sig @ D @ Sig.T + sigma ** 2 * np.eye(n_train)
    fro_norm.append( np.linalg.norm(K_hat - Approx, 'fro') )
    pcg_ = p_cg(K_hat, train_y, pmvm=np.linalg.inv(Approx))
    pcg_its.append(pcg_[2])    
    
plt.figure(figsize=[20,10])
plt.scatter(n_comps, pcg_its, color='blue', marker='x')
plt.plot(n_comps, fro_norm, color='red')
plt.plot(n_comps, [its for i in range(len(n_comps))],'--', color='black')
plt.xlabel("Matrix rank")
plt.ylabel("Frobenius norm of $K_{hat} - Approx$")
plt.title("Comparing approximation rank against accuracy.")


