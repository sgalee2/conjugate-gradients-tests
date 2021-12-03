"""
Routine for testing the Preconditioned Conjugate Gradients algorithm on various UCI data-sets using a near optimal preconditioner.

Our preconditioner M \approx K^-1 is formed using K's SVD. We choose the rank $m$ of the approximation at the start of the algorithm s.t. $m << n$. For this example, this does nothing for the scalability of PCG since we must compute the full SVD of K to form the preconditioner. However, we would expect that M is the best case scenario for an $m$ rank approximation.

"""

from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt
import pandas

ls_ = [10 ** i for i in range(-4,4)]
sig_ = [10 ** i for i in range(-2, 4)]
n = 1000
vmax = int(np.log10(n)) + 1

cg_1 =  cg_(load_protein, ls_, sig_, form_cov, ARD = 1, n=n)
pcg_1 = cg_(load_protein, ls_, sig_, form_cov, ARD = 1, n=n, p_strat = optimal_precon, rank = int(np.sqrt(n)))

vals = cg_1.values
cg_1['Iterations'] -= pcg_1['Iterations']

print(cg_1.pivot('Lengthscales', 'Noise', 'Iterations'))