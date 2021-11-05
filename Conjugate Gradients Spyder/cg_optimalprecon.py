"""
Routine for testing the Preconditioned Conjugate Gradients algorithm on various UCI data-sets using a near optimal preconditioner.

Our preconditioner M \approx K^-1 is formed using K's SVD. We choose the rank $m$ of the approximation at the start of the algorithm s.t. $m << n$. For this example, this does nothing for the scalability of PCG since we must compute the full SVD of K to form the preconditioner. However, we would expect that M is the best case scenario for an $m$ rank approximation.

"""

from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt


ls_ = [10 ** i for i in range(-4,4)]
sig_ = [10 ** i for i in range(-2, 4)]
n = 10
rank = int(np.sqrt(n))
vmax = int(np.log10(n)) + 1

pcg_p1 = cg_(load_protein, ls_, sig_, form_cov, ARD = False, n=n, p_strat = optimal_precon, rank = rank)
cg_p1  = cg_(load_protein, ls_, sig_, form_cov, ARD = False, n=n)
pcg_p2 = cg_(load_protein, ls_, sig_, form_cov, ARD = True, n=n, p_strat = optimal_precon, rank = rank)
cg_p2  = cg_(load_protein, ls_, sig_, form_cov, ARD = True, n=n)

print(cg_p1.values[:,2] - pcg_p1.values[:,2])


pcg_e1 = cg_(load_elevator, ls_, sig_, form_cov, ARD = False, n=n, p_strat = optimal_precon, rank = rank)
cg_e1  = cg_(load_elevator, ls_, sig_, form_cov, ARD = False, n=n)
pcg_e2 = cg_(load_elevator, ls_, sig_, form_cov, ARD = True, n=n, p_strat = optimal_precon, rank = rank)
cg_e2  = cg_(load_elevator, ls_, sig_, form_cov, ARD = True, n=n)


pcg_b1 = cg_(load_bike, ls_, sig_, form_cov, ARD = False, n=n, p_strat = optimal_precon, rank = rank)
cg_b1  = cg_(load_bike, ls_, sig_, form_cov, ARD = False, n=n)
pcg_b2 = cg_(load_bike, ls_, sig_, form_cov, ARD = True, n=n, p_strat = optimal_precon, rank = rank)
cg_b2  = cg_(load_bike, ls_, sig_, form_cov, ARD = True, n=n)


pcg_r1 = cg_(load_road, ls_, sig_, form_cov, ARD = False, n=n, p_strat = optimal_precon, rank = rank)
cg_r1  = cg_(load_road, ls_, sig_, form_cov, ARD = False, n=n)
pcg_r2 = cg_(load_road, ls_, sig_, form_cov, ARD = True, n=n, p_strat = optimal_precon, rank = rank)
cg_r2  = cg_(load_road, ls_, sig_, form_cov, ARD = True, n=n)


pcg_t1 = cg_(load_2dtoy, ls_, sig_, form_cov, ARD = False, n=n, p_strat = optimal_precon, rank = rank)
cg_t1  = cg_(load_2dtoy, ls_, sig_, form_cov, ARD = False, n=n)
pcg_t2 = cg_(load_2dtoy, ls_, sig_, form_cov, ARD = True, n=n, p_strat = optimal_precon, rank = rank)
cg_t2  = cg_(load_2dtoy, ls_, sig_, form_cov, ARD = True, n=n)
