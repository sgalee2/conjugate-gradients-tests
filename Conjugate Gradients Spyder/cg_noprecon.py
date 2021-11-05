"""
Routine for testing the non-preconditioned Conjugate Gradients algorithm on various UCI data-sets.

"""

from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt


    
    
ls_ = [10 ** i for i in range(-4,4)]
sig_ = [10 ** i for i in range(-2, 4)]
n = 1000
vmax = int(np.log10(n)) + 1

cg_1 = cg_(load_protein, ls_, sig_, form_cov, ARD = False, n=n).pivot('Lengthscales', 'Noise', 'Iterations')
cg_2 = cg_(load_protein, ls_, sig_, form_cov, ARD = True, n=n).pivot('Lengthscales', 'Noise', 'Iterations')

heatplot(cg_1, cg_2, "Protein data-set CG solves", ls_, sig_, vmax)

cg_1 = cg_(load_elevator, ls_, sig_, form_cov, ARD = False, n=n).pivot('Lengthscales', 'Noise', 'Iterations')
cg_2 = cg_(load_elevator, ls_, sig_, form_cov, ARD = True, n=n).pivot('Lengthscales', 'Noise', 'Iterations')

heatplot(cg_1, cg_2, "Elevator data-set CG solves", ls_, sig_, vmax)

cg_1 = cg_(load_bike, ls_, sig_, form_cov, ARD = False, n=n).pivot('Lengthscales', 'Noise', 'Iterations')
cg_2 = cg_(load_bike, ls_, sig_, form_cov, ARD = True, n=n).pivot('Lengthscales', 'Noise', 'Iterations')

heatplot(cg_1, cg_2, "Bike data-set CG solves", ls_, sig_, vmax)

cg_1 = cg_(load_road, ls_, sig_, form_cov, ARD = False, n=n).pivot('Lengthscales', 'Noise', 'Iterations')
cg_2 = cg_(load_road, ls_, sig_, form_cov, ARD = True, n=n).pivot('Lengthscales', 'Noise', 'Iterations')

heatplot(cg_1, cg_2, "Road data-set CG solves", ls_, sig_, vmax)

cg_1 = cg_(load_2dtoy, ls_, sig_, form_cov, ARD = False, n=n).pivot('Lengthscales', 'Noise', 'Iterations')
cg_2 = cg_(load_2dtoy, ls_, sig_, form_cov, ARD = True, n=n).pivot('Lengthscales', 'Noise', 'Iterations')

heatplot(cg_1, cg_2, "2-D Toy data-set CG solves", ls_, sig_, vmax)
