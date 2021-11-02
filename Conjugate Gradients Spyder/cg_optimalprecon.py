"""
Routine for testing the Preconditioned Conjugate Gradients algorithm on various UCI data-sets using a near optimal preconditioner.

Our preconditioner M \approx K^-1 is formed using K's SVD. We choose the rank $m$ of the approximation at the start of the algorithm s.t. $m << n$. For this example, this does nothing for the scalability of PCG since we must compute the full SVD of K to form the preconditioner. However, we would expect that M is the best case scenario for an $m$ rank approximation.

"""

from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt

def pcg_(data_call, lengthscale_range, noise_range, kernel, ARD=False, n=1000, p_strat=None, rank=None):
    """
    

    Parameters
    ----------
    data_call : a function that loads the data.
    lengthscale_range : test values for lengthscales.
    noise_range : test values for noise.
    kernel : a function which defines the covariance/kernel.
    ARD : Automatic Relevance Determination. The default is False.
    n : Size of system. The default is 1000.
    p_strat : preconditioning strategy. The default is None.

    Raises
    ------
    ValueError
        ARD only available for multivariate data.

    Returns
    -------
    res : number of iterations for completed solution for each pair of lengthscale & noise.

    """
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
        covar_fn = kernel(ard_num_dims=dims)
        
        #store lengthscales in array
        lengthscales = np.zeros((num_trials, dims))
        print("Beginning tests...\n")
        for i in range(num_trials):
            if i < num_trials-1:
                dist = lengthscale_range[i+1] - lengthscale_range[i]
            
            #compute lengthscales for current batch
            lengthscales[i] = abs( np.random.normal(lengthscale_range[i], dist, [1,dims]) )
            ave = np.average(lengthscales[i])
            
            #initialise lengthscales in cov
            covar_fn.lengthscale = lengthscales[i] 
            x_tens = torch.Tensor(train_x)
            
            #compute cov
            K_XX = covar_fn(x_tens, x_tens).evaluate().detach().numpy()
            M = p_strat(K_XX, rank)
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                M_inv = np.linalg.inv( M  + sigma ** 2 * np.eye(n) )
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(p_cg, K_hat, train_y, sol=sol, pmvm=M_inv)
                #store results
                results = [[np.log10(ave), np.log10(sigma), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        print("Tests complete...\n")
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        return res
    else:
        print("Initialising kernel...\n")
        covar_fn = kernel()
        print("Beginning tests...\n")
        for i in range(num_trials):
            
            #initialise lengthscales in cov
            covar_fn.lengthscale = lengthscale_range[i]
            x_tens = torch.Tensor(train_x)
            
            #compute cov
            K_XX = covar_fn(x_tens, x_tens).evaluate().detach().numpy()
            M = p_strat(K_XX, rank)
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                M_inv = np.linalg.inv( M + sigma ** 2 * np.eye(n) )
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(p_cg, K_hat, train_y, sol=sol, pmvm=M_inv)
                #store results
                results = [[np.log10(lengthscale_range[i]), np.log10(sigma), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res
    
ls_ = [10 ** i for i in range(-4,4)]
sig_ = [10 ** i for i in range(-2, 4)]
n = 100
vmax = int(np.log10(n)) + 1

cg_1 = pcg_(load_protein, ls_, sig_, RBFKernel, ARD = False, n = n, p_strat = optimal_precon, rank = 10).pivot('Lengthscales', 'Noise', 'Iterations')
cg_2 = pcg_(load_protein, ls_, sig_, RBFKernel, ARD = True, n = n, p_strat = optimal_precon, rank = 10).pivot('Lengthscales', 'Noise', 'Iterations')

heatplot(cg_1, cg_2, "egg", ls_, sig_, vmax=vmax)