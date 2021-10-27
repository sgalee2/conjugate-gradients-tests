from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt

def protein_cg(lengthscale_range, noise_range, kernel, ARD=False, n=1000):
    
    #load data
    data = load_protein(info=1)
    
    #assign training data
    print("Assigning training data...")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,4))
    
    #if ARD then we have E[lengthscales] = lengthscale_range
    if ARD:
        covar_fn = kernel(ard_num_dims=9)
        lengthscales = np.zeros((num_trials, 9))
        for i in range(num_trials):
            if i < num_trials-1:
                dist = lengthscale_range[i+1] - lengthscale_range[i]
            lengthscales[i] = abs( np.random.normal(lengthscale_range[i], dist, [1,9]).astype('float32') )
            ave = np.average(lengthscales[i])
            covar_fn.lengthscale = lengthscales[i] 
            x_tens = torch.Tensor(train_x)
            K_XX = covar_fn(x_tens, x_tens).evaluate().detach().numpy()
            for j in range(noise_length):
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                sol = np.linalg.inv(K_hat) @ train_y
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                results = [[ave, sigma, cg_run[0], cg_run[1]]]
                res = np.vstack([res, results])
        res = pandas.DataFrame( res[1:, 0:3], columns=['Lengthscales', 'Noise', 'Iterations'])
        return res

cg_ = protein_cg(np.linspace(0.0,1.0,10), [1,2,3,4,5,6,7,8], RBFKernel, ARD=1)
heat_map = cg_.pivot('Lengthscales', 'Noise', 'Iterations')
plt.figure(figsize=[10,10])
seaborn.heatmap(heat_map, cmap='Reds')