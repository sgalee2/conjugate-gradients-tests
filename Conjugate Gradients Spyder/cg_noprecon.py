"""
Routine for testing the non-preconditioned Conjugate Gradients algorithm on various UCI data-sets.

"""

from gpytorch.kernels import RBFKernel
from init import *
from linear_conjugate_gradients import *
import sklearn.datasets as dt

def protein_cg(lengthscale_range, noise_range, kernel, ARD=False, n=1000):
    
    #load data
    data = load_protein(info=1)
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    dims = len(train_x[0])
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    #if ARD then we have E[lengthscales] = lengthscale_range
    if ARD:
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
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(ave),2), round(np.log(sigma),2), cg_run[0]]]
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
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res
    
def elevator_cg(lengthscale_range, noise_range, kernel, ARD=False, n=1000):
    
    #load data
    data = load_elevator(info=1)
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    dims = len(train_x[0])
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    #if ARD then we have E[lengthscales] = lengthscale_range
    if ARD:
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
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(ave),2), round(np.log(sigma),2), cg_run[0]]]
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
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res

def bike_cg(lengthscale_range, noise_range, kernel, ARD=False, n=1000):
    
    #load data
    data = load_bike(info=1)
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    dims = len(train_x[0])
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    #if ARD then we have E[lengthscales] = lengthscale_range
    if ARD:
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
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(ave),2), round(np.log(sigma),2), cg_run[0]]]
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
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res

def road_cg(lengthscale_range, noise_range, kernel, ARD=False, n=1000):
    
    #load data
    data = load_road(info=1)
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    dims = len(train_x[0])
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    #if ARD then we have E[lengthscales] = lengthscale_range
    if ARD:
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
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(ave),2), round(np.log(sigma),2), cg_run[0]]]
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
            
            for j in range(noise_length):
                #assign noise & compute likelihood cov
                sigma = noise_range[j]
                K_hat = K_XX + sigma ** 2 * np.eye(n)
                #compute exact solution for error comparison
                sol = np.linalg.inv(K_hat) @ train_y
                #run cg
                cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
                #store results
                results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
                res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res
    
def audio_cg(lengthscale_range, noise_range, kernel,  n=1000):
    
    #load data
    data = load_audio(info=1)
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    print("Initialising kernel...\n")
    covar_fn = kernel()
    print("Beginning tests...\n")
    for i in range(num_trials):
        
        #initialise lengthscales in cov
        covar_fn.lengthscale = lengthscale_range[i]
        x_tens = torch.Tensor(train_x)
        
        #compute cov
        K_XX = covar_fn(x_tens, x_tens).evaluate().detach().numpy()
        
        for j in range(noise_length):
            #assign noise & compute likelihood cov
            sigma = noise_range[j]
            K_hat = K_XX + sigma ** 2 * np.eye(n)
            #compute exact solution for error comparison
            sol = np.linalg.inv(K_hat) @ train_y
            #run cg
            cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
            #store results
            results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
            res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res
    
def toy_cg(lengthscale_range, noise_range, kernel,  n=1000):
    
    #load data
    data = load_toy()
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    print("Initialising kernel...\n")
    covar_fn = kernel()
    print("Beginning tests...\n")
    for i in range(num_trials):
        
        #initialise lengthscales in cov
        covar_fn.lengthscale = lengthscale_range[i]
        x_tens = torch.Tensor(train_x)
        
        #compute cov
        K_XX = covar_fn(x_tens, x_tens).evaluate().detach().numpy()
        
        for j in range(noise_length):
            #assign noise & compute likelihood cov
            sigma = noise_range[j]
            K_hat = K_XX + sigma ** 2 * np.eye(n)
            #compute exact solution for error comparison
            sol = np.linalg.inv(K_hat) @ train_y
            #run cg
            cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
            #store results
            results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
            res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res
    
def tdtoy_cg(lengthscale_range, noise_range, kernel,  n=1000):
    
    #load data
    data = load_2dtoy()
    
    #assign training data
    print("Assigning training data...\n")
    train_x, train_y = data[0:n, :-1],  data[0:n, -1:]
    train_x = max_min_scale(train_x) #input points lie on [0,1]^dims
    
    #set up number of runs
    num_trials = len(lengthscale_range)
    noise_length = len(noise_range)
    res = np.zeros((1,3))
    
    print("Initialising kernel...\n")
    covar_fn = kernel()
    print("Beginning tests...\n")
    for i in range(num_trials):
        
        #initialise lengthscales in cov
        covar_fn.lengthscale = lengthscale_range[i]
        x_tens = torch.Tensor(train_x)
        
        #compute cov
        K_XX = covar_fn(x_tens, x_tens).evaluate().detach().numpy()
        
        for j in range(noise_length):
            #assign noise & compute likelihood cov
            sigma = noise_range[j]
            K_hat = K_XX + sigma ** 2 * np.eye(n)
            #compute exact solution for error comparison
            sol = np.linalg.inv(K_hat) @ train_y
            #run cg
            cg_run = cg_test(lin_cg, K_hat, train_y, sol=sol)
            #store results
            results = [[round(np.log(lengthscale_range[i]),2), round(np.log(sigma),2), cg_run[0]]]
            res = np.vstack([res, results])
        #return results as a dataframe
        res = pandas.DataFrame( res[1:], columns=['Lengthscales', 'Noise', 'Iterations'])
        print("Tests complete...\n")
        return res