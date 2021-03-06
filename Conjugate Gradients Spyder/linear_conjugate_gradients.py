"""
Our own routine for Linear Conjugate Gradients, a method for approximate solves to the system

    lhs result = rhs

where lhs is SPD.

This method only needs access to a routine to compute lhs*v for some vector v, and exact solves
are possible by setting tol = macheps

Version 0.01:
    
    - A naive LCG routine which checks if bbmv is callable or not. If not, we use numpy matmul
    
Version 0.02:
    
    - Allocates the memory for all the work vectors (4*n worth of space) at the initialisation step
    
Version 0.03:
    
    - Now with basic tests for convergence & runtimes

Version 0.10:
    
    - Confirmation lin_cg works and is at least as fast as SciPy
    - Added preconditioned routine p_cg in the same format as lin_cg for easy comparison
    - Confirmation that p_cg works albeit suboptimally for Jacobian preconditioner. Not sure if this is a preconditioner issue or algorithm issue. #it was an algorithm issue
    
Version 0.11:
    
    - Amended issues with p_cg, should now work with any viable preconditioner matrix
    
Version 1.00:
    
    - Removed non preconditioned version
    - PCG without preconditioner uses no_precon
    - Fixed some issues with numerical error in testing residual
    
"""
import numpy as np

def no_precon(x):
    return np.copy(x)

def lin_cg(
    bbmv,
    rhs,
    tol = 1e-8,
    jit = 1e-10,
    max_its = None,
    guess = None
):
    
    """
    Implements LCG solving
    
        lhs result = rhs
        
    where bbmv (v) is a routine that computes lhs*v
    
    Args:
        - bbmv    -- black box matrix vector multiplication
        - rhs     -- right hand side of system
        - tol     -- solution tolerance, exact solves occur when tol = macheps the machine precision
        - jit     -- jitter factor to prevent divisions by zero, occur when system is poorly conditioned
        - max_its -- maximum iterations of main algorithm
        - guess   -- initial guess for solution, not required but could help convergence
        
    Returns:
        - result
        - info    -- iterations completed if solution found, 0 otherwise
    
    """
    #extract some parameters
    rhs_shape = rhs.shape
    n = rhs_shape[0]
    
    #ensure rhs is an n x 1 array
    if len(rhs_shape) == 1:
        rhs = np.vstack(rhs) 
        
    if type(bbmv) == np.ndarray:
        bbmv = bbmv.dot
    elif not callable(bbmv) :
        raise RuntimeError('bbmv is not callable nor numpy array')
        
    #work vector, first column q = Ad, second column x, third column r, fourth column d
    work_vectors = np.zeros((n, 4))
    
    if guess is not None:
        work_vectors[:, 1:2] = guess
        work_vectors[:, 2:3] = rhs - bbmv(guess)
        res_norm = np.linalg.norm(work_vectors[0:n, 2], 2)
        if res_norm < tol:
            return guess, 1
    else:
        work_vectors[:, 2:3] = rhs
        res_norm = np.linalg.norm(rhs, 2)
    
    if max_its is None:
        max_its = 4 * n + 1
    work_vectors[:, 3:4] = work_vectors[:, 2:3]
    rsold = np.dot(np.transpose(work_vectors[:, 2:3]), work_vectors[:, 2:3])
    rsnew = rsold
    i = 0
    while i < max_its and rsnew > tol:
        i += 1
        work_vectors[:, 0:1] = bbmv(work_vectors[:, 3:4]) #q = A @ d
        alpha = rsold / np.dot(np.transpose(work_vectors[:, 3:4]), work_vectors[:, 0:1]) #alpha = residual / d^T @ q
        work_vectors[:, 1:2] += alpha*work_vectors[:, 3:4] #x = x + alpha*d
        if i%50 == 0:
            work_vectors[:, 2:3] = rhs - bbmv(work_vectors[:, 1:2])
            #work_vectors[:, 2:3] += - alpha*work_vectors[:, 0:1] #r = r - alpha*q
        else:
            work_vectors[:, 2:3] += - alpha*work_vectors[:, 0:1] #r = r - alpha*q
        #here we would do s = M^{-1} @ r for preconditioned C.G.
        rsnew = np.dot(np.transpose(work_vectors[:, 2:3]), work_vectors[:, 2:3]) #residual_new = norm(r)
        if np.sqrt(rsnew) < tol:
            break #convergence?
        work_vectors[:, 3:4] = work_vectors[:, 2:3] + (rsnew/rsold)*work_vectors[:, 3:4] #d = r + beta*d
        rsold = rsnew
    
    if abs(rsnew) <= tol:
        return work_vectors[:, 1:2], 1, i
    else:
        return work_vectors[:, 1:2], 0, i
    
def p_cg(mvm,
         rhs,
         tol = 10e-8,
         jit = 10e-10,
         max_its = None,
         guess = None,
         pmvm = None):
    """
    Implements Preconditioned LCG solving
    
        lhs result = rhs
        
    where mvm (v) is a routine that computes lhs*v
    
    Args:
        - mvm     -- black box matrix vector multiplication
        - rhs     -- right hand side of system
        - tol     -- solution tolerance, exact solves occur when tol = macheps the machine precision
        - jit     -- jitter factor to prevent divisions by zero, occur when system is poorly conditioned
        - max_its -- maximum iterations of main algorithm
        - guess   -- initial guess for solution, not required but could help convergence
        - pmvm    -- routine that computes M^{-1}v
        
    Returns:
        - result
        - info    -- iterations completed if solution found, 0 otherwise
    
    """
    #extract some parameters
    rhs_shape = rhs.shape
    n = rhs_shape[0]
    
    #ensure rhs is an n x 1 array
    if len(rhs_shape) == 1:
        rhs = np.vstack(rhs) 
        
    if type(mvm) == np.ndarray:
        mvm = mvm.dot
    elif not callable(mvm):
        raise TypeError('bbmv is not callable nor numpy array')
        
    if pmvm is None:
        pmvm = np.copy
    elif type(pmvm) == np.ndarray:
        pmvm = pmvm.dot
    elif not callable(pmvm):
        raise TypeError('pmvm is not of type numpy array or callable.')
        
    #work vector, first column q = Ad, second column x, third column r, fourth column d, fifth column z_new
    work_vectors = np.zeros((n, 5))
    
    if guess is not None:
        
        work_vectors[:,1:2] = guess
        work_vectors[:,2:3] = rhs - mvm(guess)
        residual = np.linalg.norm(work_vectors[:,2:3])
        
        work_vectors[:,4:5] = pmvm(work_vectors[:,2:3])
        work_vectors[:,3:4] = work_vectors[:,4:5]
        precon_res = work_vectors[:,2:3].T @ work_vectors[:,4:5]
        
    else:
        
        work_vectors[:,2:3] = rhs
        residual = np.linalg.norm(work_vectors[:,2:3])
        work_vectors[:,4:5] = pmvm(work_vectors[:,2:3])
        work_vectors[:,3:4] = work_vectors[:,4:5]
        precon_res = work_vectors[:,2:3].T @ work_vectors[:,4:5]
    
    if max_its is None:
        max_its = 10 * n + 1
    i = 0
    while i < max_its and abs(residual) > tol:
        i += 1
        work_vectors[:,0:1] = mvm(work_vectors[:,3:4])
        alpha = precon_res / np.dot(np.transpose(work_vectors[:,3:4]), work_vectors[:,0:1])
        work_vectors[:,1:2] += alpha*work_vectors[:,3:4] #x = x + alpha*d
        if i%50 == 0:
            work_vectors[:, 2:3] = rhs - mvm(work_vectors[:, 1:2]) #force an extra MVP to combat round off errors
        else:
            work_vectors[:, 2:3] += - alpha*work_vectors[:, 0:1] #r = r - alpha*q
        residual = np.linalg.norm(work_vectors[:,2:3])
        work_vectors[:,4:5] = pmvm(work_vectors[:,2:3])
        new_res = np.dot(np.transpose(work_vectors[:,2:3]), work_vectors[:,4:5])
        beta = new_res  / precon_res
        work_vectors[:,3:4] = work_vectors[:,4:5] + beta*work_vectors[:,3:4]
        precon_res = new_res
        
    if abs(residual) <= tol:
        return work_vectors[:,1:2], 1, i
    else:
        return work_vectors[:,1:2], 0, i
