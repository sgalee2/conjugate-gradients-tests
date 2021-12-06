import numpy as np
import scipy

from scipy.spatial.distance import cdist


class stationary():
    """
    Base class for stationary kernel functions
    
    A stationary kernel k(r) is a function based on r where
    
    \[ r(x, x') = \sqrt{ \\sum_{d=1}^D \\frac{(x_d - x'_d)^2}{l_d^2} }.  \]
    
    where $ L = \{ l_d \}_{d=1}^D $ is a vector of lengthscales.
    
    This class only needs you to define an evaluation method & the kernel derivatives
    """
    def __init__(self, ARD_dims=None, GPU_support=False):
        
        self.kernel_type = "stationary"
        if ARD_dims is None:
            self.ARD_dims = 1
        else:
            self.ARD_dims = ARD_dims
        self.GPU_support = GPU_support
        
        self.lengthscales = [0.619 for i in range(self.ARD_dims)]
            
        self.sig = [1.0]
        self.theta = np.hstack([self.sig, self.lengthscales])
        
    def set_lengthscales(self, lengthscales):
        if type(lengthscales) == float:
            lengthscales = [lengthscales]
        if len(lengthscales) == self.ARD_dims:
            self.lengthscales = lengthscales
        elif len(lengthscales) == 1:
            self.lengthscales = [lengthscales for i in range(self.ARD_dims)]
        else:
            raise TypeError("Number of inputs does not match ARD dimensions")
        
    def sq_dist(self, X1, X2):
        """
        Parameters
        ----------
        X1
        X2
        
        Description
        -----------
        r = r(X1, X2) = \sqrt{ \\sum_{d=1}^D \\frac{(X1_d - X2_d)^2}{lengthscales[d]^2} }
        """
        r = cdist(X1/self.lengthscales, X2/self.lengthscales, metric='sqeuclidean')
        return r
    
    def evaluate(self, X1, X2):
        """
        Parameters
        ----------
        X1
        X2

        Description
        -----------
        K = k(r) where r = r(X1, X2)

        """
        raise NotImplementedError
    
    def predict(self, x_s):
        """
        Parameters
        ----------
        x_s
        
        Description
        -----------
        Compute
        
        \[ \mu_s = \mu_ss - K_sX K_XX^-1 y, \Sigma_s = \Sigma_ss - K_sX K_XX^-1 K_Xs \]
        
        the predictive mean and variance at point x_s.
        
        """
        raise NotImplementedError

