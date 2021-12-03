# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:37:19 2021

@author: sgalee2
"""

from Preconditioner import *
import os
os.chdir('../..')
from init import *

class SVD(Preconditioner):
    """
    The Singular Valued Decomposition as a preconditioning strategy, computes the matrix
    
   \[ M = Sigma @ Diag @ Sigma.T \]
    
    of singular values Diag \& their corresponding eigenvectors.
    
    This should be the best element-by-element t-rank approximation for the given matrix, minimising
    the Frobenius norm of the difference between the two.
    """
    def __init__(self, K, rank):
        super(SVD, self).__init__("SVD")
        self.K = K
        self.rank = rank
        self.n = K.shape[0]
        
    def compute_svd(self):
        U, D, V = np.linalg.svd(self.K)
        return U, D, V
    
    def invert_svd(self, sigma):
        U, D, V = compute_svd()
        noise = sigma ** -2
        inner_inverse = np.linalg.inv( np.diag( D ** (-1) ) + noise * V @ U)
        inverse = noise * np.eye(self.n) - noise ** 2 * U @ inner_inverse @ V
        return inverse
    
