# coding: utf-8
'''
Utilities
'''
import torch
import argparse
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    return opt

def psd_proj_symm(S):
    '''
    Projects a matrix S into the PSD cone.
    Specifically: performs SVD, makes negative eigenvalues 0, and multiplies them back together to get projected matrix.

    Args:
        S (pytorch tensor matrix): symmetric matrix

    Returns:
        Matrix A which is the closest PSD matrix to S (in Frobenius norm)
    '''
    d = S.shape[0]
    U, D, V_t = torch.svd(S)
    D = torch.clamp(D, min=0, max=None).diag()
    A = torch.mm(torch.mm(U, D), U.t()) 
    return A

def inverse_covariance_weighting(means, covariances):
    '''
    Takes a set of estimates of a single quantity and weights them by the inverse of their (co)variances 
    to get the minimum (co)variance combined estimate.

    General idea explained here (https://en.wikipedia.org/wiki/Inverse-variance_weighting), 
    statement for the multivariate case found in Theorem 4.1 here (https://arxiv.org/pdf/2110.14465.pdf) 

    Args:
        means (vector of length k): d-dimensional parameter estimates 
        covariances (vector of length k): d x d covariance estimates
    
    Returns:
        optimal estimate (est) and associated covariance of the new estimate (inverse_sum)
    '''

    '''first term'''
    # get inverse of each covariance matrix
    inverse_covs = [np.linalg.inv(cov_i) for cov_i in covariances]

    # add inverses together 
    sum_inverse_covs = np.add.reduce(inverse_covs)

    # get inverse of sum
    inverse_sum = np.linalg.inv(sum_inverse_covs)

    '''second term'''
    # get element-wise product of inverse covariances and means and sum them
    cov_mean_prod = [np.dot(inverse_cov, mean) for inverse_cov,mean in zip(inverse_covs, means)]
    cov_mean_prod_sum = np.add.reduce(cov_mean_prod)

    '''return est and covariance'''
    est = np.dot(inverse_sum, cov_mean_prod_sum)
    return(est, inverse_sum)

def gaussian_tailbound(d, b):
    '''
    Calculate 1 - b probability upper bound on the L2 norm of a draw 
    from a d-dimensional standard multivariate Gaussian

    Args:
        d (positive int): dimension of the multivariate Gaussian in question
        b (float in (0,1)): probability that draw from the distribution has a 
                            larger L2 norm than our bound
    
    Returns:
        1-b probability upper bound on the L2 norm
    '''
    return ( d + 2*( d * math.log(1/b) )**0.5 + 2*math.log(1/b) )**0.5
