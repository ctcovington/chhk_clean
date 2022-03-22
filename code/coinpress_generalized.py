import torch
import utils
import numpy as np
import math
import scipy as sc

''' 
Functions for mean estimation
'''

def multivariate_mean_iterative(X, c, r, cov, t, betas, Ps, args):
    ''' 
    Privately estimate d-dimensional mean of X, iteratively improving the estimate over t steps

    Args:
        X (pytorch tensor): data for which we want the mean
        c (float): center of ball (with radius r) which contains the mean
        r (float): radius of ball (with center c) which contains the mean
        cov (matrix): Loewner upper bound on the covariance of X
        t (int): number of iterations to run of CoinPress algorithm
        betas (list of length t): failure probability (i.e. probability of clipping at least one point) at each iteration
        Ps (list of length t): privacy budget at each iteration
        args (set of arguments): arguments to be passed to multivariate_mean_step -- in this version, it's just the 
                                 dimensions of X (args.n and args.d)

    Returns:
        - final release of CoinPress (releasable but not used, this was the algorithm output in the original CoinPress paper but we now release 
                                      all t iterations and combine them via inverse covariance weighting)
        - variance of privacy mechanism at each iteration (releasable, is used)
        - total number of points clipped at each iteration (sensitive, but useful for testing)
        - CoinPress release from each of the t iterations (releasable, is used)
    '''

    '''standardize data to have empirical mean 0 and covariance upper bounded by the identity matrix (assuming cov argument is properly set)'''
    means = np.mean(np.array(X), axis = 0)
    sds = np.sqrt(torch.diag(cov))
    sqrt_cov = np.real(sc.linalg.sqrtm(cov)) # NOTE: tiny complex values sometimes arise here because of rounding errors in PSD projection -- we get rid of them
    inv_sqrt_cov = torch.from_numpy(np.linalg.inv(sqrt_cov)).float()
    if args.d == 1:
        X = ((X-means) * inv_sqrt_cov)
    else:
        X = torch.mm( X-means, inv_sqrt_cov )

    c = c / np.max(sds.tolist())
    r = r / np.max(sds.tolist())


    '''run CoinPress for t iterations'''
    cs = []
    rs = []
    noise_variances = []
    total_clipped_vec = []
    for i in range(t-1):
        c, r, noise_variance, total_clipped = multivariate_mean_step(X.clone(), c, r, betas[i], Ps[i], args)
        cs.append(c)
        rs.append(r)
        noise_variances.append(noise_variance)
        total_clipped_vec.append(total_clipped)

    c, r, last_step_noise_variance, total_clipped = multivariate_mean_step(X.clone(), c, r, betas[t-1], Ps[t-1], args)
    cs.append(c)
    rs.append(r)
    noise_variances.append(last_step_noise_variance)
    total_clipped_vec.append(total_clipped)
            
    '''scale estimates back from mean 0 and covariance <= identity'''
    if args.d == 1:
        cs = [c*sqrt_cov + means for c in cs]
    else:
        cs = [np.matmul(c, sqrt_cov)+means for c in cs]
    priv_vars = [np.diag(np.sqrt(noise_var) * sqrt_cov)**2 for noise_var in noise_variances]

    return(cs[-1], priv_vars, total_clipped_vec, cs)

def multivariate_mean_step(X, c, r, beta, p, args):
    ''' 
    Privately estimate d-dimensional mean of X

    Args:
        X (pytorch tensor): data for which we want the mean
        c (float): center of ball (with radius r) which contains the mean
        r (float): radius of ball (with center c) which contains the mean
        beta (beta): failure probability (i.e. probability of clipping at least one point)
        p (beta): privacy budget
        args (set of arguments): arguments to be passed to multivariate_mean_step -- in this version, it's just the 
                                 dimensions of X (args.n and args.d)

    Returns:
        - mean estimate (new center of shrinking ball)
        - bound estimate (new radius of shrinking ball)
        - variance of privacy mechanism
        - total number of points clipped (sensitive, used only for diagnostics)
    '''
    n = args.n 
    d = args.d
    true_d = d

    '''
    functional beta is beta/2 because of use in gamma and r
    '''
    beta = beta/2

    '''establish clipping bounds'''
    gamma_1 = utils.gaussian_tailbound(d, beta/n)
    gamma_2 = utils.gaussian_tailbound(d, beta)
    clip_thresh = r + gamma_1

    '''clip points'''
    x = X - c
    mag_x = np.linalg.norm(x, axis=1)
    outside_ball = (mag_x > clip_thresh)
    total_clipped = np.sum(outside_ball)
    x_hat = (x.T / mag_x).T
    if np.sum(outside_ball) > 0:
        X[outside_ball] = c + (x_hat[outside_ball] * clip_thresh)
    
    '''calculate sensitivity'''
    delta = 2*clip_thresh/float(n)
    sd = delta/(2*p)**0.5
    
    '''add noise (calculate private mean) and update radius of ball'''
    Y = np.random.normal(0, sd, size=d)
    c = (torch.sum(X, axis=0)/float(n) + Y).float()
    r = gamma_2 * np.sqrt( 1/float(n) + (2 * clip_thresh**2)/(n**2 * p) )

    return c, r, sd**2, total_clipped
