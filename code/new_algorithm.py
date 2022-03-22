import sample_and_aggregate as sa
import coinpress_generalized as cg 
import numpy as np
import torch
import utils
import scipy.stats as ss

def general_valid_dp(data, 
                     k, 
                     blb_sims,
                     cov_cov_u,
                     cov_c, cov_r,
                     mean_c, mean_r,
                     t_cov, t_mean, 
                     rho, rho_mean_budget_prop, rho_cov_budget_prop, 
                     beta, beta_mean_budget_prop, beta_cov_budget_prop, 
                     ci_alphas,
                     scalable, sensitivity_scaling, estimator, **kwargs):
    '''
    Estimation meta-algorithm that produces zCDP mean and covariance estimates for an arbitrary estimator.

    Args:
        data (pandas dataframe): raw data
        k (int): number of subsets into which raw data will be partitioned, note that the variance of 
                 our privacy mechanism will scale with k so large k (all else equal) is better
        blb_sims (int): number of times the bag of little bootstraps runs within each subset
        cov_cov_u (float or matrix): IMPORTANT - user-defined upper bound on the covariance of the sample covariance 
                                     of the estimator. This could be a fully specified matrix, in which case 
                                     cov_cov_u should be a Loewner upper bound on the covariance of the covariance, 
                                     or a float, in which case cov_cov_u * I_d should be a Loewner upper bound
        cov_c (float): center of ball (with radius cov_r) which contains all the diagonal elements of the covariance of the estimator
        cov_r (float): radius of ball (with center cov_r) which contains all the diagonal elements of the covariance of the estimator
        mean_c (float): center of ball (with radius mean_r) which contains all the estimator's parameter values
        mean_r (float): radius of ball (with center mean_c) which contains all the estimator's parameter values
        t_cov (int): number of steps CoinPress takes to estimate the private covariance (usually 4-7 is fine)
        t_mean (int): number of steps CoinPress takes to estimate the private mean (usually 4-7 is fine)
        rho (float): privacy budget, expressed in zCDP
        rho_mean_budget_prop (float): proportion of privacy budget allocated to mean estimation
        rho_cov_budget_prop (float): proportion of privacy budget allocated to mean estimation
        beta (float): overall probability of failure (given that our assumptions hold, what is the probability that we clip at least one 
                      point in either the mean or covariance estimation process)
        beta_mean_budget_prop (float): proportion of failure budget allocated to mean estimation
        beta_cov_budget_prop (float): proportion of failure budget allocated to covariance estimation
        ci_alphas (vector of floats): desired confidence interval level for each parameter
        scalable (boolean): does estimator accept a "weights" argument that would allow BLB to run with small subset and multinomial weights?
                            If not, BLB will need to actually resample with replacement.
        sensitivity_scaling (int): how many times do we allow each data element to be included in the the partitioning step of the BLB.
                                   CC added this for generality, but I struggle to think of a case where it should be anything but 1.
        estimator (function): function for which we want to estimate parameters. 
                              It should take "data" as its first argument, "weights" as its second (if scalable = True),
                              and then arbitrarily many kwargs (see below)
        **kwargs (set of arguments): set of all arguments other than "data" and "weights" used in the estimator


    Returns:
        est_priv_mean: private parameter estimates  
        est_priv_cov: private estimate of the covariance of the sampling distribution of the parameters
        cis: confidence interval for each parameter estimate

    '''
    
    '''set parameters'''
    # check budgets 
    rho_budget_sum = np.sum([rho_mean_budget_prop, rho_cov_budget_prop])
    beta_budget_sum = np.sum([beta_mean_budget_prop, beta_cov_budget_prop])
    if np.abs(rho_budget_sum - 1) > 10**-5:
        raise Exception('mean and cov proportions for rho budget sum to {0} -- they must sum to 1'.format(rho_budget_sum))
    if np.abs(beta_budget_sum - 1) > 10**-5:
        raise Exception('mean and cov proportions for beta budget sum to {0} -- they must sum to 1'.format(beta_budget_sum))
    
    # rescale rhos by sensitivity scaling -- zCDP guarantee scales with square of group privacy factor 
    rho = rho / sensitivity_scaling**2

    # allocate privacy budget
    rho_mean = rho * rho_mean_budget_prop
    rho_cov = rho * rho_cov_budget_prop

    # allocate failure budget
    beta_mean = beta * beta_mean_budget_prop
    beta_cov = beta * beta_cov_budget_prop
    
    '''ensure that centers are torch tensors'''
    cov_c = torch.tensor(cov_c).float()
    mean_c = torch.tensor(mean_c).float()
    
    '''run sample and aggregate with BLB to produce k mean and covariance estimates'''
    blb_mean_ests, blb_cov_ests = sa.sample_and_aggregate_plus_blb(data = data, 
                                                            k = k, 
                                                            r = blb_sims, 
                                                            sensitivity_scaling = sensitivity_scaling, 
                                                            scalable = scalable, 
                                                            estimator = estimator,
                                                            **kwargs)
    
    # set dimension and ensure blb_cov_ests are matrices
    try:
        mean_d = blb_mean_ests.shape[1]
        cov_d = blb_cov_ests[0].shape[0]
    except:
        blb_mean_ests = blb_mean_ests.reshape(-1,1)
        blb_cov_ests = blb_cov_ests.reshape(-1,1)
        mean_d = cov_d = 1
    blb_cov_ests = np.array([np.diag(c) for c in blb_cov_ests])
    d = blb_mean_ests.shape[1]
    
    if d > 1:
        cov_uppers = blb_cov_ests
    elif d == 1:
        cov_uppers = blb_cov_ests.reshape(-1,1)

    '''perform covariance estimation'''
    args = utils.parse_args()
    args.n, args.d = cov_uppers.shape
    args.overall_n = data.shape[0]
    
    '''TODO: for now, splitting beta_cov into two parts (calculation and upper bounding)'''
    beta_cov = beta_cov / 2
    args.upper_bound_cov_beta = beta_cov 

    beta_cov_est = beta_cov/2 # half for estimation, half for Loewner upper bounding after the fact
    cov_d = cov_uppers.shape[1]
    args.d = cov_d
    _, privatization_variances, _, priv_cov_ests = cg.multivariate_mean_iterative(torch.from_numpy(cov_uppers).float(), 
                                        c = cov_c, 
                                        r = cov_r,
                                        cov = torch.tensor(cov_cov_u * np.eye(cov_uppers.shape[1])),
                                        t = t_cov,
                                        betas = [1/(4*(t_cov-1))*beta_cov_est] * (t_cov-1) + [(3/4)*beta_cov_est],
                                        Ps = [1/(4*(t_cov-1))*rho_cov] * (t_cov-1) + [(3/4)*rho_cov],
                                        args = args
                                        )

    est_priv_cov, est_priv_cov_covariance = utils.inverse_covariance_weighting(priv_cov_ests, [v * np.eye(cov_d) for v in privatization_variances])

    '''
    convert estimated covariance cells back to actual matrix
    '''
    # NOTE: line below is for diagonal cov
    est_priv_cov = est_priv_cov*np.eye(cov_d)
    
    privatization_var = np.diag(est_priv_cov_covariance)
    sd = np.sqrt(privatization_var)

    # NOTE: version below is for diagonal cov
    est_priv_cov += np.eye(cov_d) * ss.norm.ppf( (1 - args.upper_bound_cov_beta / args.d), loc = 0, scale = np.sqrt(privatization_var) )
    est_priv_cov = torch.tensor(est_priv_cov)
    est_priv_cov = utils.psd_proj_symm(est_priv_cov)
    
    '''TODO: need to make some better arguments above, specifically around getting upper bound but also maybe covariance combinations'''
    
    '''perform mean estimation'''
    args.d = mean_d
    args.est_type = 'mean'
    _, privatization_variances, _, sa_means = cg.multivariate_mean_iterative(torch.from_numpy(blb_mean_ests).float(), 
                                        c = mean_c, 
                                        r = mean_r,
                                        cov = k*est_priv_cov,
                                        t = t_mean,
                                        betas = [1/(4*(t_mean-1))*beta_mean] * (t_mean-1) + [(3/4)*beta_mean],
                                        Ps = [1/(4*(t_mean-1))*rho_mean] * (t_mean-1) + [(3/4)*rho_mean],
                                        args = args
                                        )
    
    covariances_of_mean_est = [priv_var*np.eye(args.d) + np.array(est_priv_cov) for priv_var in privatization_variances]
    est_priv_mean, est_priv_mean_covariance = utils.inverse_covariance_weighting(sa_means, covariances_of_mean_est)
    priv_var = np.diag(est_priv_mean_covariance)
    
    '''get confidence intervals'''
    if d == 1:
        est_priv_mean = est_priv_mean.reshape(1)
        est_priv_cov = est_priv_cov.reshape(1)

    ci_ls = [0] * args.d
    ci_us = [0] * args.d
    for j in range(d):
        quantile = ss.norm.ppf(1 - ci_alphas[j] / 2, loc = 0, scale = np.sqrt(np.diag(est_priv_cov)[j] + priv_var[j]))
        ci_ls[j] = est_priv_mean[j] - quantile 
        ci_us[j] = est_priv_mean[j] + quantile 

    if d == 1:
        ci_ls = ci_ls[0] 
        ci_us = ci_us[0]
    cis = [(l,u) for l,u in zip(ci_ls, ci_us)]

    
    return(est_priv_mean, est_priv_cov, cis)
    
    
    
    

    