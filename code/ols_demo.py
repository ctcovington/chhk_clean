import os 
import numpy as np 
import algorithm as na 
import sklearn.datasets as skld
import statsmodels.api as sm
import pandas as pd 
import matplotlib.pyplot as plt 
import utils
import sample_and_aggregate as sa

def ols(data, weights, dependent_var):
    X = np.array(data.drop(dependent_var, axis = 1))
    y = np.array(data[dependent_var])
    ols = sm.WLS(y, X, weights)
    ols_res = ols.fit()
    return(list(ols_res.params))

'''
data setup 
'''
n_samples_list = [100_000]
k_list = [1_000]
n_features_list = [5]
rho_list = [0.1]
n_targets = 1
noise = 10
n_sims = 1
sensitivity_scaling = 1
scalable = True
estimator = ols

res_df_list = [None] * n_sims
for rho in rho_list:
    for j in range(len(n_samples_list)):
        n = n_samples_list[j]
        k = k_list[j]
        n_features = n_features_list[j]
        d = n_features
        n_informative = n_features
        true_coefs_list = []
        for i in range(n_sims):
            print('n = {0}, k = {1}, d = {2}, rho = {3}: simulation {4} of {5}'.format(n, k, d, rho, i, n_sims))
            X, y, true_coefs = skld.make_regression(n_samples = n,
                                        n_features = n_features,
                                        n_informative = n_informative,
                                        n_targets = n_targets,
                                        effective_rank = n_features - 1,
                                        noise = noise,
                                        coef = True,
                                        random_state = i)
            true_coefs_list.extend(true_coefs)
            data = pd.DataFrame(X)
            data['y'] = y

            '''
            run non-private model
            '''
            np_res = sm.OLS(y,X).fit()
            nonpriv_coefs = np_res.params
            nonpriv_cov = np_res.cov_HC0
            np_stderrs = np_res.bse
            np_ci_ls, np_ci_us = zip(*np_res.conf_int(0.05))
            nonpriv_ci_ls = list(np_ci_ls)
            nonpriv_ci_us = list(np_ci_us)

            '''
            run private model 
            '''
            blb_sims = 10
            blb_mean_ests, blb_cov_ests = sa.sample_and_aggregate_plus_blb(data = data, 
                                                                    k = k, 
                                                                    r = blb_sims, 
                                                                    sensitivity_scaling = sensitivity_scaling, 
                                                                    scalable = scalable, 
                                                                    estimator = estimator, dependent_var = 'y')
            gvdp_overest_factor = 100
            cov_cov_u = gvdp_overest_factor * np.diag(np.cov(np.array([np.diag(c) for c in blb_cov_ests]).T))
            mean_cov_u = gvdp_overest_factor * np.diag(np.cov(blb_mean_ests.T))
            cov_c = np.diag(np.mean(blb_cov_ests, axis = 0)); cov_r = gvdp_overest_factor * max(np.diag(np.mean(blb_cov_ests, axis=0)))
            mean_c = np.mean(blb_mean_ests, axis = 0); mean_r = gvdp_overest_factor * max(np.mean(blb_mean_ests, axis = 0))
            t_cov = 5; t_mean = 5
            beta = 0.1
            rho_mean_budget_prop, rho_cov_budget_prop, beta_mean_budget_prop, beta_cov_budget_prop = (0.5, 0.5, 0.5, 0.5)
            ci_alphas = [0.05] * d
            scalable = True
            sensitivity_scaling = 1
            estimator = ols 
            dependent_var = 'y'

            priv_coefs, priv_cov, priv_CIs = na.general_valid_dp(data, k, 
                                                                blb_sims,
                                                                cov_cov_u, # TODO: need to state this in a way that scales to arbitrary degree 
                                                                cov_c, cov_r,
                                                                mean_c, mean_r,
                                                                t_cov, t_mean, 
                                                                rho, rho_mean_budget_prop, rho_cov_budget_prop, 
                                                                beta, beta_mean_budget_prop, beta_cov_budget_prop, 
                                                                ci_alphas,
                                                                scalable, sensitivity_scaling, estimator,
                                                                dependent_var = dependent_var)
            priv_ci_ls, priv_ci_us = zip(*priv_CIs)
            priv_ci_ls = list(priv_ci_ls)
            priv_ci_us = list(priv_ci_us)

            res_df = pd.DataFrame({'nonpriv_coef': nonpriv_coefs, 'nonpriv_ci_lower': nonpriv_ci_ls, 'nonpriv_ci_upper': nonpriv_ci_us, 
                                'priv_coef': priv_coefs, 'priv_ci_lower': priv_ci_ls, 'priv_ci_upper': priv_ci_us,})
            res_df_list[i] = res_df

        full_df = pd.concat(res_df_list)
        coef_df = full_df[ ['nonpriv_coef', 'priv_coef'] ]
        lower_ci_df = full_df[ ['nonpriv_ci_lower', 'priv_ci_lower'] ]
        upper_ci_df = full_df[ ['nonpriv_ci_upper', 'priv_ci_upper'] ]

        coef_df = coef_df.melt()
        coef_df['variable'] = coef_df['variable'].str.replace('_coef', '')
        coef_df.columns = ['type', 'coef']

        lower_ci_df = lower_ci_df.melt()
        lower_ci_df['variable'] = lower_ci_df['variable'].str.replace('_ci_lower', '')
        lower_ci_df.columns = ['type', 'ci_lower']

        upper_ci_df = upper_ci_df.melt()
        upper_ci_df['variable'] = upper_ci_df['variable'].str.replace('_ci_upper', '')
        upper_ci_df.columns = ['type', 'ci_upper']

        true_coefs_list.extend(true_coefs_list)

        plot_df = coef_df 
        plot_df['centered_coef'] = plot_df['coef'] - pd.Series(true_coefs_list)
        plot_df['ci_lower'] = lower_ci_df['ci_lower']
        plot_df['ci_upper'] = upper_ci_df['ci_upper']
        plot_df['centered_ci_lower'] = plot_df['ci_lower'] - pd.Series(true_coefs_list)
        plot_df['centered_ci_upper'] = plot_df['ci_upper'] - pd.Series(true_coefs_list)
        plot_df['lower_error'] = np.abs( plot_df['coef'] - plot_df['ci_lower'] )
        plot_df['upper_error'] = np.abs( plot_df['coef'] - plot_df['ci_upper'] )