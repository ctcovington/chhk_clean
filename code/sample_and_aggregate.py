import numpy as np

def create_index_partition(n, k, sensitivity_scaling):
    '''
    Creates k disjoint subsets of {0, 1, ..., n-1} of as close to equal size as possible.

    Args:
        n (int): Number of elements to be partitioned.
        k (int): Number of cells (disjoint subsets) in the partition.
        sensitivity_scaling (int): Number of times whole data set can be used

    Returns:
        list: List of disjoint subsets, each represented via a list.
    '''
    permutation = np.random.permutation(list(range(n)) * sensitivity_scaling)
    partition = [ list(permutation[i::k]) for i in range(k) ]
    return(partition)

def blb(subsampled_data, overall_n, r, estimator, scalable, **kwargs):
    '''
    Runs bag of little bootstraps.

    Args:
        subsampled_data (pandas dataframe): Data on which blb should be run.
        overall_n (int): Number of elements in overall data (note that this is not the number of elements 
                         in the subsampled data).
        r (int): Number of simulations to perform.
        estimator (function): Function we are using the blb to estimate. The estimator should take 
                              the subsampled data as the first argument.
        scalable (bool): Whether or not estimator is scalable with weights argument.
        **kwargs: Arguments to estimator function. 

    Returns:
        float or list of floats: Averaged results of estimator function applied 
    '''
    blb_res = []
    subset_n = subsampled_data.shape[0]
    for i in range(r):
        if scalable == True:
            # generate sampling weights
            weights = np.random.multinomial(n = overall_n, pvals = [1/subset_n] * subset_n)

            # apply estimator to upsampled data
            ans = estimator(subsampled_data, weights = weights, **kwargs)
            # ans = estimator(subsampled_data, weights = weights, dependent_var = 'y')
        else:
            # upsample partition cell data to size of overall data
            upsampled_data = subsampled_data.sample(n = overall_n, replace = True)

            # apply estimator to upsampled data
            ans = estimator(upsampled_data, weights = np.ones(overall_n), **kwargs)
        
            # NOTE: just trying to apply estimator to subset now 
            # ans = estimator(subsampled_data, **kwargs)

        blb_res.append(ans)
    blb_res = np.array(blb_res)

    return(np.mean(blb_res, axis = 0), np.cov(blb_res.T))

def sample_and_aggregate_plus_blb(data, k, r, sensitivity_scaling, scalable, estimator, **kwargs):
    '''
    Partitions data into k cells, runs r iterations of blb on each cell using the estimator 
    to estimate the quantity of interest, and returns the k estimates.
    The function name is a bit of a misnomer, as we do not aggregate in this step,
    instead leaving it for the CoinPress estimation step.

    Args:
        data (pandas dataframe): Data on which to perform the sample-and-aggregate + blb.
        k (int): Number of cells (disjoint subsets) in the partition.
        r (int): Number of simulations to perform within blb.
        sensitivity_scaling (int): Number of times to use full data in sample and aggregate. Recommended to be 1.
        scalable (bool): Whether or not to scale up sample size within each sample-and-aggregate subset to overall data set size
        estimator (function): Function we are using the blb to estimate. The estimator should take 
                              the subsampled data as the first argument.
        **kwargs: Arguments to estimator function. 
    
    Returns:
        tuple of numpy arrays: Set of k mean/covariance estimates calculated on the partition cells. 
    '''
    # get overall number of elements
    n = data.shape[0]

    # construct index partition
    partition = create_index_partition(n, k, sensitivity_scaling)

    # iterate over cells of partition and run blb on each
    samp_and_agg_means = []
    samp_and_agg_covs = []
    for i in range(k):
        if i % 100 == 0:
            print('running blb on cell {0} of {1}'.format(i+1, k))
        data_subset = data.iloc[partition[i], :]
        blb_mean, blb_cov = blb(data_subset, n, r, estimator, scalable, **kwargs)
        samp_and_agg_means.append(blb_mean)
        samp_and_agg_covs.append(blb_cov)
    
    mean_array = np.array(samp_and_agg_means)
    cov_array = np.array(samp_and_agg_covs)
    return(mean_array, cov_array)
