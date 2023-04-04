# CHHK21 Code

This version of the [CHHK21](https://arxiv.org/abs/2110.14465) code is cleaned and simplified from the full version in the private repository. Namely, we simplify in the following ways:

1. Assume sampling distribution of the estimator (i.e. output from sample-aggregate) is sub-Gaussian
2. Assume the analyst wants only confidence intervals and not the full covariance matrix. To this end, the algorithm estimates only the variances (rather than the full covariance matrix).

- `coinpress_generalized.py`: code for CoinPress mean estimation
- `algorithm.py`: overall algorithm from CHHK21
- `ols_demo.py`: demonstration of how to run the algorithm for an OLS estimator
- `sample_and_aggregate.py`: code for sample-aggregate and bag of little bootstraps
- `utils.py`: a few utilities used in other files