#!/usr/bin/env python3
"""
9 - BIC
"""

import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing the minimum number of
            clusters to check for (inclusive)
    - kmax: positive integer containing the maximum number of
            clusters to check for (inclusive)
    - If kmax is None, kmax should be set to the maximum number
      of clusters possible
    - iterations: positive integer containing the maximum number
      of iterations for the EM algorithm
    - tol: non-negative float containing the tolerance for the
           EM algorithm
    - verbose: boolean that determines if the EM algorithm should
               print information to the standard output

    Returns:
    - best_k: best value for k based on its BIC
    - best_result: tuple containing pi, m, S
    - pi: numpy.ndarray of shape (k,) containing the cluster
          priors for the best number of clusters
    - m: numpy.ndarray of shape (k, d) containing the centroid
         means for the best number of clusters
    - S: numpy.ndarray of shape (k, d, d) containing the
         covariance matrices for the best number of clusters
    - l: numpy.ndarray of shape (kmax - kmin + 1) containing
         the log likelihood for each cluster size tested
    - b: numpy.ndarray of shape (kmax - kmin + 1) containing
         the BIC value for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None

    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None

    if kmin >= kmax:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol <= 0:
        return None, None, None, None

    if type(verbose) != bool:
        return None, None, None, None

    expectation_maximization = __import__('8-EM').expectation_maximization

    best_k = np.arange(kmin, kmax + 1)
    best_result = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S,  _, log_l = expectation_maximization(X, k, iterations, tol,
                                                       verbose)
        best_k.append(k)
        best_result.append((pi, m, S))
        logl_val.append(log_l)

        # Formula p :https://bit.ly/33Cw8lH
        cov_params = d * (d + 1) / 2.
        mean_params = d
        p = int((k * cov_params) + (k * mean_params) + k - 1)

        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    best_k = best_k[best_val]
    best_result = best_result[best_val]

    return best_k, best_result, logl_val, bic
