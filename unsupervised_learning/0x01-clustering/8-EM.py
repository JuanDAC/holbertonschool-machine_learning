#!/usr/bin/env python3
"""
8. EM
"""

import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum
      number of iterations for the algorithm
    - tol: non-negative float containing tolerance of the log
           likelihood, used to determine early stopping i.e.
           if the difference is less than or equal to tol you
           should stop the algorithm
    - verbose: boolean that determines if you should print
               information about the algorithm
    Returns:
    - pi: numpy.ndarray of shape (k,) containing the priors
          for each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid
         means for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the
         covariance matrices for each cluster
    - g: numpy.ndarray of shape (k, n) containing the
         probabilities for each data point in each cluster
    - l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if type(k) != int or k <= 0:
        return None, None, None, None, None

    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) != float or tol < 0:
        return None, None, None, None, None

    if type(verbose) != bool:
        return None, None, None, None, None

    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    pi, m, S = initialize(X, k)

    # Calculate the log likelihood
    # You may use at most 1 loop
    g, l_ = expectation(X, pi, m, S)

    for i in range(iterations):
        # Calculate the expectation step
        # You may use at most 1 loop
        g, l_ = expectation(X, pi, m, S)

        # Calculate the maximization step
        # You may use at most 1 loop
        pi, m, S = maximization(X, g)

        # Calculate the log likelihood
        # You may use at most 1 loop
        g, l_new = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i,
                l_.round(5)
            ))

        if abs(l_new - l_) <= tol:
            break

        l_ = l_new
    g, l_ = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1,
            l_.round(5)
        ))

    return pi, m, S, g, l_
