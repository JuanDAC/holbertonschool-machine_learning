#!/usr/bin/env python3
"""
6 Expectation
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - pi: numpy.ndarray of shape (k,) containing the priors
          for each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid
         means for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the covariance
         matrices for each cluster
    Returns:
    - g: numpy.ndarray of shape (k, n) containing the posterior
         probabilities for each data point in each cluster
    - l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    n, _ = X.shape
    k = pi.shape[0]

    # Calculate the posterior probabilities for each data point in each cluster
    # You should use pdf = __import__('5-pdf').pdf
    # You may use at most 1 loop
    g = np.zeros((k, n))

    denominator = 0
    for i in range(k):
        numerator = pdf(X, m[i], S[i]) * pi[i]
        g[i] = numerator
        denominator += numerator

    # Normalize the posterior probabilities
    # You may use at most 1 loop
    g = g / denominator

    # Calculate the total log likelihood
    # You may use at most 1 loop
    log = np.sum(np.log(np.sum(g, axis=0)))

    return g, log
