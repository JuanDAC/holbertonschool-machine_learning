#!/usr/bin/env python3
"""
7. Maximization
"""

import numpy as np


def maximization(X, g):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - g: numpy.ndarray of shape (k, n) containing the posterior
         probabilities for each data point in each cluster
    Returns:
    - pi: numpy.ndarray of shape (k,) containing the updated
          priors for each cluster
    - m: numpy.ndarray of shape (k, d) containing the updated
         centroid means for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the updated
         covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n = g.shape

    # Calculate the updated priors for each cluster
    # You may use at most 1 loop
    pi = np.sum(g, axis=1) / n

    # Calculate the updated centroid means for each cluster
    # You may use at most 1 loop
    m = np.matmul(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # Calculate the updated covariance matrices for each cluster
    # You may use at most 2 loops
    S = np.zeros((k, d, d))
    for i in range(k):
        X_m = X - m[i]
        S[i] = np.matmul(g[i] * X_m.T, X_m) / np.sum(g[i])

    return pi, m, S
