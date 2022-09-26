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
    if type(X) is not np.ndarray or type(g) is not np.ndarray:
        return None, None, None

    if X.ndim != 2 or g.ndim != 2 or X.shape[0] != g.shape[1]:
        return None, None, None

    if not np.all(np.isclose(g.sum(axis=0), 1)):
        return None, None, None

    try:
        gsum = g.sum(axis=1)
        # Calculate the updated priors for each cluster
        pi = gsum / X.shape[0]
        m = np.matmul(g, X) / gsum[:, np.newaxis]
        # Calculate the updated centroid means for each cluster
        S = np.ndarray((m.shape[0], m.shape[1], m.shape[1]))
        for cluster in range(g.shape[0]):
            diff = X - m[cluster]
            S[cluster] = (np.matmul((diff * g[cluster, :, np.newaxis]).T, diff)
                          / gsum[cluster])
        return pi, m, S
    except Exception:
        return None, None, None
