#!/usr/bin/env python3
"""
1. K-means
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum number
      of iterations that should be performed
    Returns:
    - C: numpy.ndarray of shape (k, d) containing the centroid
         means for each cluster
    - clss: numpy.ndarray of shape (n,) containing the index of
            the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    C = np.random.uniform(min, max, (k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, np.newaxis] - C, axis=-1), axis=1)
        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(min, max, (1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)
    return C, clss
