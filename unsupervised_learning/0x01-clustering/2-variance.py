#!/usr/bin/env python3
"""
2 - Variance
"""

import numpy as np


def variance(X, C):
    """
    Arguments: 
    - X: numpy.ndarray of shape (n, d) containing the data set
    - C: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    Returns:
    - var: total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    n, d = X.shape
    k, d = C.shape

    # Calculate the distance between each data point and each centroid
    # You should use numpy.linalg.norm exactly twice
    # You may use at most 2 loops
    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)

    # Assign each data point to the nearest cluster centroid
    clss = np.argmin(dist, axis=1)

    # Calculate the total intra-cluster variance for a data set
    # You may use at most 2 loops
    var = 0
    for i in range(k):
        var += np.sum(np.linalg.norm(X[clss == i] - C[i], axis=1) ** 2)

    return var
