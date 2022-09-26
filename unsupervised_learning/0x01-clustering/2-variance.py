#!/usr/bin/env python3
"""
2 - Variance
"""

import numpy as np


def variance(X, C):
    """
    You are not allowed to use any loops
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - C: numpy.ndarray of shape (k, d) containing the centroid
         means for each cluster
    Returns:
    - var: var, or None on failure, var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    n, d = X.shape
    k, d = C.shape

    centroids = C[:, np.newaxis]
    distances = np.sqrt(((X - centroids)**2).sum(axis=2))
    min_distances = np.min(distances, axis=0)
    veriance = np.sum(min_distances**2)

    return veriance
