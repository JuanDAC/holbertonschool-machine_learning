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
    - var: total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    n, d = X.shape
    k, d = C.shape

    # Calculate the variance
    # You may use at most 1 loop
    var = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1))

    return var
