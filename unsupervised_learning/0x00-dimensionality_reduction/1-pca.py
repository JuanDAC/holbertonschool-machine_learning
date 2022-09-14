#!/usr/bin/env python3
"""
1 pca.py - PCA
"""

import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset
    Arguments:
    - X is a numpy.ndarray of shape (n, d) where:
    - n is the number of data points
    - d is the number of dimensions in each point
    - ndim is the new dimensionality of the transformed X
    Returns:
    - T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X
    """
    u, s, vh = np.linalg.svd(X)
    W = vh.T[:, :ndim]
    T = np.matmul(X, W)
    return T
