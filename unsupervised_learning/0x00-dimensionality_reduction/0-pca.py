#!/usr/bin/env python3
"""
0  pca.py - PCA
"""
import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset
    Arguments:
    - X is a numpy.ndarray of shape (n, d) where:
    - n is the number of data points
    - d is the number of dimensions in each point

    Validation:
    - all dimensions have a mean of 0 across all data points
    - var is the fraction of the variance that the PCA transformation
      should maintain

    Returns:
    - the weights matrix, W, that maintains var fraction of X‘s original
      variance W is a numpy.ndarray of shape (d, nd) where nd is the new
      dimensionality of the transformed X
    """
    u, s, vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    d = np.argmax(cum_var >= var)
    W = vh.T[:, :d + 1]
    return W
