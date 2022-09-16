#!/usr/bin/env python3
"""
4 Initialize GMM
"""

import numpy as np


def initialize(X, k):
    """
    Arguments: 
    - X: numpy.ndarray of shape (n, d) containing the dataset
         that will be used for K-means clustering
    - k: positive integer containing the number of clusters
    Returns:
    - centroids: numpy.ndarray of shape (k, d) containing the
                 initialized centroids for each cluster, or
                 None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if type(k) != int or k <= 0:
        return None, None, None

    kmeans = __import__('1-kmeans').kmeans

    centroids, clss = kmeans(X, k)

    pi = np.full((k,), 1 / k)
    m = centroids
    S = np.full((k, X.shape[1], X.shape[1]), np.identity(X.shape[1]))

    return pi, m, S
