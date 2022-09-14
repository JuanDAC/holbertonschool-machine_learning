#!/usr/bin/env python3
"""
0 - Initialize
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
        return None
    if type(k) != int or k <= 0:
        return None
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    centroids = np.random.uniform(min, max, (k, d))
    return centroids
