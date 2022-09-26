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

    if type(k) != int or k <= 0:
        return None, None

    if type(iterations) != int or iterations <= 0:
        return None, None

    # Initialize the cluster centroids using a multivariate uniform
    # distribution (based on0-initialize.py)
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)

    _, d = X.shape
    C = np.random.uniform(min, max, size=(k, d))

    # If a cluster contains no data points during the update step,
    # reinitialize its centroid
    # You should use numpy.random.uniform exactly twice
    # You may use at most 2 loops
    for _ in range(iterations):
        # Calculate the distance between each data point and each centroid
        # You should use numpy.linalg.norm exactly twice
        # You may use at most 2 loops

        initial_c = np.copy(C)
        c_ext = C[:, np.newaxis]
        dist = np.sqrt(((X - c_ext) ** 2).sum(axis=2))

        # Assign each data point to the nearest cluster centroid
        clss = np.argmin(dist, axis=1)

        # Update the centroid of each cluster
        # If a cluster contains no data points during the update step,
        # reinitialize its centroid
        # You should use numpy.random.uniform exactly twice
        # You may use at most 2 loops
        for j in range(k):
            if X[clss == j].size == 0:
                C[j] = np.random.uniform(min, max, size=(1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        c_ext = C[:, np.newaxis]
        dist = np.sqrt(((X - c_ext) ** 2).sum(axis=2))
        clss = np.argmin(dist, axis=0)

        if (initial_c == C).all():
            break

    return C, clss
