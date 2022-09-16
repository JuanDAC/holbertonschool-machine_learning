#!/usr/bin/env python3
"""
10 - K-means
"""

import sklearn.cluster
import numpy as np


def kmeans(X, k):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer containing the number of clusters
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

    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
