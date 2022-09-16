#!/usr/bin/env python3
"""
12 - Agglomerative
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - dist: the maximum cophenetic distance for all clusters
    Returns:
    - clss: numpy.ndarray of shape (n,) containing the cluster
            indices for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if type(dist) != int or dist <= 0:
        return None

    Z = linkage(X, 'ward')
    clss = fcluster(Z, t=dist, criterion='distance')
    dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
