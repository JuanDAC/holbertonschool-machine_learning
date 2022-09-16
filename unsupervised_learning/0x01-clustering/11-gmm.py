#!/usr/bin/env python3
"""
11 - GMM
"""

import numpy as np
import sklearn.mixture


def gmm(X, k):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: number of clusters
    Returns:
    - pi: numpy.ndarray of shape (k,) containing the cluster priors
    - m: numpy.ndarray of shape (k, d) containing the centroid means
    - S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
    - clss: numpy.ndarray of shape (n,) containing the cluster indices
            for each data point
    - bic: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
           value for each cluster size tested
    """

    gm = sklearn.mixture.GaussianMixture(n_components=k)
    gm.fit(X)

    pi = gm.weights_
    m = gm.means_
    S = gm.covariances_

    clss = gm.predict(X)
    bic = gm.bic(X)

    return pi, m, S, clss, bic
