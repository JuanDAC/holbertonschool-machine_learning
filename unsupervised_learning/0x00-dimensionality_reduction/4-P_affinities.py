#!/usr/bin/env python3
"""
4. P affinities
"""

import numpy as np


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Function that calculates the symmetric P affinities of a data set
    Arguments:
    - X is a numpy.ndarray of shape (n, d) containing the dataset to be
      transformed by t-SNE
    - n is the number of data points
    - d is the number of dimensions in each point
    - perplexity is the perplexity that all Gaussian distributions should
      have
    - tol is the maximum tolerance allowed (inclusive) for the difference 
      in Shannon entropy from perplexity for all Gaussian distributions 
    Returns:
    P, a numpy.ndarray of shape (n, n) containing the symmetric P affinities 
    """
    P_init = __import__('2-P_init').P_init
    HP = __import__('3-entropy').HP

    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    sum_P = np.sum(P)
    for i in range(n):
        beta_min = None
        beta_max = None
        Hi, Pi = HP(D[i, :], betas[i])
        Hdiff = Hi - H
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                beta_min = betas[i].copy()
                if beta_max is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2
            else:
                beta_max = betas[i].copy()
                if beta_min is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2
            Hi, Pi = HP(D[i, :], betas[i])
            Hdiff = Hi - H
        P[i, :] = Pi
    P = (P + P.T) / (2 * n)
    return P
