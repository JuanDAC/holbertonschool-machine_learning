#!/usr/bin/env python3
"""
6. Gradients
"""

import numpy as np


def grads(Y, P):
    """
    Function that calculates the gradients of Y
    Arguments:
    - Y is a numpy.ndarray of shape (n, ndim) containing the low
      dimensional transformation of X
    - P is a numpy.ndarray of shape (n, n) containing the P affinities
      of X
    - n is the number of points
    - ndim is the new dimensional representation of X

    Returns: dY, Q
    - dY is a numpy.ndarray of shape (n, ndim) containing the gradients
      of Y
    - Q is a numpy.ndarray of shape (n, n) containing the Q affinities
      of Y
    """
    Q_affinities = __import__('5-Q_affinities').Q_affinities
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    PQ = P - Q
    dY = np.zeros((n, ndim))
    for i in range(n):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],
                          (ndim, 1)).T * (Y[i, :] - Y), axis=0)
    return dY, Q
