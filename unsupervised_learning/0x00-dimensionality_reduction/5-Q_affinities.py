#!/usr/bin/env python3
"""
5. Q affinities
"""

import numpy as np


def Q_affinities(Y):
    """
    Function that calculates the Q affinities
    Arguments:
    - Y is a numpy.ndarray of shape (n, ndim) containing the low
      dimensional transformation of X
    - n is the number of points
    - ndim is the new dimensional representation of X 

    Returns: Q, num
    - Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    - num is a numpy.ndarray of shape (n, n) containing the numerator 
      of the Q affinities

    """
    n, ndim = Y.shape
    sum_Y = np.sum(np.square(Y), axis=1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0
    Q = num / np.sum(num)
    return Q, num
