#!/usr/bin/env python3
"""
File that contains the markov_chain function
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain being in a
    particular state after a specified number of iterations
    Arguments:
        - P is a square 2D numpy.ndarray of shape (n, n) representing the
        - s is a numpy.ndarray of shape (1, n) representing the probability
        - t is the number of iterations that the markov chain has been through
    Returns:
        - a numpy.ndarray of shape (1, n) representing the probability of
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if not isinstance(t, int) or t <= 0:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.sum(s) != 1:
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
