#!/usr/bin/env python3
"""
File that contains the regular function
"""
import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities of a regular markov chain
    Arguments:
        - P is a square 2D numpy.ndarray of shape (n, n) representing the
            * i is the probability of transitioning from state i to state j
            * j is the probability of transitioning from state j to state i
    Returns:
        - a numpy.ndarray of shape (1, n) containing the steady state probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    n = P.shape[0]
    if n == 1:
        return np.array([[1]])
    A = np.hstack((P.T - np.eye(n), np.ones((n, 1))))
    b = np.zeros((n, 1))
    b[-1] = 1
    return np.linalg.solve(A, b)
