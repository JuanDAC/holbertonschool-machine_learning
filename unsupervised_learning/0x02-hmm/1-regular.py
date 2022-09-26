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
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.any(P < 0):
        return None
    if np.any(np.sum(P, axis=1) != 1):
        return None
    try:
        return np.linalg.solve(np.eye(P.shape[0]) - P.T, np.ones(P.shape[0]))
    except Exception:
        return None
