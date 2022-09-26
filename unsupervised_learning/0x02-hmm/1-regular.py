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

    try:
        if type(P) is not np.ndarray or len(P.shape) != 2:
            return None

        if P.shape[0] != P.shape[1]:
            return None

        if np.any(P < 0):
            return None

        if np.any(np.sum(P, axis=1) != 1):
            return None

        n = P.shape[0]
        _, evecs = np.linalg.eig(P.T)
        state = evecs / np.sum(evecs, axis=0)

        for i in np.dot(state.T, P):
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)
    except Exception:
        return None
