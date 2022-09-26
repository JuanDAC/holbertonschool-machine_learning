#!/usr/bin/env python3
"""
File that contains the regular function
"""
import numpy as np


def regular(P):
    """
    Function that determines the steady state
    probabilities of a regular markov chain
    Arguments:
        - P is a square 2D numpy.ndarray of shape (n, n) representing the
            * i is the probability of transitioning from state i to state j
            * j is the probability of transitioning from state j to state i
    Returns:
        - a numpy.ndarray of shape (1, n) containing the
            steady state probabilities, or None on failure
    """

    try:
        if type(P) is not np.ndarray or len(P.shape) != 2:
            return None

        if P.shape[0] != P.shape[1]:
            return None

        cols = P.shape[0]
        ans = np.ones((1, cols))
        eq = np.vstack([P.T - np.identity(cols), ans])
        results = np.vstack([np.zeros((cols, 1)), np.array([1])])

        statetionary = np.linalg.solve(eq.T.dot(eq), eq.T.dot(results)).T

        if len(np.argwhere(statetionary < 0)) > 0:
            return None

        return statetionary
    except Exception:
        return None
