#!/usr/bin/env python3
"""
File that contains the absorbing function
"""
import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing:
    Argumrnts:
        - P is a is a square 2D numpy.ndarray of shape (n, n)
            representing the standard transition matrix
            * P[i, j] is the probability of transitioning from
              state i to state j.
            * n is the number of states in the markov chain
    Returns:
        - True if it is absorbing, or False on failure
    """

    try:
        if type(P) is not np.ndarray:
            return False

        if len(P.shape) != 2:
            return False

        if P.shape[0] != P.shape[1]:
            return False

        for elem in np.sum(P, axis=1):
            if not np.isclose(elem, 1):
                return False

        diagonal = np.diag(P)

        if (diagonal == 1).all():
            return True

        absorb = (diagonal == 1)

        for row in range(len(diagonal)):
            for col in range(len(diagonal)):
                if P[row, col] > 0 and absorb[col]:
                    absorb[row] = 1

        if (absorb == 1).all():
            return True

        return False
    except Exception:
        return False
