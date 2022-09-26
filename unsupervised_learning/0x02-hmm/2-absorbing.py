#!/usr/bin/env python3
"""
File that contains the absorbing function
"""
import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing:
    Argumrnts:
        - P is a is a square 2D numpy.ndarray of shape (n, n) representing the standard transition matrix
            * P[i, j] is the probability of transitioning from state i to state j
            * n is the number of states in the markov chain
    Returns: 
        - True if it is absorbing, or False on failure
    """
    try:
        if type(P) is not np.ndarray or len(P.shape) != 2:
            return False

        if P.shape[0] != P.shape[1]:
            return False

        if np.any(P < 0) or np.any(P > 1):
            return False

        n = P.shape[0]
        diag = np.diag(P)
        if np.all(diag == 1):
            return True

        for i in range(n):
            if P[i, i] == 1:
                for j in range(n):
                    if i != j and P[i, j] != 0:
                        return False

        return True
    except Exception:
        return False
