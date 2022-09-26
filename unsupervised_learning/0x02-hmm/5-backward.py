#!/usr/bin/env python3
"""
File that contains the backward function
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Funtion that performs the backward algorithm for a hidden markov model
    Arguments:
        - Observation is a numpy.ndarray of shape (T,) that contains the index
            * T is the number of observations
        - Emission is a numpy.ndarray of shape (N, M) containing the emission
            * Emission[i, j] is the probability of observing j given the hidden
        - Transition is a 2D numpy.ndarray of shape (N, N) containing the
            * Transition[i, j] is the probability of transitioning from
        - Initial a numpy.ndarray of shape (N, 1) containing the probability
    Returns:
        - P, B, or None, None on failure
    """
    try:
        if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
            return None, None

        if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
            return None, None

        if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
            return None, None

        if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
            return None, None

        T = Observation.shape[0]
        N, M = Emission.shape

        if Transition.shape[0] != N or Transition.shape[1] != N:
            return None, None

        if Initial.shape[0] != N or Initial.shape[1] != 1:
            return None, None

        B = np.zeros((N, T))
        B[:, T - 1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(N):
                B[i, t] = np.sum(B[:, t + 1] * Transition[i, :]
                                 * Emission[:, Observation[t + 1]])

        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

        return P, B
    except Exception:
        return None, None
