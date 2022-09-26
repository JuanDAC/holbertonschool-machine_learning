#!/usr/bin/env python3
"""
File that contains the viterbi function
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Function should use the viterbi algorithm
        * (https://en.wikipedia.org/wiki/Viterbi_algorithm)
    Arguments:
        - Observation is a numpy.ndarray of shape (T,) that contains the index
            * of the observation
            * T is the number of observations
        - Emission is a numpy.ndarray of shape (N, M) containing the emission
            * probability of a specific observation given a hidden state
            * Emission[i, j] is the probability of observing j given the hidden
                * state i
            * N is the number of hidden states
            * M is the number of all possible observations
        - Transition is a 2D numpy.ndarray of shape (N, N) containing the
            * transition probabilities
            * Transition[i, j] is the probability of transitioning from the
                * hidden state i to j
        - Initial a numpy.ndarray of shape (N, 1) containing the probability
            * of starting in a particular hidden state
    Returns:
        - P, F, or None, None on failure
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
        N, _ = Emission.shape

        if Transition.shape[0] != N or Transition.shape[1] != N:
            return None, None

        if Initial.shape[0] != N or Initial.shape[1] != 1:
            return None, None

        V = np.zeros((N, T))
        B = np.zeros((N, T))
        V[:, 0] = Initial.T * Emission[:, Observation[0]]
        B[:, 0] = 0

        for t in range(1, T):
            for n in range(N):
                V[n, t] = np.max(V[:, t - 1] * Transition[:, n]
                                 * Emission[n, Observation[t]])
                B[n, t] = np.argmax(
                    V[:, t - 1] * Transition[:, n] * Emission[n, Observation[t]])

        path = [np.argmax(V[:, T - 1])]
        for i in range(T - 1, 0, -1):
            path.append(int(B[path[-1], i]))
        path = path[::-1]

        P = np.max(V[:, T - 1])

        return path, P
    except Exception:
        return None, None
