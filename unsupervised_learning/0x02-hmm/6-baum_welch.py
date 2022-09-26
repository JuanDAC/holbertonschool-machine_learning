#!/usr/bin/env python3
"""
File that contains the baum_welch function
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Function that performs the backward algorithm for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the
            hidden state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state
    Returns: P, B, or None, None on failure
        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
        probabilities
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j
    """
    try:
        T = Observation.shape[0]
        N, M = Emission.shape
        beta = np.zeros((N, T))
        beta[:, T - 1] = np.ones((N))

        for t in range(T - 2, -1, -1):
            for n in range(N):
                Transitions = Transition[n, :]
                Emissions = Emission[:, Observation[t + 1]]
                beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
        return P, beta
    except Exception:
        return None, None


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm for a hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation
            T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the
            hidden state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
        transition probabilities
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
        starting in a particular hidden state
    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward
        path probabilities
        F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
    """
    try:
        N = Transition.shape[0]

        T = Observation.shape[0]

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Recursion αt(j) == ∑Ni=1 αt−1(i)ai jbj(ot); 1≤j≤N,1<t≤T
        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1]
                                 * Emissions)

        # Termination P(O|λ) == ∑Ni=1 αT (i)
        P = np.sum(F[:, -1])
        return P, F
    except Exception:
        None, None


def maximization(Observations, Transition, Emission, Initial, xi, gamma):
    """
    Arguments:
        - Observations is a numpy.ndarray of shape (T,) that contains
                        the index of the observation
            * T is the number of observations
        - Transition is a numpy.ndarray of shape (M, M) containing the
                        initialized transition probabilities
            * Transition[i, j] is the probability of transitioning from
                                the hidden state i to j
            * M is the number of hidden states
        - Emission is a numpy.ndarray of shape (M, N) containing the
                    initialized emission probabilities
            * Emission[i, j] is the probability of emitting j given the
                                hidden state i
            * N is the number of output states
        - Initial a numpy.ndarray of shape (M, 1) containing the
                    initialized starting probabilities
        - xi is the numpy.ndarray of shape (M, M, T - 1) containing the
                xi statistics
        - gamma is the numpy.ndarray of shape (M, T) containing the
                gamma statistics
    Returns:
        - Transition, Emission
    """
    T = Observations.shape[0]
    M, N = Emission.shape

    for i in range(M):
        for j in range(M):
            Transition[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :T - 1])

    for i in range(M):
        for j in range(N):
            Emission[i, j] = np.sum(
                gamma[i, Observations == j]) / np.sum(gamma[i, :])

    return Transition, Emission


def expectation(Observations, Emission, Transition, Initial, alpha, beta):
    """
    Arguments:
        - Observations is a numpy.ndarray of shape (T,) that contains the index
                        of the observation
            * T is the number of observations
        - Transition is a numpy.ndarray of shape (M, M) containing the
                        initialized transition probabilities
            * Transition[i, j] is the probability of transitioning from the
                                hidden state i to j
            * M is the number of hidden states
        - Emission is a numpy.ndarray of shape (M, N) containing the initialized
                    emission probabilities
            * Emission[i, j] is the probability of emitting j given the hidden
                                state i
            * N is the number of output states
        - Initial a numpy.ndarray of shape (M, 1) containing the initialized
                    starting probabilities
        - iterations is the number of times expectation-maximization should be
                        performed
    Return:
        - xi, gamma
    """
    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = alpha[i, t] * Transition[i, j] * \
                        Emission[j, Observations[t + 1]] * \
                        beta[j, t + 1]

            xi[:, :, t] /= np.sum(xi[:, :, t])

        for t in range(T):
            for i in range(M):
                gamma[i, t] = np.sum(xi[i, :, t])

        return xi, gamma
    except Exception:
        return None, None


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for a hidden markov model
    Arguments:
        - Observations is a numpy.ndarray of shape (T,) that contains the index
                          of the observation
            * T is the number of observations
        - Transition is a numpy.ndarray of shape (M, M) containing the
                        initialized transition probabilities
            * Transition[i, j] is the probability of transitioning from the
                                hidden state i to j
            * M is the number of hidden states
        - Emission is a numpy.ndarray of shape (M, N) containing the initialized
                    emission probabilities
            * Emission[i, j] is the probability of emitting j given the hidden
                                state i
            * N is the number of output states
        - Initial a numpy.ndarray of shape (M, 1) containing the initialized
                    starting probabilities
        - iterations is the number of times expectation-maximization should be
                        performed
    Returns:
        - the converged Transition, Emission, or None, None on failure
    """
    try:
        if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
            return None, None

        if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
            return None, None

        if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
            return None, None

        if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
            return None, None

        if type(iterations) is not int or iterations < 1:
            return None, None

        T = Observations.shape[0]
        M, N = Emission.shape

        if Transition.shape[0] != M or Transition.shape[1] != M:
            return None, None

        if Initial.shape[0] != M or Initial.shape[1] != 1:
            return None, None

        for i in range(iterations):
            alpha, _ = forward(Observations, Emission, Transition, Initial)
            beta, _ = backward(Observations, Emission, Transition, Initial)
            xi, gamma = expectation(
                Observations, Emission, Transition, Initial, alpha, beta)
            Transition, Emission = maximization(
                Observations, Transition, Emission, Initial, xi, gamma)

        return Transition, Emission
    except Exception:
        return None, None
