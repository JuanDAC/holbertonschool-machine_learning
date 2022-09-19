#!/usr/bin/env python3
"""
File that contains the function posterior continuous
"""

import numpy as np


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects
    Arguments:
     - x is the number of patients that develop severe side effects
     - n is the total number of patients observed
     - P is a 1D numpy.ndarray containing the various hypothetical
       probabilities of developing severe side effects
    Returns:
     - a 1D numpy.ndarray containing the likelihood of obtaining the data,
       x and n, for each probability in P, respectively
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(x) != int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    fact = np.math.factorial(n) / (np.math.factorial(x)
                                   * np.math.factorial(n - x))

    likelihood = fact * (P ** x) * ((1 - P) ** (n - x))

    return likelihood


def intersection(x, n, P, Pr):
    """
    Function that calculates the intersection of obtaining this data with the
    various hypothetical probabilities
    Arguments:
     - x is the number of patients that develop severe side effects
     - n is the total number of patients observed
     - P is a 1D numpy.ndarray containing the various hypothetical
       probabilities of developing severe side effects
     - Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
     - a 1D numpy.ndarray containing the intersection of obtaining x and n
       with each probability in P, respectively
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(x) != int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    intersection = likelihood(x, n, P) * Pr

    return intersection


def marginal(x, n, P, Pr):
    """
    Function that calculates the marginal probability of obtaining the data
    Arguments:
     - x is the number of patients that develop severe side effects
     - n is the total number of patients observed
     - P is a 1D numpy.ndarray containing the various hypothetical
       probabilities of patients developing severe side effects
     - Pr is a 1D numpy.ndarray containing the prior beliefs about P
    Returns:
     - the marginal probability of obtaining x and n
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(x) != int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    marginal = np.sum(intersection(x, n, P, Pr))

    return marginal


def posterior(x, n, p1, p2):
    """
    Function that calculates the posterior probability that the
    probability of developing severe side effects falls within
    a specific range given the data
    Arguments:
     - x is the number of patients that develop severe side effects
     - n is the total number of patients observed
     - p1 is the lower bound on the range
     - p2 is the upper bound on the range
    Returns:
     - the posterior probability that p is within the range [p1, p2]
       given x and n
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(x) != int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if type(p1) != float or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')

    if type(p2) != float or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')

    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    P = np.linspace(0, 1, 101)
    Pr = np.ones(P.shape) / len(P)

    posterior = np.sum(intersection(x, n, P, Pr) * (P >= p1) * (P <= p2))

    return posterior
