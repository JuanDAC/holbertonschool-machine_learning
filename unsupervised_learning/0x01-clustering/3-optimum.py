#!/usr/bin/env python3
"""
3 Optimize k
"""

import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing the minimum number of
            clusters to check for (inclusive)
    - kmax: positive integer containing the maximum number of
            clusters to check for (inclusive)
    - iterations: positive integer containing the maximum number
                  of iterations for K-means
    Returns:
    - results: list containing the outputs of K-means for each
               cluster size
    - d_vars: list containing the difference in variance from
              the smallest cluster size for each cluster size
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None

        if kmax is not None and type(kmin) != int or kmin <= 0:
            return None, None

        if kmax is None:
            kmax = X.shape[0]

        if type(kmax) != int or kmax <= 0:
            return None, None

        if type(iterations) != int or iterations <= 0:
            return None, None

        if not isinstance(kmin, int) or kmin < 1 or kmin >= X.shape[0]:
            return None, None

        kmeans = __import__('1-kmeans').kmeans
        variance = __import__('2-variance').variance

        results = []
        d_vars = []

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            var = variance(X, C)
            results.append((C, clss))
            d_vars.append(var)

        d_var = [0]
        for i in range(kmax - 1):
            derivate = d_vars[i] - d_vars[i + 1]
            d_var.append(derivate + d_var[i])

        return results, d_var

    except Exception:
        return None, None
