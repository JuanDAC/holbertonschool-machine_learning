#!/usr/bin/env python3
"""
7. Maximization
"""


import numpy as np


def maximization(X, g):
    """Calculate maximization step for GMM"""
    if ((type(X) is not np.ndarray or type(g) is not np.ndarray
         or X.ndim != 2 or g.ndim != 2 or X.shape[0] != g.shape[1]
         or not np.all(np.isclose(g.sum(axis=0), 1)))):
        return None, None, None
    try:
        gsum = g.sum(axis=1)
        pi = gsum / X.shape[0]
        m = np.matmul(g, X) / gsum[:, np.newaxis]
        S = np.ndarray((m.shape[0], m.shape[1], m.shape[1]))
        for cluster in range(g.shape[0]):
            diff = X - m[cluster]
            S[cluster] = (np.matmul((diff * g[cluster, :, np.newaxis]).T, diff)
                          / gsum[cluster])
        return pi, m, S
    except Exception:
        return None, None, None
