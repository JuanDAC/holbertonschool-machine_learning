#!/usr/bin/env python3
"""
5 PDF
"""

import numpy as np


def pdf(X, m, S):
    """
    Arguments:
    - X: numpy.ndarray of shape (n, d) containing the
         data points whose PDF should be evaluated
    - m: numpy.ndarray of shape (d,) containing the
         mean of the distribution
    - S: numpy.ndarray of shape (d, d) containing the
         covariance of the distribution
    Returns:
    - P: numpy.ndarray of shape (n,) containing the PDF
         values for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    _, d1 = m.shape
    d2, _ = S.shape

    if d != d1 or d != d2:
        return None

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    fac = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    exp = np.exp(-0.5 * np.einsum('...k,kl,...l->...', X - m, inv, X - m))
    P = fac * exp

    P = np.where(P < 1e-300, 1e-300, P)
    return P
