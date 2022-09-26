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

    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape
    _, d1 = m.shape
    d2, _ = S.shape

    if d != d1 or d != d2:
        return None

    sigma_1 = np.linalg.det(S) ** (1/2)

    sigma_2 = np.linalg.inv(S)
    operation_1 = np.matmul((X - m), sigma_2)
    operation_2 = np.sum((X - m) * operation_1, axis=1)

    equation_1 = 1/(pi * sigma_1)
    equation_2 = np.exp((-1/2) * operation_2)

    PDF = equation_1 * equation_2
    P = np.squeeze(PDF)

    P = np.where(P < 1e-300, 1e-300, P)

    return P
