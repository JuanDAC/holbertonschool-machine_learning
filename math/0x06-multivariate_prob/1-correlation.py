#!/usr/bin/env python3
""" 1. Correlation of a matrix """
import numpy as np


def correlation(C):
    """
    Calculates the correlation of a matrix C where C is a numpy.ndarray
    and its rows are variables and its columns are observations

    Validations:
    - If C is not a numpy.ndarray, raise a TypeError with the message
      "C must be a numpy.ndarray"

    - If C does not have shape (d, d), raise a ValueError with the message
      "C must be a 2D square matrix"

    Returns:
    - A numpy.ndarray of shape (d, d) containing the correlation matrix
      with the name correlation

    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    depends = np.diag(C)
    depends_dimensions_increased = np.expand_dims(depends, axis=0)
    standard_x = np.sqrt(depends_dimensions_increased)
    standard_product = np.dot(standard_x.T, standard_x)
    correlation = C / standard_product

    return correlation
