#!/usr/bin/env python3
"""Function of mean and covariance"""
import numpy as np


def mean_cov(X):
    """
    This function calculates the mean and covariance of X matrix data
    where X is a numpy.ndarray of shape (n, d) containing the data set:

    Args:
        [X]: numpy.ndarray of shape (n, d) containing the data set

    Validations:
    - If the attrubute X is not a 2D numpy.ndarray, this function should raise 
    a TypeError with the message "X must be a 2D numpy.ndarray"

    - If n is less than 2, raise a ValueError with the message 
    "X must contain multiple data points"

    Returns:
      - [mean] numpy.ndarray of shape (1, d) containing the mean of the data set
      - [cov]  numpy.ndarray of shape (d, d) containing the covariance matrix of the data set 
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    dimentions_of_points = X.shape[1]
    mean = np.mean(X, axis=0).reshape(1, dimentions_of_points)
    points = (X.shape[0] - 1)
    X_mean = X - mean
    cov = np.matmul(X_mean.T, X_mean) / points

    return mean, cov
