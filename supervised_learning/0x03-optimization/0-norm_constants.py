#!/usr/bin/env python3
"""Normalization constants"""
import numpy as np


def normalization_constants(X):
    """
        calculates the normalization (standartization)
        constants of a matrix.
        ...
        Parameters
        ----------
            X: numpy.ndarray - matrix to normalize
        ...
        Returns:
        ----------
            The mean and standard deviation
            of each feature respectively.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
