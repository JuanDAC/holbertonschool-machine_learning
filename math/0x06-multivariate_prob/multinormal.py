#!/usr/bin/env python3
""" 2. Initialize"""
import numpy as np


class MultiNormal:
    """
      Class MultiNormal that represents a Multivariate Normal distribution:
    """

    def __init__(self, data):
        """
        Properties of the class:
        - data is a numpy.ndarray of shape (d, n) containing the data set
        - n is the number of data points
        - d is the number of dimensions in each data point
        Validations:
        - If data is not a 2D numpy.ndarray, raise a TypeError
          with the message "data must be a 2D numpy.ndarray"
        - If n is less than 2, raise a ValueError with the
          message "data must contain multiple data points"
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul(
            data - self.mean, (data - self.mean).T) / (data.shape[1] - 1)
