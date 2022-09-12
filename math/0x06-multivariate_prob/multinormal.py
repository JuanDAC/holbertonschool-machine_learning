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

        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True).reshape(d, 1)
        self.cov = np.matmul(
            data - self.mean, (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """
        Description:
          Calculates the PDF at a data point

        Validation:
          - If x is not a numpy.ndarray, raise a TypeError
            with the message "x must be a numpy.ndarray"

          - If x is not of shape (d, 1), raise a ValueError
            with the message "x must have the shape ({d}, 1)"

        Returns:
          - The value of the PDF
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError(
                "x must have the shape ({}, 1)".format(self.cov.shape[0]))

        x_mean = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)
        pdf = (1 / np.sqrt(((2 * np.pi) ** self.cov.shape[0]) * cov_det)) * np.exp(
            -0.5 * np.matmul(np.matmul(x_mean.T, cov_inv), x_mean))
        return pdf[0][0]
