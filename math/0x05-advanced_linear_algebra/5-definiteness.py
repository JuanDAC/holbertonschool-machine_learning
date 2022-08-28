#!/usr/bin/env python3
""" functions the matrix multiplication of two matrices with numpy """

import numpy as np


def reduce(callback, list_of_lists):
    """ reduce(callback, list_of_lists) returns the result of the
    callback function applied to the list of lists """
    current = list_of_lists[0]

    for i in range(1, len(list_of_lists)):
        current = callback(current, list_of_lists[i])
    return current


def invalid_shape(matrix):
    """ Returns True if matrix is not a list of lists """
    return len(matrix.shape) != 2 \
        or matrix.shape[0] != matrix.shape[1] \
        or np.array_equal(matrix, matrix.T) is False


def extract_property_eigenvalues(properties, value):
    """ extract_property_eigenvalues(properties, value) """
    positive, negative, zero = properties
    return (
        positive + 1 if value > 0 else positive,
        negative + 1 if value < 0 else negative,
        zero + 1 if value == 0 or value == 0. else zero
    )


def properties_eigenvalues(matrix):
    """
    properties_eigenvalues(matrix) returns the
    properties of the eigenvalues
    """
    eigenvalues, *_ = np.linalg.eig(matrix)
    return reduce(
        extract_property_eigenvalues, [(0, 0, 0), *eigenvalues]
    )


def definiteness(matrix):
    """ Returns the definiteness of a matrix """

    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    if invalid_shape(matrix):
        return None

    positive, negative, zero = properties_eigenvalues(matrix)

    if positive and zero and negative == 0:
        return "Positive semi-definite"

    if negative and zero and positive == 0:
        return "Negative semi-definite"

    if positive and negative == 0:
        return "Positive definite"

    if negative and positive == 0:
        return "Negative definite"

    return "Indefinite"
