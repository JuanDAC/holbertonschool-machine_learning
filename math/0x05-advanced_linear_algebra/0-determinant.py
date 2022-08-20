#!/usr/bin/env python3
""" determinant.py - determinant of a matrix """


def reduce(callback, list_of_lists):
    """ reduce(callback, list_of_lists) returns the result of the
    callback function applied to the list of lists """
    current = list_of_lists[0]
    for i in range(1, len(list_of_lists)):
        current = callback(current, list_of_lists[i])
    return current


def mirror(matrix, x):
    """ mirror(matrix, x) returns a matrix with the xth row removed """
    return [matrix[i][1:] for i in range(len(matrix)) if i != x]


def shape(matrix=[]):
    """ shape(matrix) returns the shape of a matrix """
    if not matrix or type(matrix) != list:
        return (0, 0, 0)

    return (len(matrix), *shape(matrix[0]))


def get_sign(x):
    """ get_sign(i, j) returns the sign of the determinant """
    return (-1) ** (x)


def validation_of_types(matrix):
    """ validation_of_types(matrix) returns True if matrix
    is not a list of lists """
    return type(matrix) != list or reduce(
        (lambda acum, l:  acum or type(l) != list),
        [False, *matrix])


def determinant(matrix):
    """ determinant(matrix) returns the determinant of a matrix """

    if validation_of_types(matrix):
        raise TypeError("matrix must be a list of lists")

    first_dimention, second_dimention, *_ = shape(matrix)

    if (first_dimention == 1 and second_dimention == 1):
        return matrix[0][0]

    if (first_dimention == 1 and second_dimention == 0):
        return 1

    if first_dimention != second_dimention:
        raise ValueError("matrix must be a square matrix")

    if first_dimention == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    vector = list(zip(*matrix))[0]

    return sum(
        vector[i]
        * get_sign(i)
        * determinant(mirror(matrix, i)) for i in range(first_dimention)
    )
