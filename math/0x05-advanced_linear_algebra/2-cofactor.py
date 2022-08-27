#!/usr/bin/env python3
""" minor of matrix """


def reduce(callback, list_of_lists):
    """ reduce(callback, list_of_lists) returns the result of the
    callback function applied to the list of lists """
    current = list_of_lists[0]

    for i in range(1, len(list_of_lists)):
        current = callback(current, list_of_lists[i])
    return current


def validation_of_types(matrix):
    """ validation_of_types(matrix) returns True if matrix
    is not a list of lists """
    return type(matrix) != list or not len(matrix) or reduce(
        (lambda acum, l:  acum or type(l) != list),
        [False, *matrix])


def transpose(matrix):
    """ Return the transposed matrix"""
    return list(map(lambda tupla: list(tupla), zip(*matrix)))


def get_sign(x):
    """ get_sign(i, j) returns the sign of the determinant """
    return (-1) ** (x)


def shape(matrix=[]):
    """ shape of matrix """
    if not matrix or type(matrix) != list:
        return (0, 0, 0)
    return (len(matrix), *shape(matrix[0]))


def mirror(matrix, x, y):
    """ mirror(matrix, x) returns a matrix with the xth row removed """
    return [
        [*matrix[i][:y], *matrix[i][y + 1:]]
        for i in range(len(matrix))
        if i != x
    ]


def all_shapes_equals(matrix):
    """ shapes of sub matrix """
    base_len = len(matrix)
    return reduce(
        (lambda acum, l: acum and base_len == len(l)),
        [True, *matrix]
    )


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
        * determinant(mirror(matrix, i, 0)) for i in range(first_dimention)
    )


def determinant_minor(matrix, x, y):
    """ determinat minor """
    mirror_of_matrix = mirror(matrix, x, y)
    if len(mirror_of_matrix) > 2:
        return determinant(mirror_of_matrix)

    return mirror_of_matrix[0][0] * mirror_of_matrix[1][1] \
        - mirror_of_matrix[0][1] * mirror_of_matrix[1][0]


def cofactor(matrix):
    """ minor of a matrix """
    if validation_of_types(matrix):
        raise TypeError("matrix must be a list of lists")

    if not all_shapes_equals(matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    first_dimention, second_dimention, *_ = shape(matrix)

    if first_dimention == 1 and second_dimention == 1:
        return [[1]]

    if first_dimention == 2 and second_dimention == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]

    return transpose([
        [
            determinant_minor(matrix, x, y) * get_sign(x + y)
            for x in range(second_dimention)
        ]
        for y in range(first_dimention)
    ])
