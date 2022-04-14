#!/usr/bin/env python3
""" concat matrices """


from operator import le
from pickle import TRUE


def is_list(matrix):
    """ returns True if the matrix is a list """
    return type(matrix) is list


def matrix_shape(matrix):
    """ Returns the shape of the matrix"""
    if type(matrix) != list or not matrix:
        return []
    return [len(matrix), *matrix_shape(matrix[0])]


def equal_without(matrix1, matrix2, without=0, index=0):
    """Returns true if the two matrices are equal without any index"""
    if without != -1:
        del matrix1[without]
        del matrix2[without]
        without = -1

    if index >= len(matrix1):
        return True

    if matrix1[index] != matrix2[index]:
        return False

    return equal_without(matrix1, matrix2, without, index + 1)


def cat_matrices2D(mat1, mat2, axis=0, firts=True):
    """ Concat matrices to a single matrix """
    if not is_list(mat1) and not is_list(mat2):
        return None
    if firts and not equal_without(matrix_shape(mat1), matrix_shape(mat2), axis):
        return None
    if (axis == 0):
        return [*mat1, *mat2]
    result = list(range(len(mat1)))
    for i in range(len(mat1)):
        result[i] = cat_matrices2D(mat1[i], mat2[i], axis - 1, False)
    return result


if __name__ == '__main__':
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]
    mat5 = cat_matrices2D(mat1, mat3, axis=1)
    print(mat5)
    mat4 = cat_matrices2D(mat1, mat2)
    print(mat4)
    mat1[0] = [9, 10]
    mat1[1].append(5)
    print(mat1)
    print(mat4)
    print(mat5)

    m1 = [[4, -7, 56, 2], [5, 106, 7, 2]]
    m2 = [[2, -6, 3], [0, -6, 3]]
    m = cat_matrices2D(m1, m2)
    print(m)
    m1 = [[484, 247], [554, 16], [5, 88]]
    m2 = [[233, -644, 325], [406, -16, 33], [765, 34, -39]]
    m = cat_matrices2D(m1, m2, axis=0)
    print(m)
    m1 = [[-54, -87, 95], [54, 16, -72]]
    m2 = [[12, 63, 79], [-10, 69, -9], [76, 45, -11]]
    m = cat_matrices2D(m1, m2, axis=1)
    print(m)
