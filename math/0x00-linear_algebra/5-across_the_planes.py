#!/usr/bin/env python3
""" function that add two matrices """


def matrix_shape(matrix):
    """ Returns the shape of the matrix"""
    if type(matrix) != list or not matrix:
        return []
    return [len(matrix), *matrix_shape(matrix[0])]


def add_arrays(arr1, arr2):
    """ add arrays of the same shape """
    if len(arr1) != len(arr2):
        return None
    return list(map(lambda x: sum(x), zip(arr1, arr2)))


def add_matrices2D(mat1, mat2):
    """ add_matrices2D """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    return list(map(lambda x: add_arrays(*x), zip(mat1, mat2)))


if __name__ == '__main__':
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))
    print(mat1)
    print(mat2)
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
