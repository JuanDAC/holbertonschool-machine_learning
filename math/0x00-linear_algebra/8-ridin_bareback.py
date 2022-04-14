#!/usr/bin/env python3
""" multipication for the multipication module """


def matrix_shape(matrix):
    """ Returns the shape of the matrix"""
    if type(matrix) != list or not matrix:
        return []
    return [len(matrix), *matrix_shape(matrix[0])]


def rotate(mat):
    """ rotate a matrix"""
    return list(zip(*mat))


def reduce(callback, until, initial=0):
    """ reduce a function to a list of functions"""
    if until <= 0:
        return initial
    return callback(until - 1) + reduce(callback, until - 1)


def mat_mul(mat1, mat2):
    """ Multiplies two matrices and returns a new matrix that contains"""
    mat = rotate(mat2)
    if matrix_shape(mat1)[1] != matrix_shape(mat2)[0]:
        return None
    product = []
    height = len(mat1)
    for i in range(height):
        row_product = []
        width = len(mat)
        for j in range(width):
            sum = reduce(lambda k: mat1[i][k] * mat[j][k], len(mat1[i]))
            row_product.append(sum)
        product.append(row_product)
    return product


if __name__ == '__main__':
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))
