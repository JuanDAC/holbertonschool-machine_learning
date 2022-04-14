#!/usr/bin/env python3
""" fuctions for matrix shape"""

def matrix_shape(matrix):
    """ Returns the shape of the matrix"""
    if type(matrix) != list or not matrix:
        return []
    return [len(matrix), *matrix_shape(matrix[0])]


if __name__ == '__main__':
    mat0 = [1, 2, 3, 4]
    print(matrix_shape(mat0))
    mat1 = [[1, 2], [3, 4]]
    print(matrix_shape(mat1))
    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    print(matrix_shape(mat2))
