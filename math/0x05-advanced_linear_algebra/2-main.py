#!/usr/bin/env python3


if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat = [[10, 4, 7, 3, -9],
           [-2, 8, 3, -5, 6],
           [5, 19, 6, 1, 25],
           [7, -30, 21, 4, -1],
           [8, 9, -10, 2, -3]]
    print(cofactor(mat))
    mat = [[5, 11, 6, 3, -20],
           [1, -9, 13, 8, 5],
           [2, 22, 4, 7, -6],
           [-10, 3, 7, -1, 9],
           [4, 8, -2, 10, 12]]
    print(cofactor(mat))
