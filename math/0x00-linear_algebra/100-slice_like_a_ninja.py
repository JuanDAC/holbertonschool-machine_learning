#!/usr/bin/env python3

import numpy as np


def np_slice(matrix, axes={}, first=True):
    for axis, slice in axes.items():
        if axis == 0:
            if len(slice) == 1:
                slice = (0, ) + slice
            print("--------------------------------")
            print(matrix)
            print("--------------------------------")
            print(matrix[slice[0]:slice[1]])
            print("--------------------------------")
            del axes[0]
        else:
            axes[axis - 1] = axes[axis]
            del axes[axis]
    if axis > 0:
        [np_slice(value, axes=axes) for _, value in enumerate(matrix)]


if __name__ == '__main__':
    """
    mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print(np_slice(mat1, axes={1: (1, 3)}))
    print(mat1) """
    mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                    [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                    [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
    print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
    print(mat2)
