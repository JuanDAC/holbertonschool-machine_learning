#!/usr/bin/env python3
""" functions of add array """


def add_arrays(arr1, arr2):
    """ add arrays of the same shape """
    if len(arr1) != len(arr2):
        return None
    return list(map(lambda x: sum(x), zip(arr1, arr2)))


if __name__ == '__main__':
    arr1 = [1, 2, 3, 4]
    arr2 = [5, 6, 7, 8]
    print(add_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
    print(add_arrays(arr1, [1, 2, 3]))
