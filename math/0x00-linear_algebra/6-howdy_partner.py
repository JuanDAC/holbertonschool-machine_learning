#!/usr/bin/env python3
""" functions for the cat array """


def cat_arrays(arr1, arr2):
    """ function for cat arrays"""
    if type(arr1) is not list and type(arr2) is not list:
        return None
    return [*arr1, *arr2]


if __name__ == '__main__':
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    print(cat_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
