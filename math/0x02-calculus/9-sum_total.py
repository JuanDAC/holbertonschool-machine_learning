#!/usr/bin/env python3
"""
Function that can be used summation of the given number of squared
"""


def summation_i_squared(n):
    """ Returns the summation of the given number of squared"""
    if n < 1 or n is None:
        return None
    from functools import reduce
    return reduce(lambda sum, x: sum + (x ** 2), range(1, n + 1))


if __name__ == '__main__':
    n = 5
    print(summation_i_squared(n))
    print(summation_i_squared(0))
    print(summation_i_squared(1))
    print(summation_i_squared(10))
