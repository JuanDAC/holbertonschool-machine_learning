#!/usr/bin/env python3
""" This function is used to generate"""


def poly_integral(poly, C=0, power=1):
    """ This function is used to generate poly integral function """
    if type(poly) is not list or type(C) not in (int, float):
        return None
    if len(poly) == 0 and power == 1:
        return None

    coefficient, *poly = [*poly, None]

    integrals = [C] if power == 1 else []

    if coefficient is None:
        return []

    if coefficient == 0 and len(integrals) == 1:
        return [*integrals, *poly_integral(poly, C, power + 1)]

    if coefficient == 0:
        return [*integrals, 0, *poly_integral(poly, C, power + 1)]

    if power != 0:
        result = (coefficient / power)
        result = result if round(result) - result != 0 else round(result)
        return [*integrals, result, *poly_integral(poly, C, power + 1)]

    return [*integrals, coefficient, *poly_integral(poly, C, power + 1)]


if __name__ == "__main__":
    """
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
    print(poly_integral([7, 4, 6, 1, 5]))
    print(poly_integral([4, 8, 2, 4, 7, 1, 9], C=5))
    print(poly_integral([0]))
    print(poly_integral([0], C=7))
    print(poly_integral(5))
    print(poly_integral([]))
    print(poly_integral([5], C=None))
    """
    print(poly_integral([6, 5, 0, 0, -3]))
    print(poly_integral([0, 0, 3, -1, 0, 6, 2], C=6))
