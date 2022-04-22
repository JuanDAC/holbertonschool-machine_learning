#!/usr/bin/env python
""" This function is used to generate"""


def poly_derivative(poly):
    """Returns the derivative of the given poly with respect to the given point"""
    if type(poly) is not list:
        return None
    derivate = []
    for i in range(len(poly)):
        if i >= 1:
            derivate.append(poly[i] * i)
    return derivate
