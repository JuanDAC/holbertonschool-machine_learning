#!/usr/bin/env python3
""" This function is used to generate"""


def poly_derivative(poly):
    """Returns the derivative of the given poly with respect to the given point"""
    if type(poly) is not list and len(poly) == 0:
        return None
    derivate = []
    for i in range(len(poly)):
        if i >= 1:
            derivate.append(poly[i] * i)
    return derivate
