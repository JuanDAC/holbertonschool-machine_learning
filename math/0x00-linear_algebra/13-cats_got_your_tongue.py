#!/usr/bin/env python3
""" Functions that can be used to concat two matrices"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concatenate the two matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
