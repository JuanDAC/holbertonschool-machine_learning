#!/usr/bin/env python3
"""One hot decoder"""


import numpy as np


def one_hot_decode(one_hot):
    """Devode One hot"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        None
