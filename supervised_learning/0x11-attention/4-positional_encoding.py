#!/usr/bin/env python3
"""
File that contains the positional_encoding function
"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
    Arguments:
      - max_seq_len is an integer representing the maximum sequence length
      - dm is the model depth
    Returns:
      - a numpy.ndarray of shape (max_seq_len, dm) containing the positional
          encoding vectors
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((max_seq_len, dm))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    return pos_encoding
