#!/usr/bin/env python3
"""
File from numpy
"""

import numpy as np
import pandas as pd


def from_file(filename, delimiter):
    """
    Function that loads data from a file as a pd.DataFrame:
    Arguments:
        - filename is the file to load from
        - delimiter is the column separator
    Returns:
        - The newly created pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
