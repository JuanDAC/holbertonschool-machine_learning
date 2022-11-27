"""
File
"""

import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray
    Arguments:
        - array is the np.ndarray from which you should create the pd.DataFrame
            * The columns of the pd.DataFrame should be labeled in alphabetical
            order and capitalized. There will not be more than 26 columns.
    Returns:
        - The newly created pd.DataFrame
    """
    columns = [chr(i) for i in range(ord('A'), ord('A') + array.shape[1])]
    return pd.DataFrame(array, columns=columns)
