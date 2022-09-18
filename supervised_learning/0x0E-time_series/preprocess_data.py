#!/usr/bin/env python3
"""File that conatins teh function preprocess_data"""


def preprocess_data(data_frame):
    """
    Function that preprocesses raw data
    and remove the rows with empty values
    """
    columns = data_frame.columns
    index = 1 if data_frame[columns[1]] > 0 else 0
    new_data_frame = data_frame[index]
    return new_data_frame
