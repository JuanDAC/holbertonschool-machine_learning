#!/usr/bin/env python3
"""Create a placeholders"""

import tensorflow as tf

try:
    tf.placeholder
except AttributeError:
    tf = tf.compat.v1


def create_placeholders(nx, classes):
    """
    ...
    Parameters
    ----------
    nx : int
        The number of feature columns in our data
    classes : int
        The number of classes in our classifier
    Returns
    _______
    placeholders : tuple 
        Placeholders named x and y, respectively
    """
    placeholders = (
        tf.placeholder(tf.float32, name="x", shape=(None, nx)),
        tf.placeholder(tf.float32, name="y", shape=(None, classes))
    )
    return placeholders
