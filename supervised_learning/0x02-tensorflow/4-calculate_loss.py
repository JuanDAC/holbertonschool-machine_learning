#!/usr/bin/env python3
"""Accuracy of neural network"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ Function to calculate cross-entropy loss using tensorflow
    ...
    Parameters
    __________
    y : Tensor
        Placeholder for the labels of the input data
    y_pred : Tensor
        Tensor containing the network’s predictions
    ...
    Return
    ______
    accuracy : Tensor
        calculate loss
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
