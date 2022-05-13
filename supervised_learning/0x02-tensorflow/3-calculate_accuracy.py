#!/usr/bin/env python3
"""Accuracy of neural network"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ Calculates the accuracy of a prediction
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
        Prediction accuracy
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
