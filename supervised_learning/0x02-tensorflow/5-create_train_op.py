#!/usr/bin/env python3
"""Accuracy of neural network"""


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """" Function to train a model with tensorflow
    ...
    Parameters
    __________
    loss : Tensor
        The loss of the network’s prediction
    alpha : Tensor
        The learning rate
    ...
    Return
    ______
    accuracy : Tensor
        Operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(learning_rate=alpha)\
        .minimize(loss)
