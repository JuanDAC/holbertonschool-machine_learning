#!/usr/bin/env python3
"""
File that contains the class LSTMCell
"""

import numpy as np


def deep_rnn_cell(rnn_cells, x_t, h_prev):
    """
    Function that performs forward propagation for one time step for
    a deep RNN
    Arguments:
        - rnn_cells is a list of RNNCell instances of length l that will
                    be used for the forward propagation
            * l is the number of layers
        - x_t is the data input for the cell
        - h_prev is a numpy.ndarray of shape (l, m, h) containing the
                previous hidden states
    Returns: h_next, y
        - h_next is a numpy.ndarray of shape (l, m, h) containing the
                next hidden states
        - y is a numpy.ndarray of shape (l, m, o) containing the outputs
    """
    l, m, h = h_prev.shape
    h_next = np.zeros((l, m, h))
    y = np.zeros((l, m, rnn_cells[-1].by.shape[1]))
    for layer in range(l):
        if layer == 0:
            h_next[layer], y[layer] = rnn_cells[layer].forward(
                h_prev[layer],
                x_t)
        else:
            h_next[layer], y[layer] = rnn_cells[layer].forward(
                h_prev[layer],
                h_next[layer - 1])
    return h_next, y


def deep_rnn(rnn_cells, X, h_0):
    """
    Function that performs forward propagation for a deep RNN
    Arguments:
        - rnn_cells is a list of RNNCell instances of length l that will be
                    used for the forward propagation
            * l is the number of layers
        - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            * t is the maximum number:w
             of time steps
            * m is the batch size
                i is the dimensionality of the data
        - h_0 is the initial hidden state, given as a numpy.ndarray of shape
                (l, m, h)
            h is the dimensionality of the hidden state
    Returns: H, Y
        - H is a numpy.ndarray containing all of the hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((l, t + 1, m, h))
    Y = np.zeros((l, t, m, rnn_cells[-1].by.shape[1]))
    H[:, 0] = h_0
    for t in range(t):
        H[:, t + 1, :, :], Y[:, t, :, :] = deep_rnn_cell(rnn_cells, X[t],
                                                         H[:, t])
    return H, Y
