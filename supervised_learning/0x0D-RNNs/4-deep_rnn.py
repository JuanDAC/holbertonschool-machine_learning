#!/usr/bin/env python3
"""
File that contains the class LSTMCell
"""

import numpy as np


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
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    for t in range(t):
        for layer in range(l):
            if layer == 0:
                H[t + 1,
                    layer], _ = rnn_cells[layer].forward(H[t, layer], X[t])
            else:
                H[t + 1, layer], _ = rnn_cells[layer].forward(
                    H[t, layer], H[t + 1, layer - 1])
    Y = np.zeros((t, m, rnn_cells[-1].by.shape[1]))
    for t in range(t):
        Y[t] = rnn_cells[-1].forward(H[t + 1, -1], H[t + 1, -2])[1]
    return H, Y
