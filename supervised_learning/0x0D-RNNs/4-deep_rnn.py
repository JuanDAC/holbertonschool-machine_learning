#!/usr/bin/env python3
"""
File that contains the deep RNN
"""

import numpy as np


def update_hidden(rnn_cells, ly, H, i, X):
    """
    Function that updates the hidden state in a deep RNN
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
    if ly == 0:
        # Update next hidden state, compute the prediction
        h_next, y_pred = rnn_cells[ly].forward(H[i, ly], X[i])
        return h_next, y_pred
    # Update next hidden state, compute the prediction
    h_next, y_pred = rnn_cells[ly].forward(H[i, ly], h_next)
    return h_next, y_pred


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
    # dimensions shape aka, time steps
    t, m, i = X.shape
    _, _, h = h_0.shape

    layers = len(rnn_cells)

    # initialize H and Y
    H = np.zeros((t+1, layers, m, h))
    H[0] = h_0
    Y = []

    # loop over time ste
    for i in range(t):
        for ly in range(layers):
            h_next, y_pred = update_hidden(rnn_cells, ly, H, i, X)
            # Save the value of the new "next" hidden state
            H[i+1, ly] = h_next
        # Store values of the prediction
        Y.append(y_pred)
    Y = np.array(Y)
    return H, Y
