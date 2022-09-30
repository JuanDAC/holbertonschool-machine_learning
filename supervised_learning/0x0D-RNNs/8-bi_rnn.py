#!/usr/bin/env python3
"""
File that contains the  bidirectional RNN 
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Function that performs forward propagation
    for a bidirectional RNN
    Arguments:
        - bi_cell is an instance of BidirectinalCell that will
                  be used for the forward propagation
         X is the data to be used, given as a numpy.ndarray
           of shape (t, m, i)
            * t is the maximum number of time steps
            * m is the batch size
            * i is the dimensionality of the data
        - h_0 is the initial hidden state in the forward direction,
                 given as a numpy.ndarray of shape (m, h)
            * h is the dimensionality of the hidden state
        - h_t is the initial hidden state in the backward direction,
              given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
        - H is a numpy.ndarray containing all of the concatenated hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.zeros((t, m, 2 * h))
    Y = []
    h_next = h_0
    for t in range(t):
        h_next, y = bi_cell.forward(h_next, X[t])
        H[t] = h_next
        Y.append(y)
    h_next = h_t
    for t in range(t - 1, -1, -1):
        h_next, y = bi_cell.backward(h_next, X[t])
        H[t] += h_next
        Y[t] += y
    Y = np.array(Y)
    return H, Y
