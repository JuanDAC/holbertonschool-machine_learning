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
        - H is a numpy.ndarray containing all of the concatenated
            hidden states
        - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    H = np.zeros((t + 1, m, 2 * h))
    Y = []
    Hf[0] = h_0
    Hb[t] = h_t
    for step in range(t):
        Hf[step + 1], _ = bi_cell.forward(Hf[step], X[step])
        Hb[t - step - 1], _ = bi_cell.backward(Hb[t - step], X[t - step - 1])
        H[step + 1] = np.concatenate((Hf[step + 1], Hb[t - step - 1]), axis=1)
    Y = bi_cell.output(H)
    return H, Y
