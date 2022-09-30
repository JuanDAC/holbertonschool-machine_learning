#!/usr/bin/env python3
"""
File that contains the class LSTMCell
"""

import numpy as np


class LSTMCell:
    """
    Class LSTMCell that represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
            Converts the weights and biases to a dictionary to
            represent the weights and biases of the cell
            Arguments:
                - i: dimensionality of the data
                - h: dimensionality of the hidden state
                - o: dimensionality of the outputs
            Public instance attributes:
                - Wf: weights for the forget gate
                - Wu: weights for the update gate
                - Wc: weights for the intermediate cell state
                - Wo: weights for the output gate
                - Wy: weights for the output
                - bf: bias for the forget gate
                - bu: bias for the update gate
                - bc: bias for the intermediate cell state
                - bo: bias for the output gate
                - by: bias for the output
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Method that performs forward propagation for one time step
        Arguments:
            - h_prev: numpy.ndarray of shape (m, h) containing the
                        previous hidden state
            - x_t: numpy.ndarray of shape (m, i) that contains the
                    data input for the cell
                * m is the batche size for the data
            - c_prev: numpy.ndarray of shape (m, h) containing the
                        previous cell state
        Returns: h_next, c_next, y
            - h_next: the next hidden state
            - c_next: the next cell state
            - y: the output of the cell
        """
        # forget gate
        f = np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                             self.Wf) + self.bf) / \
            (np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                              self.Wf) + self.bf) + 1)
        # update gate
        u = np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                             self.Wu) + self.bu) / \
            (np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                              self.Wu) + self.bu) + 1)
        # intermediate cell state
        c = np.tanh(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                              self.Wc) + self.bc)
        # next cell state
        c_next = f * c_prev + u * c
        # output gate
        o = np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                             self.Wo) + self.bo) / \
            (np.exp(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                              self.Wo) + self.bo) + 1)
        # next hidden state
        h_next = o * np.tanh(c_next)
        # output
        y = np.matmul(h_next, self.Wy) + self.by
        return h_next, c_next, y
