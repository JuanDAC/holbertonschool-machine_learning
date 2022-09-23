#!/usr/bin/env python3
"""
File: 3-gensim_to_keras.py
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Class used to create an attention block for machine translation """

    def __init__(self, units):
        """
        Arguments:
          - units is an integer representing the number of
            hidden units in the alignment model
        Public instance attributes:
          - W is a Dense layer with units units, to be applied
              to the previous decoder hidden state
          - U is a Dense layer with units units, to be applied
              to the encoder hidden states
          - V is a Dense layer with 1 units, to be applied to
              the tanh of the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Method that calculates the attention for machine translation
        Arguments:
          - s_prev is a tensor of shape (batch, units) containing
              the previous decoder hidden state
          - hidden_states is a tensor of shape (batch, input_seq_len, units)
              containing the outputs of the encoder
        Returns:
          - context, weights
        """
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
