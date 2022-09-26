#!/usr/bin/env python3
"""
File: 6-multihead_attention.py
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer
    to perform multi head attention
    """

    def __init__(self, dm, h):
        """
        Constructor that creates the following layers
        Arguments:
          - dm is an integer representing the dimensionality of the model
          - h is an integer representing the number of heads
        Public instance attributes:
          - h: the number of heads
          - dm: the dimensionality of the model
          - depth: the depth of each attention head
          - Wq: a Dense layer with dm units, used to generate the query matrix
          - Wk: a Dense layer with dm units, used to generate the key matrix
          - Wv: a Dense layer with dm units, used to generate the value matrix
          - linear: a Dense layer with dm units, used to generate the
            attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Method that splits the last dimension of x into (h, depth)
        Arguments:
          - x is a tensor with shape (batch_size, seq_len, dm) containing
          - batch_size is an integer representing the batch size
        Returns:
          - a tensor with shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Method that splits the heads of shape (batch, seq_len_q, dm) into
        multiple heads of shape (batch, seq_len_q, h, depth)
        Arguments:
          - Q is a tensor of shape (batch, seq_len_q, dk) containing the
            input to generate the query matrix
          - K is a tensor of shape (batch, seq_len_v, dk) containing the
            input to generate the key matrix
          - V is a tensor of shape (batch, seq_len_v, dv) containing the
            input to generate the value matrix
          - mask is always None
        Returns:
          - output, weights
        """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights
