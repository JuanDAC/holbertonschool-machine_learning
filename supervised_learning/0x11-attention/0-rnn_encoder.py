#!/usr/bin/env python3
"""
File contains RNNDecoder Class
"""

import numpy as np
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):

    """
    Class RNNDecoder that inherits from tensorflow.keras.layers.Layer to
    decode for machine translation:
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        Arguments:
          - vocab is an integer representing the size of the input
            vocabulary
          - embedding is an integer representing the dimensionality
            of the embedding vector
          - units is an integer representing the number of hidden
            units in the RNN cell
          - batch is an integer representing the batch size
        Public instance attributes:
          - batch - the batch size
          - units - the number of hidden units in the RNN cell
          - embedding - a keras Embedding layer that converts
            words from the vocabulary into an embedding vector
          - gru - a keras GRU layer with units units
            * Should return both the full sequence of outputs as
              well as the last hidden state
            * Recurrent weights should be initialized with
              glorot_uniform
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
