#!/usr/bin/env python3
"""
File Name: 2-rnn_decoder.py
"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class RNNcoder that inherits from tensorflow.keras.layers.Layer
    to decode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Function that initializes the hidden states for the RNN cell to a
        tensor of zeros
        Arguments:
          - vocab is an integer representing the size of the input vocabulary
          - embedding is an integer representing the dimensionality of the
            embedding vector
          - units is an integer representing the number of hidden units in
            the RNN cell
          - batch is an integer representing the batch size
        Returns:
          - A tensor of shape (batch, units) containing the initialized
            hidden states
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Method that builds the decoder for machine translation
        Arguments:
          - x is a tensor of shape (batch, input_seq_len)
            * containing the input to the decoder layer as word
            * indices within the target vocabulary
          - s_prev is a tensor of shape (batch, units)
            * containing the previous decoder hidden state
          - hidden_states is a tensor of shape (batch, input_seq_len, units)
            * containing the outputs of the encoder
        Returns:
          - y, s
        """
        context = SelfAttention(s_prev)(hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, s = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, s
