#!/usr/bin/env python3
"""
File: 7-transformer_encoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Create a class EncoderBlock that inherits from
    tensorflow.keras.layers.Layer to create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input to
        build the model
        Arguments:
          - dm is an integer representing the dimensionality of the model
          - h is an integer representing the number of heads
          - hidden is the number of hidden units in the fully connected layer
          - drop_rate is the dropout rate
        Public instance attributes:
          - mha: a MultiHeadAttention layer
          - dense_hidden: the hidden dense layer with hidden units and relu
                          activation
          - dense_output: the output dense layer with dm units
          - layernorm1: the first layer norm layer, with epsilon=1e-6
          - layernorm2: the second layer norm layer, with epsilon=1e-6
          - dropout1: the first dropout layer
          - dropout2: the second dropout layer
        """
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        super(EncoderBlock, self).__init__()

    def call(self, x, training, mask=None):
        """
        Method call should use the following masks:
        Arguments:
          - x is a tensor of shape (batch, input_seq_len, dm) containing
              the input to the encoder block
          - training is a boolean to determine if the model is training
          - mask is the mask to be applied for multi head attention
        Returns:
          - A tensor of shape (batch, input_seq_len, dm) containing the
            blocks output
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)
        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)
        output = self.dropout2(output, training=training)
        out2 = self.layernorm2(out1 + output)
        return out2
