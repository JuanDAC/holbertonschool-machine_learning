import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
    Arguments:
      - max_seq_len is an integer representing the maximum sequence length
      - dm is the model depth
    Returns:
      - a numpy.ndarray of shape (max_seq_len, dm) containing the positional
          encoding vectors
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((max_seq_len, dm))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    return pos_encoding


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention
    Arguments:
      - Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
          containing the query matrix
      - K is a tensor with its last two dimensions as (..., seq_len_v, dk)
          containing the key matrix
      - V is a tensor with its last two dimensions as (..., seq_len_v, dv)
          containing the value matrix
      - mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
          containing the optional mask, or defaulted to None
    Returns:
      - output, weights
    """
    # (..., seq_len_q, dk) * (..., dk, seq_len_v) = (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_v) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_v)

    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, dv)

    return output, attention_weights


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


class DecoderBlock(tf.keras.layers.Layer):
    """
    DecoderBlock class to create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Constructor that creates a decoder block for a transformer
        Arguments:
          - dm is an integer representing the dimensionality of the model
          - h is an integer representing the number of heads
          - hidden is the number of hidden units in the fully connected layer
          - drop_rate is the dropout rate
        Public instance attributes:
          - mha1: the first MultiHeadAttention layer
          - mha2: the second MultiHeadAttention layer
          - dense_hidden: the hidden dense layer with hidden units and relu
          - dense_output: the output dense layer with dm units
          - layernorm1: the first layer norm layer, with epsilon=1e-6
          - layernorm2: the second layer norm layer, with epsilon=1e-6
          - layernorm3: the third layer norm layer, with epsilon=1e-6
          - dropout1: the first dropout layer
          - dropout2: the second dropout layer
          - dropout3: the third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method call that builds a decoder block for a transformer
        Arguments:
          - x is a tensor of shape (batch, target_seq_len, dm) containing
              the input to the decoder block
          - encoder_output is a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
          - training is a boolean to determine if the model is training
          - look_ahead_mask is the mask to be applied to the first multi
            head attention layer
          - padding_mask is the mask to be applied to the second multi head
            attention layer
        Returns:
          - A tensor of shape (batch, target_seq_len, dm) containing the blocks
            output
        """
        # 1st MultiHeadAttention
        mha1, _ = self.mha1(x, x, x, look_ahead_mask)
        mha1 = self.dropout1(mha1, training=training)
        out1 = self.layernorm1(mha1 + x)
        # 2nd MultiHeadAttention
        mha2, _ = self.mha2(out1, encoder_output,
                            encoder_output, padding_mask)
        mha2 = self.dropout2(mha2, training=training)
        out2 = self.layernorm2(mha2 + out1)
        # Dense hidden
        dense_hidden = self.dense_hidden(out2)
        # Dense output
        dense_output = self.dense_output(dense_hidden)
        dense_output = self.dropout3(dense_output, training=training)
        # Output
        output = self.layernorm3(dense_output + out2)
        return output


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


class Encoder(tf.keras.layers.Layer):
    """
    class Encoder that inherits from tensorflow.keras.layers.Layer
    to create the encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input to
        build the model
        Arguments:
          - dm is an integer representing the dimensionality of the model
          - h is an integer representing the number of heads
          - hidden is the number of hidden units in the fully connected layer
          - input_vocab is an integer representing the size of the input
            vocabulary
          - max_seq_len is an integer representing the maximum sequence
            length possible
          - drop_rate is the dropout rate
        Public instance attributes:
          - N - the number of blocks in the encoder
          - dm - the dimensionality of the model
          - embedding - the embedding layer for the inputs
          - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
            containing the positional encodings
          - blocks - a list of length N containing all of the EncoderBlock's
          - dropout - the dropout layer, to be applied to the positional
            encodings
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Method call should use tf.cast to convert mask to tf.float32
        Arguments:
          - x is a tensor of shape (batch, input_seq_len, dm) containing the
              input to the encoder
          - training is a boolean to determine if the model is training
          - mask is the mask to be applied for multi head attention
        Returns:
          - A tensor of shape (batch, input_seq_len, dm) containing the
            encoder output
        """
        seq_len = x.shape[1]
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        output = self.dropout(embedding, training=training)
        for i in range(self.N):
            output = self.blocks[i](output, training, mask)
        return output


class Decoder(tf.keras.layers.Layer):
    """
    Decoder that inherits from tensorflow.keras.layers.Layer to
    create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input
        to build the model
        Arguments:
        - dm is an integer representing the dimensionality of the model
        - h is an integer representing the number of heads
        - hidden is the number of hidden units in the fully connected layer
        - target_vocab is an integer representing the size of the target
          vocabulary
        - max_seq_len is an integer representing the maximum sequence
          length possible
        - drop_rate is the dropout rate
        Public instance attributes:
        - N - the number of blocks in the encoder
        - dm - the dimensionality of the model
        - embedding - the embedding layer for the inputs
        - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
          containing the positional encodings
        - blocks - a list of length N containing all of the EncoderBlock‘s
        - dropout - the dropout layer, to be applied to the positional
          encodings
        """
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        super(Decoder, self).__init__()

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method call should use tf.cast to convert mask to tf.float32
        Arguments:
          - x is a tensor of shape (batch, input_seq_len, dm) containing
              the input to the encoder
          - training is a boolean to determine if the model is training
          - mask is the mask to be applied for multi head attention
        Returns:
          - A tensor of shape (batch, input_seq_len, dm) containing
            the encoder output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """
    class Transformer that inherits from tensorflow.keras.Model
    to create a transformer network
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input
        to build the model
        Arguments:
        - dm is an integer representing the dimensionality of the model
        - h is an integer representing the number of heads
        - hidden is the number of hidden units in the fully connected layer
        - target_vocab is an integer representing the size of the target
          vocabulary
        - max_seq_len is an integer representing the maximum sequence length
          possible
        - drop_rate is the dropout rate
        Public instance attributes:
        - N - the number of blocks in the encoder
        - dm - the dimensionality of the model
        - embedding - the embedding layer for the inputs
        - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
          containing the positional encodings
        - blocks - a list of length N containing all of the EncoderBlock's
        - dropout - the dropout layer, to be applied to the positional
          encodings
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Method call should use tf.cast to convert mask to tf.float32
        Arguments:
          - inputs is a tensor of shape (batch, input_seq_len) containing the
          - target is a tensor of shape (batch, target_seq_len) containing the
          - training is a boolean to determine if the model is training
          - encoder_mask is the padding mask to be applied to the encoder
          - look_ahead_mask is the look ahead mask to be applied to the decoder
          - decoder_mask is the padding mask to be applied to the decoder
        Returns:
          - A tensor of shape (batch, target_seq_len, target_vocab) containing
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
