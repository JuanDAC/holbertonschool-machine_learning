#!/usr/bin/env python3
"""
File Name: 0-vanilla.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates an autoencoder:

    Arguments:
        - input_dims is an integer containing the dimensions of the model input
        - hidden_layers is a list containing the number of nodes for each
                        hidden layer in the encoder, respectively
            * the hidden layers should be reversed for the decoder
        - latent_dims is an integer containing the dimensions of the latent
                      space representation
    Returns: encoder, decoder, auto
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the last layer in
    the decoder, which should use sigmoid
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0], activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(encoded)
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(input_decoder)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Autoencoder
    encoder = keras.Model(inputs=input_encoder, outputs=encoded)
    decoder = keras.Model(inputs=input_decoder, outputs=decoded)
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
