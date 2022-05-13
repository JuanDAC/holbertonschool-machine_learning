#!/usr/bin/env python3
"""Create a forward propagation"""


def forward_prop(x, layer_sizes=[], activations=[], index=0):
    """ Forward propagation graph for the neural network using tensorflow
    ...
    Parameters
    __________
    x : Tensor
        Imput data placeholder
    layer_sizes : list
        This is the n nodes inside the layers
    activation : list
        This have the activation function per layer
    ...
    Return
    ______
    layer:
        Prediction
    """
    if index >= len(layer_sizes):
        return x
    create_layer = __import__('1-create_layer').create_layer
    layer = create_layer(x, layer_sizes[index], activations[index])
    return forward_prop(layer, layer_sizes, activations, index + 1)
