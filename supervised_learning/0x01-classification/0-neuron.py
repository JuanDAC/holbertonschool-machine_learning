#!/usr/bin/env python
"""Only just one neuron"""

import numpy as np


class Neuron:
    """ This class represents a neuron in the network. """

    def __init__(self, nx):
        """ Construct a new Neuron. """
        if type(nx) is not int:
            raise ValueError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.b = 0
        self.A = 0
        self.W = np.random.normal(size=(1, nx), scale=1.0)
