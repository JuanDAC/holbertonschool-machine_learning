#!/usr/bin/env python3

import numpy as np


class Neuron:
    """ This class represents a neuron in the network. """

    def __init__(self, nx):
        """ Construct a new Neuron. """
        if type(nx) is not int:
            raise ValueError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__b = 0
        self.__A = 0
        self.__W = np.random.normal(size=(1, nx), scale=1.0)

    @property
    def W(self):
        """ Weight of the neuron. """
        return self.__W

    @property
    def b(self):
        """" Bias of the neuron. """
        return self.__b

    @property
    def A(self):
        """" Prediction of the neuron. """
        return self.__A
