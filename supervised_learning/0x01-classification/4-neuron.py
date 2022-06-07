#!/usr/bin/env python3
""" This class represents a neuron in the network. """

import numpy as np


class Neuron:
    """ This class represents a neuron in the network. """

    def __init__(self, nx):
        """ Construct a new Neuron. """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
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

    def forward_prop(self, X):
        """ Calculate the forward propagation of the neuron. """
        self.__A = sigmoid(np.dot(self.W, X) + self.b)
        return self.A

    def cost(self, Y, A):
        """ Const logistic regression """
        cost = Y * np.log(A) + np.log(1.0000001 - A) * (1 - Y)
        return (-cost.sum() / len(np.transpose(Y)))

    def evaluate(self, X, Y):
        """ Evaluate neuron """
        self.forward_prop(X)
        return (self.A, self.cost(Y, self.A))


def sigmoid(z):
    """Activation function"""
    return 1 / (1 + np.exp(-z))
