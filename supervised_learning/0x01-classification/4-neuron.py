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
        cost1 = Y * np.log(A)
        cost2 = (1 - Y) * np.log(1.0000001 - A)
        all_consts = cost1 + cost2
        number_of_classes = Y.shape()[1]
        return -all_consts.sum() / number_of_classes

    def evaluate(self, X, Y):
        """ Evaluate neuron """
        self.forward_prop(X)
        cost = self.cost(Y, self.A)
        # Round elements of the array to the nearest integer.
        prediction = np.rint(self.A).astype(int)
        return (prediction, cost)


def sigmoid(z):
    """Activation function"""
    return 1 / (1 + np.exp(-z))
