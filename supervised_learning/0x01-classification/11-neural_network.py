#!/usr/bin/env python3
"""Making neural network"""


import numpy as np


class NeuralNetwork:
    """ This class represents a redneuron in the network. """

    def __init__(self, nx, nodes):
        """Initialize neural network"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Prediction for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """bias for the output neuron """
        return self.__b2

    @property
    def A2(self):
        """prediction for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """Forward propagation"""
        """Ponderate weights and data"""
        sump1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(sump1)
        sump2 = np.dot(self.__W2, self.__A1) + self.__b2
        A2 = sigmoid(sump2)
        self.__A2 = A2
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """
        Cost function CROSS ENTROPY
        Cost=(labels*log(predictions)+(1-labels)*log(1-predictions))/len(labels)
        Params:
            Y: correct labels for the input data
            A: activated output of the neuron for each(prediction)
        """
        """take the error when label = 1"""
        cost1 = Y * np.log(A)
        cost2 = (1 - Y) * np.log(1.0000001 - A)
        total_cost = cost1 + cost2
        m = len(np.transpose(Y))
        cost_avg = -total_cost.sum() / m
        return cost_avg


def sigmoid(number):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-number))
