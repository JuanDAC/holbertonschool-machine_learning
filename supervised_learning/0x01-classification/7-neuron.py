#!/usr/bin/env python3
"""Only just one neuron"""

import matplotlib.pyplot as plt
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
        x = np.dot(self.__W, X) + self.__b
        self.__A = 1 / 1 + np.exp(-x)
        return self.__A

    def cost(self, Y, A):
        """ Const logistic regression """
        cost = Y * np.log(A) + np.log(1.0000001 - A) * (1 - Y)
        return (-cost.sum() / len(np.transpose(Y)))

    def evaluate(self, X, Y):
        """ Evaluate neuron """
        self.forward_prop(X)
        prediction = np.rint(self.A).astype(int)
        return (prediction, self.cost(Y, self.A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Compute the gradient descent """
        m = Y.shape[1]
        dCodz = A - Y

        weight_derivative = (1 / m) * np.matmul(X, dCodz.T)
        bias_derivative = (1 / m) * np.sum(dCodz)

        self.__b = self.b - bias_derivative * alpha
        self.__W = self.W - (weight_derivative * alpha).T

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=101
              ):
        """ Train the our neuron"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 and step > iterations:
                raise("step must be positive and <= iterations")

        costs = []

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)
            cost = self.cost(Y, self.__A)
            if i == 0 and verbose:
                print(f"Cost after 0 iterations: {cost}")
            elif i % step == 0 and verbose:
                print(f"Cost after {i} iterations: {cost}")
            costs.append(cost)

        if graph:
            plt.plot(costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)
