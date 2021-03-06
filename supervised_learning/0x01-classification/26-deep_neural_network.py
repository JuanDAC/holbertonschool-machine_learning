#!/usr/bin/env python3
"""deep neural network"""


import numpy as np
from matplotlib import pyplot as plt
import pickle


class DeepNeuralNetwork:
    """making a deep neural network"""

    def __init__(self, nx, layers):
        """initialiaze deep neural network"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        negative = list(filter(lambda x: x <= 0, layers))
        if len(negative) > 0:
            raise TypeError("layers must be a list of positive integers")
        self.cache, self.weights, self.L = {}, {}, len(layers)
        for i in range(len(layers)):
            """biases of the network should be initialized to 0's"""
            def first():
                if i == 0:
                    factor1, factor2 = np.random.randn(
                        layers[i], nx), np.sqrt(2 / nx)
                    self.weights['W' + str(i + 1)] = factor1 * factor2
                    return
                factor1, factor2 = np.random.randn(
                    layers[i], layers[i - 1]), np.sqrt(2 / layers[i - 1])
                self.weights['W' + str(i + 1)] = factor1 * factor2
            first()
            zeros = np.zeros(layers[i])
            self.weights['b' + str(i + 1)] = zeros.reshape(layers[i], 1)

    @property
    def cache(self):
        """ A dictionary to hold all intermediary values of the network"""
        return self.__cache

    @property
    def L(self):
        """The number of layers in the neural network"""
        return self.__L

    @property
    def weights(self):
        """ A dictionary to hold all weights and biased of the network"""
        return self.__weights

    @cache.setter
    def cache(self, value):
        """ A dictionary to hold all intermediary values of the network"""
        self.__cache = value

    @L.setter
    def L(self, value):
        """The number of layers in the neural network"""
        self.__L = value

    @weights.setter
    def weights(self, value):
        """ A dictionary to hold all weights and biased of the network"""
        self.__weights = value

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            w_layer = self.__weights['W' + str(i + 1)]
            b_layer = self.__weights['b' + str(i + 1)]
            activation = sigmoid(np.dot(w_layer, X) + b_layer)
            X = activation
            self.__cache['A' + str(i + 1)] = activation
        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Cost function CROSS ENTROPY
        Cost=(labels*log(predictions)+(1-labels)*log(1-predictions))/len(labels)
        Params:
            Y: correct labels for the input data
            A: activated output of the neuron for each(prediction)
        """
        cost1 = Y * np.log(A)
        cost2 = (1 - Y) * np.log(1.0000001 - A)
        total_cost = cost1 + cost2
        m = len(np.transpose(Y))
        cost_avg = -total_cost.sum() / m
        return cost_avg

    def evaluate(self, X, Y):
        """Evaluate neuron
        return: Prediction, Cost
        """
        prediction, cache = self.forward_prop(X)
        cost = self.cost(Y, prediction)
        prediction = np.rint(prediction).astype(int)
        return (prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Gradient descent
        Partial derivates of COST FUNCTION respect to Weigth and bias
        1 step
        """
        self.__cache = cache
        m = len(Y[0])
        for i in range(self.__L, 0, -1):
            if i == self.__L:
                dzl = self.__cache['A' + str(i)] - Y
            X = self.__cache['A' + str(i - 1)]
            weight_derivative_l = np.dot(X, dzl.T) / m
            bias_derivative_l = np.sum(dzl, axis=1, keepdims=True) / m
            d_sigmoid = derivate_sigmoid(self.__cache['A' + str(i - 1)])
            dzl_1 = np.dot(self.__weights['W' + str(i)].T, dzl) * d_sigmoid
            dzl = dzl_1
            wl = self.__weights['b' + str(i)]
            restw = (alpha * bias_derivative_l)
            bl = self.__weights['W' + str(i)]
            restb = (alpha * weight_derivative_l.T)
            self.__weights['b' + str(i)] = wl - restw
            self.__weights['W' + str(i)] = bl - restb

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            AL, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        result = self.evaluate(X, Y)
        return result

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        costs = []
        iterat = []
        for i in range(iterations):
            AL, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose is True and i % step == 0:
                current_cost = self.cost(Y, AL)
                costs.append(current_cost)
                iterat.append(i)
                print("Cost after {} iterations: {}".format(i, current_cost))
        if graph is True:
            plt.plot(iterat, costs, "blue")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        result = self.evaluate(X, Y)
        return result

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if type(filename) != str:
            return None
        if filename.split('.')[-1] != 'pkl':
            filename = filename + '.pkl'
        try:
            with open(filename, 'wb') as f:
                obj = pickle.dump(self, f)
                return obj
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """
        Loads a pickled deep neural object
        """
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except Exception:
            return None


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(z):
    """Sigmoid derivate"""
    return z * (1 - z)
