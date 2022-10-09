#!/usr/bin/env python3
"""
File that contains the Performs Bayesian optimization
on a noiseless 1D Gaussian process
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    A class that performs Bayesian optimization on a Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor:
        Arguments:
            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing the
                     inputs already sampled with the black-box function
            - Y_init is a numpy.ndarray of shape (t, 1) representing the
                     outputs of the black-box function for each input in X_init
            - t is the number of initial samples
            - bounds is a tuple of (min, max) representing the bounds of the
                     space in which to look for the optimal point
            - ac_samples is the number of samples that should be analyzed
                         during acquisition
            - l is the length parameter for the kernel
            - sigma_f is the standard deviation given to the output of the
                      black-box function
            - xsi is the exploration-exploitation factor for acquisition
            - minimize is a bool determining whether optimization should be
                      performed for minimization (True) or maximization (False)
        Sets the following public instance attributes:
            - f: the black-box function
            - gp: an instance of the class GaussianProcess
            - X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
                   acquisition sample points, evenly spaced between min and max
            - xsi: the exploration-exploitation factor
            - minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Public instance method that calculates the next best sample location
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
            - X_next is a numpy.ndarray of shape (1,) representing the next
                     best sample point
            - EI is a numpy.ndarray of shape (ac_samples,) containing the
                 expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is False:
            mu_sample_opt = np.amax(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        else:
            mu_sample_opt = np.amin(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        with np.errstate(divide='warn'):
            if self.minimize is True:
                imp = mu_sample_opt - mu - self.xsi
            else:
                imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """
        Public instance method that optimizes the black-box function:
        Arguments:
            - iterations is the maximum number of iterations to perform
                * If the next proposed point is one that has already been
                  sampled, optimization should be stopped early
        Returns: X_opt, Y_opt
            - X_opt is a numpy.ndarray of shape (1,) representing the optimal
                    point
            - Y_opt is a numpy.ndarray of shape (1,) representing the optimal
                    function value
        """
        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        X_opt = self.gp.X[index]
        Y_opt = self.gp.Y[index]
        return X_opt, Y_opt
