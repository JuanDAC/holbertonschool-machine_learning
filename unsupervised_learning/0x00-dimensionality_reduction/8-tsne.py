#!/usr/bin/env python3
"""
8. tsne
"""

import numpy as np


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Function that performs a t-SNE transformation
    Arguments:
    - X is a numpy.ndarray of shape (n, d) containing the dataset to be
      transformed by t-SNE
    - n is the number of data points
    - d is the number of dimensions in each point
    - ndims is the new dimensional representation of X
    - idims is the intermediate dimensional representation of X after PCA
    - perplexity is the perplexity
    - iterations is the number of iterations
    - lr is the learning rate
    Every 100 iterations, not including 0, print Cost at iteration {iteration}: {cost}
    {iteration} is the number of times Y has been updated and {cost} is the corresponding cost
    After every iteration, Y should be re-centered by subtracting its mean
    Returns:
    - Y, a numpy.ndarray of shape (n, ndim) containing the optimized low dimensional transformation of X
    You should use:
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities
    grads = __import__('6-grads').grads
    cost = __import__('7-cost').cost
    For the first 100 iterations, perform early exaggeration with an exaggeration of 4
    a(t) = 0.5 for the first 20 iterations and 0.8 thereafter
    Hint 1: See Algorithm 1 on page 9 of t-SNE. But WATCH OUT! There is a mistake in the gradient descent step
    Hint 2: See Section 3.4 starting on page 9 of t-SNE for early exaggeration
    """
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities
    grads = __import__('6-grads').grads
    cost = __import__('7-cost').cost
    n, d = X.shape
    Y = np.random.randn(n, ndims)
    Y, _ = pca(X, idims)
    P, betas, H = P_affinities(X, perplexity=perplexity, tol=1e-5)
    for i in range(iterations):
        if i < 20:
            a = 0.5
        else:
            a = 0.8
        dY, Q = grads(Y, P)
        Y = Y + lr * dY
        Y = Y - np.mean(Y, axis=0)
        if i % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i, C))
        P = a * P + (1 - a) * Q
        if (i + 1) == 100:
            P /= 4
    return Y
