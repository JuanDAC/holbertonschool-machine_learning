#!/usr/bin/env python3
"""
File: 1-q_init.py
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action:
    Arguments:
        - Q is a numpy.ndarray containing the q-table
        - state is the current state
        - epsilon is the epsilon to use for the calculation
    Returns:
        - The next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state, :])
