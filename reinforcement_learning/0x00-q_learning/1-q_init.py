#!/usr/bin/env python3
"""
File: 1-q_init.py
"""
import numpy as np

def q_init(env):
    """
    Write a function def q_init(env): that initializes the Q-table:
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    return np.zeros((env.observation_space.n, env.action_space.n))