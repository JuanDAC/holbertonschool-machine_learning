#!/usr/bin/env python3
"""
File: 3-q_learning.py
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Function that has the trained agent play an episode:
    Arguments:
        - env is the FrozenLakeEnv instance
        - Q is a numpy.ndarray containing the Q-table
        - max_steps is the maximum number of steps in the episode
            * Each state of the board should be displayed via the console
            * You should always exploit the Q-table
    Returns:
        - the total rewards for the episode
    """
    state = env.reset()
    done = False
    env.render()
    rewards = 0
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state
        rewards += reward
        if done:
            break
    return rewards
