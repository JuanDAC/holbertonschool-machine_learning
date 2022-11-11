#!/usr/bin/env python3

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Function that performs the Monte Carlo algorithm:
    Arguments:
        - env is the openAI environment instance
        - V is a numpy.ndarray of shape (s,) containing the value estimate
        - policy is a function that takes in a state and returns the next action to take
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
    Returns:
        - V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        states, rewards = [], []
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            if done:
                break
            state = new_state
        G = 0
        for i in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[i]
            if states[i] not in states[:i]:
                V[states[i]] = V[states[i]] + alpha * (G - V[states[i]])
    return V
