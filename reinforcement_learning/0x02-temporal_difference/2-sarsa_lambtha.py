#!/usr/bin/env python3

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action:
    Arguments:
        - Q is a numpy.ndarray containing the q-table
        - state is the current state
        - epsilon is the epsilon to use for the calculation
    Returns:
        - the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs SARSA(λ):
    Arguments:
        - env is the openAI environment instance
        - Q is a numpy.ndarray of shape (s,a) containing the Q table
        - lambtha is the eligibility trace factor
        - episodes is the total number of episodes to train over
        - max_steps is the maximum number of steps per episode
        - alpha is the learning rate
        - gamma is the discount rate
        - epsilon is the initial threshold for epsilon greedy
        - min_epsilon is the minimum value that epsilon should decay to
        - epsilon_decay is the decay rate for updating epsilon between episodes
    Returns:
        - Q, the updated Q table
    """
    for _ in range(episodes):
        state = env.reset()
        eligibility = np.zeros((Q.shape))
        epsilon = min_epsilon + (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay)
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            delta = reward + gamma * \
                Q[new_state, np.argmax(Q[new_state])] - Q[state, action]
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= lambtha * gamma
            if done:
                break
            state = new_state
    return Q
