#!/usr/bin/env python3
"""
File to display a game played by the agent trained.
"""

import gym
from tensorflow.keras import optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy
from rl.policy import EpsGreedyQPolicy


FrameProcessor = __import__('train').FrameProcessor
create_model = __import__('train').create_model

CANVAS_LENGTH = 4


def play():
    """
    Function that display a game played by the agent trained
    """
    env = gym.make('ALE/Breakout-v5')
    env.reset()
    nb_actions = env.action_space.n
    model = create_model(nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=CANVAS_LENGTH)
    processor = FrameProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   processor=processor, policy=policy, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000, train_interval=4,
                   delta_clip=1.)
    dqn.compile(optimizers.Adam(lr=.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == '__main__':
    play()
