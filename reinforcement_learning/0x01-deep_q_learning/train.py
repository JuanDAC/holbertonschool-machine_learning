#!/usr/bin/env python3
"""
File training an agent that can play Atari's Breakout
"""

import gym
import numpy as np
from tensorflow.keras.layers import Input, Permute, Conv2D, \
    Flatten, Dense
from tensorflow.keras import Model, optimizers
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor


CANVAS_LENGTH = 4


class FrameProcessor(Processor):
    """
    Declare the environment to play Atari's Breakout
    A frame processor to process the frames
    A memory to store the previous observations
    A policy to determine the next action to take
    A neural network model to predict the Q-values
    A DQN agent that will use the previous components
    Train the agent
    """

    def process_observation(self, observation):
        """
        Process the observation
        Arguments:
            - observation is the observation from the environment
        Returns:
            The processed observation
        """
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize((84, 110)).convert('L')
        new_observation = np.array(img)
        assert new_observation.shape == (110, 84)
        return new_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Process the state batch
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
        Process the reward
        """
        return np.clip(reward, -1., 1.)


def create_model(num_actions, input_shape=(CANVAS_LENGTH, 84, 84), lr=0.00025):
    """
    Create the model with CNN architecture for Deep Q-learning algorithm
    Arguments:
        - num_actions: the number of actions in the environment
        - input_shape: the shape of the input
        - lr: the learning rate
    Returns:
        The model with CNN architecture.
    """
    inputs = Input(shape=input_shape)
    layer = Permute((2, 1, 3))(inputs)
    layer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(layer)
    layer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(layer)
    layer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(num_actions, activation='linear')(layer)
    model = Model(inputs=inputs, outputs=layer)
    optimizer = optimizers.RMSprop(lr=lr, rho=0.95, epsilon=0.01)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def training():
    """
    Training an agent that can play Atari's Breakout
    """
    env = gym.make('ALE/Breakout-v5')
    np.random.seed(0)
    env.seed(0)
    nb_actions = env.action_space.n
    model = create_model(nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=CANVAS_LENGTH)
    processor = FrameProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=50000, target_model_update=10000,
                   policy=policy, processor=processor)
    dqn.compile(optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                metrics=['mae'])
    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
    env.close()
