# Inspired by OpenAI Gym for Format

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt



"""
env = EnvAnalyst(Environment())
agent = Agent()

for i in tqdm(range(2000)):
    observation, info = env.reset()

    agent.reset()
    agent.initial_observation(observation)

    terminated = False
    while not terminated:
        action_taken = agent.choose_action(env.action_space)
        
        observation, reward, terminated, truncated, info = env.step(action_taken)
        agent.action_feedback(observation, reward, terminated, truncated)
"""


class Agent:

    def __init__(self) -> None:
        pass

    def reset(self):
        pass

    def initial_observation(self, observation):
        pass

    def choose_action(self, action_space):
        pass

    def action_feedback(self, observation, reward, terminated, truncated):
        pass


class Environment:

    def __init__(self) -> None:
        pass

    def reset(self) -> tuple[np.ndarray, dict]:
        """returns observation, info"""
        pass
    
    @property
    def action_space(self) -> set:
        """Returns set of possible actions"""
        pass

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Returns observation, reward, terminated, truncated, info"""
        pass


class EnvAnalyst:

    def __init__(self, env: Environment) -> None:
        self.env = env

        self.original_env_reset = self.env.reset
        self.env.reset = self.reset

        self.original_env_step = self.env.step
        self.env.step = self.step

        self.action_history = []
        self.rewards = []

    def plot(self, title):
        plt.plot([np.mean(self.rewards[:i]) for i in range(len(self.rewards))])
        plt.title(title)
        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.show()

    def reset(self):
        observation, info = self.original_env_reset()

        self.action_history = []
        self.rewards = []

        return observation, info

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        observation, reward, terminated, truncated, info = self.original_env_step(action)

        self.action_history.append(action)
        self.rewards.append(reward)

        return observation, reward, terminated, truncated, info
