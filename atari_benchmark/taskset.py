import gym
import numpy as np


class Taskset:

    def __init__(self, config):
        names = config['names']
        self.config = config
        self.envs = [gym.make(name) for name in names]

    def run_episode(self, sf, policy_function):
        env = self.envs[0]
        observation = env.reset()
        total_reward = 0
        for i in range(18000):
            action = policy_function(observation.astype(np.float32))
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return -total_reward

    @property
    def K(self):
        return len(self.config['h_mains'])
