from stable_baselines3 import PPO
import numpy as np
from envs.deep_sea_treasure import DeepSeaTreasureEnv
#from gym.wrappers import RewardWrapper

import gym

class RewardWrapper(gym.Wrapper):
    def reward(self, reward):
        """Override this method to modify the reward."""
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

class PPOOracle:
    def __init__(self, env, train_steps=10000):
        self.env = env
        self.train_steps = train_steps

    def solve(self, weight_vector):
        class WeightedRewardWrapper(RewardWrapper):
            def __init__(self, env, weights):
                super().__init__(env)
                self.weights = weights

            def reward(self, reward):
                return float(np.dot(self.weights, reward))

        env_weighted = WeightedRewardWrapper(self.env, weight_vector)
        model = PPO("MlpPolicy", env_weighted, verbose=0)
        model.learn(total_timesteps=self.train_steps)

        obs = env_weighted.reset()
        done = False
        cumulative_reward = np.zeros(2)
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env_weighted.step(action)
            cumulative_reward += reward

        return cumulative_reward, model
