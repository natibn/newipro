import gym
from gym import spaces
import numpy as np

class DeepSeaTreasureEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid = [
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0, 40],
            [ 0,  0,  0,  0, 50],
            [ 0,  0,  0,  0,100]
        ]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.state = (0, 0)
        self.max_steps = 30
        self.steps = 0
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        self.state = (0, 0)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32) / np.array([self.n_rows, self.n_cols])

    def step(self, action):
        row, col = self.state
        if action == 0 and row > 0: row -= 1  # up
        if action == 1 and row < self.n_rows - 1: row += 1  # down
        if action == 2 and col > 0: col -= 1  # left
        if action == 3 and col < self.n_cols - 1: col += 1  # right

        self.state = (row, col)
        self.steps += 1

        # Two objectives: 1) treasure (maximize), 2) time (minimize)
        treasure = self.grid[row][col]
        time_penalty = -self.steps  # negative to make it a cost

        done = self.steps >= self.max_steps or treasure > 0
        reward = np.array([treasure, time_penalty], dtype=np.float32)

        return self._get_obs(), reward, done, {}
