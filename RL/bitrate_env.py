import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BitrateEnv(gym.Env):
    def __init__(self):
        super(BitrateEnv, self).__init__()

        # Define action and observation space
        # Assuming 4 bitrate levels as actions
        self.action_space = spaces.Discrete(4)

        # Observation space can be a vector of relevant metrics
        # e.g., current buffer level, last chunk download time, etc.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32)

        # Initialize environment state
        self.state = np.zeros(3)
        self.buffer = 0
        self.last_chunk_time = 1
        self.total_chunks = 10
        self.current_chunk = 0

    def reset(self):
        self.state = np.zeros(3)
        self.buffer = np.random.uniform(0, 1)
        self.last_chunk_time = np.random.uniform(0.1, 1)
        self.current_chunk = 0
        return self.state

    def step(self, action):
        # Simulate the download of a chunk with the selected bitrate
        download_time = np.random.uniform(0.1, 1) / (action + 1)

        # Update buffer
        self.buffer += self.last_chunk_time - download_time
        self.last_chunk_time = download_time

        # Update state
        self.state = np.array([self.buffer, self.last_chunk_time, action])

        # Calculate reward
        reward = self.calculate_reward(action, download_time)

        # Check if episode is done
        done = self.current_chunk >= self.total_chunks
        self.current_chunk += 1

        return self.state, reward, done, {}

    def calculate_reward(self, action, download_time):
        # Example reward calculation
        # Higher bitrates get higher reward, penalize for rebuffering
        if self.buffer < 0:
            reward = -1  # Rebuffering penalty
        else:
            reward = action - download_time  # Reward for bitrate level minus download time
        return reward
