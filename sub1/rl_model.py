import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO

PROB_ROOT = "../data/probs/"
MODEL_DIR = "../data/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

class TradingEnv(gym.Env):
    def __init__(self, file_list, fee_rate=0.0003, min_hold=15,
                 hold_bonus=0.001, early_exit_penalty=0.002,
                 switch_penalty=0.001, long_hold_bonus=0.1,
                 long_hold_candles=200, post_hold_bonus=0.01,
                 post_hold_interval=10):
        super().__init__()
        self.file_list = file_list
        self.current_file_idx = 0
        self.ptr = 0
        self.df = None

        self.fee_rate = fee_rate
        self.min_hold = min_hold
        self.hold_bonus = hold_bonus
        self.early_exit_penalty = early_exit_penalty
        self.switch_penalty = switch_penalty
        self.long_hold_bonus = long_hold_bonus
        self.long_hold_candles = long_hold_candles
        self.post_hold_bonus = post_hold_bonus
        self.post_hold_interval = post_hold_interval

        sample = pd.read_csv(self.file_list[0])
        self.feature_cols = [c for c in sample.columns if c != "Price"]
        n_features = len(self.feature_cols)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _get_obs(self):
        return self.features[self.ptr].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_file_idx = (self.current_file_idx + 1) % len(self.file_list)
        self.df = pd.read_csv(self.file_list[self.current_file_idx])

        feats = self.df[self.feature_cols].values
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
        self.features = np.nan_to_num(feats)
        self.prices = self.df["Price"].values

        self.ptr = 0
        self.position = 0
        self.steps_in_pos = 0
        self.total_reward = 0

        return self._get_obs(), {}

    def step(self, action):
        new_pos = action - 1
        price_t = self.prices[self.ptr]
        price_next = self.prices[min(self.ptr + 1, len(self.prices) - 1)]

        price_ret = (price_next - price_t) / price_t
        pnl = price_ret * self.position

        fee = 0.0
        switched = new_pos != self.position
        if switched:
            fee = self.fee_rate * abs(new_pos - self.position)

        reward = pnl

        if new_pos == self.position and new_pos != 0:
            self.steps_in_pos += 1
            if self.steps_in_pos >= self.min_hold:
                reward += self.hold_bonus * (self.steps_in_pos / self.min_hold)

            if self.steps_in_pos == self.long_hold_candles:
                reward += self.long_hold_bonus
            elif self.steps_in
