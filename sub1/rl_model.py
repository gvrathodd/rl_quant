import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO

PROB_ROOT = "../data/probs/"
PRED_ROOT = "../data/probs_pred/"
MODEL_DIR = "../data/models/"
RESULTS_DIR = "../data/results/"


def causal_zscore(feats, min_periods=30):
    """Expanding-window z-score: row t only uses rows <= t, so there is no look-ahead."""
    df = pd.DataFrame(feats)
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std()
    z = (df - mean) / (std + 1e-8)
    return np.nan_to_num(z.values)


class TradingEnv(gym.Env):
    """One episode = one trading day. Actions: 0 short, 1 flat, 2 long.

    file_list entries may be CSV paths or DataFrames with a "Price" column plus features.
    """

    def __init__(self, file_list, fee_rate=0.0003, min_hold=15,
                 hold_bonus=0.001, early_exit_penalty=0.002,
                 switch_penalty=0.001, long_hold_bonus=0.1,
                 long_hold_candles=200, post_hold_bonus=0.01,
                 post_hold_interval=10, shuffle=True):
        super().__init__()
        self.file_list = file_list
        self.shuffle = shuffle
        self.current_file_idx = -1
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

        sample = self._load(self.file_list[0])
        self.feature_cols = [c for c in sample.columns if c != "Price"]
        # features + current position + time in position
        n_obs = len(self.feature_cols) + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    @staticmethod
    def _load(item):
        return item if isinstance(item, pd.DataFrame) else pd.read_csv(item)

    def _get_obs(self):
        state = [self.position, min(self.steps_in_pos / self.long_hold_candles, 1.0)]
        return np.concatenate([self.features[self.ptr], state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.shuffle:
            self.current_file_idx = int(self.np_random.integers(len(self.file_list)))
        else:
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_list)
        self.df = self._load(self.file_list[self.current_file_idx])

        self.features = causal_zscore(self.df[self.feature_cols].values)
        self.prices = self.df["Price"].values

        self.ptr = 0
        self.position = 0
        self.steps_in_pos = 0
        self.total_reward = 0.0

        return self._get_obs(), {}

    def step(self, action):
        new_pos = int(action) - 1
        price_t = self.prices[self.ptr]
        price_next = self.prices[self.ptr + 1]

        # the position chosen at t earns the return from t to t+1
        price_ret = (price_next - price_t) / price_t
        switched = new_pos != self.position
        fee = self.fee_rate * abs(new_pos - self.position)
        net_ret = price_ret * new_pos - fee

        reward = net_ret

        if switched:
            reward -= self.switch_penalty
            if self.position != 0 and self.steps_in_pos < self.min_hold:
                reward -= self.early_exit_penalty
            self.steps_in_pos = 0
        elif new_pos != 0:
            self.steps_in_pos += 1
            if self.steps_in_pos >= self.min_hold:
                reward += self.hold_bonus * (self.steps_in_pos / self.min_hold)
            if self.steps_in_pos == self.long_hold_candles:
                reward += self.long_hold_bonus
            elif (self.steps_in_pos > self.long_hold_candles
                  and (self.steps_in_pos - self.long_hold_candles) % self.post_hold_interval == 0):
                reward += self.post_hold_bonus

        self.position = new_pos
        self.ptr += 1
        self.total_reward += reward

        terminated = self.ptr >= len(self.prices) - 1
        info = {"net_return": net_ret, "position": new_pos, "fee": fee}
        return self._get_obs(), float(reward), terminated, False, info


def run_policy(model, env, n_days):
    """Deterministic rollout over n_days episodes; returns per-step net returns and positions."""
    rows = []
    for day in range(n_days):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            rows.append({"day": day, "net_return": info["net_return"],
                         "position": info["position"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_files = sorted(glob.glob(f"{PROB_ROOT}day*.csv"))
    test_files = sorted(glob.glob(f"{PRED_ROOT}day*.csv"))

    env = TradingEnv(train_files)
    model = PPO("MlpPolicy", env, verbose=1, seed=42)
    model.learn(total_timesteps=500_000)
    model.save(os.path.join(MODEL_DIR, "ppo_trader"))

    test_env = TradingEnv(test_files, shuffle=False)
    results = run_policy(model, test_env, len(test_files))
    results.to_csv(os.path.join(RESULTS_DIR, "ppo_test_returns.csv"), index=False)
    print(f"Saved {len(results)} out-of-sample steps to {RESULTS_DIR}")
    print("Then run `python risk_report.py --returns data/results/ppo_test_returns.csv` from the repo root.")
