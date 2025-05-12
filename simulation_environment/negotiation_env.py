import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple

class NegotiationEnv(gym.Env):
    """
    Custom Gymnasium environment for IOI negotiation using historical data.
    Each episode is initialized from a sampled IOI row.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        ioi_df: pd.DataFrame,
        initial_price_col: str = "price",
        min_price_margin: float = 0.4,
        max_rounds: int = 5
    ):
        super().__init__()
        self.df = ioi_df.reset_index(drop=True)
        self.initial_price_col = initial_price_col
        self.min_price_margin = min_price_margin
        self.max_rounds = max_rounds

        # Observation: [own_price, counterparty_price, time_elapsed, negotiation_round]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1e6, 1e6, 100, max_rounds], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: 0 = ACCEPT, 1 = REJECT, 2 = COUNTER_OFFER
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.current_ioi = None
        self.round = 0
        self.time_elapsed = 0
        self.min_price = 0
        self.initial_price = 0
        self.np_random = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Sample one IOI row
        self.current_ioi = self.df.sample(1, random_state=self.np_random.integers(1e9)).iloc[0]
        self.initial_price = float(self.current_ioi[self.initial_price_col])
        self.min_price = self.initial_price * self.min_price_margin
        self.round = 0
        self.time_elapsed = 0
        self.counterparty_price = self.initial_price * (0.9 + 0.1 * self.np_random.random())

        self.state = np.array([
            self.initial_price,
            self.counterparty_price,
            self.time_elapsed,
            self.round
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        truncated = False
        reward = 0.0

        if action == 0:  # ACCEPT
            if self.counterparty_price >= self.min_price:
                reward = self._compute_reward(self.counterparty_price)
            else:
                reward = -1.0
            terminated = True

        elif action == 1:  # REJECT
            reward = -1.0
            terminated = True

        elif action == 2:  # COUNTER_OFFER
            self.round += 1
            self.time_elapsed += 1
            adjustment = 0.98 + 0.04 * self.np_random.random()
            self.counterparty_price = max(self.min_price, self.counterparty_price * adjustment)

            if self.round >= self.max_rounds:
                truncated = True
                reward = -0.5

        self.state = np.array([
            self.initial_price,
            self.counterparty_price,
            self.time_elapsed,
            self.round
        ], dtype=np.float32)

        return self.state, reward, terminated, truncated, {}

    def _compute_reward(self, price: float) -> float:
        if price < self.min_price:
            return -1.0
        negotiation_penalty = self.round / self.max_rounds
        return ((price - self.min_price) / (self.initial_price - self.min_price)) * (1 - negotiation_penalty)

    def render(self):
        print(f"Round {self.round} | Offer: ${self.counterparty_price:.2f} | Min: ${self.min_price:.2f}")

    def close(self):
        pass
