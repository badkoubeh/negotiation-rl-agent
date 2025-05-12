import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

class NegotiationEnv(gym.Env):
    """
    Custom Environment for IOI negotiation using Gymnasium interface.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, initial_price=10000, min_price=6000, max_rounds=5):
        super(NegotiationEnv, self).__init__()

        self.initial_price = initial_price
        self.min_price = min_price
        self.max_rounds = max_rounds

        # Observation space: [own_price, counterparty_price, time_elapsed, negotiation_round]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1e6, 1e6, 100, max_rounds], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: 0 = ACCEPT, 1 = REJECT, 2 = COUNTER_OFFER
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.round = 0

        # Seeding
        self.np_random = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.
        """
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.round = 0
        self.own_price = self.initial_price
        self.counterparty_price = self.initial_price * (0.9 + 0.1 * self.np_random.random())  # Slight variation
        self.time_elapsed = 0

        self.state = np.array([self.own_price, self.counterparty_price, self.time_elapsed, self.round], dtype=np.float32)
        return self.state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes an action and returns (obs, reward, terminated, truncated, info).
        """
        terminated = False
        truncated = False
        reward = 0.0

        if action == 0:  # ACCEPT
            if self.counterparty_price >= self.min_price:
                reward = self._compute_reward(self.counterparty_price)
            else:
                reward = -1.0  # Accepted a bad deal
            terminated = True

        elif action == 1:  # REJECT
            reward = -1.0
            terminated = True

        elif action == 2:  # COUNTER_OFFER
            self.round += 1
            self.time_elapsed += 1

            # Generate a new counter offer from the other party
            adjustment = 0.98 + 0.04 * self.np_random.random()
            self.counterparty_price = max(self.min_price, self.counterparty_price * adjustment)

            if self.round >= self.max_rounds:
                truncated = True  # Deal expired after too many rounds
                reward = -0.5

        # Update state
        self.state = np.array([self.own_price, self.counterparty_price, self.time_elapsed, self.round], dtype=np.float32)
        return self.state, reward, terminated, truncated, {}

    def _compute_reward(self, price: float) -> float:
        """
        Reward is higher when final price is closer to asking price (seller's perspective).
        """
        return (price - self.min_price) / (self.initial_price - self.min_price)

    def render(self):
        print(f"Round {self.round}: Counterparty offer: {self.counterparty_price:.2f}")

    def close(self):
        pass
