import numpy as np
from .base import PolicyBase


class RandomPolicy(PolicyBase):
    def __init__(self, n_agents: int, n_actions: int, seed: int = 0):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs_list, info=None):
        return [int(self.rng.integers(0, self.n_actions)) for _ in range(self.n_agents)]


