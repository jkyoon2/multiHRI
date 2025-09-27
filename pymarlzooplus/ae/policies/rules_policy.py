from typing import List

from .base import PolicyBase
from pymarlzooplus.envs.oai_agents.policies.rule_based import RuleBasedPlanner


class RulesPolicy(PolicyBase):
    """Adapter to use RuleBasedPlanner in data collection (PolicyBase interface)."""

    def __init__(self, mode: str = "default", seed: int = 0):
        self.mode = mode
        self.planner = RuleBasedPlanner(rng=None)

    def reset(self, env):
        # could adapt planner settings by mode here
        pass

    def act(self, obs_list: List, info=None) -> List[int]:
        # obs_list is unused; the planner queries env state directly when wired in collector
        # The collector should pass env to this policy to act; we hack by attaching later if needed
        raise NotImplementedError("RulesPolicy needs env reference to compute actions")


