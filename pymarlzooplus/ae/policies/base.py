from typing import List, Any


class PolicyBase:
    def reset(self, env: Any):
        pass

    def act(self, obs_list: List[Any], info: Any = None) -> List[int]:
        raise NotImplementedError


