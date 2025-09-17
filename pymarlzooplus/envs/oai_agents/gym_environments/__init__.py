"""
Overcooked Gym Environments (simplified)

현재 리포지토리에서는 base_overcooked_env만 사용합니다.
"""

from .base_overcooked_env import OvercookedGymEnv, BonusOvercookedGymEnv

__version__ = "1.0.0"

__all__ = [
    'OvercookedGymEnv',
    'BonusOvercookedGymEnv',
]
