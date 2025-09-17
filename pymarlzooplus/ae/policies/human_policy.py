import pygame
from typing import Optional
from pygame.locals import (
    K_w, K_s, K_a, K_d, K_SPACE, K_q,
    K_i, K_k, K_j, K_l, K_RSHIFT, K_p,
    K_UP, K_DOWN, K_LEFT, K_RIGHT, K_RETURN, K_RCTRL,
)
from .base import PolicyBase


class HumanPolicy(PolicyBase):
    """
    Keyboard control for one or more agents with independent key mappings.

    Action indices assumed:
      0: NORTH, 1: SOUTH, 2: WEST, 3: EAST, 4: INTERACT, 5: STAY

    Default independent keymaps (up to 3 agents):
      - Agent 0: W/S/A/D, SPACE(interact), Q(stay)
      - Agent 1: I/K/J/L, RSHIFT(interact), P(stay)
      - Agent 2: Arrow keys, ENTER(interact), RCTRL(stay)
    """

    DEFAULT_KEYMAPS = [
        {"up": K_w, "down": K_s, "left": K_a, "right": K_d, "interact": K_SPACE, "stay": K_q},
        {"up": K_i, "down": K_k, "left": K_j, "right": K_l, "interact": K_RSHIFT, "stay": K_p},
        {"up": K_UP, "down": K_DOWN, "left": K_LEFT, "right": K_RIGHT, "interact": K_RETURN, "stay": K_RCTRL},
    ]

    def __init__(self, n_agents: int, control_agent_ids=None, keymaps=None):
        self.n_agents = n_agents
        self.control_agent_ids = control_agent_ids or [0]
        # Build per-agent keymaps. Uncontrolled agents don't read keys.
        self.keymaps = [None for _ in range(n_agents)]
        for idx, agent_id in enumerate(self.control_agent_ids):
            # Assign default map per controlled agent (cycle if more than defaults)
            km = (keymaps[idx] if keymaps and idx < len(keymaps)
                  else self.DEFAULT_KEYMAPS[min(idx, len(self.DEFAULT_KEYMAPS) - 1)])
            self.keymaps[agent_id] = km
        self.last_actions = [5] * n_agents  # default STAY
        self._inited = False

    def reset(self, env):
        if not self._inited:
            pygame.init()
            self._inited = True
        self.last_actions = [5] * self.n_agents

    def _action_from_keys(self, keys, keymap) -> Optional[int]:
        # Priority: interact > movement > stay
        if keymap is None:
            return None
        if keys[keymap["interact"]]:
            return 4
        if keys[keymap["up"]]:
            return 0
        if keys[keymap["down"]]:
            return 1
        if keys[keymap["left"]]:
            return 2
        if keys[keymap["right"]]:
            return 3
        if keys[keymap["stay"]]:
            return 5
        return None

    def act(self, obs_list, info=None):
        actions = list(self.last_actions)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        keys = pygame.key.get_pressed()
        for agent_id in self.control_agent_ids:
            a = self._action_from_keys(keys, self.keymaps[agent_id])
            if a is not None:
                actions[agent_id] = a
        self.last_actions = actions
        return actions


