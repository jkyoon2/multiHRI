import os
import argparse
import time
import numpy as np
import torch as th
import pygame

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from .policies.random_policy import RandomPolicy
from .policies.human_policy import HumanPolicy


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_policy(policy_name: str, env, seed: int, human_agent_ids: str):
    n_agents = env.n_agents
    if policy_name == "random":
        return RandomPolicy(n_agents=n_agents, n_actions=env.n_actions, seed=seed)
    if policy_name == "human":
        ids = [int(x) for x in human_agent_ids.split(",")] if human_agent_ids else [0]
        return HumanPolicy(n_agents=n_agents, control_agent_ids=ids)
    raise ValueError(f"Unknown policy: {policy_name}")


def rollouts(env_name: str,
             layout_name: str,
             num_episodes: int,
             shard_size: int,
             out_dir: str,
             seed: int = 0,
             policy: str = "random",
             human_agent_ids: str = None,
             render: bool = False):
    # Minimal args to env
    env = env_REGISTRY[env_name](
        layout_name=layout_name,
        num_agents=3,
        encoding_scheme="OAI_lossless",
        reward_type="shaped",
        horizon=400,
        render=False,
    )

    rng = np.random.default_rng(seed)
    pol = make_policy(policy, env, seed, human_agent_ids)
    pol.reset(env)
    buffer = []
    shard_idx = 0
    ensure_dir(out_dir)

    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            if policy == "human" and render:
                env.render()
                pygame.time.wait(50)

            obs_list = env.get_obs()
            actions = pol.act(obs_list, info=None)
            reward, done, info = env.step(actions)

            # Collect observations per agent
            obs_list = env.get_obs()
            for obs in obs_list:
                if not isinstance(obs, np.ndarray):
                    obs = np.asarray(obs)
                # Expect (C,H,W), store as uint8/uint16 safe range
                buffer.append(obs.astype(np.int16))

            if len(buffer) >= shard_size:
                shard_path = os.path.join(out_dir, f"shard_{shard_idx}.npz")
                np.savez_compressed(shard_path, visual_obs=np.stack(buffer, axis=0))
                buffer = []
                shard_idx += 1

    if len(buffer) > 0:
        shard_path = os.path.join(out_dir, f"shard_{shard_idx}.npz")
        np.savez_compressed(shard_path, visual_obs=np.stack(buffer, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="multi_overcooked")
    parser.add_argument("--layout", type=str, default="3_chefs_smartfactory")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "ae_dataset"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", type=str, default="random", choices=["random", "human"])
    parser.add_argument("--human-agent-ids", type=str, default=None, help="e.g., 0 or 0,1")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(args.out, args.env_name, args.layout, time.strftime("%Y%m%d_%H%M%S"))
    rollouts(args.env_name, args.layout, args.episodes, args.shard_size, out_dir,
             seed=args.seed, policy=args.policy, human_agent_ids=args.human_agent_ids, render=args.render)


if __name__ == "__main__":
    main()


