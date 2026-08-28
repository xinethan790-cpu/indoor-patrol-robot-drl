#!/usr/bin/env python3
"""Run deterministic smoke episodes with the rule-based baseline."""

from __future__ import annotations

import argparse

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv
from indoor_patrol_rl.policies import ReactiveAvoidancePolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = DynamicObstacleAvoidanceEnv(max_episode_steps=300)
    policy = ReactiveAvoidancePolicy(env.lidar_beams)
    successes = 0
    collisions = 0

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        terminated = truncated = False
        info = {}
        episode_return = 0.0
        while not (terminated or truncated):
            action = policy.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        successes += int(info["is_success"])
        collisions += int(info["collision"])
        print(
            f"episode={episode + 1} return={episode_return:.2f} "
            f"success={info['is_success']} collision={info['collision']} "
            f"steps={info['step_count']}"
        )

    print(f"summary episodes={args.episodes} successes={successes} collisions={collisions}")


if __name__ == "__main__":
    main()
