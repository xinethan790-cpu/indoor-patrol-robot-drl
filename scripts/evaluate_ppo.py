#!/usr/bin/env python3
"""Evaluate a saved PPO model on deterministic prototype scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=5_000,
        help="First validation seed; final test seeds 10000-10019 are reserved.",
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = DynamicObstacleAvoidanceEnv()
    successes = 0
    collisions = 0

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
        successes += int(info["is_success"])
        collisions += int(info["collision"])

    print(
        f"episodes={args.episodes} successes={successes} collisions={collisions} "
        f"timeouts={args.episodes - successes - collisions}"
    )


if __name__ == "__main__":
    main()
