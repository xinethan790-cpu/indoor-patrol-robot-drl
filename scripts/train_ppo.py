#!/usr/bin/env python3
"""Train a small PPO policy in the fast kinematic prototype environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train/ppo.yaml"))
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/ppo_kinematic"))
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("algorithm") != "PPO":
        raise ValueError(f"Expected a PPO mapping in {args.config}.")

    seeds = config.get("seeds", [42])
    seed = int(args.seed if args.seed is not None else seeds[0])
    n_envs = int(args.n_envs if args.n_envs is not None else 4)
    n_steps = int(args.n_steps if args.n_steps is not None else config["n_steps"])
    total_timesteps = int(
        args.timesteps if args.timesteps is not None else config["total_timesteps"]
    )
    hidden_layers = [int(width) for width in config["network"]["hidden_layers"]]

    env = make_vec_env(DynamicObstacleAvoidanceEnv, n_envs=n_envs, seed=seed)
    model = PPO(
        str(config["policy"]),
        env,
        device=str(config["device"]),
        learning_rate=float(config["learning_rate"]),
        n_steps=n_steps,
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        policy_kwargs={"net_arch": hidden_layers},
        verbose=1,
        seed=seed,
        tensorboard_log="artifacts/logs/tensorboard",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    resolved_config = {
        **config,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "seed": seed,
        "source_config": str(args.config),
    }
    resolved_path = args.output.parent / f"{args.output.name}.config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    env.close()
    print(f"Saved model to {args.output}.zip and resolved config to {resolved_path}")


if __name__ == "__main__":
    main()
