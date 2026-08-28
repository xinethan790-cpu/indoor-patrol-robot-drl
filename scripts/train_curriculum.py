#!/usr/bin/env python3
"""Train or resume PPO through a staged dynamic-obstacle curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv
from indoor_patrol_rl.training import load_curriculum


def load_ppo_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("algorithm") != "PPO":
        raise ValueError(f"Expected a PPO configuration in {path}.")
    return config


def build_model(
    *,
    config: dict[str, object],
    env: object,
    seed: int,
    resume: Path | None,
) -> PPO:
    if resume is not None:
        print(f"Resuming model from {resume}", flush=True)
        return PPO.load(resume, env=env, device=str(config["device"]))
    network = config["network"]
    if not isinstance(network, dict):
        raise ValueError("PPO network configuration must be a mapping.")
    return PPO(
        str(config["policy"]),
        env,
        device=str(config["device"]),
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        policy_kwargs={"net_arch": [int(width) for width in network["hidden_layers"]]},
        tensorboard_log="artifacts/logs/tensorboard",
        seed=seed,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-config", type=Path, default=Path("configs/train/ppo.yaml"))
    parser.add_argument(
        "--curriculum", type=Path, default=Path("configs/train/curriculum.yaml")
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/models/ppo_curriculum_seed11")
    )
    parser.add_argument(
        "--stage-step-limit",
        type=int,
        help="Cap every stage for a fast integration test without changing the YAML.",
    )
    args = parser.parse_args()

    ppo_config = load_ppo_config(args.ppo_config)
    curriculum = load_curriculum(args.curriculum)
    if not 0 <= args.start_stage < len(curriculum.stages):
        raise ValueError("start-stage is outside the configured curriculum.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = args.output.parent / f"{args.output.name}_checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    model: PPO | None = None
    old_env = None

    for stage_index, stage in enumerate(curriculum.stages[args.start_stage :], args.start_stage):
        stage_seed = args.seed + stage_index * 1000
        env = make_vec_env(
            DynamicObstacleAvoidanceEnv,
            n_envs=curriculum.n_envs,
            seed=stage_seed,
            env_kwargs=stage.env_kwargs,
        )
        if model is None:
            model = build_model(
                config=ppo_config,
                env=env,
                seed=args.seed,
                resume=args.resume,
            )
        else:
            model.set_env(env)
        if old_env is not None:
            old_env.close()
        old_env = env

        stage_timesteps = stage.timesteps
        if args.stage_step_limit is not None:
            stage_timesteps = min(stage_timesteps, args.stage_step_limit)
        callback = CheckpointCallback(
            save_freq=max(curriculum.checkpoint_freq // curriculum.n_envs, 1),
            save_path=str(checkpoint_directory),
            name_prefix=f"seed{args.seed}_{stage_index:02d}_{stage.name}",
        )
        print(
            f"CURRICULUM_STAGE_START index={stage_index} name={stage.name} "
            f"timesteps={stage_timesteps} env={stage.env_kwargs}",
            flush=True,
        )
        model.learn(
            total_timesteps=stage_timesteps,
            reset_num_timesteps=False,
            callback=callback,
            progress_bar=False,
            tb_log_name=f"ppo_curriculum_seed{args.seed}",
        )
        stage_output = args.output.parent / f"{args.output.name}_stage{stage_index}_{stage.name}"
        model.save(stage_output)
        model.save(args.output)
        state = {
            "seed": args.seed,
            "completed_stage_index": stage_index,
            "completed_stage_name": stage.name,
            "model_num_timesteps": model.num_timesteps,
            "latest_model": f"{args.output}.zip",
            "ppo_config": str(args.ppo_config),
            "curriculum_config": str(args.curriculum),
            "stage_env": stage.env_kwargs,
        }
        state_path = args.output.parent / f"{args.output.name}.state.yaml"
        state_path.write_text(yaml.safe_dump(state, sort_keys=False), encoding="utf-8")
        print(
            f"CURRICULUM_STAGE_OK index={stage_index} name={stage.name} "
            f"model_steps={model.num_timesteps} model={stage_output}.zip",
            flush=True,
        )

    if old_env is not None:
        old_env.close()
    print(f"CURRICULUM_TRAINING_OK model={args.output}.zip", flush=True)


if __name__ == "__main__":
    main()
