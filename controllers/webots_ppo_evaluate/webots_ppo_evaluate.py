"""Deterministically evaluate a trained PPO model on Webots validation seeds."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

from controller import Supervisor
from stable_baselines3 import PPO

from indoor_patrol_rl.envs import WebotsPatrolEnv
from indoor_patrol_rl.evaluation import EpisodeAccumulator, write_benchmark_outputs

ALL_OBSTACLE_DEFS = ("TRAIN_OBSTACLE_0", "TRAIN_OBSTACLE_1", "TRAIN_OBSTACLE_2")


def environment_flag(name: str, *, default: bool = False) -> bool:
    """Parse an explicit environment boolean and reject ambiguous values."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off.")


def main() -> None:
    repository_root = Path(__file__).absolute().parents[2]
    supervisor = Supervisor()
    exit_code = 1
    env = None
    try:
        training_seed = int(os.environ.get("WEBOTS_MODEL_SEED", "11"))
        seed_start = int(os.environ.get("WEBOTS_EVAL_SEED_START", "5000"))
        episode_count = int(os.environ.get("WEBOTS_EVAL_EPISODES", "20"))
        obstacle_count = int(os.environ.get("WEBOTS_OBSTACLE_COUNT", "1"))
        obstacle_speed_range = (
            float(os.environ.get("WEBOTS_OBSTACLE_SPEED_MIN", "0.20")),
            float(os.environ.get("WEBOTS_OBSTACLE_SPEED_MAX", "0.42")),
        )
        horizontal_route_half_length = float(
            os.environ.get("WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH", "1.55")
        )
        vertical_route_half_length = float(
            os.environ.get("WEBOTS_VERTICAL_ROUTE_HALF_LENGTH", "1.10")
        )
        safety_shield_enabled = environment_flag("WEBOTS_SAFETY_SHIELD")
        split = os.environ.get("WEBOTS_EVAL_SPLIT", "validation")
        condition_id = os.environ.get(
            "WEBOTS_EVAL_CONDITION", f"W{obstacle_count}"
        )
        planner_label = os.environ.get(
            "WEBOTS_EVAL_PLANNER_LABEL",
            f"ppo_seed{training_seed}{'_shield' if safety_shield_enabled else ''}",
        )
        if split == "test" and os.environ.get("WEBOTS_CONFIRM_TEST") != "1":
            raise PermissionError("Held-out Webots test evaluation requires WEBOTS_CONFIRM_TEST=1.")
        if not 0 <= obstacle_count <= len(ALL_OBSTACLE_DEFS):
            raise ValueError("WEBOTS_OBSTACLE_COUNT must be between 0 and 3.")
        for def_name in ALL_OBSTACLE_DEFS[obstacle_count:]:
            supervisor.getFromDef(def_name).getField("translation").setSFVec3f([0.0, 0.0, -5.0])

        env = WebotsPatrolEnv(
            supervisor,
            obstacle_defs=ALL_OBSTACLE_DEFS[:obstacle_count],
            action_repeat=4,
            max_episode_steps=300,
            enable_safety_shield=safety_shield_enabled,
            obstacle_speed_range=obstacle_speed_range,
            horizontal_route_half_length=horizontal_route_half_length,
            vertical_route_half_length=vertical_route_half_length,
        )
        default_model = (
            f"artifacts/models/webots/seed_{training_seed}/ppo_webots_latest.zip"
        )
        model_value = os.environ.get(
            "WEBOTS_EVAL_MODEL", default_model
        )
        model_path = repository_root / model_value
        model = PPO.load(model_path, device="cpu")
        records = []
        default_run_name = (
            f"webots_{split}_seed{training_seed}_obstacles{obstacle_count}"
            f"{'_shield' if safety_shield_enabled else ''}"
        )
        run_name = os.environ.get("WEBOTS_EVAL_RUN_NAME", default_run_name)
        if Path(run_name).name != run_name:
            raise ValueError("WEBOTS_EVAL_RUN_NAME must be a single directory name.")
        output_directory = repository_root / "artifacts/results" / run_name

        for episode_index in range(episode_count):
            scenario_seed = seed_start + episode_index
            observation, initial_info = env.reset(seed=scenario_seed)
            accumulator = EpisodeAccumulator(
                episode_id=(
                    f"{split}-{condition_id}-{scenario_seed}-{planner_label}"
                ),
                split=split,
                condition_id=condition_id,
                scenario_seed=scenario_seed,
                planner=planner_label,
                reference_path_length=float(initial_info["distance_to_goal"]),
            )
            terminated = truncated = False
            while not (terminated or truncated):
                started = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                latency = time.perf_counter() - started
                observation, reward, terminated, truncated, info = env.step(action)
                accumulator.update(
                    reward=reward,
                    info=info,
                    inference_latency_s=latency,
                )
            record = accumulator.finalize()
            records.append(record)
            write_benchmark_outputs(output_directory, records)
            print(
                f"WEBOTS_EVAL_EPISODE index={episode_index} seed={scenario_seed} "
                f"outcome={record.outcome} steps={record.step_count}",
                flush=True,
            )

        summary = write_benchmark_outputs(output_directory, records)
        planner_summary = summary["by_planner"][0]
        print(
            f"WEBOTS_EVALUATION_OK episodes={episode_count} "
            f"shield={int(safety_shield_enabled)} "
            f"success_rate={planner_summary['success_rate']:.3f} "
            f"collision_rate={planner_summary['collision_rate']:.3f} "
            f"results={output_directory}",
            flush=True,
        )
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        supervisor.step(int(supervisor.getBasicTimeStep()))
        supervisor.simulationQuit(exit_code)


if __name__ == "__main__":
    main()
