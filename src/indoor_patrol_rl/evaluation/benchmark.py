"""Reproducible, planner-independent kinematic benchmark runner."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from indoor_patrol_rl.envs import DynamicObstacleAvoidanceEnv
from indoor_patrol_rl.evaluation.metrics import (
    EpisodeAccumulator,
    EpisodeRecord,
    write_episode_records,
)
from indoor_patrol_rl.policies import DWAConfig, DWAPlanner, ReactiveAvoidancePolicy


class PredictPolicy(Protocol):
    """Small common interface implemented by all benchmark policies."""

    def predict(self, observation: np.ndarray) -> np.ndarray: ...


PolicyBuilder = Callable[[DynamicObstacleAvoidanceEnv], PredictPolicy]
ProgressCallback = Callable[[EpisodeRecord, int, int], None]


@dataclass(frozen=True)
class SeedSet:
    start: int
    count: int

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.start + self.count))


@dataclass(frozen=True)
class BenchmarkCondition:
    condition_id: str
    env_kwargs: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfig:
    seed_sets: dict[str, SeedSet]
    planners: tuple[str, ...]
    conditions: tuple[BenchmarkCondition, ...]
    ppo_model: Path | None


class _StableBaselinesPolicy:
    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load and validate the portions of ``benchmark.yaml`` used by this backend."""

    with path.open(encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), str(path))

    raw_seed_sets = _mapping(raw.get("seed_sets"), "seed_sets")
    seed_sets: dict[str, SeedSet] = {}
    for name, value in raw_seed_sets.items():
        seed_mapping = _mapping(value, f"seed_sets.{name}")
        seed_set = SeedSet(start=int(seed_mapping["start"]), count=int(seed_mapping["count"]))
        if seed_set.start < 0 or seed_set.count <= 0:
            raise ValueError(f"seed_sets.{name} must have start >= 0 and count > 0.")
        seed_sets[str(name)] = seed_set
    if "validation" not in seed_sets or "test" not in seed_sets:
        raise ValueError("seed_sets must define both validation and test.")

    planners = tuple(str(name) for name in raw.get("planners", ()))
    if not planners or len(set(planners)) != len(planners):
        raise ValueError("planners must be a non-empty list of unique names.")

    backend = _mapping(raw.get("backend"), "backend")
    kinematic = _mapping(backend.get("kinematic"), "backend.kinematic")
    defaults = _mapping(kinematic.get("defaults", {}), "backend.kinematic.defaults")

    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions must be a non-empty list.")
    conditions: list[BenchmarkCondition] = []
    for index, value in enumerate(raw_conditions):
        condition = _mapping(value, f"conditions[{index}]")
        condition_id = str(condition["id"])
        overrides = _mapping(condition.get("kinematic", {}), f"conditions[{index}].kinematic")
        kwargs = {**defaults, **overrides}
        if "obstacle_speed_range" in kwargs:
            speeds = kwargs["obstacle_speed_range"]
            if not isinstance(speeds, (list, tuple)) or len(speeds) != 2:
                raise ValueError("obstacle_speed_range must contain two values.")
            kwargs["obstacle_speed_range"] = (float(speeds[0]), float(speeds[1]))
        conditions.append(BenchmarkCondition(condition_id=condition_id, env_kwargs=kwargs))
    condition_ids = [condition.condition_id for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition ids must be unique.")

    models = _mapping(raw.get("models", {}), "models")
    ppo_value = models.get("ppo")
    ppo_model = Path(str(ppo_value)) if ppo_value else None
    return BenchmarkConfig(
        seed_sets=seed_sets,
        planners=planners,
        conditions=tuple(conditions),
        ppo_model=ppo_model,
    )


def require_test_confirmation(split: str, *, confirmed: bool) -> None:
    """Protect the held-out test seeds from accidental tuning runs."""

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    if split == "test" and not confirmed:
        raise PermissionError(
            "The held-out test split requires the explicit --confirm-test flag. "
            "Use validation for model or parameter selection."
        )


def _default_policy_builders(
    *, ppo_model: Path | None
) -> dict[str, PolicyBuilder]:
    ppo_cache: dict[str, PredictPolicy] = {}

    def build_reactive(env: DynamicObstacleAvoidanceEnv) -> PredictPolicy:
        return ReactiveAvoidancePolicy(lidar_beams=env.lidar_beams)

    def build_dwa(env: DynamicObstacleAvoidanceEnv) -> PredictPolicy:
        wheel_radius = 0.033
        axle_length = 0.16
        required_wheel_speed = (
            env.max_linear_speed + 0.5 * axle_length * env.max_angular_speed
        ) / wheel_radius
        config = DWAConfig(
            lidar_beams=env.lidar_beams,
            lidar_max_range=env.lidar_range,
            goal_distance_scale=env.world_diagonal,
            max_linear_speed=env.max_linear_speed,
            max_angular_speed=env.max_angular_speed,
            max_linear_acceleration=1.5,
            max_angular_acceleration=3.0,
            wheel_radius=wheel_radius,
            axle_length=axle_length,
            max_wheel_speed=required_wheel_speed,
            control_dt=env.dt,
            robot_radius=env.robot_radius,
            safety_margin=env.safety_distance,
        )
        return DWAPlanner(config)

    def build_ppo(_env: DynamicObstacleAvoidanceEnv) -> PredictPolicy:
        if "ppo" not in ppo_cache:
            if ppo_model is None:
                raise ValueError("A PPO model path is required when planner 'ppo' is selected.")
            if not ppo_model.is_file():
                raise FileNotFoundError(
                    f"PPO model not found: {ppo_model}. Train it first or pass --ppo-model."
                )
            from stable_baselines3 import PPO

            ppo_cache["ppo"] = _StableBaselinesPolicy(PPO.load(ppo_model, device="cpu"))
        return ppo_cache["ppo"]

    return {"reactive": build_reactive, "dwa": build_dwa, "ppo": build_ppo}


def run_kinematic_benchmark(
    config: BenchmarkConfig,
    *,
    split: str = "validation",
    planners: Sequence[str] | None = None,
    episode_limit: int | None = None,
    ppo_model: Path | None = None,
    policy_builders: Mapping[str, PolicyBuilder] | None = None,
    progress: ProgressCallback | None = None,
) -> list[EpisodeRecord]:
    """Run all selected planners on identical seeded scenarios."""

    if split not in config.seed_sets:
        raise ValueError(f"Unknown split {split!r}; available: {sorted(config.seed_sets)}")
    selected = tuple(planners or config.planners)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("At least one unique planner must be selected.")
    seeds = config.seed_sets[split].seeds
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ValueError("episode_limit must be positive.")
        seeds = seeds[:episode_limit]

    builders = dict(
        policy_builders
        or _default_policy_builders(ppo_model=ppo_model or config.ppo_model)
    )
    missing = set(selected) - builders.keys()
    if missing:
        raise ValueError(f"No policy builder registered for: {sorted(missing)}")

    total = len(config.conditions) * len(seeds) * len(selected)
    records: list[EpisodeRecord] = []
    for condition in config.conditions:
        template_env = DynamicObstacleAvoidanceEnv(**condition.env_kwargs)
        policies = {name: builders[name](template_env) for name in selected}
        template_env.close()
        for scenario_seed in seeds:
            for planner_name in selected:
                env = DynamicObstacleAvoidanceEnv(**condition.env_kwargs)
                observation, _ = env.reset(seed=scenario_seed)
                reference_length = float(
                    np.linalg.norm(env.goal_position - env.robot_position)
                )
                accumulator = EpisodeAccumulator(
                    episode_id=f"{split}-{condition.condition_id}-{scenario_seed}-{planner_name}",
                    split=split,
                    condition_id=condition.condition_id,
                    scenario_seed=scenario_seed,
                    planner=planner_name,
                    reference_path_length=reference_length,
                )
                terminated = truncated = False
                while not (terminated or truncated):
                    started = time.perf_counter()
                    action = policies[planner_name].predict(observation)
                    latency = time.perf_counter() - started
                    observation, reward, terminated, truncated, info = env.step(action)
                    accumulator.update(
                        reward=reward,
                        info=info,
                        inference_latency_s=latency,
                    )
                env.close()
                record = accumulator.finalize()
                records.append(record)
                if progress is not None:
                    progress(record, len(records), total)
    return records


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _summarize_group(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    count = len(records)
    successes = [record for record in records if record.is_success]
    return {
        "episodes": count,
        "success_count": len(successes),
        "collision_count": sum(record.collision for record in records),
        "timeout_count": sum(record.timeout for record in records),
        "error_count": sum(record.outcome == "error" for record in records),
        "success_rate": sum(record.is_success for record in records) / count,
        "collision_rate": sum(record.collision for record in records) / count,
        "timeout_rate": sum(record.timeout for record in records) / count,
        "completion_time_s_mean": _mean(
            [record.completion_time_s for record in successes]
        ),
        "path_length_ratio_mean": _mean(
            [record.path_length_ratio for record in successes]
        ),
        "minimum_clearance_m_mean": _mean(
            [record.minimum_clearance_m for record in records]
        ),
        "near_collision_ratio_mean": _mean(
            [record.near_collision_ratio for record in records]
        ),
        "action_delta_sq_mean": _mean(
            [record.action_delta_sq_mean for record in records]
        ),
        "inference_latency_p95_ms": float(
            np.percentile(
                [record.inference_latency_p95_ms for record in records], 95
            )
        ),
    }


def summarize_records(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one episode record is required.")
    split_values = {record.split for record in records}
    if len(split_values) != 1:
        raise ValueError("A summary cannot mix validation and test records.")
    planners = sorted({record.planner for record in records})
    conditions = sorted({record.condition_id for record in records})
    by_planner_condition = []
    for planner in planners:
        for condition in conditions:
            group = [
                record
                for record in records
                if record.planner == planner and record.condition_id == condition
            ]
            if group:
                by_planner_condition.append(
                    {"planner": planner, "condition_id": condition, **_summarize_group(group)}
                )
    by_planner = [
        {
            "planner": planner,
            **_summarize_group([record for record in records if record.planner == planner]),
        }
        for planner in planners
    ]
    return {
        "schema_version": "1.0",
        "split": next(iter(split_values)),
        "episode_count": len(records),
        "planners": planners,
        "conditions": conditions,
        "by_planner_condition": by_planner_condition,
        "by_planner": by_planner,
    }


def write_benchmark_outputs(output_dir: Path, records: list[EpisodeRecord]) -> dict[str, Any]:
    """Write the required raw episode table and machine-readable aggregate."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_episode_records(output_dir / "episodes.csv", records)
    summary = summarize_records(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
