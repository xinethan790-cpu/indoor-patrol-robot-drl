"""Versioned, planner-independent episode metrics and CSV serialization."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Outcome = Literal["success", "collision", "timeout", "error"]


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    episode_id: str
    split: str
    condition_id: str
    scenario_seed: int
    planner: str
    outcome: Outcome
    is_success: bool
    collision: bool
    timeout: bool
    step_count: int
    elapsed_sim_s: float
    completion_time_s: float | None
    episode_return: float
    path_length_m: float
    reference_path_length_m: float | None
    path_length_ratio: float | None
    minimum_clearance_m: float
    near_collision_ratio: float
    action_delta_sq_mean: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    error_type: str | None = None


class EpisodeAccumulator:
    """Accumulate step-level data without depending on one planner implementation."""

    def __init__(
        self,
        *,
        episode_id: str,
        split: str,
        condition_id: str,
        scenario_seed: int,
        planner: str,
        reference_path_length: float | None = None,
        near_collision_threshold: float = 0.20,
    ) -> None:
        if reference_path_length is not None and reference_path_length <= 0.0:
            raise ValueError("reference_path_length must be positive when provided.")
        if near_collision_threshold <= 0.0:
            raise ValueError("near_collision_threshold must be positive.")
        self.episode_id = episode_id
        self.split = split
        self.condition_id = condition_id
        self.scenario_seed = scenario_seed
        self.planner = planner
        self.reference_path_length = reference_path_length
        self.near_collision_threshold = near_collision_threshold
        self._episode_return = 0.0
        self._clearances: list[float] = []
        self._action_deltas: list[float] = []
        self._latencies_ms: list[float] = []
        self._previous_action: np.ndarray | None = None
        self._latest_info: dict[str, Any] | None = None

    def update(
        self,
        *,
        reward: float,
        info: dict[str, Any],
        inference_latency_s: float,
    ) -> None:
        required = {
            "is_success",
            "collision",
            "timeout",
            "clearance",
            "path_length",
            "step_count",
            "elapsed_sim_time",
            "executed_action",
        }
        missing = required - info.keys()
        if missing:
            raise ValueError(f"Episode info is missing required keys: {sorted(missing)}")
        if inference_latency_s < 0.0:
            raise ValueError("inference_latency_s cannot be negative.")

        action = np.asarray(info["executed_action"], dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("executed_action must be a finite two-value vector.")
        clearance = float(info["clearance"])
        if not np.isfinite(clearance):
            raise ValueError("clearance must be finite.")

        self._episode_return += float(reward)
        self._clearances.append(clearance)
        self._latencies_ms.append(1000.0 * inference_latency_s)
        if self._previous_action is not None:
            self._action_deltas.append(float(np.sum((action - self._previous_action) ** 2)))
        self._previous_action = action.copy()
        self._latest_info = dict(info)

    def finalize(self, *, error_type: str | None = None) -> EpisodeRecord:
        if self._latest_info is None:
            raise RuntimeError("Cannot finalize an episode without any recorded steps.")
        info = self._latest_info
        collision = bool(info["collision"])
        is_success = bool(info["is_success"]) and not collision
        timeout = bool(info["timeout"]) and not collision and not is_success
        if error_type is not None:
            outcome: Outcome = "error"
            collision = is_success = timeout = False
        elif collision:
            outcome = "collision"
        elif is_success:
            outcome = "success"
        elif timeout:
            outcome = "timeout"
        else:
            outcome = "error"
            error_type = "incomplete_episode"

        elapsed = float(info["elapsed_sim_time"])
        path_length = float(info["path_length"])
        path_ratio = None
        if is_success and self.reference_path_length is not None:
            path_ratio = path_length / self.reference_path_length
        minimum_clearance = min(self._clearances)
        near_collision_ratio = float(
            np.mean(np.asarray(self._clearances) < self.near_collision_threshold)
        )
        action_delta = float(np.mean(self._action_deltas)) if self._action_deltas else 0.0
        latencies = np.asarray(self._latencies_ms, dtype=np.float64)

        return EpisodeRecord(
            schema_version="1.0",
            episode_id=self.episode_id,
            split=self.split,
            condition_id=self.condition_id,
            scenario_seed=self.scenario_seed,
            planner=self.planner,
            outcome=outcome,
            is_success=is_success,
            collision=collision,
            timeout=timeout,
            step_count=int(info["step_count"]),
            elapsed_sim_s=elapsed,
            completion_time_s=elapsed if is_success else None,
            episode_return=self._episode_return,
            path_length_m=path_length,
            reference_path_length_m=self.reference_path_length,
            path_length_ratio=path_ratio,
            minimum_clearance_m=minimum_clearance,
            near_collision_ratio=near_collision_ratio,
            action_delta_sq_mean=action_delta,
            inference_latency_p50_ms=float(np.percentile(latencies, 50)),
            inference_latency_p95_ms=float(np.percentile(latencies, 95)),
            error_type=error_type,
        )


def write_episode_records(path: Path, records: list[EpisodeRecord]) -> None:
    """Write a stable CSV schema suitable for later thesis plotting."""

    if not records:
        raise ValueError("At least one episode record is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
