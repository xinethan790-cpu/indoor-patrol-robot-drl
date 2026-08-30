"""Diagnostic-only per-episode Webots context artifact.

This deliberately lives beside, rather than inside, :class:`EpisodeRecord` so
the stable ``episodes.csv`` v1.0 contract remains unchanged.  The JSON Lines
artifact is an additional debugging aid and is never part of validation or
held-out test output.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EPISODE_CONTEXT_FILENAME = "episode_context.jsonl"
EPISODE_CONTEXT_SCHEMA_VERSION = "1.1"
RISK_TRACE_WINDOW_STEPS = 16
RISK_TRACE_SAMPLE_SEMANTICS = (
    "pre_action_sector_and_interval_action_with_post_control_state_v1"
)
RISK_TRACE_COORDINATE_SEMANTICS = (
    "world_xy__robot_root=supervisor_root__robot_gps=gps_sensor__"
    "obstacle=supervisor_root_v1"
)
RISK_TRACE_SAMPLE_FIELDS = frozenset(
    {
        "step",
        "post_elapsed_s",
        "post_robot_root_xy",
        "post_robot_gps_xy",
        "post_robot_yaw",
        "post_goal_distance",
        "post_global_clearance",
        "pre_action_front_clearance",
        "pre_action_left_clearance",
        "pre_action_right_clearance",
        "post_lidar_argmin",
        "post_obstacle_xy",
        "post_obstacle_motion",
        "interval_policy_action",
        "interval_executed_action",
        "interval_commanded_linear",
        "interval_commanded_angular",
        "post_contact_collision",
        "post_lidar_collision",
    }
)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _numeric_vector(value: Any, field: str, *, length: int) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-value array.")
    for index, item in enumerate(value):
        _finite_number(item, f"{field}[{index}]")
    return value


def _validate_trace_sample(value: Any, field: str) -> None:
    if not isinstance(value, dict) or set(value) != RISK_TRACE_SAMPLE_FIELDS:
        raise ValueError(
            f"{field} must be an object with exactly the risk-trace sample fields."
        )
    step = value["step"]
    if type(step) is not int or step < 1:
        raise ValueError(f"{field}.step must be a positive integer.")
    for name in (
        "post_elapsed_s",
        "post_robot_yaw",
        "post_goal_distance",
        "post_global_clearance",
        "pre_action_front_clearance",
        "pre_action_left_clearance",
        "pre_action_right_clearance",
        "interval_commanded_linear",
        "interval_commanded_angular",
    ):
        number = _finite_number(value[name], f"{field}.{name}")
        if name not in {
            "post_robot_yaw",
            "interval_commanded_linear",
            "interval_commanded_angular",
        } and number < 0.0:
            raise ValueError(f"{field}.{name} cannot be negative.")
    _numeric_vector(
        value["post_robot_root_xy"], f"{field}.post_robot_root_xy", length=2
    )
    _numeric_vector(
        value["post_robot_gps_xy"], f"{field}.post_robot_gps_xy", length=2
    )
    _numeric_vector(
        value["interval_policy_action"], f"{field}.interval_policy_action", length=2
    )
    _numeric_vector(
        value["interval_executed_action"],
        f"{field}.interval_executed_action",
        length=2,
    )

    lidar_argmin = value["post_lidar_argmin"]
    if not isinstance(lidar_argmin, dict) or set(lidar_argmin) != {
        "range_m",
        "bearing_rad",
    }:
        raise ValueError(
            f"{field}.post_lidar_argmin must contain range_m and bearing_rad."
        )
    if (
        _finite_number(
            lidar_argmin["range_m"], f"{field}.post_lidar_argmin.range_m"
        )
        < 0.0
    ):
        raise ValueError(f"{field}.post_lidar_argmin.range_m cannot be negative.")
    _finite_number(
        lidar_argmin["bearing_rad"], f"{field}.post_lidar_argmin.bearing_rad"
    )

    obstacle_xy = value["post_obstacle_xy"]
    if not isinstance(obstacle_xy, list):
        raise ValueError(f"{field}.post_obstacle_xy must be an array.")
    for index, position in enumerate(obstacle_xy):
        _numeric_vector(position, f"{field}.post_obstacle_xy[{index}]", length=2)
    obstacle_motion = value["post_obstacle_motion"]
    if not isinstance(obstacle_motion, list) or len(obstacle_motion) != len(obstacle_xy):
        raise ValueError(
            f"{field}.post_obstacle_motion must align one-to-one with post_obstacle_xy."
        )
    for index, motion in enumerate(obstacle_motion):
        motion_field = f"{field}.post_obstacle_motion[{index}]"
        if not isinstance(motion, dict) or set(motion) != {
            "motion_axis",
            "speed_mps",
            "direction",
        }:
            raise ValueError(f"{motion_field} has an invalid structure.")
        if motion["motion_axis"] not in {"x", "y"}:
            raise ValueError(f"{motion_field}.motion_axis must be 'x' or 'y'.")
        if _finite_number(motion["speed_mps"], f"{motion_field}.speed_mps") < 0.0:
            raise ValueError(f"{motion_field}.speed_mps cannot be negative.")
        direction = _finite_number(motion["direction"], f"{motion_field}.direction")
        if direction not in {-1.0, 1.0}:
            raise ValueError(f"{motion_field}.direction must be -1 or 1.")
    for name in ("post_contact_collision", "post_lidar_collision"):
        if type(value[name]) is not bool:
            raise ValueError(f"{field}.{name} must be a boolean.")


def validate_risk_trace(value: Any, *, field: str = "risk_trace") -> None:
    """Strictly validate one bounded diagnostic risk trace."""

    if not isinstance(value, dict) or set(value) != {
        "sample_period_s",
        "sample_semantics",
        "coordinate_semantics",
        "lidar_min_range_m",
        "lidar_max_range_m",
        "window_steps",
        "terminal_window",
        "minimum_clearance_window",
    }:
        raise ValueError(f"{field} has an invalid structure.")
    if value["sample_semantics"] != RISK_TRACE_SAMPLE_SEMANTICS:
        raise ValueError(f"{field}.sample_semantics is invalid.")
    if value["coordinate_semantics"] != RISK_TRACE_COORDINATE_SEMANTICS:
        raise ValueError(f"{field}.coordinate_semantics is invalid.")
    if _finite_number(value["sample_period_s"], f"{field}.sample_period_s") <= 0.0:
        raise ValueError(f"{field}.sample_period_s must be positive.")
    lidar_minimum = _finite_number(
        value["lidar_min_range_m"], f"{field}.lidar_min_range_m"
    )
    lidar_maximum = _finite_number(
        value["lidar_max_range_m"], f"{field}.lidar_max_range_m"
    )
    if lidar_minimum < 0.0 or lidar_minimum >= lidar_maximum:
        raise ValueError(
            f"{field} LiDAR range bounds must satisfy 0 <= minimum < maximum."
        )
    if type(value["window_steps"]) is not int or value["window_steps"] != (
        RISK_TRACE_WINDOW_STEPS
    ):
        raise ValueError(
            f"{field}.window_steps must equal {RISK_TRACE_WINDOW_STEPS}."
        )
    for window_name in ("terminal_window", "minimum_clearance_window"):
        window = value[window_name]
        window_field = f"{field}.{window_name}"
        if (
            not isinstance(window, list)
            or not window
            or len(window) > value["window_steps"]
        ):
            raise ValueError(
                f"{window_field} must contain 1-{RISK_TRACE_WINDOW_STEPS} samples."
            )
        previous_step = 0
        previous_elapsed = -1.0
        for index, sample in enumerate(window):
            sample_field = f"{window_field}[{index}]"
            _validate_trace_sample(sample, sample_field)
            if sample["step"] <= previous_step:
                raise ValueError(f"{window_field} steps must be strictly increasing.")
            elapsed = float(sample["post_elapsed_s"])
            if elapsed <= previous_elapsed:
                raise ValueError(
                    f"{window_field} post_elapsed_s values must be strictly increasing."
                )
            previous_step = sample["step"]
            previous_elapsed = elapsed
    minimum_window = value["minimum_clearance_window"]
    if minimum_window[-1]["post_global_clearance"] != min(
        sample["post_global_clearance"] for sample in minimum_window
    ):
        raise ValueError(
            f"{field}.minimum_clearance_window must end at its minimum clearance sample."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = dict(record)
    if result.get("schema_version") != EPISODE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"Episode context row {index} must use schema version "
            f"{EPISODE_CONTEXT_SCHEMA_VERSION}."
        )
    identity = result.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"Episode context row {index} identity must be an object.")
    if identity.get("split") != "diagnostic":
        raise ValueError(
            f"Episode context row {index} must belong to the diagnostic split."
        )
    episode_id = identity.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(
            f"Episode context row {index} identity.episode_id must be non-empty."
        )
    validate_risk_trace(result.get("risk_trace"), field=f"Episode context row {index}.risk_trace")
    return result


def write_episode_context_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Atomically write diagnostic context and return its provenance binding."""

    if not records:
        raise ValueError("At least one diagnostic episode context record is required.")
    validated = [_validated_record(record, index) for index, record in enumerate(records)]
    episode_ids = [record["identity"]["episode_id"] for record in validated]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Diagnostic episode context episode_id values must be unique.")

    root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        portable_path = resolved_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Episode context artifact must remain inside the repository.") from error

    lines: list[str] = []
    for index, record in enumerate(validated):
        try:
            lines.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Episode context row {index} is not finite JSON data: {error}"
            ) from error

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_name(f".{resolved_path.name}.tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(resolved_path)
    return {
        "path": portable_path,
        "schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
        "count": len(validated),
        "sha256": _sha256_file(resolved_path),
    }


__all__ = [
    "EPISODE_CONTEXT_FILENAME",
    "EPISODE_CONTEXT_SCHEMA_VERSION",
    "RISK_TRACE_COORDINATE_SEMANTICS",
    "RISK_TRACE_SAMPLE_FIELDS",
    "RISK_TRACE_SAMPLE_SEMANTICS",
    "RISK_TRACE_WINDOW_STEPS",
    "validate_risk_trace",
    "write_episode_context_jsonl",
]
