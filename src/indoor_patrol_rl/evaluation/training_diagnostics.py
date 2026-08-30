"""Deterministic diagnostics for Stable-Baselines3 Monitor episode logs."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingEpisode:
    """One completed Webots training episode from a Monitor CSV file."""

    episode_return: float
    episode_steps: int
    success: bool
    collision: bool
    timeout: bool
    minimum_clearance: float
    route_axis: str
    reverse_route: bool
    obstacle_speed_mean: float
    contact_collision: bool
    lidar_collision: bool


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = value.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Monitor row {row_number} field {field!r} is not a boolean: {value!r}.")


def _finite_float(value: str, *, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"Monitor row {row_number} field {field!r} is not numeric: {value!r}."
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"Monitor row {row_number} field {field!r} is non-finite: {value!r}.")
    return result


def read_training_monitor(path: Path) -> list[TrainingEpisode]:
    """Read and strictly validate completed episodes from an SB3 Monitor CSV."""

    required_fields = {
        "r",
        "l",
        "is_success",
        "collision",
        "timeout",
        "episode_min_clearance",
        "route_axis",
        "reverse_route",
        "obstacle_speed_mean",
        "contact_collision",
        "lidar_collision",
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        non_comment_lines = (line for line in stream if not line.startswith("#"))
        reader = csv.DictReader(non_comment_lines)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Monitor CSV is missing fields: {sorted(missing)}.")
        episodes: list[TrainingEpisode] = []
        for row_number, row in enumerate(reader, start=2):
            success = _parse_bool(row["is_success"], field="is_success", row_number=row_number)
            collision = _parse_bool(row["collision"], field="collision", row_number=row_number)
            timeout = _parse_bool(row["timeout"], field="timeout", row_number=row_number)
            if sum((success, collision, timeout)) != 1:
                raise ValueError(
                    f"Monitor row {row_number} must have exactly one terminal outcome."
                )
            route_axis = row["route_axis"].strip().casefold()
            if route_axis not in {"horizontal", "vertical"}:
                raise ValueError(f"Monitor row {row_number} has invalid route_axis {route_axis!r}.")
            episode_steps_value = _finite_float(row["l"], field="l", row_number=row_number)
            episode_steps = int(episode_steps_value)
            if episode_steps <= 0 or episode_steps != episode_steps_value:
                raise ValueError(f"Monitor row {row_number} field 'l' must be a positive integer.")
            episodes.append(
                TrainingEpisode(
                    episode_return=_finite_float(row["r"], field="r", row_number=row_number),
                    episode_steps=episode_steps,
                    success=success,
                    collision=collision,
                    timeout=timeout,
                    minimum_clearance=_finite_float(
                        row["episode_min_clearance"],
                        field="episode_min_clearance",
                        row_number=row_number,
                    ),
                    route_axis=route_axis,
                    reverse_route=_parse_bool(
                        row["reverse_route"],
                        field="reverse_route",
                        row_number=row_number,
                    ),
                    obstacle_speed_mean=_finite_float(
                        row["obstacle_speed_mean"],
                        field="obstacle_speed_mean",
                        row_number=row_number,
                    ),
                    contact_collision=_parse_bool(
                        row["contact_collision"],
                        field="contact_collision",
                        row_number=row_number,
                    ),
                    lidar_collision=_parse_bool(
                        row["lidar_collision"],
                        field="lidar_collision",
                        row_number=row_number,
                    ),
                )
            )
    if not episodes:
        raise ValueError("Monitor CSV contains no completed episodes.")
    return episodes


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _aggregate(episodes: Sequence[TrainingEpisode]) -> dict[str, int | float]:
    if not episodes:
        raise ValueError("Cannot aggregate an empty episode collection.")
    count = len(episodes)
    successes = sum(episode.success for episode in episodes)
    collisions = sum(episode.collision for episode in episodes)
    timeouts = sum(episode.timeout for episode in episodes)
    return {
        "episodes": count,
        "completed_environment_steps": sum(episode.episode_steps for episode in episodes),
        "success_count": successes,
        "collision_count": collisions,
        "timeout_count": timeouts,
        "success_rate": successes / count,
        "collision_rate": collisions / count,
        "timeout_rate": timeouts / count,
        "episode_return_mean": _mean(episode.episode_return for episode in episodes),
        "episode_steps_mean": _mean(float(episode.episode_steps) for episode in episodes),
        "minimum_clearance_mean": _mean(episode.minimum_clearance for episode in episodes),
        "obstacle_speed_mean": _mean(episode.obstacle_speed_mean for episode in episodes),
    }


def _grouped(episodes: Sequence[TrainingEpisode], key: Any) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[TrainingEpisode]] = {}
    for episode in episodes:
        groups.setdefault(str(key(episode)), []).append(episode)
    return {label: _aggregate(groups[label]) for label in sorted(groups)}


def _speed_groups(
    episodes: Sequence[TrainingEpisode],
    *,
    speed_minimum: float,
    speed_maximum: float,
    bin_count: int,
) -> dict[str, dict[str, int | float]]:
    if not (
        math.isfinite(speed_minimum)
        and math.isfinite(speed_maximum)
        and 0.0 <= speed_minimum < speed_maximum
    ):
        raise ValueError("Speed bounds must be finite, non-negative, and increasing.")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    width = (speed_maximum - speed_minimum) / bin_count
    groups: dict[str, list[TrainingEpisode]] = {}
    for episode in episodes:
        speed = episode.obstacle_speed_mean
        tolerance = 1e-12
        if speed < speed_minimum - tolerance or speed > speed_maximum + tolerance:
            raise ValueError(
                f"Obstacle speed {speed} falls outside [{speed_minimum}, {speed_maximum}]."
            )
        index = min(int((speed - speed_minimum) / width), bin_count - 1)
        lower = speed_minimum + index * width
        upper = speed_minimum + (index + 1) * width
        closing = "]" if index == bin_count - 1 else ")"
        label = f"[{lower:.6g},{upper:.6g}{closing}"
        groups.setdefault(label, []).append(episode)
    return {label: _aggregate(groups[label]) for label in sorted(groups)}


def _collision_sources(episodes: Sequence[TrainingEpisode]) -> dict[str, int | float]:
    collisions = [episode for episode in episodes if episode.collision]
    counts = {
        "contact_only": sum(
            episode.contact_collision and not episode.lidar_collision for episode in collisions
        ),
        "lidar_only": sum(
            episode.lidar_collision and not episode.contact_collision for episode in collisions
        ),
        "contact_and_lidar": sum(
            episode.contact_collision and episode.lidar_collision for episode in collisions
        ),
        "unclassified": sum(
            not episode.contact_collision and not episode.lidar_collision for episode in collisions
        ),
    }
    denominator = len(collisions)
    return {
        "collision_episodes": denominator,
        **counts,
        **{
            f"{name}_ratio": (value / denominator if denominator else 0.0)
            for name, value in counts.items()
        },
    }


def training_diagnostic_report(
    path: Path,
    *,
    speed_minimum: float,
    speed_maximum: float,
    speed_bin_count: int = 4,
    window_size: int = 100,
) -> dict[str, Any]:
    """Build an auditable failure-analysis report for one training segment."""

    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    episodes = read_training_monitor(path)
    window = min(window_size, len(episodes))
    return {
        "schema_version": 1,
        "source_monitor": str(path),
        "source_monitor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "window_size": window,
        "overall": _aggregate(episodes),
        "first_window": _aggregate(episodes[:window]),
        "last_window": _aggregate(episodes[-window:]),
        "by_route_axis": _grouped(episodes, lambda episode: episode.route_axis),
        "by_reverse_route": _grouped(episodes, lambda episode: str(episode.reverse_route).lower()),
        "by_obstacle_speed": _speed_groups(
            episodes,
            speed_minimum=speed_minimum,
            speed_maximum=speed_maximum,
            bin_count=speed_bin_count,
        ),
        "collision_detection": _collision_sources(episodes),
    }


__all__ = ["TrainingEpisode", "read_training_monitor", "training_diagnostic_report"]
