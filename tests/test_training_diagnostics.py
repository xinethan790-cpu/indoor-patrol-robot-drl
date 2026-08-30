from __future__ import annotations

import csv
from pathlib import Path

import pytest

from indoor_patrol_rl.evaluation.training_diagnostics import (
    read_training_monitor,
    training_diagnostic_report,
)

FIELDS = (
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
)


def _write_monitor(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write('#{"t_start": 1.0, "env_id": "None"}\n')
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    success: bool = False,
    collision: bool = False,
    timeout: bool = False,
    axis: str = "horizontal",
    reverse: bool = False,
    speed: float = 0.2,
    contact: bool = False,
    lidar: bool = False,
) -> dict[str, object]:
    return {
        "r": 30.0 if success else -30.0,
        "l": 100,
        "is_success": success,
        "collision": collision,
        "timeout": timeout,
        "episode_min_clearance": 0.2 if success else 0.0,
        "route_axis": axis,
        "reverse_route": reverse,
        "obstacle_speed_mean": speed,
        "contact_collision": contact,
        "lidar_collision": lidar,
    }


def test_report_groups_outcomes_routes_speeds_and_collision_sources(tmp_path: Path) -> None:
    monitor = tmp_path / "training.monitor.csv"
    _write_monitor(
        monitor,
        [
            _row(success=True, speed=0.20),
            _row(collision=True, speed=0.26, lidar=True),
            _row(
                collision=True,
                axis="vertical",
                reverse=True,
                speed=0.35,
                contact=True,
                lidar=True,
            ),
            _row(timeout=True, axis="vertical", speed=0.42),
        ],
    )

    report = training_diagnostic_report(
        monitor,
        speed_minimum=0.20,
        speed_maximum=0.42,
        speed_bin_count=4,
        window_size=2,
    )

    assert report["overall"]["episodes"] == 4
    assert report["overall"]["success_rate"] == pytest.approx(0.25)
    assert report["first_window"]["success_rate"] == pytest.approx(0.5)
    assert report["last_window"]["timeout_rate"] == pytest.approx(0.5)
    assert report["by_route_axis"]["horizontal"]["episodes"] == 2
    assert report["by_reverse_route"]["true"]["episodes"] == 1
    assert len(report["by_obstacle_speed"]) == 4
    sources = report["collision_detection"]
    assert sources["collision_episodes"] == 2
    assert sources["lidar_only"] == 1
    assert sources["contact_and_lidar"] == 1
    assert len(report["source_monitor_sha256"]) == 64


def test_reader_rejects_rows_without_exactly_one_outcome(tmp_path: Path) -> None:
    monitor = tmp_path / "invalid.monitor.csv"
    _write_monitor(monitor, [_row()])

    with pytest.raises(ValueError, match="exactly one terminal outcome"):
        read_training_monitor(monitor)


def test_report_rejects_speed_outside_declared_range(tmp_path: Path) -> None:
    monitor = tmp_path / "invalid-speed.monitor.csv"
    _write_monitor(monitor, [_row(success=True, speed=0.50)])

    with pytest.raises(ValueError, match="falls outside"):
        training_diagnostic_report(
            monitor,
            speed_minimum=0.20,
            speed_maximum=0.42,
        )
