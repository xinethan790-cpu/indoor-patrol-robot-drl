#!/usr/bin/env python3
"""Export the verified graduation patrol run for the WeChat demo replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FIXTURE_SCHEMA_VERSION = "wechat-patrol-fixture/v1"
DASHBOARD_SCHEMA_VERSION = "patrol-dashboard/v1"
EXPECTED_RUN_ID = "graduation-demo-seed7000-v1"
DEFAULT_MAX_FRAMES = 90
_DYNAMIC_OBSTACLE_SPECS = (
    ("DYNAMIC_OBSTACLE_0", 0.18),
    ("DYNAMIC_OBSTACLE_1", 0.17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).absolute().parents[1]
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=root / "artifacts/results" / EXPECTED_RUN_ID,
        help="Verified terminal patrol result directory.",
    )
    parser.add_argument(
        "--map-config",
        type=Path,
        default=root / "configs/maps/patrol_route.yaml",
        help="Static patrol map used by the run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "wechat-miniprogram/data" / f"{EXPECTED_RUN_ID}.json",
        help="Destination fixture JSON.",
    )
    parser.add_argument(
        "--module-output",
        type=Path,
        default=root / "wechat-miniprogram/data" / f"{EXPECTED_RUN_ID}.js",
        help="Destination CommonJS fixture used by the WeChat runtime.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum exported replay frames (2-120, default 90).",
    )
    return parser.parse_args()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path.name} contains a blank record at line {line_number}")
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        records.append(raw)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_signature(event: Mapping[str, Any]) -> tuple[object, ...]:
    progress = event["checkpoint_progress"]
    readings = event["latest_readings"]
    navigation = event["local_navigation"]
    return (
        event["state"],
        event["outcome"],
        progress["completed_checkpoint_count"],
        progress["current_checkpoint_id"],
        readings["checkpoint_id"],
        tuple(alert["rule_id"] for alert in event["active_alerts"]),
        navigation["local_goal"] is None,
    )


def _sample_indices(
    events: Sequence[Mapping[str, Any]],
    limit: int,
    *,
    extra_mandatory: Sequence[int] = (),
) -> list[int]:
    if len(events) <= limit:
        return list(range(len(events)))

    mandatory = {0, len(events) - 1, *extra_mandatory}
    previous = _frame_signature(events[0])
    for index, event in enumerate(events[1:], 1):
        signature = _frame_signature(event)
        if signature != previous:
            mandatory.add(index)
            mandatory.add(index - 1)
        previous = signature
    if len(mandatory) > limit:
        raise ValueError(
            f"max-frames={limit} cannot preserve {len(mandatory)} state transitions"
        )

    selected = set(mandatory)
    target_count = limit
    if target_count > len(selected):
        slots = target_count - len(selected)
        candidates = [
            round(step * (len(events) - 1) / (slots + 1))
            for step in range(1, slots + 1)
        ]
        selected.update(candidates)

    # Rounding or mandatory overlap can leave holes. Fill the widest temporal
    # gaps first so the visual replay remains smooth without losing key events.
    while len(selected) < target_count:
        ordered = sorted(selected)
        gap_start, gap_end = max(
            zip(ordered, ordered[1:], strict=False),
            key=lambda pair: pair[1] - pair[0],
        )
        if gap_end - gap_start <= 1:
            break
        selected.add((gap_start + gap_end) // 2)
    return sorted(selected)


def _empty_avoidance() -> dict[str, Any]:
    return {
        "available": False,
        "measured_control_step": None,
        "measured_at_s": None,
        "minimum_lidar_range_m": None,
        "robust_ttc_s": None,
        "closing_rate_mps": None,
        "critical_proximity": False,
        "actions": {"policy": None, "filtered": None, "executed": None},
        "safety_shield": {"active": False, "intervention": 0.0},
        "escape_guard": {
            "active": False,
            "mode": "inactive",
            "unrecoverable": False,
        },
    }


def _telemetry_key(record: Mapping[str, Any]) -> tuple[int, float]:
    return (int(record["control_step"]), round(float(record["simulation_time_s"]), 9))


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _csv_bool(value: Any, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"trajectory.csv {label} must be True or False")


def _trajectory_telemetry(record: Mapping[str, Any]) -> dict[str, Any]:
    dynamic_obstacles = []
    for ordinal, (obstacle_id, radius_m) in enumerate(_DYNAMIC_OBSTACLE_SPECS, 1):
        dynamic_obstacles.append(
            {
                "id": obstacle_id,
                "position": {
                    "x_m": float(record[f"dynamic_obstacle_{ordinal}_x_m"]),
                    "y_m": float(record[f"dynamic_obstacle_{ordinal}_y_m"]),
                },
                "radius_m": radius_m,
                # Historical trajectory evidence contains position but no
                # per-frame direction; do not reconstruct a velocity claim.
                "velocity": None,
            }
        )
    guard_mode = str(record["ttc_escape_guard_mode"])
    return {
        "dynamic_obstacles": dynamic_obstacles,
        "avoidance": {
            "available": True,
            "measured_control_step": int(record["control_step"]),
            "measured_at_s": float(record["simulation_time_s"]),
            "minimum_lidar_range_m": float(record["minimum_lidar_range_m"]),
            "robust_ttc_s": _optional_float(record["estimated_ttc_s"]),
            "closing_rate_mps": _optional_float(record["proximity_closing_rate_mps"]),
            "critical_proximity": _csv_bool(
                record["critical_proximity"], "critical_proximity"
            ),
            "actions": {
                "policy": {
                    "linear_normalized": float(record["policy_action_linear"]),
                    "angular_normalized": float(record["policy_action_angular"]),
                },
                "filtered": {
                    "linear_normalized": float(record["shielded_action_linear"]),
                    "angular_normalized": float(record["shielded_action_angular"]),
                },
                "executed": {
                    "linear_mps": float(record["executed_linear_mps"]),
                    "angular_radps": float(record["executed_angular_radps"]),
                },
            },
            "safety_shield": {
                "active": _csv_bool(
                    record["safety_shield_active"], "safety_shield_active"
                ),
                "intervention": float(record["safety_shield_intervention"]),
            },
            "escape_guard": {
                "active": _csv_bool(
                    record["ttc_escape_guard_active"], "ttc_escape_guard_active"
                ),
                "mode": guard_mode,
                "unrecoverable": guard_mode == "unrecoverable",
            },
        },
    }


def _trajectory_by_key(
    path: Path,
) -> tuple[list[dict[str, str]], dict[tuple[int, float], dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("trajectory.csv must contain at least one data row")
    telemetry = {_telemetry_key(record): _trajectory_telemetry(record) for record in records}
    if len(telemetry) != len(records):
        raise ValueError("trajectory.csv control_step/simulation_time pairs must be unique")
    return records, telemetry


def _avoidance_mandatory_indices(
    events: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
) -> list[int]:
    event_indices: dict[tuple[int, float], list[int]] = {}
    for index, event in enumerate(events):
        event_indices.setdefault(_telemetry_key(event), []).append(index)

    mandatory_rows: set[int] = {
        min(
            range(len(trajectory)),
            key=lambda index: float(trajectory[index]["minimum_lidar_range_m"]),
        )
    }
    ttc_rows = [
        index
        for index, record in enumerate(trajectory)
        if record["estimated_ttc_s"] != ""
    ]
    if ttc_rows:
        mandatory_rows.add(
            min(ttc_rows, key=lambda index: float(trajectory[index]["estimated_ttc_s"]))
        )

    for index, record in enumerate(trajectory):
        if record["ttc_escape_guard_active"] == "True" or record["critical_proximity"] == "True":
            mandatory_rows.add(index)

    # Preserve the onset and peak of every contiguous shield intervention,
    # rather than attempting to retain hundreds of near-identical active rows.
    active_episode: list[int] = []
    for index, record in enumerate(trajectory):
        if record["safety_shield_active"] == "True":
            active_episode.append(index)
        elif active_episode:
            mandatory_rows.add(active_episode[0])
            mandatory_rows.add(
                max(
                    active_episode,
                    key=lambda item: float(
                        trajectory[item]["safety_shield_intervention"]
                    ),
                )
            )
            active_episode = []
    if active_episode:
        mandatory_rows.add(active_episode[0])
        mandatory_rows.add(
            max(
                active_episode,
                key=lambda item: float(trajectory[item]["safety_shield_intervention"]),
            )
        )

    result: set[int] = set()
    for row_index in mandatory_rows:
        matches = event_indices.get(_telemetry_key(trajectory[row_index]), [])
        navigating = [index for index in matches if events[index]["state"] == "navigating"]
        result.update(navigating or matches)
    return sorted(result)


def _compact_frame(
    event: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    progress = event["checkpoint_progress"]
    navigation = event["local_navigation"]
    return {
        "sequence": event["sequence"],
        "control_step": event["control_step"],
        "simulation_time_s": event["simulation_time_s"],
        "state": event["state"],
        "outcome": event["outcome"],
        "terminal": event["state"] in {"completed", "failed"},
        "position": event["position"],
        "local_goal": navigation["local_goal"],
        "task": {
            "current_checkpoint_id": progress["current_checkpoint_id"],
            "current_checkpoint_ordinal": progress["current_checkpoint_ordinal"],
            "completed_checkpoint_count": progress["completed_checkpoint_count"],
            "checkpoint_count": progress["checkpoint_count"],
            "progress_ratio": progress["progress_ratio"],
            "remaining_distance_m": navigation["remaining_distance_m"],
        },
        "readings": event["latest_readings"],
        "active_alerts": event["active_alerts"],
        "dynamic_obstacles": list(
            event.get(
                "dynamic_obstacles",
                [] if telemetry is None else telemetry["dynamic_obstacles"],
            )
        ),
        "avoidance": dict(
            event.get(
                "avoidance",
                _empty_avoidance() if telemetry is None else telemetry["avoidance"],
            )
        ),
    }


def _sensor_timeline(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        key = (int(record["visit_index"]), str(record["sensor_id"]))
        previous = grouped.get(key)
        if previous is None or int(record["sample_index"]) > int(previous["sample_index"]):
            grouped[key] = dict(record)

    by_visit: dict[int, dict[str, Any]] = {}
    for (visit_index, sensor_id), record in grouped.items():
        visit = by_visit.setdefault(
            visit_index,
            {
                "visit_index": visit_index,
                "checkpoint_id": record["checkpoint_id"],
                "control_step": record["control_step"],
                "simulation_time_s": record["simulation_time_s"],
                "readings": {},
            },
        )
        visit["simulation_time_s"] = max(
            float(visit["simulation_time_s"]), float(record["simulation_time_s"])
        )
        visit["readings"][sensor_id] = {
            "display_name": record["display_name"],
            "value": record["observed_value"],
            "unit": record["unit"],
            "quality": record["quality"],
            "sample_index": record["sample_index"],
        }
    return [by_visit[index] for index in sorted(by_visit)]


def _alarm_timeline(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        item = dict(record)
        item.pop("schema_version", None)
        item.pop("run_id", None)
        item["event_id"] = f"alarm-{item['event_index']}"
        result.append(item)
    return result


def build_fixture(
    *,
    repository_root: Path,
    run_directory: Path,
    map_config: Path,
    max_frames: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(repository_root / "src"))
    from indoor_patrol_rl.dashboard.projector import DashboardMap, project_dashboard
    from indoor_patrol_rl.dashboard.repository import PatrolDashboardRepository

    evidence = PatrolDashboardRepository(run_directory).read()
    if not evidence.terminal or evidence.terminal_validation is None:
        raise ValueError("the WeChat fixture requires a validated terminal patrol run")
    if evidence.current.run_id != EXPECTED_RUN_ID:
        raise ValueError(
            f"expected run_id {EXPECTED_RUN_ID!r}, got {evidence.current.run_id!r}"
        )

    dashboard_map = DashboardMap.from_yaml(map_config)
    final_payload = project_dashboard(evidence, dashboard_map)
    events = _read_json_lines(run_directory / "events.jsonl")
    if any(event.get("run_id") != EXPECTED_RUN_ID for event in events):
        raise ValueError("events.jsonl contains a foreign run_id")
    if [event.get("sequence") for event in events] != list(range(len(events))):
        raise ValueError("events.jsonl sequence must be contiguous from zero")
    trajectory_path = run_directory / "trajectory.csv"
    trajectory, trajectory_telemetry = _trajectory_by_key(trajectory_path)
    indices = _sample_indices(
        events,
        max_frames,
        extra_mandatory=_avoidance_mandatory_indices(events, trajectory),
    )
    frames = [
        _compact_frame(
            events[index],
            trajectory_telemetry.get(_telemetry_key(events[index])),
        )
        for index in indices
    ]

    sample_records = _read_json_lines_as_csv(run_directory / "inspection_samples.csv")
    alarm_records = _read_json_lines(run_directory / "alarm_events.jsonl")
    if any(record.get("run_id") != EXPECTED_RUN_ID for record in alarm_records):
        raise ValueError("alarm_events.jsonl contains a foreign run_id")

    map_payload = dict(final_payload["map"])
    for key in (
        "robot",
        "local_goal",
        "trajectory",
        "trajectory_point_count",
        "trajectory_source_point_count",
        "trajectory_truncated",
        "dynamic_obstacles",
    ):
        map_payload.pop(key, None)

    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "dashboard_schema_version": DASHBOARD_SCHEMA_VERSION,
        "source": {
            "run_id": EXPECTED_RUN_ID,
            "run_directory_name": run_directory.name,
            "terminal_validation": evidence.terminal_validation,
            "map_source_file": final_payload["source"]["map_source_file"],
            "map_qualification": final_payload["source"]["map_qualification"],
            "map_boundary_note": final_payload["source"]["map_boundary_note"],
            "inspection_manifest_sha256": _sha256(
                run_directory / "inspection_manifest.json"
            ),
            "events_sha256": _sha256(run_directory / "events.jsonl"),
            "trajectory_sha256": _sha256(trajectory_path),
            "avoidance_source": "trajectory_exact_step_and_time_match",
            "source_event_count": len(events),
            "exported_frame_count": len(frames),
            "max_frame_count": max_frames,
            "qualification": "graduation_demonstration_only",
        },
        "replay": {"default_interval_ms": 160, "loop": False},
        "run": {
            "run_id": EXPECTED_RUN_ID,
            "seed": final_payload["run"]["seed"],
            "runtime_identity": final_payload["run"]["runtime_identity"],
        },
        "map": map_payload,
        "frames": frames,
        "sensor_timeline": _sensor_timeline(sample_records),
        "alarm_timeline": _alarm_timeline(alarm_records),
        "result": final_payload["result"],
    }


def _read_json_lines_as_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    numeric_int = {"scenario_seed", "sample_index", "visit_index", "control_step"}
    numeric_float = {
        "simulation_time_s",
        "position_x_m",
        "position_y_m",
        "truth_value",
        "noise_value",
        "observed_value",
    }
    for record in records:
        for key in numeric_int:
            record[key] = int(record[key])
        for key in numeric_float:
            record[key] = float(record[key])
    return records


def main() -> int:
    args = parse_args()
    if not 2 <= args.max_frames <= 120:
        raise SystemExit("--max-frames must be between 2 and 120")
    repository_root = Path(__file__).absolute().parents[1]
    fixture = build_fixture(
        repository_root=repository_root,
        run_directory=args.run_dir.expanduser().resolve(),
        map_config=args.map_config.expanduser().resolve(),
        max_frames=args.max_frames,
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = repository_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fixture, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    module_output = args.module_output.expanduser()
    if not module_output.is_absolute():
        module_output = repository_root / module_output
    module_output.parent.mkdir(parents=True, exist_ok=True)
    module_payload = json.dumps(
        fixture,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    module_output.write_text(
        f"'use strict';\n\nmodule.exports = {module_payload};\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(fixture['frames'])} replay frames for "
        f"{fixture['source']['run_id']} to {output} and {module_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
