#!/usr/bin/env python3
"""Export the verified graduation patrol run for the WeChat demo replay."""

from __future__ import annotations

import argparse
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


def _sample_indices(events: Sequence[Mapping[str, Any]], limit: int) -> list[int]:
    if len(events) <= limit:
        return list(range(len(events)))

    mandatory = {0, len(events) - 1}
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


def _compact_frame(event: Mapping[str, Any]) -> dict[str, Any]:
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
    indices = _sample_indices(events, max_frames)
    frames = [_compact_frame(events[index]) for index in indices]

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
    print(
        f"Exported {len(fixture['frames'])} replay frames for "
        f"{fixture['source']['run_id']} to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
