"""Project validated patrol evidence into the stable dashboard DTO."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from indoor_patrol_rl.application.live_status import PatrolLiveAlert, PatrolLiveSnapshot
from indoor_patrol_rl.dashboard.repository import DashboardEvidence, DashboardEvidenceError

DASHBOARD_SCHEMA_VERSION = "patrol-dashboard/v1"
MAX_TRAJECTORY_POINTS = 300
MAX_ALARM_HISTORY_ITEMS = 200
DEMONSTRATION_BOUNDARY_NOTE = (
    "本二维地图来自仿真配置，仅用于毕设演示可视化，不代表真实场地测绘或导航安全边界。"
)


@dataclass(frozen=True)
class DashboardCheckpoint:
    checkpoint_id: str
    position: tuple[float, float]


@dataclass(frozen=True)
class DashboardObstacle:
    obstacle_id: str
    center: tuple[float, float]
    size: tuple[float, float]


@dataclass(frozen=True)
class DashboardMap:
    """Static demonstration geometry loaded from the patrol YAML map."""

    origin: tuple[float, float]
    size: tuple[float, float]
    resolution_m: float
    robot_start: tuple[float, float]
    checkpoints: tuple[DashboardCheckpoint, ...]
    obstacles: tuple[DashboardObstacle, ...]
    source_file: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> DashboardMap:
        config_path = Path(path).expanduser()
        try:
            resolved = config_path.resolve(strict=True)
        except OSError as error:
            raise DashboardEvidenceError(f"地图配置不存在：{config_path}") from error
        if not resolved.is_file():
            raise DashboardEvidenceError(f"地图配置不是文件：{resolved}")
        try:
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise DashboardEvidenceError(f"地图配置无法读取：{error}") from error
        root = _mapping(raw, "map config")
        map_raw = _mapping(root.get("map"), "map")
        origin = _point(map_raw.get("origin"), "map.origin")
        size = _positive_point(map_raw.get("size"), "map.size")
        resolution = _positive_number(map_raw.get("resolution"), "map.resolution")
        robot_start = _point(root.get("robot_start"), "robot_start")

        rectangle_items = _sequence(map_raw.get("static_rectangles"), "static_rectangles")
        obstacles: list[DashboardObstacle] = []
        for index, raw_rectangle in enumerate(rectangle_items, 1):
            rectangle = _mapping(raw_rectangle, f"static_rectangles[{index}]")
            obstacles.append(
                DashboardObstacle(
                    obstacle_id=_text(rectangle.get("id"), f"static_rectangles[{index}].id"),
                    center=_point(
                        rectangle.get("center"),
                        f"static_rectangles[{index}].center",
                    ),
                    size=_positive_point(
                        rectangle.get("size"),
                        f"static_rectangles[{index}].size",
                    ),
                )
            )
        checkpoint_items = _sequence(root.get("checkpoints"), "checkpoints")
        checkpoints: list[DashboardCheckpoint] = []
        for index, raw_checkpoint in enumerate(checkpoint_items, 1):
            checkpoint = _mapping(raw_checkpoint, f"checkpoints[{index}]")
            checkpoints.append(
                DashboardCheckpoint(
                    checkpoint_id=_text(
                        checkpoint.get("id"), f"checkpoints[{index}].id"
                    ),
                    position=_point(
                        checkpoint.get("position"),
                        f"checkpoints[{index}].position",
                    ),
                )
            )
        _require_unique(
            (checkpoint.checkpoint_id for checkpoint in checkpoints), "checkpoint id"
        )
        _require_unique((obstacle.obstacle_id for obstacle in obstacles), "obstacle id")
        if not checkpoints:
            raise DashboardEvidenceError("地图配置至少需要一个巡检点。")
        _require_inside(robot_start, origin, size, "robot_start")
        for checkpoint in checkpoints:
            _require_inside(
                checkpoint.position,
                origin,
                size,
                f"checkpoint {checkpoint.checkpoint_id}",
            )
        return cls(
            origin=origin,
            size=size,
            resolution_m=resolution,
            robot_start=robot_start,
            checkpoints=tuple(checkpoints),
            obstacles=tuple(obstacles),
            source_file=resolved.name,
        )


def project_dashboard(
    evidence: DashboardEvidence,
    dashboard_map: DashboardMap,
    *,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build a JSON-ready ``patrol-dashboard/v1`` representation."""

    current = evidence.current
    _validate_map_matches_run(evidence, dashboard_map)
    refreshed = generated_at or datetime.now(UTC)
    if refreshed.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware.")
    terminal = evidence.terminal
    trajectory = _trajectory_points(evidence.snapshots, limit=MAX_TRAJECTORY_POINTS)
    progress = (
        current.completed_checkpoint_count / current.checkpoint_count
    )
    result = _terminal_result(evidence)
    readings = current.latest_readings
    active_origins = _active_alarm_origins(evidence.alarm_history)

    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at_utc": refreshed.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "run_directory_name": evidence.run_directory.name,
            "mode": "terminal" if terminal else "live",
            "alarm_history_source": evidence.alarm_history_source,
            "terminal_validation": evidence.terminal_validation,
            "map_source_file": dashboard_map.source_file,
            "map_qualification": "demonstration_only",
            "map_boundary_note": DEMONSTRATION_BOUNDARY_NOTE,
        },
        "run": {
            "run_id": current.run_id,
            "seed": current.seed,
            "state": current.state.value,
            "outcome": current.outcome.value,
            "terminal": terminal,
            "sequence": current.sequence,
            "control_step": current.control_step,
            "simulation_time_s": current.simulation_time_s,
            "runtime_identity": {
                "model_id": current.runtime_identity.model_id,
                "model_sha256": current.runtime_identity.model_sha256,
                "safety_shield_enabled": current.runtime_identity.safety_shield_enabled,
                "safety_shield_id": current.runtime_identity.safety_shield_id,
                "safety_shield_sha256": (
                    current.runtime_identity.safety_shield_sha256
                ),
            },
        },
        "map": {
            "origin": _xy(dashboard_map.origin),
            "size": {"width_m": dashboard_map.size[0], "height_m": dashboard_map.size[1]},
            "resolution_m": dashboard_map.resolution_m,
            "robot_start": _xy(dashboard_map.robot_start),
            "robot": _xy(current.position),
            "local_goal": None if current.local_goal is None else _xy(current.local_goal),
            "trajectory": trajectory,
            "trajectory_point_count": len(trajectory),
            "trajectory_source_point_count": len(evidence.snapshots),
            "trajectory_truncated": len(evidence.snapshots) > len(trajectory),
            "checkpoints": _checkpoint_projection(current, dashboard_map.checkpoints),
            "static_obstacles": [
                {
                    "id": obstacle.obstacle_id,
                    "label": (
                        "中央柜体"
                        if obstacle.obstacle_id == "central_cabinet"
                        else obstacle.obstacle_id
                    ),
                    "center": _xy(obstacle.center),
                    "size": {"width_m": obstacle.size[0], "height_m": obstacle.size[1]},
                }
                for obstacle in dashboard_map.obstacles
            ],
        },
        "task": {
            "current_checkpoint_id": current.current_checkpoint_id,
            "current_checkpoint_ordinal": current.current_checkpoint_ordinal,
            "completed_checkpoint_count": current.completed_checkpoint_count,
            "checkpoint_count": current.checkpoint_count,
            "progress_ratio": progress,
            "progress_percent": round(progress * 100.0, 1),
            "remaining_distance_m": current.remaining_distance_m,
        },
        "sensors": {
            "provenance_label": "上次巡检点采样",
            "available": readings.is_available,
            "checkpoint_id": readings.checkpoint_id,
            "sampled_at_s": readings.sampled_at_s,
            "temperature": {
                "display_name": "温度",
                "value": readings.temperature_c,
                "unit": "°C",
            },
            "smoke": {
                "display_name": "烟雾遮蔽率",
                "value": readings.smoke_obscuration_pct_per_m,
                "unit": "%/m",
            },
            "carbon_monoxide": {
                "display_name": "一氧化碳浓度",
                "value": readings.carbon_monoxide_ppm,
                "unit": "ppm",
            },
        },
        "alarms": {
            "active_count": len(current.active_alerts),
            "active": [
                _active_alert(alert, active_origins.get(alert.rule_id))
                for alert in current.active_alerts
            ],
            "history_count": len(evidence.alarm_history),
            "history_returned_count": min(
                len(evidence.alarm_history), MAX_ALARM_HISTORY_ITEMS
            ),
            "history_truncated": len(evidence.alarm_history) > MAX_ALARM_HISTORY_ITEMS,
            "history": list(reversed(evidence.alarm_history[-MAX_ALARM_HISTORY_ITEMS:])),
        },
        "result": result,
    }


def _validate_map_matches_run(
    evidence: DashboardEvidence,
    dashboard_map: DashboardMap,
) -> None:
    current = evidence.current
    if len(dashboard_map.checkpoints) != current.checkpoint_count:
        raise DashboardEvidenceError("地图巡检点数量与运行证据不一致。")
    if current.current_checkpoint_id is not None:
        assert current.current_checkpoint_ordinal is not None
        expected = dashboard_map.checkpoints[current.current_checkpoint_ordinal - 1]
        if current.current_checkpoint_id != expected.checkpoint_id:
            raise DashboardEvidenceError("当前巡检点 id/ordinal 与演示地图顺序不一致。")
    if not evidence.terminal:
        return
    assert evidence.summary is not None
    raw_records = evidence.summary.get("checkpoint_records")
    if not isinstance(raw_records, list) or len(raw_records) != len(
        dashboard_map.checkpoints
    ):
        raise DashboardEvidenceError("summary.json 的 checkpoint_records 与地图数量不一致。")
    reached_count = 0
    seen_unreached = False
    for ordinal, (raw, checkpoint) in enumerate(
        zip(raw_records, dashboard_map.checkpoints, strict=True), 1
    ):
        record = _mapping(raw, f"checkpoint_records[{ordinal}]")
        if (
            record.get("checkpoint_id") != checkpoint.checkpoint_id
            or record.get("visit_order") != ordinal
        ):
            raise DashboardEvidenceError(
                f"checkpoint_records[{ordinal}] 的 id/ordinal 与演示地图不一致。"
            )
        target = (
            _number(record.get("target_x_m"), f"checkpoint_records[{ordinal}].target_x_m"),
            _number(record.get("target_y_m"), f"checkpoint_records[{ordinal}].target_y_m"),
        )
        if not all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(target, checkpoint.position, strict=True)
        ):
            raise DashboardEvidenceError(
                f"checkpoint_records[{ordinal}] 的目标坐标与演示地图不一致。"
            )
        reached = record.get("reached")
        if not isinstance(reached, bool):
            raise DashboardEvidenceError(
                f"checkpoint_records[{ordinal}].reached 必须是布尔值。"
            )
        if reached:
            if seen_unreached:
                raise DashboardEvidenceError(
                    "checkpoint_records 的到达状态必须保持前缀连续。"
                )
            reached_count += 1
        else:
            seen_unreached = True
    if reached_count != current.completed_checkpoint_count:
        raise DashboardEvidenceError(
            "checkpoint_records 的到达数量与终态 status.json 不一致。"
        )


def _checkpoint_projection(
    current: PatrolLiveSnapshot,
    checkpoints: Sequence[DashboardCheckpoint],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, checkpoint in enumerate(checkpoints):
        if index < current.completed_checkpoint_count:
            status = "completed"
        elif checkpoint.checkpoint_id == current.current_checkpoint_id:
            status = "current"
        else:
            status = "pending"
        result.append(
            {
                "id": checkpoint.checkpoint_id,
                "ordinal": index + 1,
                "position": _xy(checkpoint.position),
                "status": status,
            }
        )
    return result


def _trajectory_points(
    snapshots: Sequence[PatrolLiveSnapshot],
    *,
    limit: int,
) -> list[dict[str, object]]:
    if limit < 2:
        raise ValueError("trajectory limit must be at least two.")
    count = len(snapshots)
    if count <= limit:
        selected = snapshots
    else:
        indexes = [round(index * (count - 1) / (limit - 1)) for index in range(limit)]
        selected = tuple(snapshots[index] for index in indexes)
    return [
        {
            "sequence": snapshot.sequence,
            "simulation_time_s": snapshot.simulation_time_s,
            "x_m": snapshot.position[0],
            "y_m": snapshot.position[1],
        }
        for snapshot in selected
    ]


def _active_alert(
    alert: PatrolLiveAlert,
    checkpoint_id: str | None,
) -> dict[str, object]:
    return {
        "rule_id": alert.rule_id,
        "sensor_id": alert.sensor_id,
        "severity": alert.severity,
        "observed_value": alert.observed_value,
        "threshold": alert.threshold,
        "unit": alert.unit,
        "message": alert.message,
        "raised_at_s": alert.raised_at_s,
        "checkpoint_id": checkpoint_id,
    }


def _active_alarm_origins(
    history: Sequence[Mapping[str, object]],
) -> dict[str, str | None]:
    origins: dict[str, str | None] = {}
    for event in history:
        rule_id = str(event["rule_id"])
        if event["event_type"] == "raised":
            checkpoint = event.get("checkpoint_id")
            origins[rule_id] = checkpoint if isinstance(checkpoint, str) else None
        else:
            origins.pop(rule_id, None)
    return origins


def _terminal_result(evidence: DashboardEvidence) -> dict[str, object]:
    if not evidence.terminal:
        return {
            "available": False,
            "validated": False,
            "validation": None,
            "outcome": None,
            "success": None,
            "failure_reason": None,
        }
    assert evidence.summary is not None
    assert evidence.inspection_summary is not None
    summary = evidence.summary
    inspection = evidence.inspection_summary
    return {
        "available": True,
        "validated": True,
        "validation": evidence.terminal_validation,
        "outcome": summary.get("outcome"),
        "success": summary.get("outcome") == "completed",
        "failure_reason": summary.get("failure_reason"),
        "checkpoints_visited": summary.get("checkpoints_visited"),
        "checkpoints_total": summary.get("checkpoints_total"),
        "collision_count": summary.get("collision_count"),
        "elapsed_simulation_time_s": summary.get("elapsed_simulation_time_s"),
        "actual_path_length_m": summary.get("actual_path_length_m"),
        "minimum_clearance_m": summary.get("minimum_clearance_m"),
        "sample_count": inspection.get("sample_count"),
        "alarms_raised": inspection.get("alarms_raised"),
        "alarms_cleared": inspection.get("alarms_cleared"),
        "highest_event_severity": inspection.get("highest_event_severity"),
    }


def _xy(point: tuple[float, float]) -> dict[str, float]:
    return {"x_m": point[0], "y_m": point[1]}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DashboardEvidenceError(f"{label} 必须是对象。")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DashboardEvidenceError(f"{label} 必须是数组。")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DashboardEvidenceError(f"{label} 必须是非空文本。")
    return value.strip()


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DashboardEvidenceError(f"{label} 必须是有限数值。")
    result = float(value)
    if not math.isfinite(result):
        raise DashboardEvidenceError(f"{label} 必须是有限数值。")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise DashboardEvidenceError(f"{label} 必须大于零。")
    return result


def _point(value: object, label: str) -> tuple[float, float]:
    items = _sequence(value, label)
    if len(items) != 2:
        raise DashboardEvidenceError(f"{label} 必须包含两个坐标。")
    return _number(items[0], f"{label}[0]"), _number(items[1], f"{label}[1]")


def _positive_point(value: object, label: str) -> tuple[float, float]:
    point = _point(value, label)
    if min(point) <= 0.0:
        raise DashboardEvidenceError(f"{label} 的两个分量都必须大于零。")
    return point


def _require_unique(values: Sequence[str] | Any, label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise DashboardEvidenceError(f"{label} 不允许重复。")


def _require_inside(
    point: tuple[float, float],
    origin: tuple[float, float],
    size: tuple[float, float],
    label: str,
) -> None:
    if not (
        origin[0] <= point[0] <= origin[0] + size[0]
        and origin[1] <= point[1] <= origin[1] + size[1]
    ):
        raise DashboardEvidenceError(f"{label} 位于演示地图边界之外。")
