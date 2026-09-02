"""Project validated patrol evidence into the stable dashboard DTO."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from indoor_patrol_rl.application.live_status import PatrolLiveAlert, PatrolLiveSnapshot
from indoor_patrol_rl.dashboard.repository import DashboardEvidence, DashboardEvidenceError
from indoor_patrol_rl.evaluation import PatrolRunSummary

DASHBOARD_SCHEMA_VERSION = "patrol-dashboard/v1"
MAX_TRAJECTORY_POINTS = 300
MAX_ALARM_HISTORY_ITEMS = 200
DEMONSTRATION_BOUNDARY_NOTE = (
    "本二维地图来自仿真配置，仅用于毕设演示可视化，不代表真实场地测绘或导航安全边界。"
)
_INSPECTION_SENSOR_PRESENTATION = {
    "temperature": ("温度", "°C"),
    "smoke_obscuration": ("烟雾遮蔽率", "%/m"),
    "carbon_monoxide": ("一氧化碳浓度", "ppm"),
}
_ALARM_SEVERITY_ORDER = {"info": 1, "warning": 2, "critical": 3}


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
                    checkpoint_id=_text(checkpoint.get("id"), f"checkpoints[{index}].id"),
                    position=_point(
                        checkpoint.get("position"),
                        f"checkpoints[{index}].position",
                    ),
                )
            )
        _require_unique((checkpoint.checkpoint_id for checkpoint in checkpoints), "checkpoint id")
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
    progress = current.completed_checkpoint_count / current.checkpoint_count
    result = _terminal_result(evidence)
    readings = current.latest_readings
    active_origins = _active_alarm_origins(evidence.alarm_history)

    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at_utc": refreshed.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "run_directory_name": evidence.run_directory.name,
            "mode": (
                "aborted"
                if evidence.runtime_abort is not None
                else ("terminal" if terminal else "live")
            ),
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
                "safety_shield_sha256": (current.runtime_identity.safety_shield_sha256),
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
            "dynamic_obstacles": [obstacle.to_dict() for obstacle in current.dynamic_obstacles],
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
            "history_returned_count": min(len(evidence.alarm_history), MAX_ALARM_HISTORY_ITEMS),
            "history_truncated": len(evidence.alarm_history) > MAX_ALARM_HISTORY_ITEMS,
            "history": list(reversed(evidence.alarm_history[-MAX_ALARM_HISTORY_ITEMS:])),
        },
        "avoidance": current.avoidance.to_dict(),
        "inspection": _inspection_projection(evidence),
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
    if not evidence.terminal or evidence.runtime_abort is not None:
        return
    assert evidence.summary is not None
    raw_records = evidence.summary.get("checkpoint_records")
    if not isinstance(raw_records, list) or len(raw_records) != len(dashboard_map.checkpoints):
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
            raise DashboardEvidenceError(f"checkpoint_records[{ordinal}].reached 必须是布尔值。")
        if reached:
            if seen_unreached:
                raise DashboardEvidenceError("checkpoint_records 的到达状态必须保持前缀连续。")
            reached_count += 1
        else:
            seen_unreached = True
    if reached_count != current.completed_checkpoint_count:
        raise DashboardEvidenceError("checkpoint_records 的到达数量与终态 status.json 不一致。")


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


def _inspection_projection(evidence: DashboardEvidence) -> dict[str, object]:
    unavailable: dict[str, object] = {
        "available": False,
        "sample_count": None,
        "visit_count": 0,
        "visits": [],
    }
    if not evidence.terminal or evidence.runtime_abort is not None:
        return unavailable
    if (
        evidence.summary is None
        or evidence.inspection_summary is None
        or evidence.inspection_samples is None
    ):
        raise DashboardEvidenceError("完整终态缺少可投影的巡检证据。")

    current = evidence.current
    patrol_summary = evidence.summary
    inspection_summary = evidence.inspection_summary
    samples = evidence.inspection_samples
    if inspection_summary.get("run_id") != current.run_id:
        raise DashboardEvidenceError("inspection summary 的 run_id 与运行证据不一致。")

    sample_count = _integer(
        inspection_summary.get("sample_count"),
        "inspection_summary.sample_count",
    )
    if sample_count != len(samples):
        raise DashboardEvidenceError(
            "inspection_summary.sample_count 与 inspection_samples.csv 不一致。"
        )
    visit_count = _integer(
        inspection_summary.get("visit_count"),
        "inspection_summary.visit_count",
    )
    raw_visits = _sequence(
        inspection_summary.get("visit_records"),
        "inspection_summary.visit_records",
    )
    if visit_count != len(raw_visits):
        raise DashboardEvidenceError("inspection_summary.visit_count 与 visit_records 不一致。")
    if visit_count != current.completed_checkpoint_count:
        raise DashboardEvidenceError("巡检 visit_count 与已到达巡检点数不一致。")
    if (
        _integer(
            inspection_summary.get("expected_checkpoint_count"),
            "inspection_summary.expected_checkpoint_count",
        )
        != current.checkpoint_count
    ):
        raise DashboardEvidenceError("巡检声明的巡检点总数与运行证据不一致。")
    if _integer(
        inspection_summary.get("sensor_count"),
        "inspection_summary.sensor_count",
    ) != len(_INSPECTION_SENSOR_PRESENTATION):
        raise DashboardEvidenceError("巡检传感器数量与看板合同不一致。")

    visits_by_index: dict[int, Mapping[str, Any]] = {}
    checkpoint_ids: set[str] = set()
    previous_control_step: int | None = None
    previous_simulation_time_s: float | None = None
    for ordinal, raw_visit in enumerate(raw_visits, 1):
        visit = _mapping(raw_visit, f"visit_records[{ordinal}]")
        visit_index = _integer(visit.get("visit_index"), f"visit_records[{ordinal}].visit_index")
        if visit_index != ordinal:
            raise DashboardEvidenceError("巡检 visit_index 必须从 1 连续递增。")
        if visit.get("run_id") != current.run_id:
            raise DashboardEvidenceError(f"visit_records[{ordinal}] 的 run_id 不一致。")
        checkpoint_id = _text(
            visit.get("checkpoint_id"),
            f"visit_records[{ordinal}].checkpoint_id",
        )
        if checkpoint_id in checkpoint_ids:
            raise DashboardEvidenceError("巡检 visit_records 不允许重复 checkpoint_id。")
        checkpoint_ids.add(checkpoint_id)
        control_step = _integer(
            visit.get("control_step"),
            f"visit_records[{ordinal}].control_step",
        )
        simulation_time_s = _number(
            visit.get("simulation_time_s"),
            f"visit_records[{ordinal}].simulation_time_s",
        )
        if simulation_time_s < 0.0:
            raise DashboardEvidenceError(
                f"visit_records[{ordinal}].simulation_time_s 不能为负数。"
            )
        if control_step > current.control_step:
            raise DashboardEvidenceError(
                f"visit_records[{ordinal}].control_step 超过终态控制步。"
            )
        if simulation_time_s > current.simulation_time_s:
            raise DashboardEvidenceError(
                f"visit_records[{ordinal}].simulation_time_s 超过终态仿真时间。"
            )
        # Each patrol visit is a separately completed checkpoint.  The controller
        # must advance at least one frame before another checkpoint can complete,
        # so equal control steps or arrival times are not valid for this contract.
        if previous_control_step is not None and control_step <= previous_control_step:
            raise DashboardEvidenceError(
                "visit_records.control_step 必须随巡检顺序严格递增。"
            )
        if (
            previous_simulation_time_s is not None
            and simulation_time_s <= previous_simulation_time_s
        ):
            raise DashboardEvidenceError(
                "visit_records.simulation_time_s 必须随巡检顺序严格递增。"
            )
        previous_control_step = control_step
        previous_simulation_time_s = simulation_time_s
        visits_by_index[visit_index] = visit

    if _integer(
        inspection_summary.get("visited_checkpoint_count"),
        "inspection_summary.visited_checkpoint_count",
    ) != len(checkpoint_ids):
        raise DashboardEvidenceError("巡检声明的已访问巡检点数不一致。")

    raw_checkpoint_records = _sequence(
        patrol_summary.get("checkpoint_records"),
        "summary.checkpoint_records",
    )
    if len(raw_checkpoint_records) != current.checkpoint_count:
        raise DashboardEvidenceError("summary.checkpoint_records 数量不一致。")
    for ordinal, visit in visits_by_index.items():
        checkpoint = _mapping(
            raw_checkpoint_records[ordinal - 1],
            f"summary.checkpoint_records[{ordinal}]",
        )
        if (
            checkpoint.get("reached") is not True
            or checkpoint.get("visit_order") != ordinal
            or checkpoint.get("checkpoint_id") != visit.get("checkpoint_id")
        ):
            raise DashboardEvidenceError(
                f"visit_records[{ordinal}] 与到达的 checkpoint 顺序不一致。"
            )

    grouped_samples: dict[int, list[dict[str, object]]] = {index: [] for index in visits_by_index}
    sample_indexes: set[int] = set()
    for row_number, sample in enumerate(samples, 1):
        label = f"inspection_samples[{row_number}]"
        if sample.get("run_id") != current.run_id:
            raise DashboardEvidenceError(f"{label} 的 run_id 不一致。")
        sample_index = _csv_integer(sample, "sample_index", label, minimum=1)
        if sample_index in sample_indexes:
            raise DashboardEvidenceError("巡检 sample_index 不允许重复。")
        sample_indexes.add(sample_index)
        visit_index = _csv_integer(sample, "visit_index", label, minimum=1)
        if visit_index not in visits_by_index:
            raise DashboardEvidenceError(f"{label} 引用了未声明的 visit_index。")
        sensor_id = sample.get("sensor_id")
        if (
            not isinstance(sensor_id, str)
            or sensor_id not in _INSPECTION_SENSOR_PRESENTATION
        ):
            raise DashboardEvidenceError(f"{label} 传感器不在看板白名单中。")
        display_name, unit = _INSPECTION_SENSOR_PRESENTATION[sensor_id]
        if (
            sample.get("quantity") != sensor_id
            or sample.get("display_name") != display_name
            or sample.get("unit") != unit
        ):
            raise DashboardEvidenceError(f"{label} 传感器元数据与看板合同不一致。")
        quality = _text(sample.get("quality"), f"{label}.quality")
        if quality != "ok":
            raise DashboardEvidenceError(f"{label}.quality 不在支持的白名单中。")
        grouped_samples[visit_index].append(
            {
                "sample_index": sample_index,
                "checkpoint_id": _text(sample.get("checkpoint_id"), f"{label}.checkpoint_id"),
                "control_step": _csv_integer(sample, "control_step", label, minimum=0),
                "simulation_time_s": _csv_number(sample, "simulation_time_s", label),
                "sensor_id": sensor_id,
                "display_name": display_name,
                "unit": unit,
                "value": _csv_number(sample, "observed_value", label),
                "quality": quality,
            }
        )
    if sorted(sample_indexes) != list(range(1, sample_count + 1)):
        raise DashboardEvidenceError("巡检 sample_index 必须从 1 连续递增。")

    event_states = _visit_alarm_states(
        inspection_summary,
        visits_by_index,
        expected_run_id=current.run_id,
    )
    visits: list[dict[str, object]] = []
    for ordinal, visit in visits_by_index.items():
        label = f"visit_records[{ordinal}]"
        checkpoint_id = str(visit["checkpoint_id"])
        control_step = _integer(visit.get("control_step"), f"{label}.control_step")
        simulation_time_s = _number(
            visit.get("simulation_time_s"),
            f"{label}.simulation_time_s",
        )
        if simulation_time_s < 0.0:
            raise DashboardEvidenceError(f"{label}.simulation_time_s 不能为负数。")
        rows = grouped_samples[ordinal]
        declared_visit_samples = _integer(visit.get("sample_count"), f"{label}.sample_count")
        if declared_visit_samples != len(rows):
            raise DashboardEvidenceError(f"{label}.sample_count 与原始样本不一致。")
        latest_by_sensor: dict[str, dict[str, object]] = {}
        sample_counts_by_sensor: dict[str, int] = {}
        for sample in rows:
            sample_time = float(sample["simulation_time_s"])
            if (
                sample["checkpoint_id"] != checkpoint_id
                or sample["control_step"] != control_step
                or (
                    sample_time < simulation_time_s
                    and not math.isclose(
                        sample_time,
                        simulation_time_s,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            ):
                raise DashboardEvidenceError(f"{label} 与其原始样本上下文不一致。")
            sensor_id = str(sample["sensor_id"])
            sample_counts_by_sensor[sensor_id] = sample_counts_by_sensor.get(sensor_id, 0) + 1
            previous = latest_by_sensor.get(sensor_id)
            if previous is None or int(sample["sample_index"]) > int(previous["sample_index"]):
                latest_by_sensor[sensor_id] = sample
        if set(latest_by_sensor) != set(_INSPECTION_SENSOR_PRESENTATION):
            raise DashboardEvidenceError(f"{label} 没有且仅有三种约定传感器。")
        if len(set(sample_counts_by_sensor.values())) != 1:
            raise DashboardEvidenceError(f"{label} 三种传感器的采样数不一致。")

        alarm_event_count, active_alarm_count, highest_active_severity = event_states[ordinal]
        if (
            _integer(visit.get("alarm_event_count"), f"{label}.alarm_event_count")
            != alarm_event_count
        ):
            raise DashboardEvidenceError(f"{label}.alarm_event_count 与告警事件不一致。")
        if (
            _integer(visit.get("active_alarm_count"), f"{label}.active_alarm_count")
            != active_alarm_count
        ):
            raise DashboardEvidenceError(f"{label}.active_alarm_count 与告警状态不一致。")
        if visit.get("highest_active_severity") != highest_active_severity:
            raise DashboardEvidenceError(f"{label}.highest_active_severity 与告警状态不一致。")

        readings = {
            sensor_id: {
                "display_name": latest_by_sensor[sensor_id]["display_name"],
                "value": latest_by_sensor[sensor_id]["value"],
                "unit": latest_by_sensor[sensor_id]["unit"],
                "quality": latest_by_sensor[sensor_id]["quality"],
                "sample_index": latest_by_sensor[sensor_id]["sample_index"],
            }
            for sensor_id in _INSPECTION_SENSOR_PRESENTATION
        }
        visits.append(
            {
                "visit_index": ordinal,
                "checkpoint_id": checkpoint_id,
                "control_step": control_step,
                "simulation_time_s": simulation_time_s,
                "alarm_event_count": alarm_event_count,
                "highest_active_severity": highest_active_severity,
                "readings": readings,
            }
        )

    return {
        "available": True,
        "sample_count": sample_count,
        "visit_count": len(visits),
        "visits": visits,
    }


def _visit_alarm_states(
    inspection_summary: Mapping[str, Any],
    visits_by_index: Mapping[int, Mapping[str, Any]],
    *,
    expected_run_id: str,
) -> dict[int, tuple[int, int, str | None]]:
    raw_events = _sequence(
        inspection_summary.get("alarm_events"),
        "inspection_summary.alarm_events",
    )
    if _integer(inspection_summary.get("event_count"), "inspection_summary.event_count") != len(
        raw_events
    ):
        raise DashboardEvidenceError("inspection_summary.event_count 与 alarm_events 不一致。")
    events_by_visit: dict[int, list[Mapping[str, Any]]] = {index: [] for index in visits_by_index}
    for event_number, raw_event in enumerate(raw_events, 1):
        event = _mapping(raw_event, f"alarm_events[{event_number}]")
        if event.get("run_id") != expected_run_id:
            raise DashboardEvidenceError("告警事件的 run_id 与运行证据不一致。")
        visit_index = _integer(
            event.get("visit_index"),
            f"alarm_events[{event_number}].visit_index",
        )
        if visit_index not in events_by_visit:
            raise DashboardEvidenceError("告警事件引用了未声明的 visit_index。")
        if event.get("checkpoint_id") != visits_by_index[visit_index].get(
            "checkpoint_id"
        ):
            raise DashboardEvidenceError("告警事件的 checkpoint_id 与 visit 不一致。")
        events_by_visit[visit_index].append(event)

    active_rules: dict[str, str] = {}
    result: dict[int, tuple[int, int, str | None]] = {}
    for visit_index in visits_by_index:
        for event in events_by_visit[visit_index]:
            rule_id = _text(event.get("rule_id"), "alarm_event.rule_id")
            severity = event.get("severity")
            if not isinstance(severity, str) or severity not in _ALARM_SEVERITY_ORDER:
                raise DashboardEvidenceError("告警事件 severity 不在白名单中。")
            event_type = event.get("event_type")
            if event_type == "raised":
                if rule_id in active_rules:
                    raise DashboardEvidenceError("告警事件重复激活同一规则。")
                active_rules[rule_id] = str(severity)
            elif event_type == "cleared":
                if active_rules.get(rule_id) != severity:
                    raise DashboardEvidenceError("告警事件清除了未激活或严重度不一致的规则。")
                del active_rules[rule_id]
            else:
                raise DashboardEvidenceError("告警事件 event_type 不在白名单中。")
        highest = (
            max(active_rules.values(), key=_ALARM_SEVERITY_ORDER.__getitem__)
            if active_rules
            else None
        )
        result[visit_index] = (
            len(events_by_visit[visit_index]),
            len(active_rules),
            highest,
        )
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DashboardEvidenceError(f"{label} 必须是非负整数。")
    return value


def _csv_integer(
    row: Mapping[str, str],
    key: str,
    label: str,
    *,
    minimum: int,
) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise DashboardEvidenceError(f"{label}.{key} 必须是整数。") from error
    if value < minimum:
        raise DashboardEvidenceError(f"{label}.{key} 必须大于等于 {minimum}。")
    return value


def _csv_number(row: Mapping[str, str], key: str, label: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise DashboardEvidenceError(f"{label}.{key} 必须是有限数值。") from error
    if not math.isfinite(value):
        raise DashboardEvidenceError(f"{label}.{key} 必须是有限数值。")
    return value


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
    if evidence.runtime_abort is not None:
        abort = evidence.runtime_abort
        return {
            "available": True,
            "validated": False,
            "validation": evidence.terminal_validation,
            "incomplete": True,
            "outcome": "aborted",
            "success": False,
            "failure_reason": abort.failure_reason,
            "failure_code": abort.reason_code,
            "failure_type": abort.exception_type,
            "checkpoints_visited": abort.checkpoints_visited,
            "checkpoints_total": abort.checkpoints_total,
            "collision_count": abort.collision_count,
            "sample_count": abort.sample_count,
            "elapsed_simulation_time_s": abort.simulation_time_s,
        }
    assert evidence.summary is not None
    assert evidence.inspection_summary is not None
    summary = _validated_patrol_summary(evidence.summary)
    inspection = evidence.inspection_summary
    return {
        "available": True,
        "validated": True,
        "validation": evidence.terminal_validation,
        "outcome": summary.outcome,
        "success": summary.outcome == "completed",
        "failure_reason": summary.failure_reason,
        "checkpoints_visited": summary.checkpoints_visited,
        "checkpoints_total": summary.checkpoints_total,
        "collision_count": summary.collision_count,
        "elapsed_simulation_time_s": summary.elapsed_simulation_time_s,
        "actual_path_length_m": summary.actual_path_length_m,
        "minimum_clearance_m": summary.minimum_clearance_m,
        "minimum_lidar_range_m": summary.minimum_lidar_range_m,
        "critical_proximity_count": summary.critical_proximity_count,
        "safety_shield_intervention_steps": summary.safety_shield_intervention_steps,
        "safety_shield_intervention_rate": summary.safety_shield_intervention_rate,
        "ttc_escape_guard_intervention_steps": (
            summary.ttc_escape_guard_intervention_steps
        ),
        "ttc_escape_guard_activation_count": summary.ttc_escape_guard_activation_count,
        "ttc_escape_guard_recovery_count": summary.ttc_escape_guard_recovery_count,
        "ttc_escape_guard_unrecoverable_count": (
            summary.ttc_escape_guard_unrecoverable_count
        ),
        "inference_latency_p50_ms": summary.inference_latency_p50_ms,
        "inference_latency_p95_ms": summary.inference_latency_p95_ms,
        "dynamic_obstacle_count": summary.dynamic_obstacle_count,
        "dynamic_obstacle_speed_max_mps": summary.dynamic_obstacle_speed_max_mps,
        "sample_count": inspection.get("sample_count"),
        "alarms_raised": inspection.get("alarms_raised"),
        "alarms_cleared": inspection.get("alarms_cleared"),
        "highest_event_severity": inspection.get("highest_event_severity"),
    }


def _validated_patrol_summary(raw: Mapping[str, Any]) -> PatrolRunSummary:
    values = {
        field.name: raw[field.name]
        for field in fields(PatrolRunSummary)
        if field.name in raw
    }
    try:
        return PatrolRunSummary(**values)
    except (TypeError, ValueError) as error:
        raise DashboardEvidenceError(f"summary.json 运行结果不满足领域约束：{error}") from error


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
