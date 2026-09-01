"""Alarm-history projections for live and completed patrol runs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from indoor_patrol_rl.application.live_status import PatrolLiveAlert, PatrolLiveSnapshot

_EVENT_TYPES = {"raised", "cleared"}
_SEVERITIES = {"info", "warning", "critical"}
_VERIFIED_EVENT_KEYS = {
    "schema_version",
    "run_id",
    "event_index",
    "event_type",
    "rule_id",
    "sensor_id",
    "severity",
    "checkpoint_id",
    "visit_index",
    "sample_index",
    "control_step",
    "simulation_time_s",
    "observed_value",
    "threshold",
    "unit",
    "message",
}


def snapshot_alarm_history(
    snapshots: Sequence[PatrolLiveSnapshot],
) -> tuple[dict[str, object], ...]:
    """Derive live alarm transitions by diffing successive active-alert sets."""

    history: list[dict[str, object]] = []
    active: dict[str, PatrolLiveAlert] = {}
    for snapshot in snapshots:
        current = {alert.rule_id: alert for alert in snapshot.active_alerts}
        for rule_id in sorted(current.keys() - active.keys()):
            alert = current[rule_id]
            history.append(
                _live_event(
                    snapshot=snapshot,
                    alert=alert,
                    event_type="raised",
                    simulation_time_s=alert.raised_at_s,
                )
            )
        for rule_id in sorted(active.keys() - current.keys()):
            history.append(
                _live_event(
                    snapshot=snapshot,
                    alert=active[rule_id],
                    event_type="cleared",
                    simulation_time_s=snapshot.simulation_time_s,
                )
            )
        active = current
    return tuple(history)


def verified_alarm_history(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_run_id: str,
) -> tuple[dict[str, object], ...]:
    """Validate and normalize events from a verified inspection bundle."""

    history: list[dict[str, object]] = []
    active_rules: set[str] = set()
    previous_time = -math.inf
    previous_step = -1
    for expected_index, raw in enumerate(records, 1):
        item = _mapping(raw, f"alarm event {expected_index}")
        if set(item) != _VERIFIED_EVENT_KEYS:
            missing = sorted(_VERIFIED_EVENT_KEYS - set(item))
            extra = sorted(set(item) - _VERIFIED_EVENT_KEYS)
            raise ValueError(
                f"alarm event {expected_index} keys are invalid; missing={missing}, extra={extra}."
            )
        if item["schema_version"] != "1.0":
            raise ValueError(f"alarm event {expected_index} schema_version must be '1.0'.")
        if item["run_id"] != expected_run_id:
            raise ValueError(
                f"alarm event {expected_index} run_id does not match status.json."
            )
        if _integer(item["event_index"], "event_index", minimum=1) != expected_index:
            raise ValueError("alarm event_index values must be contiguous from one.")
        event_type = _text(item["event_type"], "event_type")
        if event_type not in _EVENT_TYPES:
            raise ValueError(f"alarm event {expected_index} has an invalid event_type.")
        rule_id = _text(item["rule_id"], "rule_id")
        sensor_id = _text(item["sensor_id"], "sensor_id")
        severity = _text(item["severity"], "severity")
        if severity not in _SEVERITIES:
            raise ValueError(f"alarm event {expected_index} has an invalid severity.")
        checkpoint_id = _text(item["checkpoint_id"], "checkpoint_id")
        visit_index = _integer(item["visit_index"], "visit_index", minimum=1)
        sample_index = _integer(item["sample_index"], "sample_index", minimum=1)
        control_step = _integer(item["control_step"], "control_step", minimum=0)
        simulation_time = _number(item["simulation_time_s"], "simulation_time_s", minimum=0.0)
        observed_value = _number(item["observed_value"], "observed_value")
        threshold = _number(item["threshold"], "threshold")
        unit = _text(item["unit"], "unit")
        message = _text(item["message"], "message")
        if control_step < previous_step or simulation_time < previous_time:
            raise ValueError("verified alarm events must be time ordered.")
        previous_step = control_step
        previous_time = simulation_time
        if event_type == "raised":
            if rule_id in active_rules:
                raise ValueError(f"alarm event {expected_index} raises an active rule.")
            active_rules.add(rule_id)
        else:
            if rule_id not in active_rules:
                raise ValueError(f"alarm event {expected_index} clears an inactive rule.")
            active_rules.remove(rule_id)
        history.append(
            {
                "event_id": f"alarm-{expected_index}",
                "event_index": expected_index,
                "event_type": event_type,
                "rule_id": rule_id,
                "sensor_id": sensor_id,
                "severity": severity,
                "checkpoint_id": checkpoint_id,
                "visit_index": visit_index,
                "sample_index": sample_index,
                "control_step": control_step,
                "simulation_time_s": simulation_time,
                "observed_value": observed_value,
                "threshold": threshold,
                "unit": unit,
                "message": message,
            }
        )
    return tuple(history)


def active_rule_ids_from_history(
    history: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Replay normalized transitions and return the active rule IDs."""

    active: dict[str, None] = {}
    for event in history:
        rule_id = str(event["rule_id"])
        if event["event_type"] == "raised":
            active[rule_id] = None
        else:
            active.pop(rule_id, None)
    return tuple(active)


def _live_event(
    *,
    snapshot: PatrolLiveSnapshot,
    alert: PatrolLiveAlert,
    event_type: str,
    simulation_time_s: float,
) -> dict[str, object]:
    checkpoint_id = (
        snapshot.latest_readings.checkpoint_id
        or snapshot.current_checkpoint_id
    )
    return {
        "event_id": f"live-{snapshot.sequence}-{event_type}-{alert.rule_id}",
        "event_index": None,
        "event_type": event_type,
        "rule_id": alert.rule_id,
        "sensor_id": alert.sensor_id,
        "severity": alert.severity,
        "checkpoint_id": checkpoint_id,
        "visit_index": None,
        "sample_index": None,
        "control_step": snapshot.control_step,
        "simulation_time_s": simulation_time_s,
        "observed_value": alert.observed_value,
        "threshold": alert.threshold,
        "unit": alert.unit,
        "message": alert.message,
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be a finite number in range.")
    return result
