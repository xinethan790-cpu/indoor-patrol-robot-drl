"""Strict, replayable live status files for the local patrol dashboard.

The controller can publish one complete :class:`PatrolLiveSnapshot` whenever
its visible application state changes.  ``status.json`` is atomically replaced
for polling clients, while ``events.jsonl`` retains the ordered snapshots for
replay and audit.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import Any

from indoor_patrol_rl.application.task import PatrolTaskState
from indoor_patrol_rl.identifiers import validate_patrol_run_id

LIVE_STATUS_SCHEMA_VERSION = "patrol-live-status/v2"
LEGACY_LIVE_STATUS_SCHEMA_VERSION = "patrol-live-status/v1"
MAX_LIVE_DYNAMIC_OBSTACLES = 32
_SHA256_LENGTH = 64
_ALERT_SEVERITIES = {"info", "warning", "critical"}
_ESCAPE_GUARD_MODES = {"inactive", "reverse", "forward", "rotate", "unrecoverable"}


class PatrolLiveOutcome(StrEnum):
    """Run-level result paired with a task state."""

    RUNNING = "running"
    SUCCESS = "success"
    COLLISION = "collision"
    SAFETY_STOP = "safety_stop"
    TIMEOUT = "timeout"
    INSPECTION_FAILED = "inspection_failed"
    ERROR = "error"


@dataclass(frozen=True)
class PatrolLiveReadings:
    """Latest complete environmental sample displayed by the dashboard.

    An empty instance represents the period before the first inspection.  Once
    present, all three application quantities and their shared provenance are
    required so a dashboard never mixes readings from different visits.
    """

    temperature_c: float | None = None
    smoke_obscuration_pct_per_m: float | None = None
    carbon_monoxide_ppm: float | None = None
    sampled_at_s: float | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.temperature_c,
            self.smoke_obscuration_pct_per_m,
            self.carbon_monoxide_ppm,
        )
        populated = tuple(value is not None for value in values)
        if any(populated) and not all(populated):
            raise ValueError("The three latest environmental readings must be set together.")
        if not any(populated):
            if self.sampled_at_s is not None or self.checkpoint_id is not None:
                raise ValueError("Empty readings cannot contain sampling provenance.")
            return

        temperature = _finite_number(self.temperature_c, "temperature_c")
        smoke = _finite_number(
            self.smoke_obscuration_pct_per_m,
            "smoke_obscuration_pct_per_m",
        )
        carbon_monoxide = _finite_number(
            self.carbon_monoxide_ppm,
            "carbon_monoxide_ppm",
        )
        sampled_at = _non_negative_number(self.sampled_at_s, "sampled_at_s")
        checkpoint_id = _non_empty_text(self.checkpoint_id, "checkpoint_id")
        if smoke < 0.0:
            raise ValueError("smoke_obscuration_pct_per_m must be non-negative.")
        if carbon_monoxide < 0.0:
            raise ValueError("carbon_monoxide_ppm must be non-negative.")
        object.__setattr__(self, "temperature_c", temperature)
        object.__setattr__(self, "smoke_obscuration_pct_per_m", smoke)
        object.__setattr__(self, "carbon_monoxide_ppm", carbon_monoxide)
        object.__setattr__(self, "sampled_at_s", sampled_at)
        object.__setattr__(self, "checkpoint_id", checkpoint_id)

    @property
    def is_available(self) -> bool:
        return self.temperature_c is not None


@dataclass(frozen=True)
class PatrolLiveAlert:
    """One alarm that remains active at the snapshot time."""

    rule_id: str
    sensor_id: str
    severity: str
    observed_value: float
    threshold: float
    unit: str
    message: str
    raised_at_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _non_empty_text(self.rule_id, "rule_id"))
        object.__setattr__(self, "sensor_id", _non_empty_text(self.sensor_id, "sensor_id"))
        severity = _non_empty_text(self.severity, "severity")
        if severity not in _ALERT_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_ALERT_SEVERITIES)}.")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(
            self,
            "observed_value",
            _finite_number(self.observed_value, "observed_value"),
        )
        object.__setattr__(self, "threshold", _finite_number(self.threshold, "threshold"))
        object.__setattr__(self, "unit", _non_empty_text(self.unit, "unit"))
        object.__setattr__(self, "message", _non_empty_text(self.message, "message"))
        object.__setattr__(
            self,
            "raised_at_s",
            _non_negative_number(self.raised_at_s, "raised_at_s"),
        )


@dataclass(frozen=True)
class PatrolRuntimeIdentity:
    """Content identity of the policy and optional safety shield in use."""

    model_id: str
    model_sha256: str
    safety_shield_enabled: bool
    safety_shield_id: str | None = None
    safety_shield_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _non_empty_text(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "model_sha256",
            _sha256(self.model_sha256, "model_sha256"),
        )
        if not isinstance(self.safety_shield_enabled, bool):
            raise TypeError("safety_shield_enabled must be a bool.")
        if self.safety_shield_enabled:
            object.__setattr__(
                self,
                "safety_shield_id",
                _non_empty_text(self.safety_shield_id, "safety_shield_id"),
            )
            object.__setattr__(
                self,
                "safety_shield_sha256",
                _sha256(self.safety_shield_sha256, "safety_shield_sha256"),
            )
        elif self.safety_shield_id is not None or self.safety_shield_sha256 is not None:
            raise ValueError("Disabled safety shields cannot carry an identity or digest.")


@dataclass(frozen=True)
class PatrolLiveDynamicObstacle:
    """Ground-truth pose and geometry of one simulator-controlled obstacle."""

    obstacle_id: str
    position: tuple[float, float]
    radius_m: float
    velocity_mps: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obstacle_id",
            _non_empty_text(self.obstacle_id, "obstacle_id"),
        )
        object.__setattr__(self, "position", _finite_point(self.position, "position"))
        radius = _finite_number(self.radius_m, "radius_m")
        if radius <= 0.0:
            raise ValueError("radius_m must be positive.")
        object.__setattr__(self, "radius_m", radius)
        if self.velocity_mps is not None:
            object.__setattr__(
                self,
                "velocity_mps",
                _finite_point(self.velocity_mps, "velocity_mps"),
            )

    def to_dict(self) -> dict[str, Any]:
        velocity = None
        if self.velocity_mps is not None:
            velocity = {
                "x_mps": self.velocity_mps[0],
                "y_mps": self.velocity_mps[1],
            }
        return {
            "id": self.obstacle_id,
            "position": {"x_m": self.position[0], "y_m": self.position[1]},
            "radius_m": self.radius_m,
            "velocity": velocity,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, name: str) -> PatrolLiveDynamicObstacle:
        value = _mapping(raw, name)
        _exact_keys(value, {"id", "position", "radius_m", "velocity"}, name)
        position = _point_mapping(value["position"], f"{name}.position")
        velocity_raw = value["velocity"]
        velocity = None
        if velocity_raw is not None:
            velocity_mapping = _mapping(velocity_raw, f"{name}.velocity")
            _exact_keys(
                velocity_mapping,
                {"x_mps", "y_mps"},
                f"{name}.velocity",
            )
            velocity = (velocity_mapping["x_mps"], velocity_mapping["y_mps"])
        return cls(
            obstacle_id=value["id"],
            position=(position["x_m"], position["y_m"]),
            radius_m=value["radius_m"],
            velocity_mps=velocity,
        )


@dataclass(frozen=True)
class PatrolLiveAvoidance:
    """Latest measured risk and command stages used by dynamic avoidance."""

    measured_control_step: int | None = None
    measured_at_s: float | None = None
    minimum_lidar_range_m: float | None = None
    robust_ttc_s: float | None = None
    closing_rate_mps: float | None = None
    critical_proximity: bool = False
    policy_action: tuple[float, float] | None = None
    filtered_action: tuple[float, float] | None = None
    executed_action: tuple[float, float] | None = None
    safety_shield_active: bool = False
    safety_shield_intervention: float = 0.0
    escape_guard_active: bool = False
    escape_guard_mode: str = "inactive"
    escape_guard_unrecoverable: bool = False

    def __post_init__(self) -> None:
        available = self.minimum_lidar_range_m is not None
        provenance = (self.measured_control_step, self.measured_at_s)
        if available != all(value is not None for value in provenance):
            raise ValueError(
                "Avoidance measurement provenance and minimum_lidar_range_m "
                "must be set together."
            )
        if not available:
            optional_values = (
                self.robust_ttc_s,
                self.closing_rate_mps,
                self.policy_action,
                self.filtered_action,
                self.executed_action,
            )
            if any(value is not None for value in optional_values):
                raise ValueError("Unavailable avoidance telemetry cannot contain measurements.")
            if (
                self.critical_proximity
                or self.safety_shield_active
                or self.safety_shield_intervention != 0.0
                or self.escape_guard_active
                or self.escape_guard_mode != "inactive"
                or self.escape_guard_unrecoverable
            ):
                raise ValueError("Unavailable avoidance telemetry must be inactive.")
            return

        object.__setattr__(
            self,
            "measured_control_step",
            _non_negative_integer(self.measured_control_step, "measured_control_step"),
        )
        object.__setattr__(
            self,
            "measured_at_s",
            _non_negative_number(self.measured_at_s, "measured_at_s"),
        )
        minimum = _finite_number(self.minimum_lidar_range_m, "minimum_lidar_range_m")
        if minimum <= 0.0:
            raise ValueError("minimum_lidar_range_m must be positive.")
        object.__setattr__(self, "minimum_lidar_range_m", minimum)
        if self.robust_ttc_s is not None:
            object.__setattr__(
                self,
                "robust_ttc_s",
                _non_negative_number(self.robust_ttc_s, "robust_ttc_s"),
            )
        if self.closing_rate_mps is not None:
            object.__setattr__(
                self,
                "closing_rate_mps",
                _finite_number(self.closing_rate_mps, "closing_rate_mps"),
            )
        if not isinstance(self.critical_proximity, bool):
            raise TypeError("critical_proximity must be a bool.")

        if (self.policy_action is None) != (self.filtered_action is None):
            raise ValueError("policy_action and filtered_action must be set together.")
        if self.policy_action is not None:
            object.__setattr__(
                self,
                "policy_action",
                _normalized_action(self.policy_action, "policy_action"),
            )
            object.__setattr__(
                self,
                "filtered_action",
                _normalized_action(self.filtered_action, "filtered_action"),
            )
        if self.executed_action is not None:
            object.__setattr__(
                self,
                "executed_action",
                _finite_point(self.executed_action, "executed_action"),
            )

        if not isinstance(self.safety_shield_active, bool):
            raise TypeError("safety_shield_active must be a bool.")
        intervention = _non_negative_number(
            self.safety_shield_intervention,
            "safety_shield_intervention",
        )
        object.__setattr__(self, "safety_shield_intervention", intervention)
        if self.policy_action is None:
            if self.safety_shield_active or intervention != 0.0:
                raise ValueError("Safety-shield evidence requires policy and filtered actions.")
        else:
            expected_intervention = math.dist(self.policy_action, self.filtered_action)
            if not math.isclose(intervention, expected_intervention, abs_tol=1e-6):
                raise ValueError(
                    "safety_shield_intervention must equal the policy/filter difference."
                )
            if self.safety_shield_active != (intervention > 1e-6):
                raise ValueError(
                    "safety_shield_active must match a non-zero action intervention."
                )

        if not isinstance(self.escape_guard_active, bool):
            raise TypeError("escape_guard_active must be a bool.")
        mode = _non_empty_text(self.escape_guard_mode, "escape_guard_mode")
        if mode not in _ESCAPE_GUARD_MODES:
            raise ValueError(f"escape_guard_mode must be one of {sorted(_ESCAPE_GUARD_MODES)}.")
        object.__setattr__(self, "escape_guard_mode", mode)
        if not isinstance(self.escape_guard_unrecoverable, bool):
            raise TypeError("escape_guard_unrecoverable must be a bool.")
        if self.escape_guard_active != (mode != "inactive"):
            raise ValueError("escape_guard_active must match escape_guard_mode.")
        if self.escape_guard_unrecoverable != (mode == "unrecoverable"):
            raise ValueError("escape_guard_unrecoverable must match unrecoverable mode.")

    @property
    def is_available(self) -> bool:
        return self.minimum_lidar_range_m is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.is_available,
            "measured_control_step": self.measured_control_step,
            "measured_at_s": self.measured_at_s,
            "minimum_lidar_range_m": self.minimum_lidar_range_m,
            "robust_ttc_s": self.robust_ttc_s,
            "closing_rate_mps": self.closing_rate_mps,
            "critical_proximity": self.critical_proximity,
            "actions": {
                "policy": _normalized_action_dict(self.policy_action),
                "filtered": _normalized_action_dict(self.filtered_action),
                "executed": _executed_action_dict(self.executed_action),
            },
            "safety_shield": {
                "active": self.safety_shield_active,
                "intervention": self.safety_shield_intervention,
            },
            "escape_guard": {
                "active": self.escape_guard_active,
                "mode": self.escape_guard_mode,
                "unrecoverable": self.escape_guard_unrecoverable,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PatrolLiveAvoidance:
        value = _mapping(raw, "avoidance")
        _exact_keys(
            value,
            {
                "available",
                "measured_control_step",
                "measured_at_s",
                "minimum_lidar_range_m",
                "robust_ttc_s",
                "closing_rate_mps",
                "critical_proximity",
                "actions",
                "safety_shield",
                "escape_guard",
            },
            "avoidance",
        )
        if not isinstance(value["available"], bool):
            raise TypeError("avoidance.available must be a bool.")
        actions = _mapping(value["actions"], "avoidance.actions")
        _exact_keys(actions, {"policy", "filtered", "executed"}, "avoidance.actions")
        shield = _mapping(value["safety_shield"], "avoidance.safety_shield")
        _exact_keys(shield, {"active", "intervention"}, "avoidance.safety_shield")
        guard = _mapping(value["escape_guard"], "avoidance.escape_guard")
        _exact_keys(
            guard,
            {"active", "mode", "unrecoverable"},
            "avoidance.escape_guard",
        )
        result = cls(
            measured_control_step=value["measured_control_step"],
            measured_at_s=value["measured_at_s"],
            minimum_lidar_range_m=value["minimum_lidar_range_m"],
            robust_ttc_s=value["robust_ttc_s"],
            closing_rate_mps=value["closing_rate_mps"],
            critical_proximity=value["critical_proximity"],
            policy_action=_parse_normalized_action(
                actions["policy"], "avoidance.actions.policy"
            ),
            filtered_action=_parse_normalized_action(
                actions["filtered"], "avoidance.actions.filtered"
            ),
            executed_action=_parse_executed_action(
                actions["executed"], "avoidance.actions.executed"
            ),
            safety_shield_active=shield["active"],
            safety_shield_intervention=shield["intervention"],
            escape_guard_active=guard["active"],
            escape_guard_mode=guard["mode"],
            escape_guard_unrecoverable=guard["unrecoverable"],
        )
        if result.is_available != value["available"]:
            raise ValueError("avoidance.available does not match its measurements.")
        return result


@dataclass(frozen=True)
class PatrolLiveSnapshot:
    """One complete, validated dashboard view of a patrol run.

    ``current_checkpoint_ordinal`` is one-based for direct display.  Progress is
    tied to completed checkpoints and therefore must equal
    ``completed_checkpoint_count / checkpoint_count``.
    """

    run_id: str
    sequence: int
    state: PatrolTaskState
    outcome: PatrolLiveOutcome
    seed: int
    control_step: int
    simulation_time_s: float
    position: tuple[float, float]
    current_checkpoint_id: str | None
    current_checkpoint_ordinal: int | None
    completed_checkpoint_count: int
    checkpoint_count: int
    progress_ratio: float
    local_goal: tuple[float, float] | None
    remaining_distance_m: float | None
    latest_readings: PatrolLiveReadings
    active_alerts: tuple[PatrolLiveAlert, ...]
    runtime_identity: PatrolRuntimeIdentity
    dynamic_obstacles: tuple[PatrolLiveDynamicObstacle, ...] = ()
    avoidance: PatrolLiveAvoidance = field(default_factory=PatrolLiveAvoidance)
    schema_version: str = field(default=LIVE_STATUS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", validate_patrol_run_id(self.run_id))
        object.__setattr__(self, "sequence", _non_negative_integer(self.sequence, "sequence"))
        try:
            state = PatrolTaskState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported patrol state: {self.state!r}.") from error
        try:
            outcome = PatrolLiveOutcome(self.outcome)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported patrol outcome: {self.outcome!r}.") from error
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "seed", _non_negative_integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "control_step",
            _non_negative_integer(self.control_step, "control_step"),
        )
        simulation_time = _non_negative_number(self.simulation_time_s, "simulation_time_s")
        object.__setattr__(self, "simulation_time_s", simulation_time)
        object.__setattr__(self, "position", _finite_point(self.position, "position"))

        checkpoint_count = _positive_integer(self.checkpoint_count, "checkpoint_count")
        completed_count = _non_negative_integer(
            self.completed_checkpoint_count,
            "completed_checkpoint_count",
        )
        if completed_count > checkpoint_count:
            raise ValueError("completed_checkpoint_count cannot exceed checkpoint_count.")
        progress = _finite_number(self.progress_ratio, "progress_ratio")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress_ratio must be within [0, 1].")
        expected_progress = completed_count / checkpoint_count
        if not math.isclose(progress, expected_progress, abs_tol=1e-9):
            raise ValueError(
                "progress_ratio must equal completed_checkpoint_count / checkpoint_count."
            )
        object.__setattr__(self, "checkpoint_count", checkpoint_count)
        object.__setattr__(self, "completed_checkpoint_count", completed_count)
        object.__setattr__(self, "progress_ratio", progress)
        self._validate_checkpoint(state, completed_count, checkpoint_count)

        if (self.local_goal is None) != (self.remaining_distance_m is None):
            raise ValueError("local_goal and remaining_distance_m must be set together.")
        if self.local_goal is not None:
            object.__setattr__(self, "local_goal", _finite_point(self.local_goal, "local_goal"))
            object.__setattr__(
                self,
                "remaining_distance_m",
                _non_negative_number(self.remaining_distance_m, "remaining_distance_m"),
            )

        if not isinstance(self.latest_readings, PatrolLiveReadings):
            raise TypeError("latest_readings must be a PatrolLiveReadings instance.")
        if (
            self.latest_readings.sampled_at_s is not None
            and self.latest_readings.sampled_at_s > simulation_time
        ):
            raise ValueError("latest_readings cannot come from a future simulation time.")

        if not isinstance(self.active_alerts, tuple):
            object.__setattr__(self, "active_alerts", tuple(self.active_alerts))
        if not all(isinstance(alert, PatrolLiveAlert) for alert in self.active_alerts):
            raise TypeError("active_alerts must contain only PatrolLiveAlert values.")
        rule_ids = [alert.rule_id for alert in self.active_alerts]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("active_alerts cannot contain duplicate rule_id values.")
        if any(alert.raised_at_s > simulation_time for alert in self.active_alerts):
            raise ValueError("active_alerts cannot be raised in the future.")
        if not isinstance(self.runtime_identity, PatrolRuntimeIdentity):
            raise TypeError("runtime_identity must be a PatrolRuntimeIdentity instance.")
        if not isinstance(self.dynamic_obstacles, tuple):
            object.__setattr__(self, "dynamic_obstacles", tuple(self.dynamic_obstacles))
        if not all(
            isinstance(obstacle, PatrolLiveDynamicObstacle)
            for obstacle in self.dynamic_obstacles
        ):
            raise TypeError(
                "dynamic_obstacles must contain only PatrolLiveDynamicObstacle values."
            )
        if len(self.dynamic_obstacles) > MAX_LIVE_DYNAMIC_OBSTACLES:
            raise ValueError(
                "dynamic_obstacles cannot contain more than "
                f"{MAX_LIVE_DYNAMIC_OBSTACLES} values."
            )
        obstacle_ids = [obstacle.obstacle_id for obstacle in self.dynamic_obstacles]
        if len(obstacle_ids) != len(set(obstacle_ids)):
            raise ValueError("dynamic_obstacles cannot contain duplicate obstacle_id values.")
        if not isinstance(self.avoidance, PatrolLiveAvoidance):
            raise TypeError("avoidance must be a PatrolLiveAvoidance instance.")
        if (
            self.avoidance.measured_at_s is not None
            and self.avoidance.measured_at_s > simulation_time
        ):
            raise ValueError("avoidance cannot be measured in the future.")
        if (
            self.avoidance.measured_control_step is not None
            and self.avoidance.measured_control_step > self.control_step
        ):
            raise ValueError("avoidance cannot come from a future control step.")

    def _validate_checkpoint(
        self,
        state: PatrolTaskState,
        completed_count: int,
        checkpoint_count: int,
    ) -> None:
        if (self.current_checkpoint_id is None) != (self.current_checkpoint_ordinal is None):
            raise ValueError(
                "current_checkpoint_id and current_checkpoint_ordinal must be set together."
            )
        if self.current_checkpoint_id is not None:
            object.__setattr__(
                self,
                "current_checkpoint_id",
                _non_empty_text(self.current_checkpoint_id, "current_checkpoint_id"),
            )
            ordinal = _positive_integer(
                self.current_checkpoint_ordinal,
                "current_checkpoint_ordinal",
            )
            if ordinal > checkpoint_count:
                raise ValueError("current_checkpoint_ordinal exceeds checkpoint_count.")
            if ordinal != completed_count + 1:
                raise ValueError(
                    "current_checkpoint_ordinal must follow completed_checkpoint_count."
                )
            object.__setattr__(self, "current_checkpoint_ordinal", ordinal)

        if state is PatrolTaskState.COMPLETED:
            if self.outcome is not PatrolLiveOutcome.SUCCESS:
                raise ValueError("A completed patrol must have outcome='success'.")
            if completed_count != checkpoint_count or self.current_checkpoint_id is not None:
                raise ValueError("A completed patrol must have completed all checkpoints.")
        elif state is PatrolTaskState.FAILED:
            if self.outcome in {PatrolLiveOutcome.RUNNING, PatrolLiveOutcome.SUCCESS}:
                raise ValueError("A failed patrol must have a failure outcome.")
        else:
            if self.outcome is not PatrolLiveOutcome.RUNNING:
                raise ValueError("A non-terminal patrol must have outcome='running'.")
            if completed_count >= checkpoint_count or self.current_checkpoint_id is None:
                raise ValueError("A non-terminal patrol must identify its current checkpoint.")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable on-disk representation."""

        local_goal = None
        if self.local_goal is not None:
            local_goal = {"x_m": self.local_goal[0], "y_m": self.local_goal[1]}
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "state": self.state.value,
            "outcome": self.outcome.value,
            "seed": self.seed,
            "control_step": self.control_step,
            "simulation_time_s": self.simulation_time_s,
            "position": {"x_m": self.position[0], "y_m": self.position[1]},
            "checkpoint_progress": {
                "current_checkpoint_id": self.current_checkpoint_id,
                "current_checkpoint_ordinal": self.current_checkpoint_ordinal,
                "completed_checkpoint_count": self.completed_checkpoint_count,
                "checkpoint_count": self.checkpoint_count,
                "progress_ratio": self.progress_ratio,
            },
            "local_navigation": {
                "local_goal": local_goal,
                "remaining_distance_m": self.remaining_distance_m,
            },
            "latest_readings": {
                "temperature_c": self.latest_readings.temperature_c,
                "smoke_obscuration_pct_per_m": (
                    self.latest_readings.smoke_obscuration_pct_per_m
                ),
                "carbon_monoxide_ppm": self.latest_readings.carbon_monoxide_ppm,
                "sampled_at_s": self.latest_readings.sampled_at_s,
                "checkpoint_id": self.latest_readings.checkpoint_id,
            },
            "active_alerts": [
                {
                    "rule_id": alert.rule_id,
                    "sensor_id": alert.sensor_id,
                    "severity": alert.severity,
                    "observed_value": alert.observed_value,
                    "threshold": alert.threshold,
                    "unit": alert.unit,
                    "message": alert.message,
                    "raised_at_s": alert.raised_at_s,
                }
                for alert in self.active_alerts
            ],
            "runtime_identity": {
                "model_id": self.runtime_identity.model_id,
                "model_sha256": self.runtime_identity.model_sha256,
                "safety_shield_enabled": self.runtime_identity.safety_shield_enabled,
                "safety_shield_id": self.runtime_identity.safety_shield_id,
                "safety_shield_sha256": self.runtime_identity.safety_shield_sha256,
            },
            "dynamic_obstacles": [
                obstacle.to_dict() for obstacle in self.dynamic_obstacles
            ],
            "avoidance": self.avoidance.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PatrolLiveSnapshot:
        """Parse and strictly validate one on-disk snapshot mapping."""

        root = _mapping(raw, "live status")
        schema_version = root.get("schema_version")
        base_keys = {
            "schema_version",
            "run_id",
            "sequence",
            "state",
            "outcome",
            "seed",
            "control_step",
            "simulation_time_s",
            "position",
            "checkpoint_progress",
            "local_navigation",
            "latest_readings",
            "active_alerts",
            "runtime_identity",
        }
        if schema_version == LIVE_STATUS_SCHEMA_VERSION:
            _exact_keys(
                root,
                base_keys | {"dynamic_obstacles", "avoidance"},
                "live status",
            )
        elif schema_version == LEGACY_LIVE_STATUS_SCHEMA_VERSION:
            _exact_keys(root, base_keys, "live status")
        else:
            raise ValueError(
                "schema_version must be one of "
                f"{LEGACY_LIVE_STATUS_SCHEMA_VERSION!r} or {LIVE_STATUS_SCHEMA_VERSION!r}."
            )
        position_raw = _point_mapping(root["position"], "position")
        progress_raw = _mapping(root["checkpoint_progress"], "checkpoint_progress")
        _exact_keys(
            progress_raw,
            {
                "current_checkpoint_id",
                "current_checkpoint_ordinal",
                "completed_checkpoint_count",
                "checkpoint_count",
                "progress_ratio",
            },
            "checkpoint_progress",
        )
        navigation_raw = _mapping(root["local_navigation"], "local_navigation")
        _exact_keys(
            navigation_raw,
            {"local_goal", "remaining_distance_m"},
            "local_navigation",
        )
        goal_raw = navigation_raw["local_goal"]
        local_goal = None
        if goal_raw is not None:
            goal_mapping = _point_mapping(goal_raw, "local_goal")
            local_goal = (goal_mapping["x_m"], goal_mapping["y_m"])

        readings_raw = _mapping(root["latest_readings"], "latest_readings")
        _exact_keys(
            readings_raw,
            {
                "temperature_c",
                "smoke_obscuration_pct_per_m",
                "carbon_monoxide_ppm",
                "sampled_at_s",
                "checkpoint_id",
            },
            "latest_readings",
        )
        alerts_raw = _sequence(root["active_alerts"], "active_alerts")
        alerts: list[PatrolLiveAlert] = []
        for index, item in enumerate(alerts_raw, 1):
            alert_raw = _mapping(item, f"active_alerts[{index}]")
            _exact_keys(
                alert_raw,
                {
                    "rule_id",
                    "sensor_id",
                    "severity",
                    "observed_value",
                    "threshold",
                    "unit",
                    "message",
                    "raised_at_s",
                },
                f"active_alerts[{index}]",
            )
            alerts.append(PatrolLiveAlert(**alert_raw))

        runtime_raw = _mapping(root["runtime_identity"], "runtime_identity")
        _exact_keys(
            runtime_raw,
            {
                "model_id",
                "model_sha256",
                "safety_shield_enabled",
                "safety_shield_id",
                "safety_shield_sha256",
            },
            "runtime_identity",
        )
        dynamic_obstacles: tuple[PatrolLiveDynamicObstacle, ...] = ()
        avoidance = PatrolLiveAvoidance()
        if schema_version == LIVE_STATUS_SCHEMA_VERSION:
            dynamic_raw = _sequence(root["dynamic_obstacles"], "dynamic_obstacles")
            dynamic_obstacles = tuple(
                PatrolLiveDynamicObstacle.from_dict(
                    item,
                    name=f"dynamic_obstacles[{index}]",
                )
                for index, item in enumerate(dynamic_raw)
            )
            avoidance = PatrolLiveAvoidance.from_dict(root["avoidance"])
        return cls(
            run_id=root["run_id"],
            sequence=root["sequence"],
            state=root["state"],
            outcome=root["outcome"],
            seed=root["seed"],
            control_step=root["control_step"],
            simulation_time_s=root["simulation_time_s"],
            position=(position_raw["x_m"], position_raw["y_m"]),
            current_checkpoint_id=progress_raw["current_checkpoint_id"],
            current_checkpoint_ordinal=progress_raw["current_checkpoint_ordinal"],
            completed_checkpoint_count=progress_raw["completed_checkpoint_count"],
            checkpoint_count=progress_raw["checkpoint_count"],
            progress_ratio=progress_raw["progress_ratio"],
            local_goal=local_goal,
            remaining_distance_m=navigation_raw["remaining_distance_m"],
            latest_readings=PatrolLiveReadings(**readings_raw),
            active_alerts=tuple(alerts),
            runtime_identity=PatrolRuntimeIdentity(**runtime_raw),
            dynamic_obstacles=dynamic_obstacles,
            avoidance=avoidance,
        )


class PatrolLiveStatusWriter:
    """Publish validated snapshots to an atomic current file and JSONL history."""

    def __init__(self, output_directory: str | Path, *, run_id: str) -> None:
        self.run_id = validate_patrol_run_id(run_id)
        self.output_directory = Path(output_directory).expanduser()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_directory / "status.json"
        self.events_path = self.output_directory / "events.jsonl"
        self._last_snapshot = self._load_existing_state()

    @property
    def last_snapshot(self) -> PatrolLiveSnapshot | None:
        return self._last_snapshot

    def publish(self, snapshot: PatrolLiveSnapshot) -> None:
        """Append one event and atomically expose the same snapshot as current."""

        if not isinstance(snapshot, PatrolLiveSnapshot):
            raise TypeError("snapshot must be a PatrolLiveSnapshot instance.")
        if snapshot.run_id != self.run_id:
            raise ValueError(
                f"snapshot run_id {snapshot.run_id!r} does not match writer run_id "
                f"{self.run_id!r}."
            )
        _validate_successor(self._last_snapshot, snapshot)
        encoded = _encode_snapshot(snapshot)

        temporary_path = self._write_temporary_status(encoded)
        event_committed = False
        try:
            _append_single_json_line(self.events_path, encoded)
            event_committed = True
            for attempt in range(2):
                try:
                    os.replace(temporary_path, self.status_path)
                    break
                except OSError:
                    if attempt == 1:
                        raise
        except Exception:
            temporary_path.unlink(missing_ok=True)
            if event_committed:
                self._last_snapshot = snapshot
            raise
        self._last_snapshot = snapshot

    def _write_temporary_status(self, encoded: bytes) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".status-",
            suffix=".tmp",
            dir=self.output_directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path

    def _load_existing_state(self) -> PatrolLiveSnapshot | None:
        status = read_patrol_live_status(self.status_path) if self.status_path.exists() else None
        events = read_patrol_live_events(self.events_path) if self.events_path.exists() else ()
        candidates = tuple(item for item in (status, events[-1] if events else None) if item)
        for candidate in candidates:
            if candidate.run_id != self.run_id:
                raise ValueError(
                    f"Existing live status belongs to run_id {candidate.run_id!r}, "
                    f"not {self.run_id!r}."
                )
        if status is not None and events and status.sequence == events[-1].sequence:
            if status != events[-1]:
                raise ValueError("status.json and the latest event disagree at one sequence.")
        latest = max(candidates, key=lambda item: item.sequence, default=None)
        if latest is not None and (status is None or latest.sequence > status.sequence):
            temporary_path = self._write_temporary_status(_encode_snapshot(latest))
            try:
                os.replace(temporary_path, self.status_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        return latest


def read_patrol_live_status(path: str | Path) -> PatrolLiveSnapshot:
    """Read and validate ``status.json`` from a file or output directory."""

    status_path = _resolve_path(path, "status.json")
    try:
        raw = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid live status from {status_path}: {error}") from error
    return PatrolLiveSnapshot.from_dict(raw)


def read_patrol_live_events(path: str | Path) -> tuple[PatrolLiveSnapshot, ...]:
    """Read and validate the ordered snapshot history from a file or directory."""

    events_path = _resolve_path(path, "events.jsonl")
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read live events from {events_path}: {error}") from error
    if any(not line.strip() for line in lines):
        raise ValueError(f"Blank JSONL records are not allowed in {events_path}.")

    snapshots: list[PatrolLiveSnapshot] = []
    for line_number, line in enumerate(lines, 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {events_path} at line {line_number}: {error}"
            ) from error
        snapshot = PatrolLiveSnapshot.from_dict(raw)
        _validate_successor(snapshots[-1] if snapshots else None, snapshot)
        if snapshots and snapshot.run_id != snapshots[0].run_id:
            raise ValueError("All live events must belong to the same run_id.")
        snapshots.append(snapshot)
    return tuple(snapshots)


def _validate_successor(
    previous: PatrolLiveSnapshot | None,
    current: PatrolLiveSnapshot,
) -> None:
    if previous is None:
        return
    if current.run_id != previous.run_id:
        raise ValueError("run_id cannot change within one live status stream.")
    if current.sequence <= previous.sequence:
        raise ValueError("sequence must increase strictly.")
    if current.control_step < previous.control_step:
        raise ValueError("control_step must be monotonic.")
    if current.simulation_time_s < previous.simulation_time_s:
        raise ValueError("simulation_time_s must be monotonic.")
    if previous.state in {PatrolTaskState.COMPLETED, PatrolTaskState.FAILED}:
        raise ValueError("A terminal live status cannot have a successor.")
    if current.seed != previous.seed:
        raise ValueError("seed cannot change within one live status stream.")
    if current.checkpoint_count != previous.checkpoint_count:
        raise ValueError("checkpoint_count cannot change within one live status stream.")
    if current.runtime_identity != previous.runtime_identity:
        raise ValueError("runtime_identity cannot change within one live status stream.")
    progress_delta = (
        current.completed_checkpoint_count - previous.completed_checkpoint_count
    )
    if progress_delta not in {0, 1}:
        raise ValueError("completed checkpoint progress must advance by at most one.")
    if (
        progress_delta == 0
        and current.current_checkpoint_id != previous.current_checkpoint_id
    ):
        raise ValueError("current checkpoint cannot change before progress advances.")


def _encode_snapshot(snapshot: PatrolLiveSnapshot) -> bytes:
    text = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def _append_single_json_line(path: Path, encoded: bytes) -> None:
    line = encoded + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError("Could not append one complete live event record.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_path(path: str | Path, filename: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate / filename if candidate.is_dir() else candidate


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, unexpected={unexpected}.")


def _point_mapping(value: Any, name: str) -> Mapping[str, Any]:
    point = _mapping(value, name)
    _exact_keys(point, {"x_m", "y_m"}, name)
    return point


def _non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _non_negative_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    result = _non_negative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite_point(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{name} must contain exactly two coordinates.")
    return (_finite_number(value[0], f"{name}[0]"), _finite_number(value[1], f"{name}[1]"))


def _normalized_action(value: Any, name: str) -> tuple[float, float]:
    action = _finite_point(value, name)
    if any(component < -1.0 or component > 1.0 for component in action):
        raise ValueError(f"{name} components must be within [-1, 1].")
    return action


def _normalized_action_dict(value: tuple[float, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"linear_normalized": value[0], "angular_normalized": value[1]}


def _executed_action_dict(value: tuple[float, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"linear_mps": value[0], "angular_radps": value[1]}


def _parse_normalized_action(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    action = _mapping(value, name)
    _exact_keys(action, {"linear_normalized", "angular_normalized"}, name)
    return (action["linear_normalized"], action["angular_normalized"])


def _parse_executed_action(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    action = _mapping(value, name)
    _exact_keys(action, {"linear_mps", "angular_radps"}, name)
    return (action["linear_mps"], action["angular_radps"])


def _sha256(value: Any, name: str) -> str:
    text = _non_empty_text(value, name)
    is_hexadecimal = all(character in "0123456789abcdef" for character in text)
    if len(text) != _SHA256_LENGTH or not is_hexadecimal:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest.")
    return text
