"""Portable result schema for the multi-checkpoint Webots patrol demo.

This module intentionally contains no Webots imports.  It keeps model selection
and result serialization testable in ordinary Python while the live controller
is executed by Webots.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Literal, TypeVar

from indoor_patrol_rl.evaluation.checkpoint_selection import sha256_file
from indoor_patrol_rl.identifiers import validate_patrol_run_id

ModelTrainingDomain = Literal["webots", "kinematic", "unknown"]
PatrolOutcome = Literal["completed", "collision", "safety_stop", "timeout", "error"]

_FORMAL_MIN_EPISODES = 20
_FORMAL_MIN_SUCCESS_RATE = 0.75
_FORMAL_MAX_COLLISION_RATE = 0.15
_AUDIT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_AUDIT_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CSV_FILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.csv")


@dataclass(frozen=True)
class PatrolModelSelection:
    """Resolved model plus the evidence domain in which it was trained."""

    path: Path
    training_domain: ModelTrainingDomain
    selection_reason: str
    model_sha256: str
    selection_manifest: Path | None = None


@dataclass(frozen=True)
class PatrolTrajectorySample:
    """One post-action sample from the real Webots controller loop."""

    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    yaw_rad: float
    checkpoint_id: str
    waypoint_index: int
    local_target_x_m: float
    local_target_y_m: float
    remaining_path_m: float
    requested_linear_mps: float
    requested_angular_radps: float
    executed_linear_mps: float
    executed_angular_radps: float
    minimum_lidar_range_m: float
    clearance_m: float
    collision: bool
    inference_latency_ms: float
    policy_action_linear: float | None = None
    policy_action_angular: float | None = None
    shielded_action_linear: float | None = None
    shielded_action_angular: float | None = None
    safety_shield_active: bool = False
    safety_shield_intervention: float = 0.0
    shield_front_clearance_m: float | None = None
    shield_left_clearance_m: float | None = None
    shield_right_clearance_m: float | None = None
    minimum_lidar_beam_index: int | None = None
    minimum_lidar_angle_rad: float | None = None
    dynamic_obstacle_1_x_m: float | None = None
    dynamic_obstacle_1_y_m: float | None = None
    dynamic_obstacle_2_x_m: float | None = None
    dynamic_obstacle_2_y_m: float | None = None
    critical_proximity: bool = False
    proximity_closing_rate_mps: float | None = None
    estimated_ttc_s: float | None = None
    ttc_escape_guard_active: bool = False
    ttc_escape_guard_mode: Literal[
        "inactive",
        "reverse",
        "forward",
        "rotate",
        "unrecoverable",
    ] = "inactive"
    ttc_escape_guard_target_linear_mps: float | None = None
    ttc_escape_guard_target_angular_radps: float | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent shield evidence."""

        action_values = (
            self.policy_action_linear,
            self.policy_action_angular,
            self.shielded_action_linear,
            self.shielded_action_angular,
        )
        if any(value is not None for value in action_values) and not all(
            value is not None for value in action_values
        ):
            raise ValueError("Policy and shielded action components must be recorded together.")
        for name, value in zip(
            (
                "policy_action_linear",
                "policy_action_angular",
                "shielded_action_linear",
                "shielded_action_angular",
            ),
            action_values,
            strict=True,
        ):
            if value is not None:
                _finite_range(value, name, minimum=-1.0, maximum=1.0)

        clearance_values = (
            self.shield_front_clearance_m,
            self.shield_left_clearance_m,
            self.shield_right_clearance_m,
        )
        if any(value is not None for value in clearance_values) and not all(
            value is not None for value in clearance_values
        ):
            raise ValueError("Front, left, and right shield clearances must be recorded together.")
        if all(value is not None for value in action_values) != all(
            value is not None for value in clearance_values
        ):
            raise ValueError("Action vectors and sector clearances must be recorded together.")
        for name, value in zip(
            (
                "shield_front_clearance_m",
                "shield_left_clearance_m",
                "shield_right_clearance_m",
            ),
            clearance_values,
            strict=True,
        ):
            if value is not None:
                _finite_range(value, name, minimum=0.0)

        if type(self.safety_shield_active) is not bool:
            raise ValueError("safety_shield_active must be a boolean.")
        intervention = _finite_range(
            self.safety_shield_intervention,
            "safety_shield_intervention",
            minimum=0.0,
        )
        if self.safety_shield_active and intervention <= 0.0:
            raise ValueError("An active safety shield must record a positive intervention.")
        if all(value is not None for value in action_values):
            expected = math.hypot(
                float(self.shielded_action_linear) - float(self.policy_action_linear),
                float(self.shielded_action_angular) - float(self.policy_action_angular),
            )
            if not math.isclose(intervention, expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(
                    "safety_shield_intervention must equal the action-vector difference."
                )
        elif intervention != 0.0 or self.safety_shield_active:
            raise ValueError("Shield intervention evidence requires both action vectors.")

        if type(self.collision) is not bool:
            raise ValueError("collision must be a boolean.")
        if type(self.critical_proximity) is not bool:
            raise ValueError("critical_proximity must be a boolean.")
        if self.collision and self.critical_proximity:
            raise ValueError("Physical collision and critical proximity are mutually exclusive.")

        if self.proximity_closing_rate_mps is not None:
            _finite_range(
                self.proximity_closing_rate_mps,
                "proximity_closing_rate_mps",
            )
        if self.estimated_ttc_s is not None:
            _finite_range(self.estimated_ttc_s, "estimated_ttc_s", minimum=0.0)

        if type(self.ttc_escape_guard_active) is not bool:
            raise ValueError("ttc_escape_guard_active must be a boolean.")
        if self.ttc_escape_guard_mode not in {
            "inactive",
            "reverse",
            "forward",
            "rotate",
            "unrecoverable",
        }:
            raise ValueError("ttc_escape_guard_mode is not supported.")
        guard_targets = (
            self.ttc_escape_guard_target_linear_mps,
            self.ttc_escape_guard_target_angular_radps,
        )
        if any(value is not None for value in guard_targets) and not all(
            value is not None for value in guard_targets
        ):
            raise ValueError("TTC escape guard target velocities must be recorded together.")
        for name, value in zip(
            (
                "ttc_escape_guard_target_linear_mps",
                "ttc_escape_guard_target_angular_radps",
            ),
            guard_targets,
            strict=True,
        ):
            if value is not None:
                _finite_range(value, name)
        if self.ttc_escape_guard_active:
            if self.ttc_escape_guard_mode == "inactive":
                raise ValueError("An active TTC escape guard must use a non-inactive mode.")
            if not all(value is not None for value in guard_targets):
                raise ValueError("An active TTC escape guard requires target velocities.")
        elif self.ttc_escape_guard_mode != "inactive" or any(
            value is not None for value in guard_targets
        ):
            raise ValueError(
                "An inactive TTC escape guard must use inactive mode without target velocities."
            )

        _validate_scene_diagnostic(
            minimum_lidar_beam_index=self.minimum_lidar_beam_index,
            minimum_lidar_angle_rad=self.minimum_lidar_angle_rad,
            dynamic_obstacle_1_x_m=self.dynamic_obstacle_1_x_m,
            dynamic_obstacle_1_y_m=self.dynamic_obstacle_1_y_m,
            dynamic_obstacle_2_x_m=self.dynamic_obstacle_2_x_m,
            dynamic_obstacle_2_y_m=self.dynamic_obstacle_2_y_m,
        )


@dataclass(frozen=True)
class PatrolTTCEscapeGuardEventRecord:
    """One guard update, including dwell updates absent from navigation trajectory."""

    event_index: int
    control_step: int
    simulation_time_s: float
    phase: Literal["navigation", "dwell"]
    active: bool
    mode: Literal[
        "inactive",
        "reverse",
        "forward",
        "rotate",
        "unrecoverable",
    ]
    target_linear_mps: float | None
    target_angular_radps: float | None
    counts_as_intervention_step: bool

    def __post_init__(self) -> None:
        if isinstance(self.event_index, bool) or not isinstance(self.event_index, int):
            raise ValueError("event_index must be an integer.")
        if self.event_index < 1:
            raise ValueError("event_index must be positive.")
        if isinstance(self.control_step, bool) or not isinstance(self.control_step, int):
            raise ValueError("control_step must be an integer.")
        if self.control_step < 0:
            raise ValueError("control_step cannot be negative.")
        _finite_range(self.simulation_time_s, "simulation_time_s", minimum=0.0)
        if self.phase not in {"navigation", "dwell"}:
            raise ValueError("phase must be 'navigation' or 'dwell'.")
        if type(self.counts_as_intervention_step) is not bool:
            raise ValueError("counts_as_intervention_step must be a boolean.")
        _validate_ttc_escape_guard_state(
            active=self.active,
            mode=self.mode,
            target_linear_mps=self.target_linear_mps,
            target_angular_radps=self.target_angular_radps,
        )
        if self.counts_as_intervention_step and not self.active:
            raise ValueError("Only an active guard update can count as an intervention step.")
        if self.phase == "navigation" and self.active != self.counts_as_intervention_step:
            raise ValueError(
                "A navigation guard update counts exactly when the guard is active."
            )


@dataclass(frozen=True)
class PatrolCheckpointRecord:
    """Per-checkpoint progress and path-efficiency record."""

    checkpoint_id: str
    visit_order: int
    target_x_m: float
    target_y_m: float
    reached: bool
    reached_control_step: int | None
    arrival_simulation_time_s: float | None
    segment_control_steps: int
    actual_segment_path_m: float
    planned_segment_path_m: float | None
    segment_path_ratio: float | None
    segment_minimum_clearance_m: float | None


@dataclass(frozen=True)
class PatrolCollisionRecord:
    """A physical collision latched by Webots contact points."""

    collision_index: int
    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    detection_source: str
    minimum_lidar_range_m: float
    minimum_lidar_beam_index: int | None = None
    minimum_lidar_angle_rad: float | None = None
    dynamic_obstacle_1_x_m: float | None = None
    dynamic_obstacle_1_y_m: float | None = None
    dynamic_obstacle_2_x_m: float | None = None
    dynamic_obstacle_2_y_m: float | None = None

    def __post_init__(self) -> None:
        """Require a physical-contact event and reject partial scene diagnostics."""

        _validate_terminal_scene_record(
            index=self.collision_index,
            index_name="collision_index",
            control_step=self.control_step,
            simulation_time_s=self.simulation_time_s,
            position_x_m=self.position_x_m,
            position_y_m=self.position_y_m,
            detection_source=self.detection_source,
            expected_detection_source="contact",
            minimum_lidar_range_m=self.minimum_lidar_range_m,
        )

        _validate_scene_diagnostic(
            minimum_lidar_beam_index=self.minimum_lidar_beam_index,
            minimum_lidar_angle_rad=self.minimum_lidar_angle_rad,
            dynamic_obstacle_1_x_m=self.dynamic_obstacle_1_x_m,
            dynamic_obstacle_1_y_m=self.dynamic_obstacle_1_y_m,
            dynamic_obstacle_2_x_m=self.dynamic_obstacle_2_x_m,
            dynamic_obstacle_2_y_m=self.dynamic_obstacle_2_y_m,
        )


@dataclass(frozen=True)
class PatrolCriticalProximityRecord:
    """One audited proximity event before physical contact.

    ``lidar_threshold`` records the rising edge of an interval at the monitor's
    hard critical boundary and therefore requires at least one consecutive
    critical frame.  The robot may subsequently recover, so this event is not
    necessarily terminal.
    ``ttc_escape_guard_timeout`` records a bounded application escape attempt
    that could not establish its release clearance; it may stop before the hard
    boundary and must retain the observed (possibly zero) critical-frame count.
    """

    proximity_index: int
    control_step: int
    simulation_time_s: float
    position_x_m: float
    position_y_m: float
    detection_source: str
    minimum_lidar_range_m: float
    consecutive_frames: int
    closing_rate_mps: float | None = None
    estimated_ttc_s: float | None = None
    minimum_lidar_beam_index: int | None = None
    minimum_lidar_angle_rad: float | None = None
    dynamic_obstacle_1_x_m: float | None = None
    dynamic_obstacle_1_y_m: float | None = None
    dynamic_obstacle_2_x_m: float | None = None
    dynamic_obstacle_2_y_m: float | None = None

    def __post_init__(self) -> None:
        """Require truthful proximity provenance and complete diagnostics."""

        supported_sources = {
            "lidar_threshold",
            "ttc_escape_guard_timeout",
        }
        if self.detection_source not in supported_sources:
            raise ValueError(
                "detection_source must be 'lidar_threshold' or "
                "'ttc_escape_guard_timeout' for a proximity safety stop."
            )

        _validate_terminal_scene_record(
            index=self.proximity_index,
            index_name="proximity_index",
            control_step=self.control_step,
            simulation_time_s=self.simulation_time_s,
            position_x_m=self.position_x_m,
            position_y_m=self.position_y_m,
            detection_source=self.detection_source,
            expected_detection_source=self.detection_source,
            minimum_lidar_range_m=self.minimum_lidar_range_m,
        )
        if isinstance(self.consecutive_frames, bool) or not isinstance(
            self.consecutive_frames, int
        ):
            raise ValueError("consecutive_frames must be an integer.")
        minimum_frames = 1 if self.detection_source == "lidar_threshold" else 0
        if self.consecutive_frames < minimum_frames:
            if self.detection_source == "lidar_threshold":
                raise ValueError(
                    "consecutive_frames must be positive for lidar_threshold."
                )
            raise ValueError("consecutive_frames cannot be negative.")
        closing_rate = None
        if self.closing_rate_mps is not None:
            closing_rate = _finite_range(self.closing_rate_mps, "closing_rate_mps")
        if self.estimated_ttc_s is not None:
            _finite_range(self.estimated_ttc_s, "estimated_ttc_s", minimum=0.0)
            if self.estimated_ttc_s > 0.0 and (closing_rate is None or closing_rate <= 0.0):
                raise ValueError("A positive estimated_ttc_s requires a positive closing_rate_mps.")
        _validate_scene_diagnostic(
            minimum_lidar_beam_index=self.minimum_lidar_beam_index,
            minimum_lidar_angle_rad=self.minimum_lidar_angle_rad,
            dynamic_obstacle_1_x_m=self.dynamic_obstacle_1_x_m,
            dynamic_obstacle_1_y_m=self.dynamic_obstacle_1_y_m,
            dynamic_obstacle_2_x_m=self.dynamic_obstacle_2_x_m,
            dynamic_obstacle_2_y_m=self.dynamic_obstacle_2_y_m,
        )


@dataclass
class PatrolCriticalProximityEdgeTracker:
    """Latch one audit event per continuous hard-critical LiDAR interval."""

    active: bool = False

    def __post_init__(self) -> None:
        if type(self.active) is not bool:
            raise ValueError("active must be a boolean.")

    def update(self, critical: bool) -> bool:
        """Return true only on a false-to-true transition and update the latch."""

        if type(critical) is not bool:
            raise ValueError("critical must be a boolean.")
        rising_edge = critical and not self.active
        self.active = critical
        return rising_edge


@dataclass(frozen=True)
class PatrolRunSummary:
    """Machine-readable aggregate for one Webots multi-point patrol run."""

    schema_version: str
    run_id: str
    backend: str
    planner: str
    scenario_seed: int
    model_path: str
    model_sha256: str
    model_training_domain: ModelTrainingDomain
    model_selection_reason: str
    outcome: PatrolOutcome
    failure_reason: str | None
    checkpoints_total: int
    checkpoints_visited: int
    completion_ratio: float
    collision_count: int
    control_step_count: int
    physics_step_count: int
    elapsed_simulation_time_s: float
    actual_path_length_m: float
    planned_attempted_path_length_m: float
    nominal_route_length_m: float
    completed_path_ratio: float | None
    minimum_lidar_range_m: float
    minimum_clearance_m: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    trajectory_file: str
    checkpoints_file: str
    collisions_file: str
    system_deployment_id: str | None = None
    system_deployment_profile_sha256: str | None = None
    patrol_scenario_qualified: bool = False
    deployment_evidence_scope: str | None = None
    safety_shield_id: str | None = None
    safety_shield_sha256: str | None = None
    safety_shield_enabled: bool = False
    safety_shield_intervention_steps: int = 0
    safety_shield_intervention_rate: float = 0.0
    critical_proximity_count: int = 0
    proximity_events_file: str = "proximity_events.csv"
    patrol_scenario_id: str | None = None
    patrol_scenario_mode: Literal["operational", "stress"] | None = None
    patrol_scenario_config_sha256: str | None = None
    dynamic_obstacle_count: int | None = None
    dynamic_obstacle_speed_max_mps: float | None = None
    ttc_escape_guard_enabled: bool = False
    ttc_escape_guard_id: str | None = None
    ttc_escape_guard_evidence_scope: str | None = None
    ttc_escape_guard_intervention_steps: int = 0
    ttc_escape_guard_activation_count: int = 0
    ttc_escape_guard_recovery_count: int = 0
    ttc_escape_guard_unrecoverable_count: int = 0
    ttc_escape_guard_events_file: str = "ttc_escape_guard_events.csv"

    def __post_init__(self) -> None:
        """Validate the deployment identity and aggregate intervention evidence."""

        object.__setattr__(self, "run_id", validate_patrol_run_id(self.run_id))
        if type(self.safety_shield_enabled) is not bool:
            raise ValueError("safety_shield_enabled must be a boolean.")
        identity_values = (
            self.system_deployment_id,
            self.system_deployment_profile_sha256,
            self.safety_shield_id,
            self.safety_shield_sha256,
        )
        if not self.safety_shield_enabled:
            if any(value is not None for value in identity_values):
                raise ValueError(
                    "A disabled safety shield cannot carry deployment or shield identity."
                )
        else:
            _audit_identity(self.safety_shield_id, "safety_shield_id")
            _audit_sha256(self.safety_shield_sha256, "safety_shield_sha256")

        deployment_values = (
            self.system_deployment_id,
            self.system_deployment_profile_sha256,
        )
        if any(value is not None for value in deployment_values) and not all(
            value is not None for value in deployment_values
        ):
            raise ValueError("System deployment ID and profile SHA-256 must be recorded together.")
        if all(value is not None for value in deployment_values):
            _audit_identity(self.system_deployment_id, "system_deployment_id")
            _audit_sha256(
                self.system_deployment_profile_sha256,
                "system_deployment_profile_sha256",
            )

        if type(self.patrol_scenario_qualified) is not bool:
            raise ValueError("patrol_scenario_qualified must be a boolean.")
        if self.deployment_evidence_scope is not None:
            _audit_identity(
                self.deployment_evidence_scope,
                "deployment_evidence_scope",
            )
        if all(value is not None for value in deployment_values):
            if self.deployment_evidence_scope is None:
                raise ValueError("A system deployment identity must declare its evidence scope.")
        if self.patrol_scenario_qualified:
            raise ValueError(
                "patrol_scenario_qualified cannot be true without a qualification issuer."
            )

        if isinstance(self.safety_shield_intervention_steps, bool) or not isinstance(
            self.safety_shield_intervention_steps, int
        ):
            raise ValueError("safety_shield_intervention_steps must be an integer.")
        if self.safety_shield_intervention_steps < 0:
            raise ValueError("safety_shield_intervention_steps cannot be negative.")
        if self.safety_shield_intervention_steps > self.control_step_count:
            raise ValueError("safety_shield_intervention_steps cannot exceed control_step_count.")
        intervention_rate = _finite_range(
            self.safety_shield_intervention_rate,
            "safety_shield_intervention_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if not self.safety_shield_enabled and (
            self.safety_shield_intervention_steps != 0 or intervention_rate != 0.0
        ):
            raise ValueError("A disabled safety shield cannot record interventions.")
        if (self.safety_shield_intervention_steps == 0) != (intervention_rate == 0.0):
            raise ValueError("Shield intervention steps and rate must both be zero or positive.")

        if isinstance(self.collision_count, bool) or not isinstance(self.collision_count, int):
            raise ValueError("collision_count must be an integer.")
        if self.collision_count < 0:
            raise ValueError("collision_count cannot be negative.")
        if isinstance(self.critical_proximity_count, bool) or not isinstance(
            self.critical_proximity_count, int
        ):
            raise ValueError("critical_proximity_count must be an integer.")
        if self.critical_proximity_count < 0:
            raise ValueError("critical_proximity_count cannot be negative.")
        patrol_output_files = {
            "trajectory_file": self.trajectory_file,
            "checkpoints_file": self.checkpoints_file,
            "collisions_file": self.collisions_file,
            "proximity_events_file": self.proximity_events_file,
            "ttc_escape_guard_events_file": self.ttc_escape_guard_events_file,
        }
        for name, value in patrol_output_files.items():
            _portable_csv_filename(value, name)
        if len(set(patrol_output_files.values())) != len(patrol_output_files):
            raise ValueError("Patrol output CSV filenames must be mutually distinct.")

        if self.outcome == "collision":
            if self.collision_count < 1:
                raise ValueError("A collision outcome requires contact records.")
        elif self.outcome == "safety_stop":
            if self.collision_count != 0 or self.critical_proximity_count < 1:
                raise ValueError(
                    "A safety_stop outcome requires proximity records and no contact collision."
                )
        elif self.collision_count != 0:
            raise ValueError("Only a collision outcome may contain contact records.")

        scenario_identity = (
            self.patrol_scenario_id,
            self.patrol_scenario_mode,
            self.patrol_scenario_config_sha256,
            self.dynamic_obstacle_count,
            self.dynamic_obstacle_speed_max_mps,
        )
        if any(value is not None for value in scenario_identity) and not all(
            value is not None for value in scenario_identity
        ):
            raise ValueError("Patrol scenario identity fields must be recorded together.")
        if all(value is not None for value in scenario_identity):
            _audit_identity(self.patrol_scenario_id, "patrol_scenario_id")
            if self.patrol_scenario_mode not in {"operational", "stress"}:
                raise ValueError("patrol_scenario_mode must be 'operational' or 'stress'.")
            _audit_sha256(
                self.patrol_scenario_config_sha256,
                "patrol_scenario_config_sha256",
            )
            if isinstance(self.dynamic_obstacle_count, bool) or not isinstance(
                self.dynamic_obstacle_count, int
            ):
                raise ValueError("dynamic_obstacle_count must be an integer.")
            if self.dynamic_obstacle_count < 0:
                raise ValueError("dynamic_obstacle_count cannot be negative.")
            maximum_speed = _finite_range(
                self.dynamic_obstacle_speed_max_mps,
                "dynamic_obstacle_speed_max_mps",
                minimum=0.0,
            )
            if (self.dynamic_obstacle_count == 0) != (maximum_speed == 0.0):
                raise ValueError(
                    "Dynamic-obstacle count and maximum speed must both be zero or positive."
                )

        if type(self.ttc_escape_guard_enabled) is not bool:
            raise ValueError("ttc_escape_guard_enabled must be a boolean.")
        guard_identity = (
            self.ttc_escape_guard_id,
            self.ttc_escape_guard_evidence_scope,
        )
        if self.ttc_escape_guard_enabled:
            _audit_identity(self.ttc_escape_guard_id, "ttc_escape_guard_id")
            _audit_identity(
                self.ttc_escape_guard_evidence_scope,
                "ttc_escape_guard_evidence_scope",
            )
        elif any(value is not None for value in guard_identity):
            raise ValueError("A disabled TTC escape guard cannot carry identity or evidence scope.")

        if isinstance(self.control_step_count, bool) or not isinstance(
            self.control_step_count, int
        ):
            raise ValueError("control_step_count must be an integer.")
        if self.control_step_count < 0:
            raise ValueError("control_step_count cannot be negative.")
        guard_counts = {
            "ttc_escape_guard_intervention_steps": self.ttc_escape_guard_intervention_steps,
            "ttc_escape_guard_activation_count": self.ttc_escape_guard_activation_count,
            "ttc_escape_guard_recovery_count": self.ttc_escape_guard_recovery_count,
            "ttc_escape_guard_unrecoverable_count": self.ttc_escape_guard_unrecoverable_count,
        }
        for name, value in guard_counts.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        if (
            self.ttc_escape_guard_activation_count
            > self.ttc_escape_guard_intervention_steps
        ):
            raise ValueError(
                "ttc_escape_guard_activation_count cannot exceed intervention steps."
            )
        if (self.ttc_escape_guard_intervention_steps == 0) != (
            self.ttc_escape_guard_activation_count == 0
        ):
            raise ValueError(
                "TTC escape guard intervention steps and activations must both be zero or positive."
            )
        if (
            self.ttc_escape_guard_recovery_count
            + self.ttc_escape_guard_unrecoverable_count
            > self.ttc_escape_guard_activation_count
        ):
            raise ValueError(
                "TTC escape guard recoveries plus unrecoverable outcomes cannot exceed activations."
            )
        if not self.ttc_escape_guard_enabled and any(guard_counts.values()):
            raise ValueError("A disabled TTC escape guard cannot record intervention evidence.")


def _validate_terminal_scene_record(
    *,
    index: int,
    index_name: str,
    control_step: int,
    simulation_time_s: float,
    position_x_m: float,
    position_y_m: float,
    detection_source: str,
    expected_detection_source: str,
    minimum_lidar_range_m: float,
) -> None:
    """Validate shared identity, timing, position, and source semantics."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError(f"{index_name} must be a positive integer.")
    if isinstance(control_step, bool) or not isinstance(control_step, int) or control_step < 0:
        raise ValueError("control_step must be a non-negative integer.")
    _finite_range(simulation_time_s, "simulation_time_s", minimum=0.0)
    _finite_range(position_x_m, "position_x_m")
    _finite_range(position_y_m, "position_y_m")
    _finite_range(minimum_lidar_range_m, "minimum_lidar_range_m", minimum=0.0)
    if detection_source != expected_detection_source:
        raise ValueError(f"detection_source must be {expected_detection_source!r} for this record.")


def _validate_scene_diagnostic(
    *,
    minimum_lidar_beam_index: int | None,
    minimum_lidar_angle_rad: float | None,
    dynamic_obstacle_1_x_m: float | None,
    dynamic_obstacle_1_y_m: float | None,
    dynamic_obstacle_2_x_m: float | None,
    dynamic_obstacle_2_y_m: float | None,
) -> None:
    """Validate one optional all-beam LiDAR and two-obstacle scene snapshot."""

    lidar_diagnostic = (minimum_lidar_beam_index, minimum_lidar_angle_rad)
    if any(value is not None for value in lidar_diagnostic) and not all(
        value is not None for value in lidar_diagnostic
    ):
        raise ValueError("Minimum LiDAR beam index and angle must be recorded together.")
    if minimum_lidar_beam_index is not None:
        if isinstance(minimum_lidar_beam_index, bool) or not isinstance(
            minimum_lidar_beam_index, int
        ):
            raise ValueError("minimum_lidar_beam_index must be an integer.")
        if minimum_lidar_beam_index < 0:
            raise ValueError("minimum_lidar_beam_index cannot be negative.")
        _finite_range(
            minimum_lidar_angle_rad,
            "minimum_lidar_angle_rad",
            minimum=-math.pi,
            maximum=math.pi,
        )

    obstacle_positions = (
        dynamic_obstacle_1_x_m,
        dynamic_obstacle_1_y_m,
        dynamic_obstacle_2_x_m,
        dynamic_obstacle_2_y_m,
    )
    if any(value is not None for value in obstacle_positions) and not all(
        value is not None for value in obstacle_positions
    ):
        raise ValueError("Both dynamic-obstacle x/y positions must be recorded together.")
    for name, value in zip(
        (
            "dynamic_obstacle_1_x_m",
            "dynamic_obstacle_1_y_m",
            "dynamic_obstacle_2_x_m",
            "dynamic_obstacle_2_y_m",
        ),
        obstacle_positions,
        strict=True,
    ):
        if value is not None:
            _finite_range(value, name)


def _finite_range(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return one finite numeric audit value after optional bound checks."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return number


def _validate_ttc_escape_guard_state(
    *,
    active: bool,
    mode: str,
    target_linear_mps: float | None,
    target_angular_radps: float | None,
) -> None:
    if type(active) is not bool:
        raise ValueError("active must be a boolean.")
    if mode not in {"inactive", "reverse", "forward", "rotate", "unrecoverable"}:
        raise ValueError("mode is not supported.")
    targets = (target_linear_mps, target_angular_radps)
    if any(value is not None for value in targets) and not all(
        value is not None for value in targets
    ):
        raise ValueError("TTC escape guard target velocities must be recorded together.")
    for name, value in zip(
        ("target_linear_mps", "target_angular_radps"),
        targets,
        strict=True,
    ):
        if value is not None:
            _finite_range(value, name)
    if active:
        if mode == "inactive":
            raise ValueError("An active TTC escape guard must use a non-inactive mode.")
        if not all(value is not None for value in targets):
            raise ValueError("An active TTC escape guard requires target velocities.")
        linear = float(target_linear_mps)
        angular = float(target_angular_radps)
        if mode == "reverse" and not (linear < 0.0 and angular == 0.0):
            raise ValueError("Reverse guard mode requires negative linear and zero angular target.")
        if mode == "forward" and not (linear > 0.0 and angular == 0.0):
            raise ValueError("Forward guard mode requires positive linear and zero angular target.")
        if mode == "rotate" and not (linear == 0.0 and angular != 0.0):
            raise ValueError("Rotate guard mode requires zero linear and nonzero angular target.")
        if mode == "unrecoverable" and (linear != 0.0 or angular != 0.0):
            raise ValueError("Unrecoverable guard mode requires zero velocity targets.")
    elif mode != "inactive" or any(value is not None for value in targets):
        raise ValueError(
            "An inactive TTC escape guard must use inactive mode without target velocities."
        )


def _portable_csv_filename(value: object, name: str) -> str:
    if not isinstance(value, str) or _CSV_FILE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable CSV basename.")
    if Path(value).name != value:
        raise ValueError(f"{name} must be a portable CSV basename.")
    return value


def _audit_identity(value: str | None, name: str) -> str:
    """Validate a compact portable identity token used by audit outputs."""

    if not isinstance(value, str) or _AUDIT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a non-empty portable identity token.")
    return value


def _audit_sha256(value: str | None, name: str) -> str:
    """Validate a canonical lowercase SHA-256 audit identity."""

    if not isinstance(value, str) or _AUDIT_SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def select_patrol_model(
    repository_root: Path,
    explicit_model: str | Path | None = None,
    *,
    selection_manifest: str | Path | None = (
        "artifacts/results/webots_validation_w1_selection_seed11/selection.json"
    ),
    allow_unvalidated_fallback: bool = False,
    webots_model: str | Path = "artifacts/models/webots/seed_11/ppo_webots_latest.zip",
    warm_start_model: str | Path = (
        "artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip"
    ),
) -> PatrolModelSelection:
    """Resolve an immutable, validation-selected model unless explicitly overridden.

    Relative paths are resolved inside ``repository_root``.  An explicitly
    requested missing model is an error rather than a silent fallback.  Mutable
    ``latest`` and kinematic warm-start files are available only when the caller
    explicitly enables an unvalidated exploratory fallback.
    """

    root = repository_root.resolve()
    if explicit_model is not None:
        explicit_path = _inside_root(root, explicit_model)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Explicit patrol model does not exist: {explicit_path}")
        if not allow_unvalidated_fallback:
            raise PermissionError(
                "An explicit patrol model bypasses formal validation. Set "
                "WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL=1 for an exploratory run."
            )
        return PatrolModelSelection(
            explicit_path,
            _infer_training_domain(explicit_path),
            "UNVALIDATED explicit WEBOTS_PATROL_MODEL override",
            sha256_file(explicit_path),
        )

    if selection_manifest is not None:
        manifest_path = _inside_root(root, selection_manifest)
        if manifest_path.is_file():
            return _selection_from_manifest(root, manifest_path)
        if not allow_unvalidated_fallback:
            raise FileNotFoundError(
                "The formal patrol selection manifest does not exist: "
                f"{manifest_path}. Run checkpoint selection first, or explicitly allow "
                "an unvalidated exploratory model."
            )
    elif not allow_unvalidated_fallback:
        raise FileNotFoundError(
            "No formal patrol selection manifest was configured. Pass an explicit model "
            "for an exploratory run or explicitly allow an unvalidated fallback."
        )

    webots_path = _inside_root(root, webots_model)
    if webots_path.is_file():
        return PatrolModelSelection(
            webots_path,
            "webots",
            "UNVALIDATED fallback to mutable Webots latest checkpoint",
            sha256_file(webots_path),
        )

    warm_start_path = _inside_root(root, warm_start_model)
    if warm_start_path.is_file():
        return PatrolModelSelection(
            warm_start_path,
            "kinematic",
            "UNVALIDATED fallback to kinematic stage-2 warm start",
            sha256_file(warm_start_path),
        )
    raise FileNotFoundError(
        f"No patrol model is available. Expected either {webots_path} or {warm_start_path}."
    )


def _selection_from_manifest(
    repository_root: Path,
    manifest_path: Path,
) -> PatrolModelSelection:
    """Load a formal validation selection and verify the selected model digest."""

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read patrol selection manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise ValueError("Patrol selection manifest must contain a JSON object.")
    if raw.get("schema_version") != 1:
        raise ValueError("Patrol selection manifest schema_version must be 1.")
    contract = raw.get("validation_contract")
    if not isinstance(contract, Mapping) or contract.get("split") != "validation":
        raise ValueError("Patrol deployment accepts only a validation selection manifest.")
    episodes = _manifest_integer(contract.get("episodes"), "validation_contract.episodes")
    if episodes < _FORMAL_MIN_EPISODES:
        raise ValueError(
            f"Formal patrol selection requires at least {_FORMAL_MIN_EPISODES} episodes."
        )
    manifest_gate = raw.get("gate")
    if not isinstance(manifest_gate, Mapping):
        raise ValueError("Patrol selection manifest is missing its validation gate.")
    min_success_rate = _manifest_rate(
        manifest_gate.get("min_success_rate"), "gate.min_success_rate"
    )
    max_collision_rate = _manifest_rate(
        manifest_gate.get("max_collision_rate"), "gate.max_collision_rate"
    )
    if min_success_rate < _FORMAL_MIN_SUCCESS_RATE:
        raise ValueError(
            f"Formal patrol selection requires min_success_rate >= {_FORMAL_MIN_SUCCESS_RATE:.2f}."
        )
    if max_collision_rate > _FORMAL_MAX_COLLISION_RATE:
        raise ValueError(
            "Formal patrol selection requires max_collision_rate <= "
            f"{_FORMAL_MAX_COLLISION_RATE:.2f}."
        )
    if manifest_gate.get("requires_model_file") is not True:
        raise ValueError("Formal patrol selection must require the model file.")
    if raw.get("selection_status") != "formal_selected":
        raise ValueError("Patrol selection manifest has no gate-passing formal selection.")
    selected = raw.get("formal_selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Patrol selection manifest is missing formal_selected details.")
    if selected.get("gate_passed") is not True:
        raise ValueError("formal_selected.gate_passed must be true.")
    selected_episodes = _manifest_integer(selected.get("episodes"), "formal_selected.episodes")
    if selected_episodes != episodes:
        raise ValueError("formal_selected.episodes must match validation_contract.episodes.")
    selected_success_rate = _manifest_rate(
        selected.get("success_rate"), "formal_selected.success_rate"
    )
    selected_collision_rate = _manifest_rate(
        selected.get("collision_rate"), "formal_selected.collision_rate"
    )
    if selected_success_rate < max(min_success_rate, _FORMAL_MIN_SUCCESS_RATE):
        raise ValueError("formal_selected.success_rate does not pass the formal gate.")
    if selected_collision_rate > min(max_collision_rate, _FORMAL_MAX_COLLISION_RATE):
        raise ValueError("formal_selected.collision_rate does not pass the formal gate.")
    contract_variant = contract.get("model_variant")
    if raw.get("model_variant") != contract_variant:
        raise ValueError("Manifest and validation contract model_variant values disagree.")
    if selected.get("model_variant") != contract_variant:
        raise ValueError("formal_selected.model_variant does not match the contract.")

    raw_model_path = selected.get("model_path")
    expected_sha256 = selected.get("model_sha256")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise ValueError("formal_selected.model_path must be a non-empty string.")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
    ):
        raise ValueError("formal_selected.model_sha256 must be a SHA-256 digest.")

    model_path = _inside_root(repository_root, raw_model_path)
    try:
        model_path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("A formal selected model must remain inside the repository.") from error
    try:
        model_path.relative_to((repository_root / "artifacts/models/webots").resolve())
    except ValueError as error:
        raise ValueError(
            "A formal selected model must be inside artifacts/models/webots."
        ) from error
    if not model_path.is_file():
        raise FileNotFoundError(f"Formal selected patrol model does not exist: {model_path}")
    actual_sha256 = sha256_file(model_path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ValueError(
            "Formal selected patrol model failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    return PatrolModelSelection(
        path=model_path,
        training_domain="webots",
        selection_reason="gate-passing formal validation selection",
        model_sha256=actual_sha256,
        selection_manifest=manifest_path,
    )


def _manifest_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _manifest_rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite rate between zero and one.")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{label} must be a finite rate between zero and one.")
    return rate


def write_patrol_outputs(
    output_directory: Path,
    *,
    trajectory: list[PatrolTrajectorySample],
    checkpoints: list[PatrolCheckpointRecord],
    collisions: list[PatrolCollisionRecord],
    summary: PatrolRunSummary,
    critical_proximities: list[PatrolCriticalProximityRecord] | None = None,
    ttc_escape_guard_events: list[PatrolTTCEscapeGuardEventRecord] | None = None,
) -> dict[str, Path]:
    """Exclusively write cross-checked patrol evidence and its JSON aggregate."""

    if not checkpoints:
        raise ValueError("A patrol run must contain at least one checkpoint record.")
    proximity_records = list(critical_proximities or ())
    guard_event_records = list(ttc_escape_guard_events or ())
    if summary.collision_count != len(collisions):
        raise ValueError("summary.collision_count must match the contact collision records.")
    if summary.critical_proximity_count != len(proximity_records):
        raise ValueError(
            "summary.critical_proximity_count must match the critical proximity records."
        )
    _validate_trajectory_audit(summary, trajectory)
    _validate_proximity_event_audit(trajectory, proximity_records)
    _validate_guard_event_audit(
        summary,
        trajectory,
        guard_event_records,
        stream_was_provided=ttc_escape_guard_events is not None,
    )

    payload = asdict(summary)
    payload["minimum_lidar_diagnostic"] = _minimum_lidar_diagnostic(trajectory)
    payload["checkpoint_records"] = [asdict(record) for record in checkpoints]
    payload["collision_records"] = [asdict(record) for record in collisions]
    payload["critical_proximity_records"] = [asdict(record) for record in proximity_records]
    payload["ttc_escape_guard_event_records"] = [
        asdict(record) for record in guard_event_records
    ]
    summary_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"

    output_root = Path(output_directory).expanduser().resolve()
    output_paths = {
        "trajectory": _contained_output_path(output_root, summary.trajectory_file),
        "checkpoints": _contained_output_path(output_root, summary.checkpoints_file),
        "collisions": _contained_output_path(output_root, summary.collisions_file),
        "proximity_events": _contained_output_path(
            output_root,
            summary.proximity_events_file,
        ),
        "ttc_escape_guard_events": _contained_output_path(
            output_root,
            summary.ttc_escape_guard_events_file,
        ),
        "summary": _contained_output_path(output_root, "summary.json"),
    }
    existing = [
        path
        for path in output_paths.values()
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "Patrol evidence files are immutable and cannot be overwritten: "
            + ", ".join(str(path) for path in existing)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_paths["trajectory"]
    checkpoints_path = output_paths["checkpoints"]
    collisions_path = output_paths["collisions"]
    proximity_events_path = output_paths["proximity_events"]
    guard_events_path = output_paths["ttc_escape_guard_events"]
    summary_path = output_paths["summary"]
    _write_dataclass_csv(trajectory_path, trajectory, PatrolTrajectorySample)
    _write_dataclass_csv(checkpoints_path, checkpoints, PatrolCheckpointRecord)
    _write_dataclass_csv(collisions_path, collisions, PatrolCollisionRecord)
    _write_dataclass_csv(
        proximity_events_path,
        proximity_records,
        PatrolCriticalProximityRecord,
    )
    _write_dataclass_csv(
        guard_events_path,
        guard_event_records,
        PatrolTTCEscapeGuardEventRecord,
    )
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(summary_text)
    return {
        "trajectory": trajectory_path,
        "checkpoints": checkpoints_path,
        "collisions": collisions_path,
        "proximity_events": proximity_events_path,
        "ttc_escape_guard_events": guard_events_path,
        "summary": summary_path,
    }


def _validate_trajectory_audit(
    summary: PatrolRunSummary,
    trajectory: list[PatrolTrajectorySample],
) -> None:
    if len(trajectory) != summary.control_step_count:
        raise ValueError(
            "Trajectory must contain exactly one sample for every policy control step."
        )
    for expected_control_step, sample in enumerate(trajectory, start=1):
        if isinstance(sample.control_step, bool) or not isinstance(sample.control_step, int):
            raise ValueError("Trajectory control_step values must be integers.")
        if sample.control_step != expected_control_step:
            raise ValueError(
                "Trajectory control_step values must be the contiguous sequence from one "
                "through summary.control_step_count."
            )

    shield_steps = sum(sample.safety_shield_active for sample in trajectory)
    if summary.safety_shield_intervention_steps != shield_steps:
        raise ValueError(
            "summary.safety_shield_intervention_steps must match active trajectory samples."
        )
    expected_rate = (
        shield_steps / summary.control_step_count
        if summary.control_step_count > 0
        else 0.0
    )
    if not math.isclose(
        summary.safety_shield_intervention_rate,
        expected_rate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "summary.safety_shield_intervention_rate must equal active shield samples divided "
            "by control_step_count."
        )


def _validate_proximity_event_audit(
    trajectory: list[PatrolTrajectorySample],
    records: list[PatrolCriticalProximityRecord],
) -> None:
    """Require enough threshold events to cover trajectory critical rising edges."""

    previous_critical = False
    trajectory_rising_edges = 0
    for sample in trajectory:
        trajectory_rising_edges += int(sample.critical_proximity and not previous_critical)
        previous_critical = sample.critical_proximity
    threshold_events = sum(
        record.detection_source == "lidar_threshold" for record in records
    )
    if threshold_events < trajectory_rising_edges:
        raise ValueError(
            "Every critical-proximity rising edge in trajectory evidence requires a "
            "lidar_threshold proximity event record."
        )


def _validate_guard_event_audit(
    summary: PatrolRunSummary,
    trajectory: list[PatrolTrajectorySample],
    events: list[PatrolTTCEscapeGuardEventRecord],
    *,
    stream_was_provided: bool,
) -> None:
    summary_counts = (
        summary.ttc_escape_guard_intervention_steps,
        summary.ttc_escape_guard_activation_count,
        summary.ttc_escape_guard_recovery_count,
        summary.ttc_escape_guard_unrecoverable_count,
    )
    if not summary.ttc_escape_guard_enabled:
        if events:
            raise ValueError("A disabled TTC escape guard cannot carry guard event records.")
        if any(sample.ttc_escape_guard_active for sample in trajectory):
            raise ValueError("A disabled TTC escape guard cannot be active in trajectory evidence.")
        return
    if (summary.control_step_count > 0 or any(summary_counts)) and not stream_was_provided:
        raise ValueError("An enabled TTC escape guard requires its complete guard event stream.")

    previous_control_step = -1
    previous_simulation_time = -math.inf
    for expected_index, event in enumerate(events, start=1):
        if event.event_index != expected_index:
            raise ValueError("TTC escape guard event_index values must be contiguous from one.")
        if event.control_step < previous_control_step:
            raise ValueError("TTC escape guard control_step values cannot move backwards.")
        if event.simulation_time_s < previous_simulation_time:
            raise ValueError("TTC escape guard simulation_time_s values cannot move backwards.")
        if event.control_step > summary.control_step_count:
            raise ValueError(
                "TTC escape guard event control_step cannot exceed control_step_count."
            )
        previous_control_step = event.control_step
        previous_simulation_time = event.simulation_time_s

    navigation_events = [event for event in events if event.phase == "navigation"]
    if len(navigation_events) != len(trajectory):
        raise ValueError(
            "Every navigation trajectory sample must have exactly one guard event record."
        )
    for sample, event in zip(trajectory, navigation_events, strict=True):
        if (
            event.control_step != sample.control_step
            or event.active != sample.ttc_escape_guard_active
            or event.mode != sample.ttc_escape_guard_mode
            or not _optional_float_matches(
                event.target_linear_mps,
                sample.ttc_escape_guard_target_linear_mps,
            )
            or not _optional_float_matches(
                event.target_angular_radps,
                sample.ttc_escape_guard_target_angular_radps,
            )
        ):
            raise ValueError(
                "Navigation guard events must match the corresponding trajectory guard evidence."
            )

    intervention_steps = sum(event.counts_as_intervention_step for event in events)
    activation_count = 0
    recovery_count = 0
    unrecoverable_count = 0
    previous_active = False
    previous_unrecoverable = False
    for event in events:
        is_unrecoverable = event.mode == "unrecoverable"
        activation_count += int(not previous_active and event.active)
        recovery_count += int(previous_active and not event.active)
        unrecoverable_count += int(not previous_unrecoverable and is_unrecoverable)
        previous_active = event.active
        previous_unrecoverable = is_unrecoverable
    derived_counts = (
        intervention_steps,
        activation_count,
        recovery_count,
        unrecoverable_count,
    )
    if summary_counts != derived_counts:
        raise ValueError(
            "TTC escape guard aggregate counts must match the complete guard event stream: "
            f"summary={summary_counts}, derived={derived_counts}."
        )


def _optional_float_matches(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _minimum_lidar_diagnostic(
    trajectory: list[PatrolTrajectorySample],
) -> dict[str, int | float | None] | None:
    """Return the earliest trajectory sample with the lowest all-beam range."""

    if not trajectory:
        return None
    sample = min(trajectory, key=lambda item: item.minimum_lidar_range_m)
    return {
        "control_step": sample.control_step,
        "simulation_time_s": sample.simulation_time_s,
        "minimum_lidar_range_m": sample.minimum_lidar_range_m,
        "minimum_lidar_beam_index": sample.minimum_lidar_beam_index,
        "minimum_lidar_angle_rad": sample.minimum_lidar_angle_rad,
        "robot_position_x_m": sample.position_x_m,
        "robot_position_y_m": sample.position_y_m,
        "dynamic_obstacle_1_x_m": sample.dynamic_obstacle_1_x_m,
        "dynamic_obstacle_1_y_m": sample.dynamic_obstacle_1_y_m,
        "dynamic_obstacle_2_x_m": sample.dynamic_obstacle_2_x_m,
        "dynamic_obstacle_2_y_m": sample.dynamic_obstacle_2_y_m,
    }


def portable_model_path(path: Path, repository_root: Path) -> str:
    """Return a repository-relative model path when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


RecordT = TypeVar("RecordT")


def _write_dataclass_csv(
    path: Path,
    records: list[RecordT],
    record_type: type[RecordT],
) -> None:
    field_names = [field.name for field in fields(record_type)]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _contained_output_path(output_root: Path, filename: str) -> Path:
    candidate = (output_root / filename).resolve()
    try:
        candidate.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            f"Patrol output file must stay inside the output directory: {filename!r}."
        ) from error
    return candidate


def _inside_root(repository_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _infer_training_domain(path: Path) -> ModelTrainingDomain:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "webots" in parts or "ppo_webots" in name:
        return "webots"
    if "curriculum" in name or "kinematic" in name:
        return "kinematic"
    return "unknown"
