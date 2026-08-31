"""Patrol-specific, unqualified TTC escape guard with hysteresis.

This application protection is deliberately separate from the qualified R2a
model-plus-shield deployment.  It does not alter, extend, or inherit that
system's identity or validation claims.  The guard temporarily takes over
velocity targets when an omnidirectional proximity assessment predicts an
imminent encounter, and latches an unrecoverable result if escape does not
complete within its bounded intervention window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from indoor_patrol_rl.application.proximity import CriticalProximityAssessment

PATROL_TTC_ESCAPE_GUARD_SCHEMA_VERSION = 1
PATROL_TTC_ESCAPE_GUARD_ID = "patrol-ttc-escape-guard-v2"
UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE = "unqualified_patrol_application_escape_guard"

PatrolTTCEscapeMode = Literal[
    "inactive",
    "reverse",
    "forward",
    "rotate",
    "unrecoverable",
]
PatrolTTCEscapeActivationReason = Literal[
    "distance_threshold",
    "ttc_threshold",
    "distance_and_ttc",
]


@dataclass(frozen=True, slots=True)
class PatrolTTCEscapeGuardConfig:
    """Thresholds and bounded commands for :class:`PatrolTTCEscapeGuard`."""

    ttc_activation_s: float = 2.5
    distance_activation_m: float = 0.30
    ttc_activation_max_distance_m: float = 0.45
    release_distance_m: float = 0.42
    release_confirmation_s: float = 0.20
    rear_escape_front_clearance_m: float = 0.40
    front_threat_rear_clearance_m: float = 0.40
    reverse_linear_mps: float = -0.20
    forward_linear_mps: float = 0.20
    rotation_angular_radps: float = 0.80
    forward_watchdog_s: float = 2.00
    forward_maximum_regression_m: float = 0.03
    maximum_active_duration_s: float = 5.5

    def __post_init__(self) -> None:
        for name in (
            "ttc_activation_s",
            "distance_activation_m",
            "ttc_activation_max_distance_m",
            "release_distance_m",
            "release_confirmation_s",
            "rear_escape_front_clearance_m",
            "front_threat_rear_clearance_m",
            "reverse_linear_mps",
            "forward_linear_mps",
            "rotation_angular_radps",
            "forward_watchdog_s",
            "forward_maximum_regression_m",
            "maximum_active_duration_s",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), label=name))
        if self.ttc_activation_s <= 0.0:
            raise ValueError("ttc_activation_s must be positive.")
        if self.distance_activation_m <= 0.0:
            raise ValueError("distance_activation_m must be positive.")
        if self.ttc_activation_max_distance_m <= self.distance_activation_m:
            raise ValueError("ttc_activation_max_distance_m must exceed distance_activation_m.")
        if self.release_distance_m <= self.distance_activation_m:
            raise ValueError("release_distance_m must exceed distance_activation_m.")
        if self.release_confirmation_s <= 0.0:
            raise ValueError("release_confirmation_s must be positive.")
        if self.rear_escape_front_clearance_m <= 0.0:
            raise ValueError("rear_escape_front_clearance_m must be positive.")
        if self.front_threat_rear_clearance_m <= 0.0:
            raise ValueError("front_threat_rear_clearance_m must be positive.")
        if self.reverse_linear_mps >= 0.0:
            raise ValueError("reverse_linear_mps must be negative.")
        if self.forward_linear_mps <= 0.0:
            raise ValueError("forward_linear_mps must be positive.")
        if self.rotation_angular_radps <= 0.0:
            raise ValueError("rotation_angular_radps must be positive.")
        if self.forward_watchdog_s <= 0.0:
            raise ValueError("forward_watchdog_s must be positive.")
        if self.forward_maximum_regression_m <= 0.0:
            raise ValueError("forward_maximum_regression_m must be positive.")
        if self.maximum_active_duration_s <= 0.0:
            raise ValueError("maximum_active_duration_s must be positive.")


@dataclass(frozen=True, slots=True)
class PatrolTTCEscapeDecision:
    """One auditable guard decision and its target chassis velocities."""

    active: bool
    mode: PatrolTTCEscapeMode
    target_linear_mps: float
    target_angular_radps: float
    activation_reason: PatrolTTCEscapeActivationReason | None
    active_duration_s: float
    guard_id: str = PATROL_TTC_ESCAPE_GUARD_ID
    schema_version: int = PATROL_TTC_ESCAPE_GUARD_SCHEMA_VERSION
    evidence_scope: str = UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise TypeError("active must be a bool.")
        if self.mode not in {
            "inactive",
            "reverse",
            "forward",
            "rotate",
            "unrecoverable",
        }:
            raise ValueError("mode is not a supported patrol TTC escape mode.")
        if self.activation_reason not in {
            None,
            "distance_threshold",
            "ttc_threshold",
            "distance_and_ttc",
        }:
            raise ValueError("activation_reason is not supported.")
        linear = _finite_float(self.target_linear_mps, label="target_linear_mps")
        angular = _finite_float(self.target_angular_radps, label="target_angular_radps")
        duration = _finite_float(self.active_duration_s, label="active_duration_s")
        if duration < 0.0:
            raise ValueError("active_duration_s cannot be negative.")
        if self.guard_id != PATROL_TTC_ESCAPE_GUARD_ID:
            raise ValueError("guard_id must identify the patrol TTC escape guard.")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PATROL_TTC_ESCAPE_GUARD_SCHEMA_VERSION
        ):
            raise ValueError("schema_version is unsupported.")
        if self.evidence_scope != UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE:
            raise ValueError("evidence_scope must retain the unqualified application label.")

        if self.mode == "inactive":
            if self.active or linear != 0.0 or angular != 0.0:
                raise ValueError("Inactive decisions must be inactive with zero velocities.")
            if self.activation_reason is not None or duration != 0.0:
                raise ValueError("Inactive decisions cannot retain activation state.")
        else:
            if not self.active or self.activation_reason is None:
                raise ValueError("Escape decisions must be active and retain their trigger.")
            if self.mode == "reverse" and not (linear < 0.0 and angular == 0.0):
                raise ValueError("Reverse mode requires negative linear and zero angular velocity.")
            if self.mode == "forward" and not (linear > 0.0 and angular == 0.0):
                raise ValueError("Forward mode requires positive linear and zero angular velocity.")
            if self.mode == "rotate" and not (linear == 0.0 and angular != 0.0):
                raise ValueError("Rotate mode requires zero linear and nonzero angular velocity.")
            if self.mode == "unrecoverable" and (linear != 0.0 or angular != 0.0):
                raise ValueError("Unrecoverable mode must command zero velocities.")

        object.__setattr__(self, "target_linear_mps", linear)
        object.__setattr__(self, "target_angular_radps", angular)
        object.__setattr__(self, "active_duration_s", duration)

    @property
    def unrecoverable(self) -> bool:
        """Whether the bounded escape interval expired without safe release."""

        return self.mode == "unrecoverable"


class PatrolTTCEscapeGuard:
    """Stateful patrol escape guard with activation/release hysteresis."""

    def __init__(self, config: PatrolTTCEscapeGuardConfig | None = None) -> None:
        if config is not None and not isinstance(config, PatrolTTCEscapeGuardConfig):
            raise TypeError("config must be a PatrolTTCEscapeGuardConfig value.")
        self.config = config or PatrolTTCEscapeGuardConfig()
        self._active = False
        self._unrecoverable = False
        self._active_duration_s = 0.0
        self._activation_reason: PatrolTTCEscapeActivationReason | None = None
        self._release_confirmation_elapsed_s = 0.0
        self._forward_watchdog_elapsed_s = 0.0
        self._forward_baseline_distance_m: float | None = None
        self._force_rotate = False
        self._latched_rotation_angular_radps: float | None = None
        self._latched_translation_mode: Literal["forward", "reverse"] | None = None

    @property
    def active(self) -> bool:
        """Whether patrol commands are currently overridden or terminally latched."""

        return self._active

    @property
    def unrecoverable(self) -> bool:
        """Whether the maximum escape duration expired."""

        return self._unrecoverable

    def reset(self) -> None:
        """Clear intervention and timeout state before a new patrol run."""

        self._active = False
        self._unrecoverable = False
        self._active_duration_s = 0.0
        self._activation_reason = None
        self._release_confirmation_elapsed_s = 0.0
        self._forward_watchdog_elapsed_s = 0.0
        self._forward_baseline_distance_m = None
        self._force_rotate = False
        self._latched_rotation_angular_radps = None
        self._latched_translation_mode = None

    def update(
        self,
        assessment: CriticalProximityAssessment,
        *,
        front_clearance_m: float,
        rear_clearance_m: float,
        dt_s: float,
    ) -> PatrolTTCEscapeDecision:
        """Advance the guard and return one bounded target-velocity decision."""

        if not isinstance(assessment, CriticalProximityAssessment):
            raise TypeError("assessment must be a CriticalProximityAssessment value.")
        front_clearance = _finite_float(front_clearance_m, label="front_clearance_m")
        if front_clearance < 0.0:
            raise ValueError("front_clearance_m cannot be negative.")
        rear_clearance = _finite_float(rear_clearance_m, label="rear_clearance_m")
        if rear_clearance < 0.0:
            raise ValueError("rear_clearance_m cannot be negative.")
        dt = _finite_float(dt_s, label="dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive.")

        if self._unrecoverable:
            return self._unrecoverable_decision()

        if not self._active:
            activation_reason = self._activation_trigger(assessment)
            if activation_reason is None:
                return self._inactive_decision()
            self._active = True
            self._active_duration_s = dt
            self._activation_reason = activation_reason
            self._release_confirmation_elapsed_s = 0.0
            self._forward_watchdog_elapsed_s = 0.0
            self._forward_baseline_distance_m = None
            self._force_rotate = False
            self._latched_rotation_angular_radps = None
            self._latched_translation_mode = None
            if self._active_duration_s >= self.config.maximum_active_duration_s:
                self._active_duration_s = self.config.maximum_active_duration_s
                self._unrecoverable = True
                return self._unrecoverable_decision()
        else:
            release_clearance_m = self._release_clearance_m(
                assessment,
                front_clearance_m=front_clearance,
                rear_clearance_m=rear_clearance,
            )
            if release_clearance_m >= self.config.release_distance_m and not assessment.approaching:
                self._release_confirmation_elapsed_s += dt
                if self._release_confirmation_elapsed_s >= self.config.release_confirmation_s:
                    self.reset()
                    return self._inactive_decision()
            else:
                self._release_confirmation_elapsed_s = 0.0
            next_duration = self._active_duration_s + dt
            if next_duration >= self.config.maximum_active_duration_s:
                self._active_duration_s = self.config.maximum_active_duration_s
                self._unrecoverable = True
                return self._unrecoverable_decision()
            self._active_duration_s = next_duration

        return self._escape_decision(
            assessment,
            front_clearance,
            rear_clearance,
            dt_s=dt,
        )

    def _activation_trigger(
        self,
        assessment: CriticalProximityAssessment,
    ) -> PatrolTTCEscapeActivationReason | None:
        distance_triggered = assessment.minimum_distance_m <= self.config.distance_activation_m
        ttc_triggered = (
            assessment.approaching
            and assessment.ttc_s is not None
            and assessment.ttc_s <= self.config.ttc_activation_s
            and assessment.minimum_distance_m <= self.config.ttc_activation_max_distance_m
        )
        if distance_triggered and ttc_triggered:
            return "distance_and_ttc"
        if distance_triggered:
            return "distance_threshold"
        if ttc_triggered:
            return "ttc_threshold"
        return None

    def _escape_decision(
        self,
        assessment: CriticalProximityAssessment,
        front_clearance_m: float,
        rear_clearance_m: float,
        *,
        dt_s: float,
    ) -> PatrolTTCEscapeDecision:
        assert self._activation_reason is not None
        angle = assessment.minimum_angle_rad
        if self._force_rotate:
            mode: PatrolTTCEscapeMode = "rotate"
            target_linear = 0.0
            target_angular = self._latched_rotation_away_from(angle)
        elif self._latched_translation_mode == "reverse":
            if rear_clearance_m >= self.config.front_threat_rear_clearance_m:
                mode = "reverse"
                target_linear = self.config.reverse_linear_mps
                target_angular = 0.0
            else:
                self._force_rotate = True
                mode = "rotate"
                target_linear = 0.0
                target_angular = self._latched_rotation_away_from(angle)
        elif self._latched_translation_mode == "forward":
            if front_clearance_m >= self.config.rear_escape_front_clearance_m:
                mode = "forward"
                target_linear = self.config.forward_linear_mps
                target_angular = 0.0
            else:
                self._force_rotate = True
                mode = "rotate"
                target_linear = 0.0
                target_angular = self._latched_rotation_away_from(angle)
        elif (
            abs(angle) <= math.pi / 2.0
            and rear_clearance_m >= self.config.front_threat_rear_clearance_m
        ):
            self._latched_translation_mode = "reverse"
            mode = "reverse"
            target_linear = self.config.reverse_linear_mps
            target_angular = 0.0
        elif abs(angle) <= math.pi / 2.0:
            self._force_rotate = True
            mode = "rotate"
            target_linear = 0.0
            target_angular = self._latched_rotation_away_from(angle)
        elif front_clearance_m >= self.config.rear_escape_front_clearance_m:
            self._latched_translation_mode = "forward"
            mode = "forward"
            target_linear = self.config.forward_linear_mps
            target_angular = 0.0
        else:
            self._force_rotate = True
            mode = "rotate"
            target_linear = 0.0
            target_angular = self._latched_rotation_away_from(angle)

        if mode == "forward":
            threat_distance = (
                assessment.tracked_distance_m
                if assessment.tracked_distance_m is not None
                else assessment.minimum_distance_m
            )
            if self._forward_baseline_distance_m is None:
                self._forward_baseline_distance_m = threat_distance
                self._forward_watchdog_elapsed_s = 0.0
            else:
                self._forward_watchdog_elapsed_s += dt_s
                if self._forward_watchdog_elapsed_s >= self.config.forward_watchdog_s:
                    regression = self._forward_baseline_distance_m - threat_distance
                    if (
                        regression > self.config.forward_maximum_regression_m
                        or assessment.approaching
                    ):
                        self._force_rotate = True
                        mode = "rotate"
                        target_linear = 0.0
                        target_angular = self._latched_rotation_away_from(angle)
                    else:
                        self._forward_baseline_distance_m = threat_distance
                        self._forward_watchdog_elapsed_s = 0.0
        elif not self._force_rotate:
            self._forward_baseline_distance_m = None
            self._forward_watchdog_elapsed_s = 0.0
        return PatrolTTCEscapeDecision(
            active=True,
            mode=mode,
            target_linear_mps=target_linear,
            target_angular_radps=target_angular,
            activation_reason=self._activation_reason,
            active_duration_s=self._active_duration_s,
        )

    def _release_clearance_m(
        self,
        assessment: CriticalProximityAssessment,
        *,
        front_clearance_m: float,
        rear_clearance_m: float,
    ) -> float:
        if not self._force_rotate:
            if self._latched_translation_mode == "forward":
                return front_clearance_m
            if self._latched_translation_mode == "reverse":
                return rear_clearance_m
        return assessment.minimum_distance_m

    def _rotation_away_from(self, threat_angle_rad: float) -> float:
        threat_sign = 1.0 if threat_angle_rad >= 0.0 else -1.0
        return -threat_sign * self.config.rotation_angular_radps

    def _latched_rotation_away_from(self, threat_angle_rad: float) -> float:
        if self._latched_rotation_angular_radps is None:
            self._latched_rotation_angular_radps = self._rotation_away_from(threat_angle_rad)
        return self._latched_rotation_angular_radps

    @staticmethod
    def _inactive_decision() -> PatrolTTCEscapeDecision:
        return PatrolTTCEscapeDecision(
            active=False,
            mode="inactive",
            target_linear_mps=0.0,
            target_angular_radps=0.0,
            activation_reason=None,
            active_duration_s=0.0,
        )

    def _unrecoverable_decision(self) -> PatrolTTCEscapeDecision:
        assert self._activation_reason is not None
        return PatrolTTCEscapeDecision(
            active=True,
            mode="unrecoverable",
            target_linear_mps=0.0,
            target_angular_radps=0.0,
            activation_reason=self._activation_reason,
            active_duration_s=self._active_duration_s,
        )


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number, not a boolean.")
    if not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result
