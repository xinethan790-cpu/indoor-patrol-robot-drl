"""Auditable LiDAR safety filtering for forward-only navigation actions.

The shield uses the shared policy scan convention: beam zero points forward
and positive beam indices rotate toward the robot's left.  Its first action
component is also interpreted using the project convention, where ``-1`` is
zero forward velocity and ``+1`` is maximum forward velocity.  Consequently,
the shield can stop the robot but never requests reverse motion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

SAFETY_SHIELD_CONFIG_ENV_PREFIX = "WEBOTS_SAFETY_SHIELD_"
SAFETY_SHIELD_CONFIG_ENV_FIELDS = {
    f"{SAFETY_SHIELD_CONFIG_ENV_PREFIX}{field.upper()}": field
    for field in (
        "front_stop_distance",
        "front_slowdown_distance",
        "side_stop_distance",
        "side_slowdown_distance",
        "front_half_angle",
        "side_inner_angle",
        "side_outer_angle",
        "clearance_quantile",
        "side_speed_floor",
        "side_turn_weight",
        "turn_gain",
        "turn_blend",
        "side_deadband",
        "intervention_epsilon",
    )
}


@dataclass(frozen=True)
class SafetyShieldConfig:
    """Geometric thresholds and blending gains for :class:`LidarSafetyShield`."""

    lidar_max_range: float = 3.5
    front_stop_distance: float = 0.18
    front_slowdown_distance: float = 0.60
    side_stop_distance: float = 0.12
    side_slowdown_distance: float = 0.32
    front_half_angle: float = math.radians(30.0)
    side_inner_angle: float = math.radians(35.0)
    side_outer_angle: float = math.radians(105.0)
    clearance_quantile: float = 0.15
    side_speed_floor: float = 0.35
    side_turn_weight: float = 0.55
    turn_gain: float = 0.90
    turn_blend: float = 0.90
    side_deadband: float = 0.03
    intervention_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Safety-shield configuration values must be finite.")
        positive = (
            self.lidar_max_range,
            self.front_stop_distance,
            self.front_slowdown_distance,
            self.side_stop_distance,
            self.side_slowdown_distance,
            self.front_half_angle,
            self.side_inner_angle,
            self.side_outer_angle,
            self.turn_gain,
            self.intervention_epsilon,
        )
        if min(positive) <= 0.0:
            raise ValueError("Safety-shield distances, angles, and gains must be positive.")
        if self.front_stop_distance >= self.front_slowdown_distance:
            raise ValueError("front_stop_distance must be below front_slowdown_distance.")
        if self.side_stop_distance >= self.side_slowdown_distance:
            raise ValueError("side_stop_distance must be below side_slowdown_distance.")
        if not 0.0 < self.front_half_angle < math.pi / 2.0:
            raise ValueError("front_half_angle must be between zero and pi/2.")
        if not self.front_half_angle <= self.side_inner_angle < self.side_outer_angle < math.pi:
            raise ValueError(
                "Side angles must begin outside the front sector and end before pi."
            )
        if not 0.0 <= self.clearance_quantile <= 0.5:
            raise ValueError("clearance_quantile must be between zero and 0.5.")
        for name, value in (
            ("side_speed_floor", self.side_speed_floor),
            ("side_turn_weight", self.side_turn_weight),
            ("turn_gain", self.turn_gain),
            ("turn_blend", self.turn_blend),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if self.side_deadband < 0.0:
            raise ValueError("side_deadband cannot be negative.")


def safety_shield_config_from_env(
    environ: Mapping[str, str],
    *,
    lidar_max_range: float = 3.5,
) -> SafetyShieldConfig:
    """Resolve an explicit, auditable shield configuration from environment values.

    The LiDAR range is hardware-derived by live controllers and deliberately has
    no environment override.  Every other dataclass field has one exact
    ``WEBOTS_SAFETY_SHIELD_*`` name.  Unknown names in that namespace are
    rejected so a typo cannot silently fall back to a less conservative default.
    Angle values use radians, matching :class:`SafetyShieldConfig`.
    """

    unsupported = sorted(
        name
        for name in environ
        if name.startswith(SAFETY_SHIELD_CONFIG_ENV_PREFIX)
        and name not in SAFETY_SHIELD_CONFIG_ENV_FIELDS
    )
    if unsupported:
        raise ValueError(
            "Unsupported safety-shield configuration environment variable(s): "
            f"{unsupported}."
        )
    overrides: dict[str, float] = {}
    for name, field in SAFETY_SHIELD_CONFIG_ENV_FIELDS.items():
        raw_value = environ.get(name)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite float.") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite float.")
        overrides[field] = value
    return SafetyShieldConfig(lidar_max_range=float(lidar_max_range), **overrides)


@dataclass(frozen=True)
class SafetyShieldResult:
    """One filtering decision, including enough diagnostics for later auditing."""

    enabled: bool
    active: bool
    requested_action: np.ndarray
    filtered_action: np.ndarray
    intervention: float
    front_clearance: float
    left_clearance: float
    right_clearance: float
    speed_fraction_limit: float
    normalized_speed_limit: float
    avoidance_direction: int

    def as_info(self) -> dict[str, float | bool | int]:
        """Return stable scalar fields suitable for Gym info and Monitor CSV."""

        return {
            "shield_enabled": self.enabled,
            "shield_active": self.active,
            "shield_intervention": self.intervention,
            "shield_front_clearance": self.front_clearance,
            "shield_left_clearance": self.left_clearance,
            "shield_right_clearance": self.right_clearance,
            "shield_speed_fraction_limit": self.speed_fraction_limit,
            "shield_normalized_speed_limit": self.normalized_speed_limit,
            "shield_avoidance_direction": self.avoidance_direction,
        }


class LidarSafetyShield:
    """Progressively slow and steer a forward-only action away from close obstacles.

    This is an application-layer safety heuristic, not a formal guarantee.  A
    low percentile is used within each sector so a compact obstacle influences
    the decision without allowing one isolated invalid beam to dominate it.
    """

    def __init__(self, config: SafetyShieldConfig | None = None) -> None:
        self.config = config or SafetyShieldConfig()

    @staticmethod
    def _smooth_fraction(value: float, stop: float, clear: float) -> float:
        scaled = float(np.clip((value - stop) / (clear - stop), 0.0, 1.0))
        return scaled * scaled * (3.0 - 2.0 * scaled)

    def _sector_clearances(self, raw_lidar: np.ndarray) -> tuple[float, float, float]:
        values = np.asarray(raw_lidar, dtype=np.float64).reshape(-1)
        if values.size < 8:
            raise ValueError("The safety shield requires at least eight LiDAR beams.")
        values = np.nan_to_num(
            values,
            nan=self.config.lidar_max_range,
            posinf=self.config.lidar_max_range,
            neginf=0.0,
        )
        values = np.clip(values, 0.0, self.config.lidar_max_range)
        angles = np.arange(values.size, dtype=np.float64) * (2.0 * math.pi / values.size)
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi
        front = values[np.abs(angles) <= self.config.front_half_angle]
        left = values[
            (angles >= self.config.side_inner_angle)
            & (angles <= self.config.side_outer_angle)
        ]
        right = values[
            (angles <= -self.config.side_inner_angle)
            & (angles >= -self.config.side_outer_angle)
        ]
        if min(front.size, left.size, right.size) == 0:
            raise ValueError("The LiDAR scan is too sparse for the configured shield sectors.")
        quantile = self.config.clearance_quantile
        return (
            float(np.quantile(front, quantile)),
            float(np.quantile(left, quantile)),
            float(np.quantile(right, quantile)),
        )

    def filter_action(
        self,
        raw_lidar: np.ndarray | list[float],
        action: np.ndarray | list[float],
        *,
        enabled: bool = True,
    ) -> SafetyShieldResult:
        """Filter a normalized action and return the action plus decision evidence."""

        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (2,):
            raise ValueError("action must have shape (2,).")
        requested = np.clip(requested, -1.0, 1.0).astype(np.float32)
        front, left, right = self._sector_clearances(np.asarray(raw_lidar))

        front_fraction = self._smooth_fraction(
            front,
            self.config.front_stop_distance,
            self.config.front_slowdown_distance,
        )
        side_fraction = self._smooth_fraction(
            min(left, right),
            self.config.side_stop_distance,
            self.config.side_slowdown_distance,
        )
        side_speed_limit = self.config.side_speed_floor + (
            1.0 - self.config.side_speed_floor
        ) * side_fraction
        speed_fraction_limit = min(front_fraction, side_speed_limit)
        normalized_speed_limit = 2.0 * speed_fraction_limit - 1.0

        side_difference = left - right
        if side_difference > self.config.side_deadband:
            avoidance_direction = 1
        elif side_difference < -self.config.side_deadband:
            avoidance_direction = -1
        elif abs(float(requested[1])) > self.config.intervention_epsilon:
            avoidance_direction = 1 if requested[1] > 0.0 else -1
        else:
            avoidance_direction = 1

        front_severity = 1.0 - front_fraction
        side_severity = 1.0 - side_fraction
        turn_severity = max(
            front_severity,
            self.config.side_turn_weight * side_severity,
        )
        turn_blend = self.config.turn_blend * turn_severity
        avoidance_turn = self.config.turn_gain * avoidance_direction

        candidate = requested.copy()
        candidate[0] = min(float(candidate[0]), normalized_speed_limit)
        candidate[1] = (1.0 - turn_blend) * float(candidate[1]) + (
            turn_blend * avoidance_turn
        )
        candidate = np.clip(candidate, -1.0, 1.0).astype(np.float32)
        filtered = candidate if enabled else requested.copy()
        intervention = float(np.linalg.norm(filtered - requested))
        active = bool(enabled and intervention > self.config.intervention_epsilon)
        return SafetyShieldResult(
            enabled=bool(enabled),
            active=active,
            requested_action=requested.copy(),
            filtered_action=filtered,
            intervention=intervention,
            front_clearance=front,
            left_clearance=left,
            right_clearance=right,
            speed_fraction_limit=speed_fraction_limit,
            normalized_speed_limit=normalized_speed_limit,
            avoidance_direction=avoidance_direction,
        )
