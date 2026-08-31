"""Auditable, simulator-independent patrol action execution pipeline.

The pipeline keeps the application controller's command order explicit:
validate the policy action, apply the LiDAR safety shield, map the normalized
action to body velocities, rate-limit those velocities, and finally map them
to wheel commands.  It also reports the body velocity that is physically
represented by the possibly saturated wheel commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from indoor_patrol_rl.core import (
    DifferentialDriveMapper,
    LidarSafetyShield,
    SafetyShieldResult,
    VelocityRateLimiter,
)


class PatrolControlMode(StrEnum):
    """Safety operating mode for one patrol action pipeline.

    ``PINNED_COMPONENT`` forces the evidence-bound model/shield command path;
    it does not claim that the surrounding four-point patrol scenario is
    qualified.  The only supported shield-disabled path is deliberately named
    as an experimental mode so it cannot be mistaken for the pinned path.
    """

    PINNED_COMPONENT = "pinned_component"
    EXPERIMENTAL_SHIELD_DISABLED = "experimental_shield_disabled"

    @property
    def shield_enabled(self) -> bool:
        """Return the fixed shield state implied by this operating mode."""

        return self is PatrolControlMode.PINNED_COMPONENT


@dataclass(frozen=True, slots=True)
class PatrolActionResult:
    """All command stages and safety evidence for one policy action."""

    policy_action: np.ndarray
    shielded_action: np.ndarray
    shield_result: SafetyShieldResult
    requested_linear_mps: float
    requested_angular_radps: float
    rate_limited_linear_mps: float
    rate_limited_angular_radps: float
    executable_linear_mps: float
    executable_angular_radps: float
    left_wheel_radps: float
    right_wheel_radps: float

    @property
    def previous_action(self) -> np.ndarray:
        """Return the shielded action to feed into the next observation."""

        return self.shielded_action.copy()

    @property
    def shield_diagnostics(self) -> dict[str, float | bool | int]:
        """Return stable scalar shield diagnostics for logs and dashboards."""

        return self.shield_result.as_info()


@dataclass(frozen=True, slots=True, kw_only=True)
class PatrolActionPipeline:
    """Execute normalized patrol actions through the fixed application order."""

    safety_shield: LidarSafetyShield
    mapper: DifferentialDriveMapper
    rate_limiter: VelocityRateLimiter
    mode: PatrolControlMode = PatrolControlMode.PINNED_COMPONENT

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PatrolControlMode):
            raise TypeError("mode must be a PatrolControlMode value.")

    def execute(
        self,
        *,
        raw_lidar: np.ndarray | list[float],
        policy_action: np.ndarray | list[float],
        current_linear_mps: float,
        current_angular_radps: float,
    ) -> PatrolActionResult:
        """Execute one action and return every auditable command stage.

        ``policy_action`` must be a finite two-element vector.  Finite values
        outside the policy bounds are clipped before they reach the shield.
        Current velocities are the previously executable body velocities, not
        unsaturated requests.
        """

        action = _validated_policy_action(policy_action)
        current_linear, current_angular = _finite_pair(
            current_linear_mps,
            current_angular_radps,
            label="current body velocity",
        )

        shield_result = self.safety_shield.filter_action(
            raw_lidar,
            action,
            enabled=self.mode.shield_enabled,
        )
        _validate_shield_contract(
            shield_result,
            policy_action=action,
            expected_enabled=self.mode.shield_enabled,
        )
        shielded_action = np.asarray(
            shield_result.filtered_action,
            dtype=np.float32,
        ).copy()

        requested_linear, requested_angular = _finite_pair(
            *self.mapper.normalized_to_velocity(shielded_action),
            label="mapped velocity request",
        )
        rate_limited_linear, rate_limited_angular = _finite_pair(
            *self.rate_limiter.limit(
                requested_linear,
                requested_angular,
                current_linear,
                current_angular,
            ),
            label="rate-limited velocity request",
        )
        left_wheel, right_wheel = _finite_pair(
            *self.mapper.velocity_to_wheels(
                rate_limited_linear,
                rate_limited_angular,
            ),
            label="wheel velocity command",
        )

        executable_linear = 0.5 * self.mapper.wheel_radius * (
            left_wheel + right_wheel
        )
        executable_angular = (
            self.mapper.wheel_radius
            * (right_wheel - left_wheel)
            / self.mapper.axle_length
        )
        executable_linear, executable_angular = _finite_pair(
            executable_linear,
            executable_angular,
            label="executable body velocity",
        )

        return PatrolActionResult(
            policy_action=action.copy(),
            shielded_action=shielded_action,
            shield_result=shield_result,
            requested_linear_mps=requested_linear,
            requested_angular_radps=requested_angular,
            rate_limited_linear_mps=rate_limited_linear,
            rate_limited_angular_radps=rate_limited_angular,
            executable_linear_mps=executable_linear,
            executable_angular_radps=executable_angular,
            left_wheel_radps=left_wheel,
            right_wheel_radps=right_wheel,
        )


def _validated_policy_action(
    policy_action: np.ndarray | list[float],
) -> np.ndarray:
    try:
        action = np.asarray(policy_action, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("policy_action must contain two finite numeric values.") from error
    if action.shape != (2,):
        raise ValueError("policy_action must have shape (2,).")
    if not np.isfinite(action).all():
        raise ValueError("policy_action values must be finite.")
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=True)


def _finite_pair(first: float, second: float, *, label: str) -> tuple[float, float]:
    try:
        values = (float(first), float(second))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} values must be finite numbers.") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} values must be finite numbers.")
    return values


def _validate_shield_contract(
    result: SafetyShieldResult,
    *,
    policy_action: np.ndarray,
    expected_enabled: bool,
) -> None:
    if not isinstance(result, SafetyShieldResult):
        raise TypeError("safety shield must return SafetyShieldResult.")
    requested = np.asarray(result.requested_action, dtype=np.float32)
    filtered = np.asarray(result.filtered_action, dtype=np.float32)
    if requested.shape != (2,) or filtered.shape != (2,):
        raise RuntimeError("Safety shield returned an action with an invalid shape.")
    if not np.isfinite(requested).all() or not np.isfinite(filtered).all():
        raise RuntimeError("Safety shield returned a non-finite action.")
    if not np.array_equal(requested, policy_action):
        raise RuntimeError("Safety shield diagnostics do not match the policy action.")
    if np.any(filtered < -1.0) or np.any(filtered > 1.0):
        raise RuntimeError("Safety shield returned an action outside [-1, 1].")
    if result.enabled is not expected_enabled:
        raise RuntimeError("Safety shield did not honor the selected patrol control mode.")
