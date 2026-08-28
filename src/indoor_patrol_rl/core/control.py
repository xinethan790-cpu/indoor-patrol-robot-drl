"""Normalized continuous actions and differential-drive conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityRateLimiter:
    """Apply shared acceleration limits at the environment control boundary."""

    max_linear_acceleration: float
    max_angular_acceleration: float
    control_dt: float

    def __post_init__(self) -> None:
        if min(
            self.max_linear_acceleration,
            self.max_angular_acceleration,
            self.control_dt,
        ) <= 0.0:
            raise ValueError("Velocity-rate limits and control_dt must be positive.")

    def limit(
        self,
        target_linear: float,
        target_angular: float,
        current_linear: float,
        current_angular: float,
    ) -> tuple[float, float]:
        linear_delta = self.max_linear_acceleration * self.control_dt
        angular_delta = self.max_angular_acceleration * self.control_dt
        linear = current_linear + float(
            np.clip(target_linear - current_linear, -linear_delta, linear_delta)
        )
        angular = current_angular + float(
            np.clip(target_angular - current_angular, -angular_delta, angular_delta)
        )
        return linear, angular


@dataclass(frozen=True)
class DifferentialDriveMapper:
    max_linear_speed: float
    max_angular_speed: float
    wheel_radius: float
    axle_length: float
    max_wheel_speed: float

    def __post_init__(self) -> None:
        if min(
            self.max_linear_speed,
            self.max_angular_speed,
            self.wheel_radius,
            self.axle_length,
            self.max_wheel_speed,
        ) <= 0.0:
            raise ValueError("All differential-drive parameters must be positive.")

    def normalized_to_velocity(self, action: np.ndarray) -> tuple[float, float]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,).")
        linear = float((action[0] + 1.0) * 0.5 * self.max_linear_speed)
        angular = float(action[1] * self.max_angular_speed)
        return linear, angular

    def velocity_to_wheels(self, linear: float, angular: float) -> tuple[float, float]:
        left = (linear - 0.5 * self.axle_length * angular) / self.wheel_radius
        right = (linear + 0.5 * self.axle_length * angular) / self.wheel_radius
        peak_speed = max(abs(left), abs(right))
        if peak_speed > self.max_wheel_speed:
            # Preserve the requested curvature when either wheel saturates.
            scale = self.max_wheel_speed / peak_speed
            left *= scale
            right *= scale
        return float(left), float(right)

    def normalized_to_wheels(self, action: np.ndarray) -> tuple[float, float]:
        return self.velocity_to_wheels(*self.normalized_to_velocity(action))
