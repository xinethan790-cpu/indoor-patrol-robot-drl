"""Build the shared 79-dimensional patrol policy observation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces


@dataclass
class ObservationBuilder:
    lidar_beams: int = 36
    lidar_max_range: float = 3.5
    goal_distance_scale: float = 2.0
    lidar_delta_scale: float = 0.25
    _previous_lidar: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lidar_beams < 8:
            raise ValueError("lidar_beams must be at least eight.")
        scales = {
            "lidar_max_range": self.lidar_max_range,
            "goal_distance_scale": self.goal_distance_scale,
            "lidar_delta_scale": self.lidar_delta_scale,
        }
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in scales.values()):
            raise ValueError("Observation scales must be positive.")
        self._previous_lidar = np.full(self.lidar_beams, self.lidar_max_range, dtype=np.float64)

    @property
    def space(self) -> spaces.Box:
        low = np.concatenate(
            [
                np.zeros(self.lidar_beams, dtype=np.float32),
                -np.ones(self.lidar_beams, dtype=np.float32),
                np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            ]
        )
        high = np.ones(2 * self.lidar_beams + 7, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, raw_lidar: np.ndarray | list[float]) -> None:
        self._previous_lidar = self.sectorize(raw_lidar)

    def sectorize(self, raw_lidar: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(raw_lidar, dtype=np.float64).reshape(-1)
        if values.size < self.lidar_beams:
            raise ValueError("The raw LiDAR scan has fewer samples than requested sectors.")
        values = np.nan_to_num(
            values,
            nan=self.lidar_max_range,
            posinf=self.lidar_max_range,
            neginf=0.0,
        )
        values = np.clip(values, 0.0, self.lidar_max_range)
        sectors = [float(chunk.min()) for chunk in np.array_split(values, self.lidar_beams)]
        return np.asarray(sectors, dtype=np.float64)

    def build(
        self,
        *,
        raw_lidar: np.ndarray | list[float],
        goal_distance: float,
        goal_bearing: float,
        linear_speed: float,
        max_linear_speed: float,
        angular_speed: float,
        max_angular_speed: float,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        scalar_fields = {
            "goal_distance": goal_distance,
            "goal_bearing": goal_bearing,
            "linear_speed": linear_speed,
            "max_linear_speed": max_linear_speed,
            "angular_speed": angular_speed,
            "max_angular_speed": max_angular_speed,
        }
        validated_scalars: dict[str, float] = {}
        for name, raw_value in scalar_fields.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Observation field {name} must be numeric.") from error
            if not math.isfinite(value):
                raise ValueError(f"Observation field {name} must be finite; got {value!r}.")
            validated_scalars[name] = value
        if validated_scalars["goal_distance"] < 0.0:
            raise ValueError("Observation field goal_distance cannot be negative.")
        if (
            validated_scalars["max_linear_speed"] <= 0.0
            or validated_scalars["max_angular_speed"] <= 0.0
        ):
            raise ValueError("Observation maximum speeds must be positive.")

        lidar = self.sectorize(raw_lidar)
        lidar_delta = np.clip(
            (lidar - self._previous_lidar) / self.lidar_delta_scale,
            -1.0,
            1.0,
        )
        action = np.asarray(previous_action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError("previous_action must have shape (2,).")
        if not np.isfinite(action).all():
            raise ValueError("Observation field previous_action must contain finite values.")
        action = np.clip(action, -1.0, 1.0)

        tail = np.array(
            [
                min(
                    max(
                        validated_scalars["goal_distance"] / self.goal_distance_scale,
                        0.0,
                    ),
                    1.0,
                ),
                np.sin(validated_scalars["goal_bearing"]),
                np.cos(validated_scalars["goal_bearing"]),
                np.clip(
                    validated_scalars["linear_speed"]
                    / validated_scalars["max_linear_speed"],
                    0.0,
                    1.0,
                ),
                np.clip(
                    validated_scalars["angular_speed"]
                    / validated_scalars["max_angular_speed"],
                    -1.0,
                    1.0,
                ),
                action[0],
                action[1],
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            [
                (lidar / self.lidar_max_range).astype(np.float32),
                lidar_delta.astype(np.float32),
                tail,
            ]
        )
        observation = np.clip(observation, self.space.low, self.space.high).astype(np.float32)
        if not np.isfinite(observation).all():
            indices = np.flatnonzero(~np.isfinite(observation)).tolist()
            raise RuntimeError(f"Built patrol observation is non-finite at indices {indices}.")
        self._previous_lidar = lidar
        return observation
