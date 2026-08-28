"""Deterministic Dynamic Window Approach baseline for local obstacle avoidance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DWAConfig:
    """Physical limits and trajectory-scoring weights for a DWA controller."""

    lidar_beams: int = 36
    lidar_max_range: float = 3.5
    goal_distance_scale: float = 4.0
    max_linear_speed: float = 0.22
    max_angular_speed: float = 1.5
    max_linear_acceleration: float = 0.50
    max_angular_acceleration: float = 3.0
    wheel_radius: float = 0.033
    axle_length: float = 0.16
    max_wheel_speed: float = 6.0
    control_dt: float = 0.128
    prediction_horizon: float = 1.5
    simulation_dt: float = 0.1
    linear_samples: int = 7
    angular_samples: int = 15
    robot_radius: float = 0.11
    collision_margin: float = 0.02
    safety_margin: float = 0.15
    goal_weight: float = 1.5
    clearance_weight: float = 2.5
    speed_weight: float = 0.25
    smoothness_weight: float = 0.15

    def __post_init__(self) -> None:
        positive_values = (
            self.lidar_max_range,
            self.goal_distance_scale,
            self.max_linear_speed,
            self.max_angular_speed,
            self.max_linear_acceleration,
            self.max_angular_acceleration,
            self.wheel_radius,
            self.axle_length,
            self.max_wheel_speed,
            self.control_dt,
            self.prediction_horizon,
            self.simulation_dt,
            self.robot_radius,
            self.collision_margin,
            self.safety_margin,
        )
        if self.lidar_beams < 8 or min(positive_values) <= 0.0:
            raise ValueError("DWA dimensions and physical parameters must be positive.")
        if self.linear_samples < 2 or self.angular_samples < 3:
            raise ValueError("DWA needs at least 2 linear and 3 angular samples.")
        if min(
            self.goal_weight,
            self.clearance_weight,
            self.speed_weight,
            self.smoothness_weight,
        ) < 0.0:
            raise ValueError("DWA scoring weights cannot be negative.")


@dataclass(frozen=True)
class DWAResult:
    """Selected command and diagnostics used by evaluation and visualisation."""

    action: np.ndarray
    linear_velocity: float
    angular_velocity: float
    cost: float
    minimum_clearance: float
    evaluated_trajectories: int
    feasible_trajectories: int


class DWAPlanner:
    """Sample dynamically reachable commands and score their local trajectories.

    The planner consumes the same 79-value observation as PPO. LiDAR beam zero
    points forward and increasing beam indices turn counter-clockwise (left).
    """

    def __init__(self, config: DWAConfig | None = None) -> None:
        self.config = config or DWAConfig()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return a normalized ``[linear, angular]`` action like an SB3 policy."""

        return self.plan(observation).action

    def plan(self, observation: np.ndarray) -> DWAResult:
        observation = np.asarray(observation, dtype=np.float32)
        expected_size = 2 * self.config.lidar_beams + 7
        if observation.shape != (expected_size,):
            raise ValueError(f"Expected observation shape ({expected_size},).")
        if not np.isfinite(observation).all():
            raise ValueError("DWA observation must contain only finite values.")

        scan, goal, current = self._decode_observation(observation)
        obstacle_points = self._obstacle_points(scan)
        linear_values, angular_values = self._dynamic_window(*current)

        best: tuple[float, float, float, float] | None = None
        evaluated = 0
        feasible = 0
        for linear in linear_values:
            for angular in angular_values:
                evaluated += 1
                if not self._wheel_feasible(float(linear), float(angular)):
                    continue
                trajectory = self._simulate(float(linear), float(angular))
                clearance = self._minimum_clearance(trajectory, obstacle_points)
                if clearance <= self.config.collision_margin:
                    continue
                feasible += 1
                cost = self._trajectory_cost(
                    trajectory=trajectory,
                    goal=goal,
                    clearance=clearance,
                    linear=float(linear),
                    angular=float(angular),
                    current=current,
                )
                candidate = (cost, -float(linear), abs(float(angular)), float(angular))
                if best is None or candidate < (best[0], -best[1], abs(best[2]), best[2]):
                    best = (cost, float(linear), float(angular), clearance)

        if best is None:
            return self._emergency_command(goal, current, evaluated)

        cost, linear, angular, clearance = best
        return DWAResult(
            action=self._normalize_action(linear, angular),
            linear_velocity=linear,
            angular_velocity=angular,
            cost=cost,
            minimum_clearance=clearance,
            evaluated_trajectories=evaluated,
            feasible_trajectories=feasible,
        )

    def _decode_observation(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        beam_count = self.config.lidar_beams
        scan = np.clip(observation[:beam_count], 0.0, 1.0) * self.config.lidar_max_range
        offset = 2 * beam_count
        distance = float(np.clip(observation[offset], 0.0, 1.0))
        distance *= self.config.goal_distance_scale
        bearing = math.atan2(float(observation[offset + 1]), float(observation[offset + 2]))
        goal = np.array(
            [distance * math.cos(bearing), distance * math.sin(bearing)],
            dtype=np.float64,
        )
        linear = float(np.clip(observation[offset + 3], 0.0, 1.0))
        linear *= self.config.max_linear_speed
        angular = float(np.clip(observation[offset + 4], -1.0, 1.0))
        angular *= self.config.max_angular_speed
        return scan, goal, (linear, angular)

    def _obstacle_points(self, scan: np.ndarray) -> np.ndarray:
        angles = 2.0 * math.pi * (np.arange(scan.size) + 0.5) / scan.size
        detected = scan < self.config.lidar_max_range * 0.995
        return np.column_stack(
            (scan[detected] * np.cos(angles[detected]), scan[detected] * np.sin(angles[detected]))
        )

    def _dynamic_window(
        self, current_linear: float, current_angular: float
    ) -> tuple[np.ndarray, np.ndarray]:
        linear_delta = self.config.max_linear_acceleration * self.config.control_dt
        angular_delta = self.config.max_angular_acceleration * self.config.control_dt
        linear_lower = max(0.0, current_linear - linear_delta)
        linear_upper = min(self.config.max_linear_speed, current_linear + linear_delta)
        angular_lower = max(-self.config.max_angular_speed, current_angular - angular_delta)
        angular_upper = min(self.config.max_angular_speed, current_angular + angular_delta)
        linear_values = np.linspace(linear_lower, linear_upper, self.config.linear_samples)
        angular_values = np.linspace(angular_lower, angular_upper, self.config.angular_samples)
        linear_values = np.unique(np.append(linear_values, current_linear))
        angular_values = np.unique(np.append(angular_values, current_angular))
        if angular_lower <= 0.0 <= angular_upper:
            angular_values = np.unique(np.append(angular_values, 0.0))
        return linear_values, angular_values

    def _wheel_feasible(self, linear: float, angular: float) -> bool:
        left = (linear - 0.5 * self.config.axle_length * angular) / self.config.wheel_radius
        right = (linear + 0.5 * self.config.axle_length * angular) / self.config.wheel_radius
        return max(abs(left), abs(right)) <= self.config.max_wheel_speed + 1e-9

    def _simulate(self, linear: float, angular: float) -> np.ndarray:
        step_count = int(math.ceil(self.config.prediction_horizon / self.config.simulation_dt))
        trajectory = np.zeros((step_count + 1, 3), dtype=np.float64)
        for index in range(1, step_count + 1):
            previous = trajectory[index - 1]
            heading = previous[2] + angular * self.config.simulation_dt
            trajectory[index] = (
                previous[0] + linear * math.cos(heading) * self.config.simulation_dt,
                previous[1] + linear * math.sin(heading) * self.config.simulation_dt,
                heading,
            )
        return trajectory

    def _minimum_clearance(
        self, trajectory: np.ndarray, obstacle_points: np.ndarray
    ) -> float:
        if obstacle_points.size == 0:
            return self.config.lidar_max_range
        deltas = trajectory[:, None, :2] - obstacle_points[None, :, :]
        centre_distances = np.linalg.norm(deltas, axis=2)
        return float(centre_distances.min() - self.config.robot_radius)

    def _trajectory_cost(
        self,
        *,
        trajectory: np.ndarray,
        goal: np.ndarray,
        clearance: float,
        linear: float,
        angular: float,
        current: tuple[float, float],
    ) -> float:
        goal_scale = max(float(np.linalg.norm(goal)), self.config.robot_radius)
        goal_cost = float(np.linalg.norm(trajectory[-1, :2] - goal)) / goal_scale
        clearance_cost = 1.0 - float(
            np.clip(clearance / self.config.safety_margin, 0.0, 1.0)
        )
        speed_cost = 1.0 - linear / self.config.max_linear_speed
        smoothness_cost = 0.5 * (
            abs(linear - current[0]) / self.config.max_linear_speed
            + abs(angular - current[1]) / self.config.max_angular_speed
        )
        return float(
            self.config.goal_weight * goal_cost
            + self.config.clearance_weight * clearance_cost
            + self.config.speed_weight * speed_cost
            + self.config.smoothness_weight * smoothness_cost
        )

    def _emergency_command(
        self,
        goal: np.ndarray,
        current: tuple[float, float],
        evaluated: int,
    ) -> DWAResult:
        linear_values, angular_values = self._dynamic_window(*current)
        bearing = math.atan2(float(goal[1]), float(goal[0]))
        target_angular = float(np.clip(bearing, angular_values[0], angular_values[-1]))
        angular_candidates = sorted(angular_values, key=lambda value: abs(value - target_angular))
        command = next(
            (
                (float(linear), float(angular))
                for linear in linear_values
                for angular in angular_candidates
                if self._wheel_feasible(float(linear), float(angular))
            ),
            (0.0, 0.0),
        )
        linear, angular = command
        return DWAResult(
            action=self._normalize_action(linear, angular),
            linear_velocity=linear,
            angular_velocity=angular,
            cost=math.inf,
            minimum_clearance=0.0,
            evaluated_trajectories=evaluated,
            feasible_trajectories=0,
        )

    def _normalize_action(self, linear: float, angular: float) -> np.ndarray:
        action = np.array(
            [
                2.0 * linear / self.config.max_linear_speed - 1.0,
                angular / self.config.max_angular_speed,
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0).astype(np.float32)
