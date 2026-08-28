"""Fast 2-D kinematic environment for prototyping the Webots observation contract.

This environment is intentionally lightweight. It lets us validate observation,
action, reward, seeding, training, and evaluation code before the same contract is
connected to the slower Webots simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from indoor_patrol_rl.core import ObservationBuilder, RewardConfig, navigation_reward


@dataclass
class MovingObstacle:
    """Circular obstacle with a constant velocity and wall reflection."""

    position: np.ndarray
    velocity: np.ndarray
    radius: float


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class DynamicObstacleAvoidanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous-control navigation among moving circular obstacles.

    Action values are normalized to ``[-1, 1]``:

    - ``action[0]`` maps to linear velocity ``[0, max_linear_speed]``;
    - ``action[1]`` maps to angular velocity
      ``[-max_angular_speed, max_angular_speed]``.

    The observation contains normalized full-circle LiDAR ranges, normalized goal
    distance, sine/cosine of goal bearing, and normalized current velocities.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        width: float = 10.0,
        height: float = 8.0,
        lidar_beams: int = 36,
        lidar_range: float = 6.0,
        obstacle_count: int = 5,
        obstacle_speed_range: tuple[float, float] = (0.15, 0.55),
        max_episode_steps: int = 500,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        if width <= 2.0 or height <= 2.0:
            raise ValueError("The world must be larger than 2 m in each dimension.")
        if lidar_beams < 8:
            raise ValueError("lidar_beams must be at least 8.")
        if obstacle_count < 0:
            raise ValueError("obstacle_count cannot be negative.")

        self.width = float(width)
        self.height = float(height)
        self.lidar_beams = int(lidar_beams)
        self.lidar_range = float(lidar_range)
        self.obstacle_count = int(obstacle_count)
        self.obstacle_speed_range = obstacle_speed_range
        self.max_episode_steps = int(max_episode_steps)
        self.dt = float(dt)

        self.robot_radius = 0.25
        self.obstacle_radius = 0.28
        self.goal_radius = 0.35
        self.safety_distance = 0.60
        self.max_linear_speed = 0.8
        self.max_angular_speed = 1.6
        self.velocity_response = 0.45
        self.world_diagonal = float(np.hypot(self.width, self.height))

        self._observation_builder = ObservationBuilder(
            lidar_beams=self.lidar_beams,
            lidar_max_range=self.lidar_range,
            goal_distance_scale=self.world_diagonal,
        )
        self.observation_space = self._observation_builder.space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._reward_config = RewardConfig(
            safety_weight=-0.12,
            smoothness_weight=-0.01,
            goal_reward=20.0,
            collision_penalty=-20.0,
        )

        self.robot_position = np.zeros(2, dtype=np.float64)
        self.robot_heading = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.goal_position = np.zeros(2, dtype=np.float64)
        self.obstacles: list[MovingObstacle] = []
        self.step_count = 0
        self.path_length = 0.0
        self.episode_min_clearance = math.inf
        self._previous_goal_distance = 0.0
        self._previous_action = np.array([-1.0, 0.0], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        count = int(options.get("obstacle_count", self.obstacle_count))
        if count < 0:
            raise ValueError("obstacle_count cannot be negative.")

        margin = 0.8
        self.robot_position = self._sample_point(margin)
        self.goal_position = self._sample_point(margin)
        for _ in range(200):
            if np.linalg.norm(self.goal_position - self.robot_position) >= 4.0:
                break
            self.goal_position = self._sample_point(margin)

        self.robot_heading = float(self.np_random.uniform(-np.pi, np.pi))
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.obstacles = []

        for _ in range(count):
            position = self._sample_obstacle_position()
            angle = float(self.np_random.uniform(-np.pi, np.pi))
            speed = float(self.np_random.uniform(*self.obstacle_speed_range))
            velocity = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            self.obstacles.append(MovingObstacle(position, velocity, self.obstacle_radius))

        self.step_count = 0
        self.path_length = 0.0
        self.episode_min_clearance = self._minimum_clearance()
        self._previous_goal_distance = self._goal_distance()
        self._observation_builder.reset(self._lidar_scan())
        self._previous_action = np.array([-1.0, 0.0], dtype=np.float32)
        observation = self._observation()
        return observation, self._info(
            is_success=False,
            collision=False,
            timeout=False,
            reward_terms={},
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        old_position = self.robot_position.copy()
        previous_action = self._previous_action.copy()

        target_linear = float((action[0] + 1.0) * 0.5 * self.max_linear_speed)
        target_angular = float(action[1] * self.max_angular_speed)
        alpha = self.velocity_response
        self.linear_speed += alpha * (target_linear - self.linear_speed)
        self.angular_speed += alpha * (target_angular - self.angular_speed)

        self.robot_heading = _wrap_angle(self.robot_heading + self.angular_speed * self.dt)
        direction = np.array(
            [np.cos(self.robot_heading), np.sin(self.robot_heading)], dtype=np.float64
        )
        self.robot_position += direction * self.linear_speed * self.dt
        self.path_length += float(np.linalg.norm(self.robot_position - old_position))

        self._move_obstacles()
        self.step_count += 1

        collision = self._has_collision()
        distance = self._goal_distance()
        is_success = bool(distance <= self.goal_radius and not collision)
        terminated = bool(collision or is_success)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)

        progress = self._previous_goal_distance - distance
        self._previous_goal_distance = distance
        min_clearance = self._minimum_clearance()
        self.episode_min_clearance = min(self.episode_min_clearance, min_clearance)
        reward, reward_terms = navigation_reward(
            progress=progress,
            min_clearance=min_clearance,
            safety_distance=self.safety_distance,
            action=action,
            previous_action=previous_action,
            success=is_success,
            collision=collision,
            timeout=truncated,
            config=self._reward_config,
        )

        self._previous_action = action.copy()
        observation = self._observation()
        info = self._info(
            is_success=is_success,
            collision=collision,
            timeout=truncated,
            reward_terms=reward_terms,
        )
        return observation, float(reward), terminated, truncated, info

    def _sample_point(self, margin: float) -> np.ndarray:
        return self.np_random.uniform(
            low=np.array([margin, margin]),
            high=np.array([self.width - margin, self.height - margin]),
        ).astype(np.float64)

    def _sample_obstacle_position(self) -> np.ndarray:
        margin = self.obstacle_radius + 0.25
        for _ in range(500):
            point = self._sample_point(margin)
            far_from_robot = np.linalg.norm(point - self.robot_position) >= 1.2
            far_from_goal = np.linalg.norm(point - self.goal_position) >= 1.0
            far_from_others = all(
                np.linalg.norm(point - obstacle.position) >= 2.5 * self.obstacle_radius
                for obstacle in self.obstacles
            )
            if far_from_robot and far_from_goal and far_from_others:
                return point
        raise RuntimeError("Could not place obstacles without overlap; reduce obstacle_count.")

    def _move_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle.position += obstacle.velocity * self.dt
            for axis, upper in ((0, self.width), (1, self.height)):
                lower_limit = obstacle.radius
                upper_limit = upper - obstacle.radius
                if obstacle.position[axis] < lower_limit:
                    obstacle.position[axis] = lower_limit
                    obstacle.velocity[axis] = abs(obstacle.velocity[axis])
                elif obstacle.position[axis] > upper_limit:
                    obstacle.position[axis] = upper_limit
                    obstacle.velocity[axis] = -abs(obstacle.velocity[axis])

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_position - self.robot_position))

    def _goal_bearing(self) -> float:
        vector = self.goal_position - self.robot_position
        return _wrap_angle(float(np.arctan2(vector[1], vector[0])) - self.robot_heading)

    def _observation(self) -> np.ndarray:
        return self._observation_builder.build(
            raw_lidar=self._lidar_scan(),
            goal_distance=self._goal_distance(),
            goal_bearing=self._goal_bearing(),
            linear_speed=self.linear_speed,
            max_linear_speed=self.max_linear_speed,
            angular_speed=self.angular_speed,
            max_angular_speed=self.max_angular_speed,
            previous_action=self._previous_action,
        )

    def _lidar_scan(self) -> np.ndarray:
        result = np.full(self.lidar_beams, self.lidar_range, dtype=np.float64)
        for index in range(self.lidar_beams):
            angle = self.robot_heading + 2.0 * np.pi * index / self.lidar_beams
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            result[index] = min(
                self.lidar_range,
                self._ray_distance_to_walls(direction),
                *(self._ray_distance_to_circle(direction, obstacle) for obstacle in self.obstacles),
            )
        return result

    def _ray_distance_to_walls(self, direction: np.ndarray) -> float:
        distances: list[float] = []
        epsilon = 1e-12
        if abs(direction[0]) > epsilon:
            for wall_x in (0.0, self.width):
                distance = (wall_x - self.robot_position[0]) / direction[0]
                if distance >= 0.0:
                    y = self.robot_position[1] + distance * direction[1]
                    if 0.0 <= y <= self.height:
                        distances.append(float(distance))
        if abs(direction[1]) > epsilon:
            for wall_y in (0.0, self.height):
                distance = (wall_y - self.robot_position[1]) / direction[1]
                if distance >= 0.0:
                    x = self.robot_position[0] + distance * direction[0]
                    if 0.0 <= x <= self.width:
                        distances.append(float(distance))
        return min(distances, default=self.lidar_range)

    def _ray_distance_to_circle(
        self, direction: np.ndarray, obstacle: MovingObstacle
    ) -> float:
        relative = self.robot_position - obstacle.position
        b = 2.0 * float(np.dot(direction, relative))
        c = float(np.dot(relative, relative) - obstacle.radius**2)
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            return self.lidar_range
        root = float(np.sqrt(discriminant))
        candidates = [
            distance
            for distance in ((-b - root) / 2.0, (-b + root) / 2.0)
            if distance >= 0.0
        ]
        return min(candidates, default=self.lidar_range)

    def _minimum_clearance(self) -> float:
        wall_clearance = min(
            self.robot_position[0],
            self.width - self.robot_position[0],
            self.robot_position[1],
            self.height - self.robot_position[1],
        ) - self.robot_radius
        obstacle_clearances = [
            float(np.linalg.norm(self.robot_position - obstacle.position))
            - self.robot_radius
            - obstacle.radius
            for obstacle in self.obstacles
        ]
        return min([float(wall_clearance), *obstacle_clearances])

    def _has_collision(self) -> bool:
        return bool(self._minimum_clearance() <= 0.0)

    def _info(
        self,
        *,
        is_success: bool,
        collision: bool,
        timeout: bool,
        reward_terms: dict[str, float],
    ) -> dict[str, Any]:
        clearance = self._minimum_clearance()
        executed_action = np.array(
            [
                2.0 * self.linear_speed / self.max_linear_speed - 1.0,
                self.angular_speed / self.max_angular_speed,
            ],
            dtype=np.float32,
        )
        return {
            "is_success": is_success,
            "collision": collision,
            "timeout": timeout,
            "distance_to_goal": self._goal_distance(),
            "clearance": clearance,
            "episode_min_clearance": self.episode_min_clearance,
            "min_clearance": self.episode_min_clearance,
            "path_length": self.path_length,
            "step_count": self.step_count,
            "elapsed_sim_time": self.step_count * self.dt,
            "executed_action": executed_action,
            "reward_terms": reward_terms,
        }
