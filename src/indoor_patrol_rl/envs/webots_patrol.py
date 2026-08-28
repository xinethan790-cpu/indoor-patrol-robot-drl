"""Gymnasium adapter executed inside a Webots Supervisor controller.

The module deliberately avoids importing the Webots ``controller`` package at
module import time. A live Supervisor instance is injected by the Webots entry
controller, so pure Python tests and GitHub CI can still import this package.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from indoor_patrol_rl.core import (
    DifferentialDriveMapper,
    LidarSafetyShield,
    ObservationBuilder,
    RewardConfig,
    SafetyShieldConfig,
    SafetyShieldResult,
    VelocityRateLimiter,
    navigation_reward,
)


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _webots_scan_to_policy_order(raw_scan: np.ndarray) -> np.ndarray:
    """Return a scan with beam 0 forward and positive indices turning left.

    The Robotis LDS-01 range image is centred on the forward direction, but its
    index direction is the opposite of the kinematic environment's convention.
    Keeping this conversion at the adapter boundary gives both environments the
    same observation semantics.
    """

    raw_scan = np.asarray(raw_scan, dtype=np.float64).reshape(-1)
    if raw_scan.size == 0:
        raise ValueError("The Webots LiDAR returned an empty scan.")
    centre = raw_scan.size // 2
    indices = (centre - np.arange(raw_scan.size)) % raw_scan.size
    return raw_scan[indices]


class WebotsPatrolEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous TurtleBot3 navigation with one Supervisor-controlled obstacle."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        supervisor: Any,
        *,
        obstacle_defs: tuple[str, ...] = ("TRAIN_OBSTACLE_0",),
        goal_def: str = "PATROL_GOAL",
        action_repeat: int = 4,
        max_episode_steps: int = 300,
        enable_safety_shield: bool = False,
        safety_shield: LidarSafetyShield | None = None,
        safety_shield_intervention_penalty: float = 0.02,
        obstacle_speed_range: tuple[float, float] = (0.20, 0.42),
        safety_distance: float = 0.20,
        proximity_speed_distance: float = 0.60,
        proximity_speed_weight: float = 0.0,
        reward_config: RewardConfig | None = None,
        horizontal_route_half_length: float = 1.55,
        vertical_route_half_length: float = 1.10,
    ) -> None:
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive.")
        if safety_shield_intervention_penalty < 0.0:
            raise ValueError("safety_shield_intervention_penalty cannot be negative.")
        if not 0.0 <= obstacle_speed_range[0] <= obstacle_speed_range[1]:
            raise ValueError("obstacle_speed_range must be non-negative and ordered.")
        if min(safety_distance, proximity_speed_distance) <= 0.0:
            raise ValueError("Reward safety distances must be positive.")
        if proximity_speed_weight < 0.0:
            raise ValueError("proximity_speed_weight cannot be negative.")
        if min(horizontal_route_half_length, vertical_route_half_length) <= 0.0:
            raise ValueError("Route half lengths must be positive.")

        self.supervisor = supervisor
        self.basic_time_step = int(supervisor.getBasicTimeStep())
        self.action_repeat = action_repeat
        self.max_episode_steps = max_episode_steps

        self.lidar = supervisor.getDevice("LDS-01")
        self.gps = supervisor.getDevice("gps")
        self.imu = supervisor.getDevice("imu")
        for sensor in (self.lidar, self.gps, self.imu):
            sensor.enable(self.basic_time_step)

        self.left_motor = supervisor.getDevice("left wheel motor")
        self.right_motor = supervisor.getDevice("right wheel motor")
        for motor in (self.left_motor, self.right_motor):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        self.robot_node = supervisor.getSelf()
        self.robot_translation = self.robot_node.getField("translation")
        self.robot_rotation = self.robot_node.getField("rotation")
        self.goal_node = self._required_node(goal_def)
        self.goal_translation = self.goal_node.getField("translation")
        self.obstacle_nodes = [self._required_node(def_name) for def_name in obstacle_defs]
        self.obstacle_translations = [node.getField("translation") for node in self.obstacle_nodes]

        self.mapper = DifferentialDriveMapper(
            max_linear_speed=0.22,
            max_angular_speed=1.5,
            wheel_radius=0.033,
            axle_length=0.16,
            max_wheel_speed=6.0,
        )
        self.rate_limiter = VelocityRateLimiter(
            max_linear_acceleration=0.50,
            max_angular_acceleration=3.0,
            control_dt=self.basic_time_step * self.action_repeat / 1000.0,
        )
        self.observation_builder = ObservationBuilder(
            lidar_beams=36,
            lidar_max_range=float(self.lidar.getMaxRange()),
            goal_distance_scale=4.0,
        )
        self.observation_space = self.observation_builder.space
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.reward_config = reward_config or RewardConfig()
        self.obstacle_speed_range = tuple(float(value) for value in obstacle_speed_range)
        self.safety_distance = float(safety_distance)
        self.proximity_speed_distance = float(proximity_speed_distance)
        self.proximity_speed_weight = float(proximity_speed_weight)
        self.horizontal_route_half_length = float(horizontal_route_half_length)
        self.vertical_route_half_length = float(vertical_route_half_length)
        self.enable_safety_shield = bool(enable_safety_shield)
        self.safety_shield = safety_shield or LidarSafetyShield(
            SafetyShieldConfig(lidar_max_range=float(self.lidar.getMaxRange()))
        )
        self.safety_shield_intervention_penalty = safety_shield_intervention_penalty

        self.goal_position = np.zeros(2, dtype=np.float64)
        self.obstacle_speeds = np.zeros(len(self.obstacle_nodes), dtype=np.float64)
        self.obstacle_directions = np.ones(len(self.obstacle_nodes), dtype=np.float64)
        self.obstacle_axes = np.ones(len(self.obstacle_nodes), dtype=np.int8)
        self.previous_action = np.array([-1.0, 0.0], dtype=np.float32)
        self.last_policy_action = self.previous_action.copy()
        self.last_shielded_action = self.previous_action.copy()
        self.last_shield_result: SafetyShieldResult | None = None
        self.shield_active_steps = 0
        self.shield_intervention_sum = 0.0
        self.previous_goal_distance = 0.0
        self.previous_position = np.zeros(2, dtype=np.float64)
        self.commanded_linear = 0.0
        self.commanded_angular = 0.0
        self.step_count = 0
        self.path_length = 0.0
        self.minimum_clearance = math.inf
        self.current_clearance = math.inf
        self.physics_step_count = 0
        self._simulation_ended = False
        self.current_scenario_seed: int | None = None
        self.current_route_axis = "unknown"
        self.current_reverse_route = False
        self.last_contact_collision = False
        self.last_lidar_collision = False

        if self.supervisor.step(self.basic_time_step) == -1:
            raise RuntimeError("Webots ended while initializing the patrol environment.")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.current_scenario_seed = seed
        self._stop()

        route_axis = str(
            options.get("route_axis", self.np_random.choice(["horizontal", "vertical"]))
        )
        if "start_y" in options or "goal_y" in options:
            route_axis = "horizontal"
        if route_axis == "horizontal":
            start_offset = float(options.get("start_y", self.np_random.uniform(-0.55, 0.55)))
            goal_offset = float(options.get("goal_y", self.np_random.uniform(-0.55, 0.55)))
            start = np.array(
                [-self.horizontal_route_half_length, start_offset], dtype=np.float64
            )
            self.goal_position = np.array(
                [self.horizontal_route_half_length, goal_offset], dtype=np.float64
            )
            obstacle_axis = 1
        elif route_axis == "vertical":
            start_offset = float(options.get("start_x", self.np_random.uniform(-0.70, 0.70)))
            goal_offset = float(options.get("goal_x", self.np_random.uniform(-0.70, 0.70)))
            start = np.array(
                [start_offset, -self.vertical_route_half_length], dtype=np.float64
            )
            self.goal_position = np.array(
                [goal_offset, self.vertical_route_half_length], dtype=np.float64
            )
            obstacle_axis = 0
        else:
            raise ValueError("route_axis must be 'horizontal' or 'vertical'.")
        reverse_route = bool(options.get("reverse_route", self.np_random.integers(0, 2)))
        if reverse_route:
            start, self.goal_position = self.goal_position.copy(), start.copy()
        start_yaw = float(options.get("start_yaw", self.np_random.uniform(-math.pi, math.pi)))
        self.current_route_axis = route_axis
        self.current_reverse_route = reverse_route
        self.robot_translation.setSFVec3f([float(start[0]), float(start[1]), 0.0])
        self.robot_rotation.setSFRotation([0.0, 0.0, 1.0, start_yaw])
        self.goal_translation.setSFVec3f(
            [float(self.goal_position[0]), float(self.goal_position[1]), 0.025]
        )
        self.robot_node.resetPhysics()

        for index, (node, translation) in enumerate(
            zip(self.obstacle_nodes, self.obstacle_translations, strict=True)
        ):
            self.obstacle_axes[index] = obstacle_axis
            if obstacle_axis == 1:
                x = float(self.np_random.uniform(-0.55, 0.55))
                y = float(self.np_random.uniform(-1.05, 1.05))
            else:
                x = float(self.np_random.uniform(-1.55, 1.55))
                y = float(self.np_random.uniform(-0.45, 0.45))
            translation.setSFVec3f([x, y, 0.36])
            node.resetPhysics()
            self.obstacle_speeds[index] = float(
                self.np_random.uniform(*self.obstacle_speed_range)
            )
            self.obstacle_directions[index] = float(self.np_random.choice([-1.0, 1.0]))

        self.supervisor.simulationResetPhysics()
        for _ in range(2):
            if self.supervisor.step(self.basic_time_step) == -1:
                raise RuntimeError("Webots ended while resetting the patrol environment.")

        self.previous_action = np.array([-1.0, 0.0], dtype=np.float32)
        self.last_policy_action = self.previous_action.copy()
        self.last_shielded_action = self.previous_action.copy()
        self.shield_active_steps = 0
        self.shield_intervention_sum = 0.0
        self.commanded_linear = 0.0
        self.commanded_angular = 0.0
        self.step_count = 0
        self.path_length = 0.0
        self.minimum_clearance = math.inf
        self.current_clearance = math.inf
        self.physics_step_count = 0
        self._simulation_ended = False
        self.last_contact_collision = False
        self.last_lidar_collision = False
        self.previous_position = self._position()
        self.previous_goal_distance = self._goal_distance()
        scan = self._range_image()
        self.current_clearance = max(0.0, float(scan.min()) - 0.10)
        self.minimum_clearance = self.current_clearance
        self.observation_builder.reset(scan)
        # Preserve sector diagnostics at reset without claiming an intervention
        # before a policy action has actually been submitted.
        self.last_shield_result = self.safety_shield.filter_action(
            scan,
            self.previous_action,
            enabled=False,
        )
        observation = self._observation(scan)
        return observation, self._info(False, False, False, {})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        policy_action = np.asarray(action, dtype=np.float32)
        if policy_action.shape != (2,):
            raise ValueError("action must have shape (2,).")
        policy_action = np.clip(policy_action, -1.0, 1.0)
        pre_action_scan = self._range_image()
        shield_result = self.safety_shield.filter_action(
            pre_action_scan,
            policy_action,
            enabled=self.enable_safety_shield,
        )
        shielded_action = shield_result.filtered_action
        self.last_policy_action = policy_action.copy()
        self.last_shielded_action = shielded_action.copy()
        self.last_shield_result = shield_result
        self.shield_active_steps += int(shield_result.active)
        self.shield_intervention_sum += shield_result.intervention

        previous_action = self.previous_action.copy()
        requested_linear, requested_angular = self.mapper.normalized_to_velocity(
            shielded_action
        )
        requested_linear, requested_angular = self.rate_limiter.limit(
            requested_linear,
            requested_angular,
            self.commanded_linear,
            self.commanded_angular,
        )
        left_speed, right_speed = self.mapper.velocity_to_wheels(
            requested_linear, requested_angular
        )
        # Report the velocity that the saturated wheel pair can actually execute,
        # rather than the infeasible pre-saturation request.
        self.commanded_linear = 0.5 * self.mapper.wheel_radius * (left_speed + right_speed)
        self.commanded_angular = (
            self.mapper.wheel_radius
            * (right_speed - left_speed)
            / self.mapper.axle_length
        )
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        self._simulation_ended = False
        contact_collision = False
        for _ in range(self.action_repeat):
            self._move_obstacles()
            if self.supervisor.step(self.basic_time_step) == -1:
                self._simulation_ended = True
                break
            self.physics_step_count += 1
            contact_collision = contact_collision or self._has_contact_collision()
            if contact_collision:
                self._stop()
                break

        if self._simulation_ended:
            self._stop()
            raise RuntimeError("Webots ended while stepping the patrol environment.")

        self.step_count += 1
        position = self._position()
        self.path_length += float(np.linalg.norm(position - self.previous_position))
        self.previous_position = position

        scan = self._range_image()
        finite_scan = scan[np.isfinite(scan)]
        lidar_minimum = float(finite_scan.min()) if finite_scan.size else math.inf
        contact_collision = bool(contact_collision or self._has_contact_collision())
        lidar_collision = bool(
            lidar_minimum <= float(self.lidar.getMinRange()) + 0.01
        )
        self.last_contact_collision = contact_collision
        self.last_lidar_collision = lidar_collision
        collision = bool(contact_collision or lidar_collision)
        clearance = 0.0 if collision else max(0.0, lidar_minimum - 0.10)
        self.current_clearance = clearance
        self.minimum_clearance = min(self.minimum_clearance, clearance)
        distance = self._goal_distance()
        success = bool(distance <= 0.22 and not collision)
        timeout = bool(self.step_count >= self.max_episode_steps or self._simulation_ended)
        terminated = bool(success or collision)
        truncated = bool(timeout and not terminated)

        progress = self.previous_goal_distance - distance
        self.previous_goal_distance = distance
        reward, reward_terms = navigation_reward(
            progress=progress,
            min_clearance=clearance,
            safety_distance=self.safety_distance,
            action=shielded_action,
            previous_action=previous_action,
            success=success,
            collision=collision,
            timeout=truncated,
            config=self.reward_config,
        )
        intervention_term = (
            -self.safety_shield_intervention_penalty * shield_result.intervention
        )
        reward_terms["shield_intervention"] = intervention_term
        reward += intervention_term
        proximity_ratio = max(
            0.0,
            (self.proximity_speed_distance - clearance)
            / self.proximity_speed_distance,
        )
        speed_fraction = max(
            0.0,
            self.commanded_linear / self.mapper.max_linear_speed,
        )
        proximity_speed_term = (
            -self.proximity_speed_weight * proximity_ratio**2 * speed_fraction**2
        )
        reward_terms["proximity_speed"] = proximity_speed_term
        reward += proximity_speed_term
        # The policy observes the previous target that was actually admitted by
        # the shield.  Measured command velocities remain separate observation
        # fields, while evaluation metrics use ``executed_action`` from info.
        self.previous_action = shielded_action.copy()
        observation = self._observation(scan)
        return observation, reward, terminated, truncated, self._info(
            success, collision, truncated, reward_terms
        )

    def close(self) -> None:
        self._stop()

    def _required_node(self, def_name: str) -> Any:
        node = self.supervisor.getFromDef(def_name)
        if node is None:
            raise RuntimeError(f"Required Webots DEF node {def_name!r} was not found.")
        return node

    def _stop(self) -> None:
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def _position(self) -> np.ndarray:
        values = self.gps.getValues()
        return np.asarray(values[:2], dtype=np.float64)

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_position - self._position()))

    def _goal_bearing(self) -> float:
        position = self._position()
        desired = math.atan2(
            self.goal_position[1] - position[1], self.goal_position[0] - position[0]
        )
        yaw = float(self.imu.getRollPitchYaw()[2])
        return _wrap_angle(desired - yaw)

    def _range_image(self) -> np.ndarray:
        raw = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)
        minimum = float(self.lidar.getMinRange())
        maximum = float(self.lidar.getMaxRange())
        raw = np.nan_to_num(raw, nan=maximum, posinf=maximum, neginf=minimum)
        raw = np.clip(raw, minimum, maximum)
        return _webots_scan_to_policy_order(raw)

    def _has_collision(self, lidar_minimum: float) -> bool:
        # The wheels always touch the floor, so only contacts above floor level
        # count as collisions. The LiDAR fallback covers thin geometry that may
        # not yield a retained contact point at the sampled control tick.
        return self._has_contact_collision() or bool(
            lidar_minimum <= float(self.lidar.getMinRange()) + 0.01
        )

    def _has_contact_collision(self) -> bool:
        contacts = self.robot_node.getContactPoints(includeDescendants=True)
        return any(float(contact.getPoint()[2]) > 0.025 for contact in contacts)

    def _observation(self, scan: np.ndarray) -> np.ndarray:
        return self.observation_builder.build(
            raw_lidar=scan,
            goal_distance=self._goal_distance(),
            goal_bearing=self._goal_bearing(),
            linear_speed=self.commanded_linear,
            max_linear_speed=self.mapper.max_linear_speed,
            angular_speed=self.commanded_angular,
            max_angular_speed=self.mapper.max_angular_speed,
            previous_action=self.previous_action,
        )

    def _move_obstacles(self) -> None:
        seconds = self.basic_time_step / 1000.0
        for index, translation in enumerate(self.obstacle_translations):
            position = translation.getSFVec3f()
            axis = int(self.obstacle_axes[index])
            lower, upper = (-1.55, 1.55) if axis == 0 else (-1.05, 1.05)
            position[axis] += (
                self.obstacle_directions[index] * self.obstacle_speeds[index] * seconds
            )
            if position[axis] >= upper:
                position[axis] = upper
                self.obstacle_directions[index] = -1.0
            elif position[axis] <= lower:
                position[axis] = lower
                self.obstacle_directions[index] = 1.0
            translation.setSFVec3f(position)

    def _info(
        self,
        success: bool,
        collision: bool,
        timeout: bool,
        reward_terms: dict[str, float],
    ) -> dict[str, Any]:
        executed_action = np.array(
            [
                2.0 * self.commanded_linear / self.mapper.max_linear_speed - 1.0,
                self.commanded_angular / self.mapper.max_angular_speed,
            ],
            dtype=np.float32,
        )
        shield_info: dict[str, float | bool | int]
        if self.last_shield_result is None:
            shield_info = {
                "shield_enabled": self.enable_safety_shield,
                "shield_active": False,
                "shield_intervention": 0.0,
            }
        else:
            shield_info = self.last_shield_result.as_info()
            shield_info["shield_enabled"] = self.enable_safety_shield
            if self.step_count == 0:
                shield_info["shield_active"] = False
                shield_info["shield_intervention"] = 0.0
        denominator = max(self.step_count, 1)
        return {
            "is_success": success,
            "collision": collision,
            "timeout": timeout,
            "distance_to_goal": self._goal_distance(),
            "path_length": self.path_length,
            "clearance": self.current_clearance,
            "episode_min_clearance": self.minimum_clearance,
            "min_clearance": self.minimum_clearance,
            "step_count": self.step_count,
            "elapsed_sim_time": self.physics_step_count * self.basic_time_step / 1000.0,
            "scenario_seed": self.current_scenario_seed,
            "route_axis": self.current_route_axis,
            "reverse_route": self.current_reverse_route,
            "obstacle_speed_mean": (
                float(self.obstacle_speeds.mean())
                if self.obstacle_speeds.size
                else 0.0
            ),
            "contact_collision": self.last_contact_collision,
            "lidar_collision": self.last_lidar_collision,
            # Auditable action semantics:
            # policy_action -> shielded_action -> rate/wheel-limited executed_action.
            "policy_action": self.last_policy_action.copy(),
            "shielded_action": self.last_shielded_action.copy(),
            "executed_action": executed_action,
            "reward_terms": reward_terms,
            "episode_shield_intervention_ratio": self.shield_active_steps / denominator,
            "episode_shield_intervention_mean": self.shield_intervention_sum / denominator,
            **shield_info,
        }
