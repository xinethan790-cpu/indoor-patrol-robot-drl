"""Gymnasium adapter executed inside a Webots Supervisor controller.

The module deliberately avoids importing the Webots ``controller`` package at
module import time. A live Supervisor instance is injected by the Webots entry
controller, so pure Python tests and GitHub CI can still import this package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def validate_obstacle_cross_lanes(
    value: Any,
    *,
    expected_count: int,
    minimum_separation: float = 0.0,
) -> tuple[float, ...]:
    """Validate optional fixed cross-axis lanes without depending on Webots APIs."""

    if expected_count < 0:
        raise ValueError("expected_count cannot be negative.")
    if minimum_separation < 0.0 or not math.isfinite(minimum_separation):
        raise ValueError("minimum_separation must be finite and non-negative.")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("obstacle_cross_lanes must be a numeric sequence.")
    if len(value) != expected_count:
        raise ValueError(f"obstacle_cross_lanes must contain exactly {expected_count} values.")
    lanes: list[float] = []
    for index, raw_lane in enumerate(value):
        if isinstance(raw_lane, bool) or not isinstance(raw_lane, (int, float)):
            raise ValueError(f"obstacle_cross_lanes[{index}] must be numeric.")
        lane = float(raw_lane)
        if not math.isfinite(lane):
            raise ValueError(f"obstacle_cross_lanes[{index}] must be finite.")
        lanes.append(lane)
    for first in range(len(lanes)):
        for second in range(first + 1, len(lanes)):
            if abs(lanes[first] - lanes[second]) <= minimum_separation:
                raise ValueError("obstacle_cross_lanes do not satisfy the minimum separation.")
    return tuple(lanes)


def _sample_obstacle_position(
    random_generator: Any,
    *,
    obstacle_axis: int,
    cross_lane: float | None = None,
) -> tuple[float, float]:
    """Sample exactly as before, optionally overriding only the cross-axis value."""

    if obstacle_axis == 1:
        random_x = float(random_generator.uniform(-0.55, 0.55))
        random_y = float(random_generator.uniform(-1.05, 1.05))
        return (float(cross_lane) if cross_lane is not None else random_x), random_y
    if obstacle_axis == 0:
        random_x = float(random_generator.uniform(-1.55, 1.55))
        random_y = float(random_generator.uniform(-0.45, 0.45))
        return random_x, (float(cross_lane) if cross_lane is not None else random_y)
    raise ValueError("obstacle_axis must be 0 or 1.")


def _validate_horizontal_route_probability(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("horizontal_route_probability must be between zero and one.")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "horizontal_route_probability must be between zero and one."
        ) from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("horizontal_route_probability must be between zero and one.")
    return probability


def _sample_route_axis(
    random_generator: np.random.Generator,
    horizontal_route_probability: float,
) -> str:
    if horizontal_route_probability == 0.5:
        # Preserve the established seed-to-scenario mapping at the default.
        return str(random_generator.choice(["horizontal", "vertical"]))
    return (
        "horizontal"
        if random_generator.random() < horizontal_route_probability
        else "vertical"
    )


class WebotsPatrolEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous TurtleBot3 navigation with one Supervisor-controlled obstacle."""

    metadata = {"render_modes": []}
    reset_translation_tolerance = 1e-6
    reset_start_xy_tolerance = 0.02
    reset_gps_root_xy_limit = 0.06
    reset_yaw_tolerance = 0.05
    reset_linear_speed_limit = 0.50
    reset_angular_speed_limit = 5.0

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
        horizontal_route_probability: float = 0.5,
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
        validated_horizontal_route_probability = (
            _validate_horizontal_route_probability(horizontal_route_probability)
        )

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
        self.horizontal_route_probability = validated_horizontal_route_probability
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
        self.current_start_position = np.zeros(2, dtype=np.float64)
        self.current_start_yaw = 0.0
        self.last_contact_collision = False
        self.last_lidar_collision = False
        self.sensor_sample_tick = 0
        self.reset_attempt_count = 0
        self.successful_reset_count = 0
        self.reset_finite_check_count = 0
        self.reset_readback_check_count = 0
        self.maximum_reset_translation_error = 0.0
        self.maximum_reset_start_xy_error = 0.0
        self.maximum_reset_gps_root_xy_distance = 0.0
        self.maximum_reset_yaw_error = 0.0
        self.maximum_reset_linear_speed = 0.0
        self.maximum_reset_angular_speed = 0.0
        if self.supervisor.step(self.basic_time_step) == -1:
            raise RuntimeError("Webots ended while initializing the patrol environment.")
        self.sensor_sample_tick += 1
        self._position()
        self._roll_pitch_yaw()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.current_scenario_seed = seed
        self.reset_attempt_count += 1
        self._stop()

        route_axis = (
            str(options["route_axis"])
            if "route_axis" in options
            else _sample_route_axis(
                self.np_random,
                self.horizontal_route_probability,
            )
        )
        if "start_y" in options or "goal_y" in options:
            route_axis = "horizontal"
        if route_axis == "horizontal":
            start_offset = float(options.get("start_y", self.np_random.uniform(-0.55, 0.55)))
            goal_offset = float(options.get("goal_y", self.np_random.uniform(-0.55, 0.55)))
            start = np.array([-self.horizontal_route_half_length, start_offset], dtype=np.float64)
            self.goal_position = np.array(
                [self.horizontal_route_half_length, goal_offset], dtype=np.float64
            )
            obstacle_axis = 1
        elif route_axis == "vertical":
            start_offset = float(options.get("start_x", self.np_random.uniform(-0.70, 0.70)))
            goal_offset = float(options.get("goal_x", self.np_random.uniform(-0.70, 0.70)))
            start = np.array([start_offset, -self.vertical_route_half_length], dtype=np.float64)
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
        self.current_start_position = start.copy()
        self.current_start_yaw = start_yaw
        self._restore_articulated_robot_pose(start, start_yaw)
        self.goal_translation.setSFVec3f(
            [float(self.goal_position[0]), float(self.goal_position[1]), 0.025]
        )

        raw_cross_lanes = options.get("obstacle_cross_lanes")
        obstacle_cross_lanes = (
            validate_obstacle_cross_lanes(
                raw_cross_lanes,
                expected_count=len(self.obstacle_nodes),
                minimum_separation=float(options.get("obstacle_lane_min_separation", 0.0)),
            )
            if raw_cross_lanes is not None
            else None
        )

        for index, (node, translation) in enumerate(
            zip(self.obstacle_nodes, self.obstacle_translations, strict=True)
        ):
            self.obstacle_axes[index] = obstacle_axis
            cross_lane = obstacle_cross_lanes[index] if obstacle_cross_lanes is not None else None
            x, y = _sample_obstacle_position(
                self.np_random,
                obstacle_axis=obstacle_axis,
                cross_lane=cross_lane,
            )
            translation.setSFVec3f([x, y, 0.36])
            node.resetPhysics()
            self.obstacle_speeds[index] = float(self.np_random.uniform(*self.obstacle_speed_range))
            self.obstacle_directions[index] = float(self.np_random.choice([-1.0, 1.0]))

        # Each moved dynamic node is reset above. A global physics reset here is
        # both redundant and unsafe for the articulated TurtleBot: repeated
        # world-wide ODE resets after teleporting the robot have produced a
        # fully non-finite robot transform/velocity on the next control step.
        self.step_count = 0
        self.physics_step_count = 0
        self._simulation_ended = False
        for _ in range(2):
            if self.supervisor.step(self.basic_time_step) == -1:
                raise RuntimeError("Webots ended while resetting the patrol environment.")
            self.sensor_sample_tick += 1
            self._finite_supervisor_robot_state()
            self.reset_finite_check_count += 1
        self._validate_reset_readback()

        self.previous_action = np.array([-1.0, 0.0], dtype=np.float32)
        self.last_policy_action = self.previous_action.copy()
        self.last_shielded_action = self.previous_action.copy()
        self.shield_active_steps = 0
        self.shield_intervention_sum = 0.0
        self.commanded_linear = 0.0
        self.commanded_angular = 0.0
        self.path_length = 0.0
        self.minimum_clearance = math.inf
        self.current_clearance = math.inf
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
        self.successful_reset_count += 1
        return observation, self._info(False, False, False, {})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        policy_action = np.asarray(action, dtype=np.float32)
        if policy_action.shape != (2,):
            raise ValueError("action must have shape (2,).")
        if not np.isfinite(policy_action).all():
            raise ValueError(f"action must contain finite values; got {policy_action.tolist()}.")
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
        requested_linear, requested_angular = self.mapper.normalized_to_velocity(shielded_action)
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
            self.mapper.wheel_radius * (right_speed - left_speed) / self.mapper.axle_length
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
            self.sensor_sample_tick += 1
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
        lidar_collision = bool(lidar_minimum <= float(self.lidar.getMinRange()) + 0.01)
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
        intervention_term = -self.safety_shield_intervention_penalty * shield_result.intervention
        reward_terms["shield_intervention"] = intervention_term
        reward += intervention_term
        proximity_ratio = max(
            0.0,
            (self.proximity_speed_distance - clearance) / self.proximity_speed_distance,
        )
        speed_fraction = max(
            0.0,
            self.commanded_linear / self.mapper.max_linear_speed,
        )
        proximity_speed_term = -self.proximity_speed_weight * proximity_ratio**2 * speed_fraction**2
        reward_terms["proximity_speed"] = proximity_speed_term
        reward += proximity_speed_term
        if not math.isfinite(reward):
            raise RuntimeError(
                "Webots patrol reward became non-finite; "
                f"progress={progress!r}, clearance={clearance!r}, "
                f"distance={distance!r}, terms={reward_terms!r}."
            )
        # The policy observes the previous target that was actually admitted by
        # the shield.  Measured command velocities remain separate observation
        # fields, while evaluation metrics use ``executed_action`` from info.
        self.previous_action = shielded_action.copy()
        observation = self._observation(scan)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._info(success, collision, truncated, reward_terms),
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

    def _restore_articulated_robot_pose(
        self, start: np.ndarray, start_yaw: float
    ) -> None:
        """Restore every TurtleBot joint before applying the next episode pose."""

        self.robot_node.loadState("__init__")
        for motor in (self.left_motor, self.right_motor):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.robot_translation.setSFVec3f([float(start[0]), float(start[1]), 0.0])
        self.robot_rotation.setSFRotation([0.0, 0.0, 1.0, float(start_yaw)])
        self.robot_node.resetPhysics()
        self.robot_node.setVelocity([0.0] * 6)

    def _position(self) -> np.ndarray:
        values = np.asarray(self.gps.getValues(), dtype=np.float64).reshape(-1)
        if values.size < 2:
            raise RuntimeError(
                f"Webots GPS returned {values.size} values; at least two are required."
            )
        position = values[:2]
        if not np.isfinite(position).all():
            self._raise_non_finite_pose_sensor("GPS position", position)
        return position

    def _roll_pitch_yaw(self) -> np.ndarray:
        values = np.asarray(self.imu.getRollPitchYaw(), dtype=np.float64).reshape(-1)
        if values.size < 3:
            raise RuntimeError(
                f"Webots IMU returned {values.size} values; roll, pitch, and yaw are required."
            )
        orientation = values[:3]
        if not np.isfinite(orientation).all():
            self._raise_non_finite_pose_sensor("IMU roll/pitch/yaw", orientation)
        return orientation

    def _raise_non_finite_pose_sensor(self, name: str, values: np.ndarray) -> None:
        try:
            state = self._finite_supervisor_robot_state()
        except RuntimeError as state_error:
            raise RuntimeError(
                f"Webots {name} was non-finite at sensor tick {self.sensor_sample_tick}: "
                f"{values.tolist()}; {state_error}"
            ) from state_error
        raise RuntimeError(
            f"Webots {name} was non-finite at sensor tick {self.sensor_sample_tick}: "
            f"{values.tolist()}; Supervisor robot state remained finite: {state}."
        )

    def _finite_supervisor_robot_state(self) -> dict[str, Any]:
        """Return an auditable snapshot, failing if any physical-state field is invalid."""

        translation = np.asarray(self.robot_translation.getSFVec3f(), dtype=np.float64)
        rotation = np.asarray(self.robot_rotation.getSFRotation(), dtype=np.float64)
        position = np.asarray(self.robot_node.getPosition(), dtype=np.float64)
        velocity = np.asarray(self.robot_node.getVelocity(), dtype=np.float64)
        expected_shapes = {
            "translation": (translation, (3,)),
            "rotation": (rotation, (4,)),
            "position": (position, (3,)),
            "velocity": (velocity, (6,)),
        }
        malformed = {
            name: values.shape
            for name, (values, expected_shape) in expected_shapes.items()
            if values.shape != expected_shape
        }
        if malformed:
            raise RuntimeError(
                f"Supervisor robot-state snapshot had malformed fields: {malformed}."
            )

        contact_points = self._contact_points()
        contact_heights = contact_points[:, 2]
        state = {
            "translation": translation.tolist(),
            "rotation": rotation.tolist(),
            "position": position.tolist(),
            "velocity": velocity.tolist(),
            "contact_count": int(contact_points.shape[0]),
            "contact_points": contact_points.tolist(),
            "elevated_contact_count": int(np.count_nonzero(contact_heights > 0.025)),
            "env_step": self.step_count,
            "physics_step": self.physics_step_count,
            "sensor_tick": self.sensor_sample_tick,
            "route_axis": self.current_route_axis,
            "reverse_route": self.current_reverse_route,
            "start_position": self.current_start_position.tolist(),
            "start_yaw": self.current_start_yaw,
            "goal_position": self.goal_position.tolist(),
        }
        non_finite_fields = [
            name
            for name, (field_values, _) in expected_shapes.items()
            if not np.isfinite(field_values).all()
        ]
        if non_finite_fields:
            raise RuntimeError(
                "Supervisor robot physical state was non-finite in "
                f"{non_finite_fields}: {state}."
            )
        return state

    def _validate_reset_readback(self) -> None:
        """Reject a finite but misplaced, moving, or colliding episode reset."""

        state = self._finite_supervisor_robot_state()
        translation = np.asarray(state["translation"], dtype=np.float64)
        position = np.asarray(state["position"], dtype=np.float64)
        velocity = np.asarray(state["velocity"], dtype=np.float64)
        gps_position = self._position()
        orientation = self._roll_pitch_yaw()
        metrics = {
            "translation_error": float(np.linalg.norm(translation - position)),
            "start_xy_error": float(np.linalg.norm(position[:2] - self.current_start_position)),
            # TurtleBot3 Burger mounts extension-slot sensors 0.03 m behind the
            # root origin, so the GPS/root XY distance is expected to be non-zero.
            "gps_root_xy_distance": float(np.linalg.norm(gps_position - position[:2])),
            "yaw_error": abs(_wrap_angle(float(orientation[2]) - self.current_start_yaw)),
            "linear_speed": float(np.linalg.norm(velocity[:3])),
            "angular_speed": float(np.linalg.norm(velocity[3:])),
        }
        limits = {
            "translation_error": self.reset_translation_tolerance,
            "start_xy_error": self.reset_start_xy_tolerance,
            "gps_root_xy_distance": self.reset_gps_root_xy_limit,
            "yaw_error": self.reset_yaw_tolerance,
            "linear_speed": self.reset_linear_speed_limit,
            "angular_speed": self.reset_angular_speed_limit,
        }
        violations = {
            name: {"actual": metrics[name], "limit": limit}
            for name, limit in limits.items()
            if metrics[name] > limit
        }
        if state["elevated_contact_count"]:
            violations["elevated_contact_count"] = {
                "actual": state["elevated_contact_count"],
                "limit": 0,
            }
        if violations:
            raise RuntimeError(
                "Webots reset readback failed physical consistency checks: "
                f"violations={violations}, state={state}, gps={gps_position.tolist()}, "
                f"imu={orientation.tolist()}."
            )
        self.reset_readback_check_count += 1
        self.maximum_reset_translation_error = max(
            self.maximum_reset_translation_error, metrics["translation_error"]
        )
        self.maximum_reset_start_xy_error = max(
            self.maximum_reset_start_xy_error, metrics["start_xy_error"]
        )
        self.maximum_reset_gps_root_xy_distance = max(
            self.maximum_reset_gps_root_xy_distance, metrics["gps_root_xy_distance"]
        )
        self.maximum_reset_yaw_error = max(
            self.maximum_reset_yaw_error, metrics["yaw_error"]
        )
        self.maximum_reset_linear_speed = max(
            self.maximum_reset_linear_speed, metrics["linear_speed"]
        )
        self.maximum_reset_angular_speed = max(
            self.maximum_reset_angular_speed, metrics["angular_speed"]
        )

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Collect failure-safe pose and episode context for controller artifacts."""

        def sensor_values(read: Any) -> list[float] | str:
            try:
                return np.asarray(read(), dtype=np.float64).reshape(-1).tolist()
            except Exception as error:  # pragma: no cover - only hardware/API failure
                return f"unavailable: {type(error).__name__}: {error}"

        try:
            supervisor_state: dict[str, Any] | str = self._finite_supervisor_robot_state()
        except RuntimeError as error:
            supervisor_state = str(error)
        return {
            "sensor_tick": self.sensor_sample_tick,
            "env_step": self.step_count,
            "physics_step": self.physics_step_count,
            "route_axis": self.current_route_axis,
            "reverse_route": self.current_reverse_route,
            "start_position": self.current_start_position.tolist(),
            "start_yaw": self.current_start_yaw,
            "goal_position": self.goal_position.tolist(),
            "gps_raw": sensor_values(self.gps.getValues),
            "imu_raw": sensor_values(self.imu.getRollPitchYaw),
            "supervisor_state": supervisor_state,
            "reset_health": self.reset_health_diagnostics(),
        }

    def reset_health_diagnostics(self) -> dict[str, int | float]:
        return {
            "attempts": self.reset_attempt_count,
            "successful": self.successful_reset_count,
            "finite_settle_checks": self.reset_finite_check_count,
            "readback_checks": self.reset_readback_check_count,
            "maximum_translation_error": self.maximum_reset_translation_error,
            "maximum_start_xy_error": self.maximum_reset_start_xy_error,
            "maximum_gps_root_xy_distance": self.maximum_reset_gps_root_xy_distance,
            "maximum_yaw_error": self.maximum_reset_yaw_error,
            "maximum_linear_speed": self.maximum_reset_linear_speed,
            "maximum_angular_speed": self.maximum_reset_angular_speed,
        }

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal_position - self._position()))

    def _goal_bearing(self) -> float:
        position = self._position()
        desired = math.atan2(
            self.goal_position[1] - position[1], self.goal_position[0] - position[0]
        )
        yaw = float(self._roll_pitch_yaw()[2])
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
        return bool(np.any(self._contact_points()[:, 2] > 0.025))

    def _contact_points(self) -> np.ndarray:
        contacts = self.robot_node.getContactPoints(includeDescendants=True)
        points: list[np.ndarray] = []
        for index, contact in enumerate(contacts):
            point = np.asarray(contact.getPoint(), dtype=np.float64)
            if point.shape != (3,):
                raise RuntimeError(
                    f"Webots robot contact point {index} had shape {point.shape}, expected (3,)."
                )
            points.append(point)
        result = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)
        if not np.isfinite(result).all():
            raise RuntimeError(
                "Webots robot contact state became non-finite; "
                f"points={result.tolist()}, sensor_tick={self.sensor_sample_tick}, "
                f"env_step={self.step_count}, physics_step={self.physics_step_count}."
            )
        return result

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
        distance_to_goal = self._goal_distance()
        return {
            "is_success": success,
            "collision": collision,
            "timeout": timeout,
            "distance_to_goal": distance_to_goal,
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
                float(self.obstacle_speeds.mean()) if self.obstacle_speeds.size else 0.0
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
