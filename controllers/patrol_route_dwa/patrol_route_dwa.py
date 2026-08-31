"""A* global routing plus DWA local avoidance in a multi-checkpoint Webots demo."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import yaml
from controller import Supervisor

from indoor_patrol_rl.core import DifferentialDriveMapper, ObservationBuilder, VelocityRateLimiter
from indoor_patrol_rl.envs.webots_patrol import _webots_scan_to_policy_order
from indoor_patrol_rl.planning import (
    AxisAlignedRectangle,
    WaypointManager,
    grid_from_rectangles_with_clearances,
    plan_world_path,
)
from indoor_patrol_rl.policies import DWAConfig, DWAPlanner


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def finish(robot: Supervisor, exit_code: int) -> None:
    if os.environ.get("WEBOTS_AUTO_QUIT") == "1":
        robot.simulationQuit(exit_code)
    else:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


def main() -> None:
    repository_root = Path(__file__).absolute().parents[2]
    with (repository_root / "configs/maps/patrol_route.yaml").open(encoding="utf-8") as stream:
        route_config = yaml.safe_load(stream)

    map_config = route_config["map"]
    rectangles = [
        AxisAlignedRectangle(tuple(item["center"]), tuple(item["size"]))
        for item in map_config["static_rectangles"]
    ]
    grid = grid_from_rectangles_with_clearances(
        origin=tuple(map_config["origin"]),
        size=tuple(map_config["size"]),
        resolution=float(map_config["resolution"]),
        rectangles=rectangles,
        obstacle_inflation_radius=float(map_config["obstacle_inflation_radius"]),
        border_inflation_radius=float(map_config["border_inflation_radius"]),
    )
    checkpoints = [
        (item["id"], tuple(item["position"])) for item in route_config["checkpoints"]
    ]

    robot = Supervisor()
    basic_time_step = int(robot.getBasicTimeStep())
    action_repeat = 4
    control_dt = basic_time_step * action_repeat / 1000.0
    lidar = robot.getDevice("LDS-01")
    gps = robot.getDevice("gps")
    imu = robot.getDevice("imu")
    for sensor in (lidar, gps, imu):
        sensor.enable(basic_time_step)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    mapper = DifferentialDriveMapper(0.22, 1.5, 0.033, 0.16, 6.0)
    limiter = VelocityRateLimiter(0.50, 3.0, control_dt)
    observation_builder = ObservationBuilder(36, float(lidar.getMaxRange()), 4.0)
    planner = DWAPlanner(DWAConfig(control_dt=control_dt))
    waypoint_manager = WaypointManager(waypoint_radius=0.16, final_radius=0.22)
    local_target_field = robot.getFromDef("LOCAL_TARGET").getField("translation")
    robot_node = robot.getSelf()

    for _ in range(2):
        if robot.step(basic_time_step) == -1:
            raise RuntimeError("Webots ended while initializing the route demo.")

    def position() -> tuple[float, float]:
        values = gps.getValues()
        return float(values[0]), float(values[1])

    checkpoint_index = 0
    active_path = plan_world_path(grid, position(), checkpoints[checkpoint_index][1])
    waypoint_state = waypoint_manager.reset(active_path)
    print(
        f"ROUTE_PLANNED checkpoint={checkpoints[0][0]} "
        f"cells={len(active_path.cells)} waypoints={len(active_path.waypoints)} "
        f"length={active_path.length:.2f}",
        flush=True,
    )

    raw_scan = np.asarray(lidar.getRangeImage(), dtype=np.float64)
    observation_builder.reset(_webots_scan_to_policy_order(raw_scan))
    current_linear = 0.0
    current_angular = 0.0
    previous_action = np.array([-1.0, 0.0], dtype=np.float32)
    minimum_lidar_range = math.inf

    for control_step in range(3200):
        current_position = position()
        waypoint_state = waypoint_manager.update(current_position)
        if waypoint_state.completed:
            checkpoint_id, _ = checkpoints[checkpoint_index]
            print(f"POINT_VISITED id={checkpoint_id} step={control_step}", flush=True)
            checkpoint_index += 1
            if checkpoint_index == len(checkpoints):
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                print(
                    f"PATROL_ROUTE_OK points={len(checkpoints)} steps={control_step} "
                    f"min_lidar_range={minimum_lidar_range:.3f}",
                    flush=True,
                )
                robot.step(basic_time_step)
                finish(robot, 0)
                return
            active_path = plan_world_path(
                grid, current_position, checkpoints[checkpoint_index][1]
            )
            waypoint_state = waypoint_manager.reset(active_path)
            print(
                f"ROUTE_PLANNED checkpoint={checkpoints[checkpoint_index][0]} "
                f"cells={len(active_path.cells)} waypoints={len(active_path.waypoints)} "
                f"length={active_path.length:.2f}",
                flush=True,
            )

        if waypoint_state.target is None:
            raise RuntimeError("Route demo has no active local target.")
        target = waypoint_state.target
        local_target_field.setSFVec3f([float(target[0]), float(target[1]), 0.035])

        raw_scan = np.asarray(lidar.getRangeImage(), dtype=np.float64)
        scan = _webots_scan_to_policy_order(raw_scan)
        finite_scan = scan[np.isfinite(scan)]
        if finite_scan.size:
            minimum_lidar_range = min(minimum_lidar_range, float(finite_scan.min()))
        dx = target[0] - current_position[0]
        dy = target[1] - current_position[1]
        bearing = wrap_angle(math.atan2(dy, dx) - float(imu.getRollPitchYaw()[2]))
        observation = observation_builder.build(
            raw_lidar=scan,
            goal_distance=math.hypot(dx, dy),
            goal_bearing=bearing,
            linear_speed=current_linear,
            max_linear_speed=mapper.max_linear_speed,
            angular_speed=current_angular,
            max_angular_speed=mapper.max_angular_speed,
            previous_action=previous_action,
        )
        action = planner.predict(observation)
        requested_linear, requested_angular = mapper.normalized_to_velocity(action)
        limited_linear, limited_angular = limiter.limit(
            requested_linear,
            requested_angular,
            current_linear,
            current_angular,
        )
        left_speed, right_speed = mapper.velocity_to_wheels(limited_linear, limited_angular)
        current_linear = 0.5 * mapper.wheel_radius * (left_speed + right_speed)
        current_angular = (
            mapper.wheel_radius * (right_speed - left_speed) / mapper.axle_length
        )
        previous_action = np.array(
            [
                2.0 * current_linear / mapper.max_linear_speed - 1.0,
                current_angular / mapper.max_angular_speed,
            ],
            dtype=np.float32,
        )
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        for _ in range(action_repeat):
            if robot.step(basic_time_step) == -1:
                raise RuntimeError("Webots ended before the route demo completed.")
            contacts = robot_node.getContactPoints(includeDescendants=True)
            if any(float(contact.getPoint()[2]) > 0.025 for contact in contacts):
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                print(f"PATROL_ROUTE_COLLISION step={control_step}", flush=True)
                finish(robot, 2)
                return

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    print(f"PATROL_ROUTE_TIMEOUT visited={checkpoint_index}", flush=True)
    finish(robot, 3)


if __name__ == "__main__":
    main()
