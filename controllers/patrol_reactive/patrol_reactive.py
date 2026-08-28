"""Point-to-point navigation with a simple LiDAR rule baseline."""

from __future__ import annotations

import math
import os

from controller import Supervisor


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def finite_min(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite, default=math.inf)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def finish(robot: Supervisor, exit_code: int) -> None:
    """Quit headless tests, but pause an interactive Webots demo."""
    if os.environ.get("WEBOTS_AUTO_QUIT") == "1":
        robot.simulationQuit(exit_code)
    else:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


def main() -> None:
    robot = Supervisor()
    time_step = int(robot.getBasicTimeStep())
    target = (1.65, 0.0)

    lidar = robot.getDevice("LDS-01")
    gps = robot.getDevice("gps")
    imu = robot.getDevice("imu")
    lidar.enable(time_step)
    gps.enable(time_step)
    imu.enable(time_step)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    wheel_radius = 0.033
    axle_length = 0.16
    max_wheel_speed = 6.0
    min_clearance = math.inf

    for step_index in range(520):
        if robot.step(time_step) == -1:
            raise RuntimeError("Webots ended before the dynamic demo completed.")

        position = gps.getValues()
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        distance = math.hypot(dx, dy)
        if distance <= 0.22:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print(
                f"PATROL_DEMO_OK steps={step_index} min_clearance={min_clearance:.3f}",
                flush=True,
            )
            robot.step(time_step)
            finish(robot, 0)
            return

        ranges = list(lidar.getRangeImage())
        resolution = len(ranges)
        center = resolution // 2
        half_window = max(4, resolution // 24)
        front_minimum = finite_min(ranges[center - half_window : center + half_window])
        min_clearance = min(min_clearance, finite_min(ranges))

        yaw = imu.getRollPitchYaw()[2]
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - yaw)

        if front_minimum < 0.55:
            linear_velocity = 0.02
            angular_velocity = 1.1
        else:
            angular_velocity = clamp(2.0 * heading_error, -1.0, 1.0)
            linear_velocity = 0.14 * max(0.25, 1.0 - abs(heading_error) / math.pi)

        left_speed = (linear_velocity - 0.5 * axle_length * angular_velocity) / wheel_radius
        right_speed = (linear_velocity + 0.5 * axle_length * angular_velocity) / wheel_radius
        left_motor.setVelocity(clamp(left_speed, -max_wheel_speed, max_wheel_speed))
        right_motor.setVelocity(clamp(right_speed, -max_wheel_speed, max_wheel_speed))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    print(f"PATROL_DEMO_TIMEOUT min_clearance={min_clearance:.3f}", flush=True)
    robot.step(time_step)
    finish(robot, 2)


if __name__ == "__main__":
    main()
