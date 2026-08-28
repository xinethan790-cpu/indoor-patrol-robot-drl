"""Verify TurtleBot3 wheel motors and LDS-01 LiDAR on Apple Silicon."""

from __future__ import annotations

import math

from controller import Supervisor


def finite_min(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite, default=math.inf)


def main() -> None:
    robot = Supervisor()
    time_step = int(robot.getBasicTimeStep())

    lidar = robot.getDevice("LDS-01")
    lidar.enable(time_step)
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    if robot.step(time_step) == -1:
        raise RuntimeError("Webots ended before LiDAR initialization.")

    resolution = lidar.getHorizontalResolution()
    max_range = lidar.getMaxRange()
    observed_minimum = math.inf

    for _ in range(80):
        ranges = list(lidar.getRangeImage())
        observed_minimum = min(observed_minimum, finite_min(ranges))

        quarter = max(1, resolution // 4)
        front = ranges[quarter : 3 * quarter]
        front_minimum = finite_min(front)
        if front_minimum < 0.55:
            left_motor.setVelocity(-0.8)
            right_motor.setVelocity(0.8)
        else:
            left_motor.setVelocity(1.2)
            right_motor.setVelocity(1.2)

        if robot.step(time_step) == -1:
            raise RuntimeError("Webots ended before the TurtleBot3 smoke test completed.")

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    print(
        "TURTLEBOT_SMOKE_OK "
        f"lidar_resolution={resolution} max_range={max_range:.2f} "
        f"observed_minimum={observed_minimum:.3f}",
        flush=True,
    )
    robot.step(time_step)
    robot.simulationQuit(0)


if __name__ == "__main__":
    main()
