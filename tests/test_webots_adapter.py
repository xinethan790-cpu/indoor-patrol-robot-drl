from __future__ import annotations

import numpy as np
import pytest

from indoor_patrol_rl.envs.webots_patrol import (
    WebotsPatrolEnv,
    _sample_obstacle_position,
    _sample_route_axis,
    _webots_scan_to_policy_order,
    validate_obstacle_cross_lanes,
)


def test_default_route_axis_sampling_preserves_seed_semantics() -> None:
    actual_generator = np.random.default_rng(2026)
    expected_generator = np.random.default_rng(2026)

    actual = _sample_route_axis(actual_generator, 0.5)
    expected = str(expected_generator.choice(["horizontal", "vertical"]))

    assert actual == expected
    assert actual_generator.random() == expected_generator.random()


@pytest.mark.parametrize(
    ("probability", "expected_axis"),
    [(0.0, "vertical"), (1.0, "horizontal")],
)
def test_route_axis_sampling_honors_training_probability(
    probability: float,
    expected_axis: str,
) -> None:
    generator = np.random.default_rng(123)

    assert _sample_route_axis(generator, probability) == expected_axis


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01, float("nan"), float("inf"), -float("inf"), True, "invalid"],
)
def test_webots_env_rejects_invalid_horizontal_route_probability(
    probability: object,
) -> None:
    with pytest.raises(ValueError, match="horizontal_route_probability"):
        WebotsPatrolEnv(object(), horizontal_route_probability=probability)  # type: ignore[arg-type]


class _FixedSensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def getValues(self) -> list[float]:
        return self.values

    def getRollPitchYaw(self) -> list[float]:
        return self.values


class _FixedField:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def getSFVec3f(self) -> list[float]:
        return self.values

    def getSFRotation(self) -> list[float]:
        return self.values


class _FixedNode:
    def __init__(
        self,
        position: list[float],
        contacts: list[object] | None = None,
        velocity: list[float] | None = None,
    ) -> None:
        self.position = position
        self.contacts = contacts or []
        self.velocity = velocity or [0.1, 0.0, 0.0, 0.0, 0.0, 0.1]

    def getPosition(self) -> list[float]:
        return self.position

    def getVelocity(self) -> list[float]:
        return self.velocity

    def getContactPoints(self, *, includeDescendants: bool) -> list[object]:
        assert includeDescendants is True
        return self.contacts


class _FixedContact:
    def __init__(self, point: list[float]) -> None:
        self.point = point

    def getPoint(self) -> list[float]:
        return self.point


class _EventMotor:
    def __init__(self, name: str, events: list[tuple[object, ...]]) -> None:
        self.name = name
        self.events = events

    def setPosition(self, value: float) -> None:
        self.events.append((self.name, "position", value))

    def setVelocity(self, value: float) -> None:
        self.events.append((self.name, "velocity", value))


class _EventField:
    def __init__(self, name: str, events: list[tuple[object, ...]]) -> None:
        self.name = name
        self.events = events

    def setSFVec3f(self, value: list[float]) -> None:
        self.events.append((self.name, tuple(value)))

    def setSFRotation(self, value: list[float]) -> None:
        self.events.append((self.name, tuple(value)))


class _EventNode:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def loadState(self, name: str) -> None:
        self.events.append(("node", "load", name))

    def resetPhysics(self) -> None:
        self.events.append(("node", "reset_physics"))

    def setVelocity(self, value: list[float]) -> None:
        self.events.append(("node", "velocity", tuple(value)))


def test_webots_scan_uses_forward_zero_and_left_positive_order() -> None:
    # In the Webots LDS image, forward is raw index 4 and a left-hand return is
    # at the smaller raw index 2. The shared policy convention places them at
    # output indices 0 and 2 respectively.
    raw = np.arange(8, dtype=np.float64)

    ordered = _webots_scan_to_policy_order(raw)

    np.testing.assert_array_equal(ordered, [4, 3, 2, 1, 0, 7, 6, 5])


def test_articulated_reset_restores_joints_before_pose_and_physics() -> None:
    events: list[tuple[object, ...]] = []
    env = object.__new__(WebotsPatrolEnv)
    env.robot_node = _EventNode(events)
    env.left_motor = _EventMotor("left", events)
    env.right_motor = _EventMotor("right", events)
    env.robot_translation = _EventField("translation", events)
    env.robot_rotation = _EventField("rotation", events)

    env._restore_articulated_robot_pose(np.array([-1.55, 0.25]), 0.4)

    assert events == [
        ("node", "load", "__init__"),
        ("left", "position", float("inf")),
        ("left", "velocity", 0.0),
        ("right", "position", float("inf")),
        ("right", "velocity", 0.0),
        ("translation", (-1.55, 0.25, 0.0)),
        ("rotation", (0.0, 0.0, 1.0, 0.4)),
        ("node", "reset_physics"),
        ("node", "velocity", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ]


@pytest.mark.parametrize("obstacle_axis", [0, 1])
def test_default_obstacle_sampling_is_bitwise_compatible(obstacle_axis: int) -> None:
    actual_generator = np.random.default_rng(1234)
    expected_generator = np.random.default_rng(1234)

    actual = _sample_obstacle_position(
        actual_generator,
        obstacle_axis=obstacle_axis,
    )
    if obstacle_axis == 1:
        expected = (
            float(expected_generator.uniform(-0.55, 0.55)),
            float(expected_generator.uniform(-1.05, 1.05)),
        )
    else:
        expected = (
            float(expected_generator.uniform(-1.55, 1.55)),
            float(expected_generator.uniform(-0.45, 0.45)),
        )

    assert actual == expected
    assert actual_generator.random() == expected_generator.random()


def test_formal_lane_override_preserves_random_number_consumption() -> None:
    default_generator = np.random.default_rng(4321)
    lane_generator = np.random.default_rng(4321)

    default = _sample_obstacle_position(default_generator, obstacle_axis=1)
    lane = _sample_obstacle_position(
        lane_generator,
        obstacle_axis=1,
        cross_lane=-0.77,
    )

    assert lane == (-0.77, default[1])
    assert lane_generator.random() == default_generator.random()


def test_five_formal_lanes_are_validated() -> None:
    lanes = validate_obstacle_cross_lanes(
        [0.00, -0.39, 0.39, 0.75, -0.77],
        expected_count=5,
        minimum_separation=0.35,
    )
    assert lanes == (0.00, -0.39, 0.39, 0.75, -0.77)


@pytest.mark.parametrize(
    "value",
    [
        [0.0, 0.2],
        [0.0, float("nan"), 0.8],
        [0.0, 0.4],
        "0.0,0.4,0.8",
    ],
)
def test_invalid_formal_lanes_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        validate_obstacle_cross_lanes(
            value,
            expected_count=3,
            minimum_separation=0.35,
        )


def _pose_guard_env() -> WebotsPatrolEnv:
    env = object.__new__(WebotsPatrolEnv)
    env.sensor_sample_tick = 17
    env.gps = _FixedSensor([float("nan"), 0.4, 0.0])
    env.imu = _FixedSensor([0.0, float("inf"), float("nan")])
    env.robot_translation = _FixedField([0.25, 0.35, 0.0])
    env.robot_rotation = _FixedField([0.0, 0.0, 1.0, -0.45])
    env.robot_node = _FixedNode([0.25, 0.35, 0.0])
    env.step_count = 4
    env.physics_step_count = 16
    env.current_route_axis = "horizontal"
    env.current_reverse_route = False
    env.current_start_position = np.array([-1.55, 0.1])
    env.current_start_yaw = 0.2
    env.goal_position = np.array([1.55, -0.1])
    return env


def test_webots_pose_sensors_fail_closed_with_finite_supervisor_context() -> None:
    env = _pose_guard_env()

    with pytest.raises(RuntimeError, match="GPS position.*Supervisor robot state remained finite"):
        env._position()
    with pytest.raises(
        RuntimeError, match="IMU roll/pitch/yaw.*Supervisor robot state remained finite"
    ):
        env._roll_pitch_yaw()


def test_webots_gps_reports_non_finite_supervisor_physics() -> None:
    env = _pose_guard_env()
    env.gps = _FixedSensor([float("nan"), float("nan"), 0.0])
    env.robot_translation = _FixedField([float("nan"), 0.4, 0.0])
    env.robot_node = _FixedNode([float("nan"), 0.4, 0.0])

    with pytest.raises(RuntimeError, match="GPS position.*physical state.*non-finite"):
        env._position()


def test_webots_pose_guard_rejects_non_finite_contact_state() -> None:
    env = _pose_guard_env()
    env.robot_node = _FixedNode(
        [0.25, 0.35, 0.0], contacts=[_FixedContact([0.0, 0.0, float("nan")])]
    )

    with pytest.raises(RuntimeError, match="contact state.*non-finite"):
        env._position()


def test_reset_readback_rejects_finite_but_misplaced_pose() -> None:
    env = _pose_guard_env()
    env.gps = _FixedSensor([0.22, 0.35, 0.153])
    env.imu = _FixedSensor([0.0, 0.0, 0.2])
    env.robot_node = _FixedNode([0.25, 0.35, 0.0], velocity=[0.0] * 6)
    env.reset_readback_check_count = 0
    env.maximum_reset_translation_error = 0.0
    env.maximum_reset_start_xy_error = 0.0
    env.maximum_reset_gps_root_xy_distance = 0.0
    env.maximum_reset_yaw_error = 0.0
    env.maximum_reset_linear_speed = 0.0
    env.maximum_reset_angular_speed = 0.0

    with pytest.raises(RuntimeError, match="reset readback.*start_xy_error"):
        env._validate_reset_readback()


def test_reset_readback_records_consistent_pose_metrics() -> None:
    env = _pose_guard_env()
    env.current_start_position = np.array([0.25, 0.35])
    env.current_start_yaw = 0.2
    env.gps = _FixedSensor([0.22, 0.35, 0.153])
    env.imu = _FixedSensor([0.0, 0.0, 0.2])
    env.robot_node = _FixedNode([0.25, 0.35, 0.0], velocity=[0.0] * 6)
    env.reset_readback_check_count = 0
    env.maximum_reset_translation_error = 0.0
    env.maximum_reset_start_xy_error = 0.0
    env.maximum_reset_gps_root_xy_distance = 0.0
    env.maximum_reset_yaw_error = 0.0
    env.maximum_reset_linear_speed = 0.0
    env.maximum_reset_angular_speed = 0.0

    env._validate_reset_readback()

    assert env.reset_readback_check_count == 1
    assert env.maximum_reset_translation_error == 0.0
    assert env.maximum_reset_start_xy_error == 0.0
    assert env.maximum_reset_gps_root_xy_distance == pytest.approx(0.03)
    assert env.maximum_reset_yaw_error == 0.0


def test_webots_collision_check_rejects_non_finite_contact_state() -> None:
    env = _pose_guard_env()
    env.robot_node = _FixedNode(
        [0.25, 0.35, 0.0], contacts=[_FixedContact([0.0, 0.0, float("nan")])]
    )

    with pytest.raises(RuntimeError, match="contact state became non-finite"):
        env._has_contact_collision()


def test_webots_collision_check_rejects_non_finite_contact_xy() -> None:
    env = _pose_guard_env()
    env.robot_node = _FixedNode(
        [0.25, 0.35, 0.0], contacts=[_FixedContact([float("nan"), 0.0, 0.0])]
    )

    with pytest.raises(RuntimeError, match="contact state became non-finite"):
        env._has_contact_collision()
