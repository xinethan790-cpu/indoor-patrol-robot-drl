from __future__ import annotations

import numpy as np
import pytest

from indoor_patrol_rl.core import (
    DifferentialDriveMapper,
    ObservationBuilder,
    RewardConfig,
    VelocityRateLimiter,
    navigation_reward,
)


def test_observation_builder_handles_invalid_lidar_and_has_79_values() -> None:
    builder = ObservationBuilder(lidar_beams=36, lidar_max_range=3.5)
    scan = np.linspace(0.2, 3.5, 360)
    scan[3] = np.nan
    scan[5] = np.inf
    builder.reset(scan)
    observation = builder.build(
        raw_lidar=scan,
        goal_distance=1.0,
        goal_bearing=0.25,
        linear_speed=0.1,
        max_linear_speed=0.35,
        angular_speed=-0.2,
        max_angular_speed=1.5,
        previous_action=np.array([0.0, -0.1], dtype=np.float32),
    )
    assert observation.shape == (79,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert builder.space.contains(observation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goal_distance", float("nan")),
        ("goal_bearing", float("inf")),
        ("linear_speed", float("-inf")),
        ("max_linear_speed", float("nan")),
        ("angular_speed", float("nan")),
        ("max_angular_speed", float("inf")),
    ],
)
def test_observation_builder_rejects_named_non_finite_scalar_fields(
    field: str, value: float
) -> None:
    builder = ObservationBuilder(lidar_beams=36, lidar_max_range=3.5)
    arguments: dict[str, object] = {
        "raw_lidar": np.full(360, 2.0),
        "goal_distance": 1.0,
        "goal_bearing": 0.25,
        "linear_speed": 0.1,
        "max_linear_speed": 0.35,
        "angular_speed": -0.2,
        "max_angular_speed": 1.5,
        "previous_action": np.zeros(2, dtype=np.float32),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        builder.build(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_observation_builder_rejects_non_finite_previous_action(invalid_value: float) -> None:
    builder = ObservationBuilder(lidar_beams=36, lidar_max_range=3.5)

    with pytest.raises(ValueError, match="previous_action.*finite"):
        builder.build(
            raw_lidar=np.full(360, 2.0),
            goal_distance=1.0,
            goal_bearing=0.25,
            linear_speed=0.1,
            max_linear_speed=0.35,
            angular_speed=-0.2,
            max_angular_speed=1.5,
            previous_action=np.array([0.0, invalid_value], dtype=np.float32),
        )


def test_differential_drive_mapping_is_symmetric_for_straight_motion() -> None:
    mapper = DifferentialDriveMapper(0.35, 1.5, 0.033, 0.16, 6.67)
    left, right = mapper.normalized_to_wheels(np.array([0.0, 0.0], dtype=np.float32))
    assert left == right
    assert 0.0 < left <= 6.67


def test_differential_drive_saturation_preserves_curvature() -> None:
    mapper = DifferentialDriveMapper(0.35, 1.5, 0.033, 0.16, 6.67)
    raw_left = (0.35 - 0.5 * 0.16 * 1.5) / 0.033
    raw_right = (0.35 + 0.5 * 0.16 * 1.5) / 0.033

    left, right = mapper.velocity_to_wheels(0.35, 1.5)

    assert max(abs(left), abs(right)) == pytest.approx(6.67)
    assert left / right == pytest.approx(raw_left / raw_right)


def test_velocity_rate_limiter_applies_shared_acceleration_limits() -> None:
    limiter = VelocityRateLimiter(0.5, 3.0, 0.128)

    linear, angular = limiter.limit(0.22, -1.5, 0.0, 0.0)

    assert linear == pytest.approx(0.064)
    assert angular == pytest.approx(-0.384)


def test_collision_reward_takes_priority_over_success() -> None:
    total, terms = navigation_reward(
        progress=0.1,
        min_clearance=0.0,
        safety_distance=0.2,
        action=np.zeros(2, dtype=np.float32),
        previous_action=np.zeros(2, dtype=np.float32),
        success=True,
        collision=True,
        timeout=False,
        config=RewardConfig(),
    )
    assert terms["terminal"] == -30.0
    assert total < 0.0
