from __future__ import annotations

import numpy as np
import pytest

from indoor_patrol_rl.envs.webots_patrol import (
    _sample_obstacle_position,
    _webots_scan_to_policy_order,
    validate_obstacle_cross_lanes,
)


def test_webots_scan_uses_forward_zero_and_left_positive_order() -> None:
    # In the Webots LDS image, forward is raw index 4 and a left-hand return is
    # at the smaller raw index 2. The shared policy convention places them at
    # output indices 0 and 2 respectively.
    raw = np.arange(8, dtype=np.float64)

    ordered = _webots_scan_to_policy_order(raw)

    np.testing.assert_array_equal(ordered, [4, 3, 2, 1, 0, 7, 6, 5])


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
