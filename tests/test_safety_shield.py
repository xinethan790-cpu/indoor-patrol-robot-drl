from __future__ import annotations

import numpy as np
import pytest

from indoor_patrol_rl.core import (
    SAFETY_SHIELD_CONFIG_ENV_FIELDS,
    LidarSafetyShield,
    SafetyShieldConfig,
    safety_shield_config_from_env,
)


def scan_with_clearances(*, front: float, left: float, right: float) -> np.ndarray:
    scan = np.full(360, 3.5, dtype=np.float64)
    scan[:31] = front
    scan[-30:] = front
    scan[35:106] = left
    scan[-105:-34] = right
    return scan


def test_far_obstacles_do_not_change_policy_action() -> None:
    shield = LidarSafetyShield()
    action = np.array([0.55, -0.25], dtype=np.float32)

    result = shield.filter_action(np.full(360, 3.5), action)

    np.testing.assert_array_equal(result.filtered_action, action)
    assert not result.active
    assert result.intervention == pytest.approx(0.0)
    assert result.speed_fraction_limit == pytest.approx(1.0)


def test_close_front_obstacle_progressively_stops_and_turns_to_open_side() -> None:
    shield = LidarSafetyShield()
    action = np.array([0.8, 0.0], dtype=np.float32)

    result = shield.filter_action(
        scan_with_clearances(front=0.20, left=1.5, right=0.45),
        action,
    )

    assert result.active
    assert -1.0 <= result.filtered_action[0] < -0.9
    assert result.filtered_action[1] > 0.5
    assert result.avoidance_direction == 1
    assert result.front_clearance == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("left", "right", "expected_sign"),
    [(0.14, 1.2, -1.0), (1.2, 0.14, 1.0)],
)
def test_side_obstacle_turns_toward_the_more_open_side(
    left: float,
    right: float,
    expected_sign: float,
) -> None:
    shield = LidarSafetyShield()

    result = shield.filter_action(
        scan_with_clearances(front=1.5, left=left, right=right),
        [0.6, 0.0],
    )

    assert np.sign(result.filtered_action[1]) == expected_sign
    assert result.active


def test_filter_is_continuous_bounded_and_never_requests_reverse_velocity() -> None:
    shield = LidarSafetyShield()
    action = np.array([1.0, -0.2], dtype=np.float32)
    distances = np.linspace(0.10, 0.70, 301)
    outputs = np.asarray(
        [
            shield.filter_action(
                scan_with_clearances(front=float(distance), left=1.0, right=0.8),
                action,
            ).filtered_action
            for distance in distances
        ]
    )

    assert np.all(outputs >= -1.0)
    assert np.all(outputs <= 1.0)
    # Adjacent 2 mm clearance changes cannot produce a discontinuous command jump.
    assert float(np.abs(np.diff(outputs, axis=0)).max()) < 0.03
    # Normalized -1 is zero forward velocity in this project; no value requests reverse.
    assert float(outputs[:, 0].min()) == pytest.approx(-1.0)


def test_disabled_filter_is_passthrough_but_retains_clearance_diagnostics() -> None:
    shield = LidarSafetyShield()
    action = np.array([0.7, 0.1], dtype=np.float32)

    result = shield.filter_action(
        scan_with_clearances(front=0.15, left=0.2, right=1.0),
        action,
        enabled=False,
    )

    np.testing.assert_array_equal(result.filtered_action, action)
    assert not result.enabled
    assert not result.active
    assert result.intervention == pytest.approx(0.0)
    assert result.front_clearance == pytest.approx(0.15)


def test_environment_parser_covers_every_non_hardware_config_field() -> None:
    configurable_fields = set(SafetyShieldConfig.__dataclass_fields__) - {
        "lidar_max_range"
    }

    assert set(SAFETY_SHIELD_CONFIG_ENV_FIELDS.values()) == configurable_fields
    assert safety_shield_config_from_env({}) == SafetyShieldConfig()


def test_environment_parser_resolves_overrides_and_hardware_range() -> None:
    config = safety_shield_config_from_env(
        {
            "WEBOTS_SAFETY_SHIELD_CLEARANCE_QUANTILE": "0.05",
            "WEBOTS_SAFETY_SHIELD_FRONT_STOP_DISTANCE": "0.24",
            "WEBOTS_SAFETY_SHIELD_FRONT_SLOWDOWN_DISTANCE": "0.72",
            "WEBOTS_SAFETY_SHIELD_SIDE_STOP_DISTANCE": "0.16",
            "WEBOTS_SAFETY_SHIELD_SIDE_SLOWDOWN_DISTANCE": "0.42",
        },
        lidar_max_range=8.0,
    )

    assert config.lidar_max_range == pytest.approx(8.0)
    assert config.clearance_quantile == pytest.approx(0.05)
    assert config.front_stop_distance == pytest.approx(0.24)
    assert config.front_slowdown_distance == pytest.approx(0.72)
    assert config.side_stop_distance == pytest.approx(0.16)
    assert config.side_slowdown_distance == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"WEBOTS_SAFETY_SHIELD_FRONT_STOP_DISTNACE": "0.2"}, "Unsupported"),
        ({"WEBOTS_SAFETY_SHIELD_CLEARANCE_QUANTILE": "nan"}, "finite float"),
        (
            {
                "WEBOTS_SAFETY_SHIELD_FRONT_STOP_DISTANCE": "0.8",
                "WEBOTS_SAFETY_SHIELD_FRONT_SLOWDOWN_DISTANCE": "0.7",
            },
            "front_stop_distance",
        ),
    ],
)
def test_environment_parser_rejects_typos_nonfinite_and_invalid_geometry(
    environ: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        safety_shield_config_from_env(environ)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SafetyShieldConfig(turn_gain=value)
