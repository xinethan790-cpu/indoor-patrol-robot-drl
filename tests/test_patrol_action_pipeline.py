from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from indoor_patrol_rl.application import (
    PatrolActionPipeline,
    PatrolControlMode,
)
from indoor_patrol_rl.core import (
    DifferentialDriveMapper,
    LidarSafetyShield,
    SafetyShieldResult,
    VelocityRateLimiter,
)


def _clear_scan() -> np.ndarray:
    return np.full(360, 3.5, dtype=np.float64)


class _RecordingShield:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def filter_action(
        self,
        raw_lidar: np.ndarray | list[float],
        action: np.ndarray | list[float],
        *,
        enabled: bool = True,
    ) -> SafetyShieldResult:
        requested = np.asarray(action, dtype=np.float32).copy()
        self.calls.append(("shield", requested.copy(), enabled))
        filtered = (
            np.array([0.25, -0.50], dtype=np.float32)
            if enabled
            else requested.copy()
        )
        return SafetyShieldResult(
            enabled=enabled,
            active=enabled,
            requested_action=requested,
            filtered_action=filtered,
            intervention=float(np.linalg.norm(filtered - requested)),
            front_clearance=1.0,
            left_clearance=1.0,
            right_clearance=1.0,
            speed_fraction_limit=1.0,
            normalized_speed_limit=1.0,
            avoidance_direction=1,
        )


class _RecordingMapper:
    wheel_radius = 0.1
    axle_length = 0.5

    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def normalized_to_velocity(self, action: np.ndarray) -> tuple[float, float]:
        self.calls.append(("map", action.copy()))
        return 0.4, -0.8

    def velocity_to_wheels(self, linear: float, angular: float) -> tuple[float, float]:
        self.calls.append(("wheels", linear, angular))
        return 1.0, 3.0


class _RecordingLimiter:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def limit(
        self,
        target_linear: float,
        target_angular: float,
        current_linear: float,
        current_angular: float,
    ) -> tuple[float, float]:
        self.calls.append(
            (
                "limit",
                target_linear,
                target_angular,
                current_linear,
                current_angular,
            )
        )
        return 0.2, -0.3


def test_pipeline_uses_fixed_order_and_clips_before_pinned_shield() -> None:
    calls: list[tuple[str, Any]] = []
    pipeline = PatrolActionPipeline(
        safety_shield=_RecordingShield(calls),  # type: ignore[arg-type]
        mapper=_RecordingMapper(calls),  # type: ignore[arg-type]
        rate_limiter=_RecordingLimiter(calls),  # type: ignore[arg-type]
    )

    result = pipeline.execute(
        raw_lidar=_clear_scan(),
        policy_action=[2.0, -3.0],
        current_linear_mps=0.1,
        current_angular_radps=-0.1,
    )

    assert [call[0] for call in calls] == ["shield", "map", "limit", "wheels"]
    np.testing.assert_array_equal(calls[0][1], [1.0, -1.0])
    assert calls[0][2] is True
    np.testing.assert_array_equal(result.policy_action, [1.0, -1.0])
    np.testing.assert_array_equal(result.shielded_action, [0.25, -0.50])
    assert result.requested_linear_mps == pytest.approx(0.4)
    assert result.requested_angular_radps == pytest.approx(-0.8)
    assert result.rate_limited_linear_mps == pytest.approx(0.2)
    assert result.rate_limited_angular_radps == pytest.approx(-0.3)
    assert result.left_wheel_radps == pytest.approx(1.0)
    assert result.right_wheel_radps == pytest.approx(3.0)
    assert result.executable_linear_mps == pytest.approx(0.2)
    assert result.executable_angular_radps == pytest.approx(0.4)


def test_experimental_disabled_mode_is_explicit_and_retains_diagnostics() -> None:
    pipeline = PatrolActionPipeline(
        safety_shield=LidarSafetyShield(),
        mapper=DifferentialDriveMapper(0.35, 1.5, 0.033, 0.16, 6.67),
        rate_limiter=VelocityRateLimiter(10.0, 10.0, 1.0),
        mode=PatrolControlMode.EXPERIMENTAL_SHIELD_DISABLED,
    )
    scan = _clear_scan()
    scan[:31] = 0.1
    scan[-30:] = 0.1

    result = pipeline.execute(
        raw_lidar=scan,
        policy_action=[0.7, 0.1],
        current_linear_mps=0.0,
        current_angular_radps=0.0,
    )

    assert not result.shield_result.enabled
    assert not result.shield_result.active
    assert not result.shield_diagnostics["shield_enabled"]
    assert result.shield_result.front_clearance == pytest.approx(0.1)
    np.testing.assert_array_equal(result.shielded_action, result.policy_action)


def test_wheel_saturation_is_reflected_in_executable_body_velocity() -> None:
    mapper = DifferentialDriveMapper(
        max_linear_speed=0.5,
        max_angular_speed=1.0,
        wheel_radius=0.1,
        axle_length=0.5,
        max_wheel_speed=1.0,
    )
    pipeline = PatrolActionPipeline(
        safety_shield=LidarSafetyShield(),
        mapper=mapper,
        rate_limiter=VelocityRateLimiter(10.0, 10.0, 1.0),
    )

    result = pipeline.execute(
        raw_lidar=_clear_scan(),
        policy_action=[1.0, 1.0],
        current_linear_mps=0.0,
        current_angular_radps=0.0,
    )

    assert result.rate_limited_linear_mps == pytest.approx(0.5)
    assert result.rate_limited_angular_radps == pytest.approx(1.0)
    assert max(abs(result.left_wheel_radps), abs(result.right_wheel_radps)) == pytest.approx(
        1.0
    )
    assert result.executable_linear_mps < result.rate_limited_linear_mps
    assert result.executable_angular_radps < result.rate_limited_angular_radps
    assert result.executable_linear_mps == pytest.approx(
        0.5 * mapper.wheel_radius * (result.left_wheel_radps + result.right_wheel_radps)
    )


def test_previous_action_is_a_safe_copy_of_the_shielded_action() -> None:
    pipeline = PatrolActionPipeline(
        safety_shield=LidarSafetyShield(),
        mapper=DifferentialDriveMapper(0.35, 1.5, 0.033, 0.16, 6.67),
        rate_limiter=VelocityRateLimiter(10.0, 10.0, 1.0),
    )
    result = pipeline.execute(
        raw_lidar=_clear_scan(),
        policy_action=[0.4, -0.2],
        current_linear_mps=0.0,
        current_angular_radps=0.0,
    )

    previous_action = result.previous_action
    np.testing.assert_array_equal(previous_action, result.shielded_action)
    previous_action[0] = -1.0
    assert result.shielded_action[0] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "policy_action",
    ([0.0], [0.0, 0.0, 0.0], [float("nan"), 0.0], [0.0, float("inf")]),
)
def test_invalid_policy_action_fails_before_the_shield(
    policy_action: list[float],
) -> None:
    calls: list[tuple[str, Any]] = []
    pipeline = PatrolActionPipeline(
        safety_shield=_RecordingShield(calls),  # type: ignore[arg-type]
        mapper=_RecordingMapper(calls),  # type: ignore[arg-type]
        rate_limiter=_RecordingLimiter(calls),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="policy_action"):
        pipeline.execute(
            raw_lidar=_clear_scan(),
            policy_action=policy_action,
            current_linear_mps=0.0,
            current_angular_radps=0.0,
        )

    assert calls == []


def test_invalid_mode_cannot_silently_disable_a_pinned_component_pipeline() -> None:
    with pytest.raises(TypeError, match="PatrolControlMode"):
        PatrolActionPipeline(
            safety_shield=LidarSafetyShield(),
            mapper=DifferentialDriveMapper(0.35, 1.5, 0.033, 0.16, 6.67),
            rate_limiter=VelocityRateLimiter(10.0, 10.0, 1.0),
            mode=False,  # type: ignore[arg-type]
        )
