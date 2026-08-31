from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from indoor_patrol_rl.application import (
    CriticalProximityConfig,
    CriticalProximityMonitor,
)


def _scan(minimum: float, *, beam_index: int = 0, beam_count: int = 8) -> np.ndarray:
    scan = np.full(beam_count, 3.5, dtype=np.float64)
    scan[beam_index] = minimum
    return scan


def _cluster_scan(
    minimum: float,
    *,
    beam_index: int = 0,
    beam_count: int = 360,
) -> np.ndarray:
    """Return a unique raw minimum backed by a five-beam threat cluster."""

    scan = np.full(beam_count, 3.5, dtype=np.float64)
    scan[beam_index] = minimum
    scan[(beam_index - 1) % beam_count] = minimum + 0.001
    scan[(beam_index + 1) % beam_count] = minimum + 0.001
    scan[(beam_index - 2) % beam_count] = minimum + 0.002
    scan[(beam_index + 2) % beam_count] = minimum + 0.002
    return scan


def _monitor(
    *,
    threshold: float = 0.13,
    approaching_epsilon_mps: float = 1e-6,
) -> CriticalProximityMonitor:
    return CriticalProximityMonitor(
        CriticalProximityConfig(
            critical_threshold_m=threshold,
            approaching_epsilon_mps=approaching_epsilon_mps,
        )
    )


def test_config_factory_uses_caller_supplied_lidar_minimum_plus_one_centimetre() -> None:
    config = CriticalProximityConfig.from_lidar_min_range(0.12)

    assert config.critical_threshold_m == pytest.approx(0.13)
    assert config.threat_association_max_angle_rad == pytest.approx(
        math.radians(8.0)
    )
    assert config.maximum_plausible_closing_rate_mps == pytest.approx(0.50)
    assert config.approaching_confirmation_s == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("beam_index", "expected_angle"),
    [
        (0, 0.0),
        (2, math.pi / 2.0),
        (4, -math.pi),
        (6, -math.pi / 2.0),
    ],
)
def test_policy_order_beam_is_reported_with_wrapped_angle(
    beam_index: int,
    expected_angle: float,
) -> None:
    assessment = _monitor().update(
        _scan(0.5, beam_index=beam_index),
        dt_s=0.1,
    )

    assert assessment.minimum_distance_m == pytest.approx(0.5)
    assert assessment.minimum_beam_index == beam_index
    assert assessment.minimum_angle_rad == pytest.approx(expected_angle)
    assert assessment.beam_count == 8


def test_persistent_same_cluster_requires_confirmation_before_rate_and_ttc() -> None:
    monitor = _monitor()

    first = monitor.update(_cluster_scan(0.50), dt_s=0.1)
    pending = monitor.update(_cluster_scan(0.47), dt_s=0.1)
    confirmed = monitor.update(_cluster_scan(0.44), dt_s=0.1)

    assert first.tracked_distance_m == pytest.approx(0.501)
    assert first.threat_association_valid is False
    assert first.closing_rate_valid is False
    assert first.approaching_confirmation_elapsed_s == 0.0
    assert first.closing_rate_mps is None
    assert first.ttc_s is None

    assert pending.threat_association_valid is True
    assert pending.closing_rate_valid is True
    assert pending.approaching_confirmation_elapsed_s == pytest.approx(0.1)
    assert pending.approaching_confirmed is False
    assert pending.closing_rate_mps is None
    assert pending.ttc_s is None
    assert pending.ttc_valid is False

    assert confirmed.approaching_confirmation_elapsed_s == pytest.approx(0.2)
    assert confirmed.approaching_confirmation_required_s == pytest.approx(0.2)
    assert confirmed.approaching_confirmed is True
    assert confirmed.closing_rate_mps == pytest.approx(0.30)
    assert confirmed.ttc_s == pytest.approx((0.441 - 0.13) / 0.30)
    assert confirmed.ttc_valid is True
    assert confirmed.approaching


def test_cross_beam_global_minimum_change_cannot_create_ttc_spike() -> None:
    monitor = _monitor()
    monitor.update(_cluster_scan(0.50, beam_index=0), dt_s=0.1)

    switched = monitor.update(
        _cluster_scan(0.40, beam_index=90),
        dt_s=0.1,
    )

    assert switched.minimum_beam_index == 90
    assert switched.threat_association_valid is False
    assert switched.closing_rate_valid is False
    assert switched.approaching_confirmation_elapsed_s == 0.0
    assert switched.closing_rate_mps is None
    assert switched.ttc_s is None


def test_single_beam_spike_keeps_hard_critical_without_temporal_ttc() -> None:
    monitor = _monitor()
    monitor.update(np.full(360, 0.50), dt_s=0.1)
    scan = np.full(360, 0.50)
    scan[0] = 0.12

    spike = monitor.update(scan, dt_s=0.1)

    assert spike.minimum_distance_m == pytest.approx(0.12)
    assert spike.tracked_distance_m == pytest.approx(0.50)
    assert spike.critical
    assert spike.consecutive_critical_frames == 1
    assert spike.threat_association_valid is True
    assert spike.closing_rate_valid is True
    assert spike.closing_rate_mps is None
    assert spike.ttc_s is None
    assert spike.ttc_valid is False


def test_implausibly_fast_closing_cluster_is_rejected() -> None:
    monitor = _monitor()
    monitor.update(_cluster_scan(0.50), dt_s=0.1)

    assessment = monitor.update(_cluster_scan(0.40), dt_s=0.1)

    assert assessment.threat_association_valid is True
    assert assessment.closing_rate_valid is False
    assert assessment.approaching_confirmation_elapsed_s == 0.0
    assert assessment.closing_rate_mps is None
    assert assessment.ttc_s is None


def test_implausibly_fast_opening_cluster_is_also_marked_invalid() -> None:
    monitor = _monitor()
    monitor.update(_cluster_scan(0.40), dt_s=0.1)

    assessment = monitor.update(_cluster_scan(0.50), dt_s=0.1)

    assert assessment.threat_association_valid is True
    assert assessment.closing_rate_valid is False
    assert assessment.approaching_confirmation_elapsed_s == 0.0
    assert assessment.closing_rate_mps is None
    assert assessment.ttc_s is None


def test_maximum_plausible_closing_rate_is_inclusive() -> None:
    monitor = _monitor()
    monitor.update(_cluster_scan(0.50), dt_s=0.1)
    monitor.update(_cluster_scan(0.45), dt_s=0.1)

    confirmed = monitor.update(_cluster_scan(0.40), dt_s=0.1)

    assert confirmed.closing_rate_valid is True
    assert confirmed.approaching_confirmed is True
    assert confirmed.closing_rate_mps == pytest.approx(0.50)


def test_association_uses_circular_angle_and_inclusive_eight_degree_limit() -> None:
    accepted_monitor = _monitor()
    accepted_monitor.update(_cluster_scan(0.50, beam_index=358), dt_s=0.1)
    accepted = accepted_monitor.update(
        _cluster_scan(0.48, beam_index=6),
        dt_s=0.1,
    )

    rejected_monitor = _monitor()
    rejected_monitor.update(_cluster_scan(0.50, beam_index=358), dt_s=0.1)
    rejected = rejected_monitor.update(
        _cluster_scan(0.48, beam_index=7),
        dt_s=0.1,
    )

    assert accepted.threat_association_valid is True
    assert accepted.approaching_confirmation_elapsed_s == pytest.approx(0.1)
    assert rejected.threat_association_valid is False
    assert rejected.approaching_confirmation_elapsed_s == 0.0


def test_nonapproaching_sample_resets_confirmation_and_suppresses_rate() -> None:
    monitor = _monitor(approaching_epsilon_mps=1e-3)
    monitor.update(_cluster_scan(0.50), dt_s=0.1)
    pending = monitor.update(_cluster_scan(0.48), dt_s=0.1)
    stationary = monitor.update(_cluster_scan(0.47995), dt_s=0.1)
    receding = monitor.update(_cluster_scan(0.50), dt_s=0.1)

    assert pending.approaching_confirmation_elapsed_s == pytest.approx(0.1)
    assert stationary.closing_rate_valid is True
    assert stationary.approaching_confirmation_elapsed_s == 0.0
    assert stationary.closing_rate_mps is None
    assert stationary.ttc_s is None
    assert not stationary.approaching
    assert receding.closing_rate_valid is True
    assert receding.approaching_confirmation_elapsed_s == 0.0
    assert receding.closing_rate_mps is None


def test_critical_threshold_is_inclusive_and_independent_of_confirmation() -> None:
    assessment = _monitor(threshold=0.13).update(_scan(0.13), dt_s=0.1)

    assert assessment.critical
    assert assessment.consecutive_critical_frames == 1
    assert assessment.critical_threshold_m == pytest.approx(0.13)
    assert assessment.ttc_s is None
    assert not assessment.approaching


def test_critical_streak_remains_raw_minimum_based() -> None:
    monitor = _monitor()

    first = monitor.update(_scan(0.12, beam_index=4), dt_s=0.1)
    second = monitor.update(_scan(0.11, beam_index=4), dt_s=0.1)
    cleared = monitor.update(_scan(0.20, beam_index=4), dt_s=0.1)

    assert first.consecutive_critical_frames == 1
    assert second.consecutive_critical_frames == 2
    assert cleared.consecutive_critical_frames == 0


def test_assessment_audit_fields_are_grouped_and_strict() -> None:
    monitor = _monitor()
    first = monitor.update(_cluster_scan(0.50), dt_s=0.1)
    monitor.update(_cluster_scan(0.47), dt_s=0.1)
    confirmed = monitor.update(_cluster_scan(0.44), dt_s=0.1)

    with pytest.raises(ValueError, match="must be supplied together"):
        replace(first, ttc_valid=None)
    with pytest.raises(ValueError, match="configured confirmation time"):
        replace(confirmed, approaching_confirmation_elapsed_s=0.19)
    with pytest.raises(ValueError, match="suppress"):
        replace(first, closing_rate_mps=0.1)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_config_rejects_invalid_threshold(value: float) -> None:
    with pytest.raises(ValueError, match="critical_threshold_m"):
        CriticalProximityConfig(critical_threshold_m=value)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf"), True])
def test_factory_rejects_invalid_lidar_minimum(value: float) -> None:
    error_type = TypeError if value is True else ValueError
    with pytest.raises(error_type, match="lidar_min_range_m"):
        CriticalProximityConfig.from_lidar_min_range(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_beam_count", 4),
        ("threat_association_max_angle_rad", 0.0),
        ("threat_association_max_angle_rad", math.pi + 0.01),
        ("maximum_plausible_closing_rate_mps", 0.0),
        ("approaching_confirmation_s", 0.0),
    ],
)
def test_config_rejects_invalid_tracking_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(CriticalProximityConfig(0.13), **{field: value})


def test_config_rejects_invalid_margin_beam_count_and_epsilon() -> None:
    with pytest.raises(ValueError, match="margin_m"):
        CriticalProximityConfig.from_lidar_min_range(0.12, margin_m=-0.01)
    with pytest.raises(TypeError, match="minimum_beam_count"):
        CriticalProximityConfig(0.13, minimum_beam_count=True)
    with pytest.raises(ValueError, match="minimum_beam_count"):
        CriticalProximityConfig(0.13, minimum_beam_count=1)
    with pytest.raises(ValueError, match="approaching_epsilon_mps"):
        CriticalProximityConfig(0.13, approaching_epsilon_mps=-1.0)


@pytest.mark.parametrize(
    "scan",
    [
        [],
        [1.0] * 7,
        [[1.0] * 8],
        [1.0] * 7 + [float("nan")],
        [1.0] * 7 + [float("inf")],
        [1.0] * 7 + [0.0],
        [1.0] * 7 + [-0.1],
        ["bad"] * 8,
    ],
)
def test_monitor_rejects_malformed_or_nonphysical_scans(scan: object) -> None:
    with pytest.raises(ValueError, match="policy_order_lidar"):
        _monitor().update(scan, dt_s=0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("dt_s", [0.0, -0.1, float("nan"), float("inf"), True])
def test_monitor_rejects_invalid_time_step(dt_s: float) -> None:
    error_type = TypeError if dt_s is True else ValueError
    with pytest.raises(error_type, match="dt_s"):
        _monitor().update(_scan(0.5), dt_s=dt_s)


def test_invalid_update_does_not_mutate_temporal_state() -> None:
    monitor = _monitor(threshold=0.45)
    first = monitor.update(_cluster_scan(0.40), dt_s=0.1)
    assert first.consecutive_critical_frames == 1

    with pytest.raises(ValueError, match="dt_s"):
        monitor.update(_cluster_scan(0.36), dt_s=0.0)

    pending = monitor.update(_cluster_scan(0.36), dt_s=0.1)
    confirmed = monitor.update(_cluster_scan(0.32), dt_s=0.1)
    assert pending.consecutive_critical_frames == 2
    assert pending.closing_rate_mps is None
    assert pending.approaching_confirmation_elapsed_s == pytest.approx(0.1)
    assert confirmed.consecutive_critical_frames == 3
    assert confirmed.closing_rate_mps == pytest.approx(0.4)


def test_beam_count_change_requires_reset() -> None:
    monitor = _monitor()
    monitor.update(_scan(0.5, beam_count=8), dt_s=0.1)

    with pytest.raises(ValueError, match="beam count changed"):
        monitor.update(_scan(0.4, beam_count=10), dt_s=0.1)

    monitor.reset()
    assessment = monitor.update(_scan(0.4, beam_count=10), dt_s=0.1)
    assert assessment.beam_count == 10
    assert assessment.closing_rate_mps is None
    assert assessment.consecutive_critical_frames == 0
    assert assessment.threat_association_valid is False
