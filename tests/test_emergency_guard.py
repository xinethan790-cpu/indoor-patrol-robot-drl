from __future__ import annotations

import math
from dataclasses import replace

import pytest

from indoor_patrol_rl.application import (
    PATROL_TTC_ESCAPE_GUARD_ID,
    PATROL_TTC_ESCAPE_GUARD_SCHEMA_VERSION,
    UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE,
    CriticalProximityAssessment,
    PatrolTTCEscapeGuard,
    PatrolTTCEscapeGuardConfig,
)


def _angle(beam_index: int, beam_count: int = 8) -> float:
    raw = beam_index * (2.0 * math.pi / beam_count)
    return (raw + math.pi) % (2.0 * math.pi) - math.pi


def _assessment(
    distance_m: float,
    *,
    beam_index: int = 0,
    beam_count: int = 8,
    closing_rate_mps: float | None = None,
    ttc_s: float | None = None,
    threshold_m: float = 0.13,
) -> CriticalProximityAssessment:
    """Build a legacy-compatible assessment to test the guard API boundary."""

    critical = distance_m <= threshold_m
    if critical:
        ttc_s = 0.0
    return CriticalProximityAssessment(
        minimum_distance_m=distance_m,
        minimum_beam_index=beam_index,
        minimum_angle_rad=_angle(beam_index, beam_count),
        beam_count=beam_count,
        critical_threshold_m=threshold_m,
        critical=critical,
        consecutive_critical_frames=1 if critical else 0,
        closing_rate_mps=closing_rate_mps,
        ttc_s=ttc_s,
    )


def test_default_config_carries_tuned_patrol_values() -> None:
    config = PatrolTTCEscapeGuardConfig()

    assert config.ttc_activation_s == pytest.approx(2.5)
    assert config.distance_activation_m == pytest.approx(0.30)
    assert config.ttc_activation_max_distance_m == pytest.approx(0.45)
    assert config.release_distance_m == pytest.approx(0.42)
    assert config.release_confirmation_s == pytest.approx(0.20)
    assert config.rear_escape_front_clearance_m == pytest.approx(0.40)
    assert config.front_threat_rear_clearance_m == pytest.approx(0.40)
    assert config.reverse_linear_mps == pytest.approx(-0.20)
    assert config.forward_linear_mps == pytest.approx(0.20)
    assert config.forward_watchdog_s == pytest.approx(2.00)
    assert config.forward_maximum_regression_m == pytest.approx(0.03)
    assert config.maximum_active_duration_s == pytest.approx(5.5)


def test_inactive_decision_carries_versioned_unqualified_patrol_identity() -> None:
    guard = PatrolTTCEscapeGuard()

    decision = guard.update(
        _assessment(0.80),
        front_clearance_m=1.0,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert not decision.active
    assert decision.mode == "inactive"
    assert decision.target_linear_mps == 0.0
    assert decision.target_angular_radps == 0.0
    assert decision.activation_reason is None
    assert decision.active_duration_s == 0.0
    assert not decision.unrecoverable
    assert decision.guard_id == PATROL_TTC_ESCAPE_GUARD_ID
    assert decision.schema_version == PATROL_TTC_ESCAPE_GUARD_SCHEMA_VERSION == 1
    assert decision.evidence_scope == UNQUALIFIED_PATROL_TTC_ESCAPE_GUARD_SCOPE
    assert "unqualified" in decision.evidence_scope
    assert "r2a" not in decision.guard_id.casefold()


def test_ttc_activation_against_front_threat_commands_reverse() -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.44, closing_rate_mps=0.20, ttc_s=1.55),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.active
    assert decision.mode == "reverse"
    assert decision.target_linear_mps == pytest.approx(-0.20)
    assert decision.target_angular_radps == 0.0
    assert decision.activation_reason == "ttc_threshold"
    assert decision.active_duration_s == pytest.approx(0.1)


def test_ttc_is_not_usable_beyond_raw_minimum_distance_cap() -> None:
    guard = PatrolTTCEscapeGuard()

    capped = guard.update(
        _assessment(0.451, closing_rate_mps=0.20, ttc_s=1.0),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert capped.mode == "inactive"


def test_ttc_distance_cap_is_inclusive() -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.45, closing_rate_mps=0.20, ttc_s=1.6),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.mode == "reverse"
    assert decision.activation_reason == "ttc_threshold"


@pytest.mark.parametrize("beam_index", [0, 2, 6])
def test_distance_activation_front_hemisphere_is_inclusive(beam_index: int) -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.30, beam_index=beam_index),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.mode == "reverse"
    assert decision.activation_reason == "distance_threshold"


def test_simultaneous_distance_and_ttc_trigger_is_audited() -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.12, closing_rate_mps=0.10),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.mode == "reverse"
    assert decision.activation_reason == "distance_and_ttc"


def test_front_threat_rotates_instead_of_reversing_when_rear_is_blocked() -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.25),
        front_clearance_m=0.60,
        rear_clearance_m=0.399,
        dt_s=0.1,
    )

    assert decision.mode == "rotate"
    assert decision.target_linear_mps == 0.0
    assert decision.target_angular_radps == pytest.approx(-0.80)


def test_rear_threat_crossing_front_hemisphere_stays_forward() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    crossed = guard.update(
        _assessment(0.25, beam_index=1),
        front_clearance_m=0.40,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert activated.mode == crossed.mode == "forward"
    assert crossed.target_linear_mps == pytest.approx(0.20)


def test_front_threat_crossing_rear_hemisphere_stays_reverse() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.25, beam_index=1),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    crossed = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.40,
        dt_s=0.1,
    )

    assert activated.mode == crossed.mode == "reverse"
    assert crossed.target_linear_mps == pytest.approx(-0.20)


def test_latched_forward_falls_back_permanently_when_front_is_blocked() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    blocked = guard.update(
        _assessment(0.25, beam_index=1),
        front_clearance_m=0.399,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    clearance_restored = guard.update(
        _assessment(0.25, beam_index=7),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert activated.mode == "forward"
    assert blocked.mode == clearance_restored.mode == "rotate"
    assert blocked.target_angular_radps == pytest.approx(-0.80)
    assert clearance_restored.target_angular_radps == pytest.approx(-0.80)


def test_latched_reverse_falls_back_permanently_when_rear_is_blocked() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.25, beam_index=1),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    blocked = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.399,
        dt_s=0.1,
    )
    clearance_restored = guard.update(
        _assessment(0.25, beam_index=5),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert activated.mode == "reverse"
    assert blocked.mode == clearance_restored.mode == "rotate"
    assert blocked.target_angular_radps == pytest.approx(-0.80)
    assert clearance_restored.target_angular_radps == pytest.approx(-0.80)


@pytest.mark.parametrize("beam_index", [3, 5])
def test_rear_threat_uses_faster_forward_escape_when_front_is_clear(
    beam_index: int,
) -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.25, beam_index=beam_index),
        front_clearance_m=0.40,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.mode == "forward"
    assert decision.target_linear_mps == pytest.approx(0.20)
    assert decision.target_angular_radps == 0.0


@pytest.mark.parametrize(
    ("beam_index", "expected_angular_radps"),
    [(3, -0.80), (5, 0.80)],
)
def test_blocked_front_rotates_away_from_rear_threat(
    beam_index: int,
    expected_angular_radps: float,
) -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.25, beam_index=beam_index),
        front_clearance_m=0.399,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.mode == "rotate"
    assert decision.target_linear_mps == 0.0
    assert decision.target_angular_radps == pytest.approx(expected_angular_radps)


def test_rotation_direction_is_latched_across_pi_until_release() -> None:
    guard = PatrolTTCEscapeGuard()
    first = guard.update(
        _assessment(0.25, beam_index=179, beam_count=360),
        front_clearance_m=0.399,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    crossed_pi = guard.update(
        _assessment(0.25, beam_index=181, beam_count=360),
        front_clearance_m=0.399,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert first.mode == crossed_pi.mode == "rotate"
    assert first.target_angular_radps == pytest.approx(-0.80)
    assert crossed_pi.target_angular_radps == pytest.approx(-0.80)

    released = guard.update(
        _assessment(0.50, beam_index=181, beam_count=360),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.20,
    )
    reactivated = guard.update(
        _assessment(0.25, beam_index=181, beam_count=360),
        front_clearance_m=0.399,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert released.mode == "inactive"
    assert reactivated.mode == "rotate"
    assert reactivated.target_angular_radps == pytest.approx(0.80)


def test_static_wall_requires_two_tenths_safe_clearance_before_release() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.29, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    first_safe = guard.update(
        _assessment(0.42, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    released = guard.update(
        _assessment(0.42, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert activated.active_duration_s == pytest.approx(0.1)
    assert first_safe.active
    assert first_safe.active_duration_s == pytest.approx(0.2)
    assert released.mode == "inactive"
    assert not guard.active


def test_forward_release_uses_front_clearance_despite_close_side_obstacle() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    first_safe = guard.update(
        _assessment(0.35, beam_index=2),
        front_clearance_m=0.43,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    released = guard.update(
        _assessment(0.35, beam_index=2),
        front_clearance_m=0.43,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert activated.mode == first_safe.mode == "forward"
    assert first_safe.active
    assert released.mode == "inactive"
    assert not guard.active


def test_release_confirmation_resets_when_threat_approaches_again() -> None:
    guard = PatrolTTCEscapeGuard()
    guard.update(_assessment(0.29), front_clearance_m=0.60, rear_clearance_m=0.60, dt_s=0.1)
    first_safe = guard.update(
        _assessment(0.43),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    approaching = guard.update(
        _assessment(0.43, closing_rate_mps=0.10, ttc_s=3.0),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    safe_again = guard.update(
        _assessment(0.43),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    released = guard.update(
        _assessment(0.43),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert first_safe.active
    assert approaching.active
    assert safe_again.active
    assert released.mode == "inactive"


def test_below_release_distance_never_accumulates_release_time() -> None:
    guard = PatrolTTCEscapeGuard()
    guard.update(_assessment(0.29), front_clearance_m=0.60, rear_clearance_m=0.60, dt_s=0.1)

    holding = guard.update(
        _assessment(0.419),
        front_clearance_m=0.60,
        rear_clearance_m=0.419,
        dt_s=1.0,
    )

    assert holding.active
    assert holding.mode == "reverse"


def test_safe_release_wins_at_timeout_boundary() -> None:
    guard = PatrolTTCEscapeGuard(PatrolTTCEscapeGuardConfig(maximum_active_duration_s=4.0))
    guard.update(_assessment(0.25), front_clearance_m=0.6, rear_clearance_m=0.60, dt_s=0.1)

    released = guard.update(
        _assessment(0.60),
        front_clearance_m=0.6,
        rear_clearance_m=0.60,
        dt_s=4.0,
    )

    assert released.mode == "inactive"
    assert not released.unrecoverable


def test_timeout_counts_first_active_step_and_latches_until_reset() -> None:
    guard = PatrolTTCEscapeGuard(PatrolTTCEscapeGuardConfig(maximum_active_duration_s=4.0))
    unsafe = _assessment(0.40)
    first = guard.update(_assessment(0.25), front_clearance_m=0.6, rear_clearance_m=0.60, dt_s=0.1)
    holding = guard.update(unsafe, front_clearance_m=0.6, rear_clearance_m=0.40, dt_s=3.8)
    timed_out = guard.update(unsafe, front_clearance_m=0.6, rear_clearance_m=0.40, dt_s=0.1)
    still_latched = guard.update(
        _assessment(0.80),
        front_clearance_m=1.0,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert first.active_duration_s == pytest.approx(0.1)
    assert holding.active
    assert holding.active_duration_s == pytest.approx(3.9)
    assert timed_out.active
    assert timed_out.mode == "unrecoverable"
    assert timed_out.unrecoverable
    assert timed_out.target_linear_mps == 0.0
    assert timed_out.target_angular_radps == 0.0
    assert timed_out.active_duration_s == pytest.approx(4.0)
    assert still_latched.mode == "unrecoverable"
    assert guard.active and guard.unrecoverable

    guard.reset()
    recovered = guard.update(
        _assessment(0.80),
        front_clearance_m=1.0,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    assert recovered.mode == "inactive"
    assert not guard.active and not guard.unrecoverable


def test_first_active_step_can_reach_fail_closed_timeout() -> None:
    decision = PatrolTTCEscapeGuard(
        PatrolTTCEscapeGuardConfig(maximum_active_duration_s=4.0)
    ).update(
        _assessment(0.25),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=4.0,
    )

    assert decision.mode == "unrecoverable"
    assert decision.active_duration_s == pytest.approx(4.0)


def test_forward_watchdog_keeps_forward_when_distance_is_stable_for_two_seconds() -> None:
    guard = PatrolTTCEscapeGuard()
    activated = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    pending = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=1.0,
    )
    stable = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=1.0,
    )

    assert activated.mode == pending.mode == stable.mode == "forward"
    assert stable.target_linear_mps == pytest.approx(0.20)


def test_forward_watchdog_rotates_after_more_than_three_centimetres_regression() -> None:
    guard = PatrolTTCEscapeGuard()
    guard.update(
        _assessment(0.30, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    regressed = guard.update(
        _assessment(0.26, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=2.0,
    )

    assert regressed.mode == "rotate"
    assert regressed.target_angular_radps == pytest.approx(-0.80)


def test_seed_7001_replay_releases_after_front_clearance_confirmation() -> None:
    guard = PatrolTTCEscapeGuard()
    activation = guard.update(
        _assessment(
            0.3175089359283447,
            beam_index=110,
            beam_count=360,
            closing_rate_mps=0.17754919826984406,
            ttc_s=1.1227742621248202,
        ),
        front_clearance_m=0.44836708903312683,
        rear_clearance_m=0.60,
        dt_s=0.128,
    )
    first_safe = guard.update(
        _assessment(
            0.30551716685295105,
            beam_index=114,
            beam_count=360,
        ),
        front_clearance_m=0.4541640281677246,
        rear_clearance_m=0.60,
        dt_s=0.128,
    )
    released = guard.update(
        _assessment(
            0.2967277765274048,
            beam_index=119,
            beam_count=360,
        ),
        front_clearance_m=0.4519695043563843,
        rear_clearance_m=0.60,
        dt_s=0.128,
    )

    assert activation.mode == first_safe.mode == "forward"
    assert released.mode == "inactive"


def test_forward_watchdog_continues_when_distance_improves_by_three_centimetres() -> None:
    guard = PatrolTTCEscapeGuard(PatrolTTCEscapeGuardConfig(forward_watchdog_s=0.50))
    guard.update(
        _assessment(0.25, beam_index=5),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    guard.update(
        _assessment(0.27, beam_index=5),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.25,
    )
    effective = guard.update(
        _assessment(0.28, beam_index=5),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.25,
    )

    assert effective.mode == "forward"
    assert effective.target_linear_mps == pytest.approx(0.20)


def test_forward_watchdog_latches_rotate_when_assessment_is_still_closing() -> None:
    guard = PatrolTTCEscapeGuard(PatrolTTCEscapeGuardConfig(forward_watchdog_s=0.50))
    guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    guard.update(
        _assessment(0.27, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.25,
    )
    failed = guard.update(
        _assessment(
            0.29,
            beam_index=3,
            closing_rate_mps=0.10,
            ttc_s=1.6,
        ),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.25,
    )

    assert failed.mode == "rotate"


def test_forward_watchdog_latch_clears_only_after_release() -> None:
    guard = PatrolTTCEscapeGuard(PatrolTTCEscapeGuardConfig(forward_watchdog_s=0.50))
    guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )
    guard.update(
        _assessment(0.20, beam_index=3),
        front_clearance_m=0.41,
        rear_clearance_m=0.60,
        dt_s=0.5,
    )
    released = guard.update(
        _assessment(0.45, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.2,
    )
    reactivated = guard.update(
        _assessment(0.25, beam_index=3),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert released.mode == "inactive"
    assert reactivated.mode == "forward"


def test_ttc_above_activation_threshold_does_not_activate() -> None:
    decision = PatrolTTCEscapeGuard().update(
        _assessment(0.44, closing_rate_mps=0.10, ttc_s=2.5001),
        front_clearance_m=0.60,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    assert decision.mode == "inactive"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ttc_activation_s", 0.0),
        ("distance_activation_m", 0.0),
        ("ttc_activation_max_distance_m", 0.30),
        ("release_distance_m", 0.30),
        ("release_confirmation_s", 0.0),
        ("rear_escape_front_clearance_m", 0.0),
        ("front_threat_rear_clearance_m", 0.0),
        ("reverse_linear_mps", 0.0),
        ("forward_linear_mps", 0.0),
        ("rotation_angular_radps", 0.0),
        ("forward_watchdog_s", 0.0),
        ("forward_maximum_regression_m", 0.0),
        ("maximum_active_duration_s", 0.0),
        ("maximum_active_duration_s", math.inf),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(PatrolTTCEscapeGuardConfig(), **{field: value})


@pytest.mark.parametrize("front_clearance_m", [-0.1, math.nan, math.inf, True])
def test_invalid_front_clearance_does_not_mutate_state(
    front_clearance_m: float,
) -> None:
    guard = PatrolTTCEscapeGuard()
    guard.update(_assessment(0.25), front_clearance_m=0.6, rear_clearance_m=0.60, dt_s=0.1)

    with pytest.raises((TypeError, ValueError), match="front_clearance_m"):
        guard.update(
            _assessment(0.40),
            front_clearance_m=front_clearance_m,
            rear_clearance_m=0.40,
            dt_s=0.25,
        )

    decision = guard.update(
        _assessment(0.40),
        front_clearance_m=0.6,
        rear_clearance_m=0.40,
        dt_s=0.25,
    )
    assert decision.active_duration_s == pytest.approx(0.35)


@pytest.mark.parametrize("rear_clearance_m", [-0.1, math.nan, math.inf, True])
def test_invalid_rear_clearance_does_not_mutate_state(
    rear_clearance_m: float,
) -> None:
    guard = PatrolTTCEscapeGuard()
    guard.update(
        _assessment(0.25),
        front_clearance_m=0.6,
        rear_clearance_m=0.60,
        dt_s=0.1,
    )

    with pytest.raises((TypeError, ValueError), match="rear_clearance_m"):
        guard.update(
            _assessment(0.40),
            front_clearance_m=0.6,
            rear_clearance_m=rear_clearance_m,
            dt_s=0.25,
        )

    decision = guard.update(
        _assessment(0.40),
        front_clearance_m=0.6,
        rear_clearance_m=0.40,
        dt_s=0.25,
    )
    assert decision.active_duration_s == pytest.approx(0.35)


@pytest.mark.parametrize("dt_s", [0.0, -0.1, math.nan, math.inf, True])
def test_invalid_time_step_is_rejected(dt_s: float) -> None:
    with pytest.raises((TypeError, ValueError), match="dt_s"):
        PatrolTTCEscapeGuard().update(
            _assessment(0.25),
            front_clearance_m=0.6,
            rear_clearance_m=0.60,
            dt_s=dt_s,
        )


def test_guard_rejects_wrong_config_and_assessment_types() -> None:
    with pytest.raises(TypeError, match="config"):
        PatrolTTCEscapeGuard(config=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="assessment"):
        PatrolTTCEscapeGuard().update(
            object(),  # type: ignore[arg-type]
            front_clearance_m=0.6,
            rear_clearance_m=0.60,
            dt_s=0.1,
        )
