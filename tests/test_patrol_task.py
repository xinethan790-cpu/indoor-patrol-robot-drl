from __future__ import annotations

import math
from pathlib import Path

import pytest

from indoor_patrol_rl.application import (
    PatrolTaskCheckpoint,
    PatrolTaskMachine,
    PatrolTaskState,
    validate_checkpoint_layout,
)
from indoor_patrol_rl.inspection import EnvironmentalInspection, load_inspection_config


def _four_points() -> list[PatrolTaskCheckpoint]:
    return [
        PatrolTaskCheckpoint("P1", (0.0, 0.0)),
        PatrolTaskCheckpoint("P2", (1.0, 0.0)),
        PatrolTaskCheckpoint("P3", (1.0, 1.0)),
        PatrolTaskCheckpoint("P4", (0.0, 1.0)),
    ]


def test_four_checkpoints_advance_in_order_only_after_successful_sampling() -> None:
    machine = PatrolTaskMachine(
        _four_points(), arrival_radius_m=0.2, dwell_time_s=1.0, timeout_s=60.0
    )
    now = 0.0
    control_step = 0

    for index, checkpoint in enumerate(_four_points()):
        entered = machine.update(
            position=checkpoint.position,
            collision=False,
            simulation_time_s=now,
            control_step=control_step,
        )
        assert entered.state is PatrolTaskState.DWELLING
        assert entered.checkpoint_id == checkpoint.checkpoint_id
        assert not entered.sampling_requested

        now += 0.99
        control_step += 1
        waiting = machine.update(
            position=checkpoint.position,
            collision=False,
            simulation_time_s=now,
            control_step=control_step,
        )
        assert waiting.state is PatrolTaskState.DWELLING
        assert waiting.dwell_elapsed_s == pytest.approx(0.99)

        now += 0.01
        control_step += 1
        sampling = machine.update(
            position=checkpoint.position,
            collision=False,
            simulation_time_s=now,
            control_step=control_step,
        )
        assert sampling.state is PatrolTaskState.SAMPLING
        assert sampling.sampling_requested
        assert checkpoint.checkpoint_id not in sampling.completed_checkpoint_ids

        # A normal controller step cannot accidentally complete the checkpoint.
        still_sampling = machine.update(
            position=checkpoint.position,
            collision=False,
            simulation_time_s=now,
            control_step=control_step,
        )
        assert still_sampling.state is PatrolTaskState.SAMPLING
        assert not still_sampling.sampling_requested
        assert checkpoint.checkpoint_id not in still_sampling.completed_checkpoint_ids

        acknowledged = machine.record_sampling_result(
            success=True,
            simulation_time_s=now,
            control_step=control_step,
        )
        assert acknowledged.checkpoint_completed_id == checkpoint.checkpoint_id
        assert acknowledged.completed_checkpoint_ids == tuple(
            item.checkpoint_id for item in _four_points()[: index + 1]
        )
        if index < 3:
            assert acknowledged.state is PatrolTaskState.NAVIGATING
            assert acknowledged.checkpoint_id == _four_points()[index + 1].checkpoint_id
        else:
            assert acknowledged.state is PatrolTaskState.COMPLETED
            assert acknowledged.checkpoint_id is None
            assert acknowledged.completion_ratio == 1.0

        now += 0.25
        control_step += 1


def test_dwell_must_be_continuous_inside_arrival_radius() -> None:
    machine = PatrolTaskMachine(
        [PatrolTaskCheckpoint("P1", (0.0, 0.0))],
        arrival_radius_m=0.2,
        dwell_time_s=1.0,
        timeout_s=10.0,
    )

    assert machine.update(
        position=(0.0, 0.0), collision=False, simulation_time_s=0.0, control_step=0
    ).state is PatrolTaskState.DWELLING
    left = machine.update(
        position=(0.3, 0.0), collision=False, simulation_time_s=0.8, control_step=1
    )
    assert left.state is PatrolTaskState.NAVIGATING
    assert left.dwell_elapsed_s == 0.0

    assert machine.update(
        position=(0.0, 0.0), collision=False, simulation_time_s=1.0, control_step=2
    ).state is PatrolTaskState.DWELLING
    too_soon = machine.update(
        position=(0.0, 0.0), collision=False, simulation_time_s=1.8, control_step=3
    )
    assert too_soon.state is PatrolTaskState.DWELLING
    requested = machine.update(
        position=(0.0, 0.0), collision=False, simulation_time_s=2.0, control_step=4
    )
    assert requested.state is PatrolTaskState.SAMPLING
    assert requested.sampling_requested


def test_collision_enters_failed_state_without_completing_current_checkpoint() -> None:
    machine = PatrolTaskMachine(_four_points(), timeout_s=30.0)

    result = machine.update(
        position=(-2.0, 0.0), collision=True, simulation_time_s=0.0, control_step=0
    )

    assert result.state is PatrolTaskState.FAILED
    assert result.failure_reason == "collision"
    assert result.completed_checkpoint_ids == ()
    assert result.completion_ratio == 0.0
    assert result.checkpoint_id == "P1"
    assert machine.current_checkpoint == _four_points()[0]


def test_total_task_timeout_is_measured_from_first_controller_update() -> None:
    machine = PatrolTaskMachine(_four_points(), timeout_s=5.0)
    machine.update(
        position=(-2.0, 0.0), collision=False, simulation_time_s=10.0, control_step=1
    )

    result = machine.update(
        position=(-2.0, 0.0), collision=False, simulation_time_s=15.0, control_step=2
    )

    assert result.state is PatrolTaskState.FAILED
    assert result.failure_reason == "timeout"
    assert result.completed_checkpoint_ids == ()


def test_failed_sampling_does_not_complete_checkpoint() -> None:
    machine = PatrolTaskMachine(
        [PatrolTaskCheckpoint("P1", (0.0, 0.0))], dwell_time_s=0.5, timeout_s=10.0
    )
    machine.update(
        position=(0.0, 0.0), collision=False, simulation_time_s=0.0, control_step=0
    )
    request = machine.update(
        position=(0.0, 0.0), collision=False, simulation_time_s=0.5, control_step=1
    )
    assert request.sampling_requested

    result = machine.record_sampling_result(
        success=False,
        failure_reason="inspection_bundle_write_failed",
        simulation_time_s=0.5,
        control_step=1,
    )

    assert result.state is PatrolTaskState.FAILED
    assert result.failure_reason == "inspection_bundle_write_failed"
    assert result.completed_checkpoint_ids == ()
    assert result.completion_ratio == 0.0


def test_sampling_result_is_rejected_before_sampling_state() -> None:
    machine = PatrolTaskMachine([PatrolTaskCheckpoint("P1", (0.0, 0.0))])

    with pytest.raises(RuntimeError, match="only while state is SAMPLING"):
        machine.record_sampling_result(
            success=True, simulation_time_s=0.0, control_step=0
        )


def test_robot_cannot_skip_a_checkpoint_by_reaching_a_later_position() -> None:
    machine = PatrolTaskMachine(_four_points(), arrival_radius_m=0.2)

    result = machine.update(
        position=(1.0, 1.0), collision=False, simulation_time_s=0.0, control_step=0
    )

    assert result.state is PatrolTaskState.NAVIGATING
    assert result.checkpoint_id == "P1"
    assert result.completed_checkpoint_ids == ()


def test_duplicate_checkpoint_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        PatrolTaskMachine(
            [
                PatrolTaskCheckpoint("P1", (0.0, 0.0)),
                PatrolTaskCheckpoint("P1", (1.0, 0.0)),
            ]
        )


@pytest.mark.parametrize(
    "position",
    [
        (math.nan, 0.0),
        (0.0, math.inf),
        (0.0, -math.inf),
    ],
)
def test_non_finite_checkpoint_coordinates_are_rejected(
    position: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        PatrolTaskCheckpoint("P1", position)


def test_controller_input_rejects_non_finite_position_and_backward_clock() -> None:
    machine = PatrolTaskMachine([PatrolTaskCheckpoint("P1", (0.0, 0.0))])
    with pytest.raises(ValueError, match="finite"):
        machine.update(
            position=(math.nan, 0.0),
            collision=False,
            simulation_time_s=0.0,
            control_step=0,
        )

    machine.update(
        position=(1.0, 0.0), collision=False, simulation_time_s=1.0, control_step=1
    )
    with pytest.raises(ValueError, match="monotonic"):
        machine.update(
            position=(1.0, 0.0),
            collision=False,
            simulation_time_s=0.5,
            control_step=2,
        )


def test_checkpoint_layout_accepts_matching_world_markers() -> None:
    validate_checkpoint_layout(
        _four_points(),
        {checkpoint.checkpoint_id: checkpoint.position for checkpoint in _four_points()},
    )


def test_checkpoint_layout_rejects_missing_and_drifted_world_markers() -> None:
    positions = {
        checkpoint.checkpoint_id: checkpoint.position for checkpoint in _four_points()
    }
    positions.pop("P4")
    with pytest.raises(ValueError, match="missing=.*P4"):
        validate_checkpoint_layout(_four_points(), positions)

    positions["P4"] = (0.02, 1.0)
    with pytest.raises(ValueError, match="differs from task config"):
        validate_checkpoint_layout(_four_points(), positions)


def test_production_four_point_task_generates_expected_inspection_chain() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_inspection_config(
        repository_root / "configs/inspection/environmental_patrol_v1.yaml"
    )
    route = [
        PatrolTaskCheckpoint("P1", (1.60, 0.0)),
        PatrolTaskCheckpoint("P2", (1.60, 1.15)),
        PatrolTaskCheckpoint("P3", (-1.60, 1.15)),
        PatrolTaskCheckpoint("P4", (-1.60, -1.15)),
    ]
    machine = PatrolTaskMachine(
        route,
        arrival_radius_m=0.22,
        dwell_time_s=(config.samples_per_visit - 1) * config.sample_interval_s,
        timeout_s=30.0,
    )
    inspection = EnvironmentalInspection(config, run_id="application-flow", scenario_seed=7000)
    simulation_time_s = 0.0
    control_step = 0

    for visit_index, checkpoint in enumerate(route, 1):
        entered = machine.update(
            position=checkpoint.position,
            collision=False,
            simulation_time_s=simulation_time_s,
            control_step=control_step,
        )
        assert entered.state is PatrolTaskState.DWELLING
        physics_steps = 0
        while not entered.sampling_requested:
            simulation_time_s += 0.032
            physics_steps += 1
            entered = machine.update(
                position=checkpoint.position,
                collision=False,
                simulation_time_s=simulation_time_s,
                control_step=control_step,
            )
        assert physics_steps == 32
        inspection.sample_visit(
            checkpoint.checkpoint_id,
            visit_index,
            control_step,
            simulation_time_s - inspection.sampling_duration_s,
            checkpoint.position,
        )
        result = machine.record_sampling_result(
            success=True,
            simulation_time_s=simulation_time_s,
            control_step=control_step,
        )
        assert result.checkpoint_completed_id == checkpoint.checkpoint_id
        control_step += 1

    assert machine.state is PatrolTaskState.COMPLETED
    assert len(inspection.visits) == 4
    assert len(inspection.samples) == 36
    assert [(event.event_type, event.rule_id) for event in inspection.events] == [
        ("raised", "high_temperature"),
        ("cleared", "high_temperature"),
        ("raised", "smoke_detected"),
        ("raised", "high_carbon_monoxide"),
        ("cleared", "smoke_detected"),
        ("cleared", "high_carbon_monoxide"),
    ]
    assert inspection.active_alarm_rule_ids == ()
