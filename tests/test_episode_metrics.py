from __future__ import annotations

import csv

import numpy as np
import pytest

from indoor_patrol_rl.evaluation import EpisodeAccumulator, write_episode_records


def step_info(
    *,
    step: int,
    action: tuple[float, float],
    clearance: float,
    success: bool = False,
    collision: bool = False,
    timeout: bool = False,
) -> dict[str, object]:
    return {
        "is_success": success,
        "collision": collision,
        "timeout": timeout,
        "clearance": clearance,
        "path_length": 1.5,
        "step_count": step,
        "elapsed_sim_time": step * 0.1,
        "executed_action": np.asarray(action, dtype=np.float32),
    }


def test_episode_metrics_compute_success_values() -> None:
    accumulator = EpisodeAccumulator(
        episode_id="validation-G0-5000-dwa",
        split="validation",
        condition_id="G0",
        scenario_seed=5000,
        planner="dwa",
        reference_path_length=1.0,
    )
    accumulator.update(
        reward=1.0,
        info=step_info(step=1, action=(-0.5, 0.0), clearance=0.30),
        inference_latency_s=0.001,
    )
    accumulator.update(
        reward=2.0,
        info=step_info(step=2, action=(0.5, 0.0), clearance=0.10, success=True),
        inference_latency_s=0.003,
    )

    record = accumulator.finalize()

    assert record.outcome == "success"
    assert record.episode_return == 3.0
    assert record.completion_time_s == pytest.approx(0.2)
    assert record.path_length_ratio == pytest.approx(1.5)
    assert record.minimum_clearance_m == pytest.approx(0.10)
    assert record.near_collision_ratio == pytest.approx(0.5)
    assert record.action_delta_sq_mean == pytest.approx(1.0)
    assert record.inference_latency_p50_ms == pytest.approx(2.0)


def test_collision_has_priority_over_simultaneous_success() -> None:
    accumulator = EpisodeAccumulator(
        episode_id="collision",
        split="validation",
        condition_id="G0",
        scenario_seed=5001,
        planner="reactive",
    )
    accumulator.update(
        reward=-30.0,
        info=step_info(
            step=1,
            action=(-1.0, 0.0),
            clearance=0.0,
            success=True,
            collision=True,
        ),
        inference_latency_s=0.0,
    )

    record = accumulator.finalize()

    assert record.outcome == "collision"
    assert record.collision and not record.is_success
    assert record.completion_time_s is None
    assert record.path_length_ratio is None


def test_episode_records_round_trip_through_csv(tmp_path) -> None:
    accumulator = EpisodeAccumulator(
        episode_id="timeout",
        split="validation",
        condition_id="G0",
        scenario_seed=5002,
        planner="dwa",
    )
    accumulator.update(
        reward=-1.0,
        info=step_info(
            step=3,
            action=(-1.0, 0.0),
            clearance=0.4,
            timeout=True,
        ),
        inference_latency_s=0.002,
    )
    output = tmp_path / "episodes.csv"

    write_episode_records(output, [accumulator.finalize()])

    with output.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["outcome"] == "timeout"
    assert rows[0]["scenario_seed"] == "5002"
