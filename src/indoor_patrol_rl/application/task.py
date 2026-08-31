"""Webots-independent state machine for a multi-checkpoint patrol task.

The navigation controller calls :meth:`PatrolTaskMachine.update` once per
simulation step.  Reaching a checkpoint does not complete it: the robot must
remain inside the arrival radius for the configured dwell time, then complete
the requested application-layer inspection successfully.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real


class PatrolTaskState(StrEnum):
    """Lifecycle states of one ordered patrol task."""

    NAVIGATING = "navigating"
    DWELLING = "dwelling"
    SAMPLING = "sampling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PatrolTaskCheckpoint:
    """One uniquely identified target in the required visit order."""

    checkpoint_id: str
    position: tuple[float, float]

    def __post_init__(self) -> None:
        checkpoint_id = _non_empty_text(self.checkpoint_id, "checkpoint_id")
        position = _finite_point(self.position, f"checkpoint {checkpoint_id!r} position")
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class PatrolTaskUpdate:
    """Snapshot returned after a simulation step or a sampling result.

    ``sampling_requested`` is an edge-triggered flag.  It is true exactly when
    the update enters ``SAMPLING``; the caller should invoke its
    ``EnvironmentalInspection`` instance and then call
    :meth:`PatrolTaskMachine.record_sampling_result`.
    """

    state: PatrolTaskState
    checkpoint_id: str | None
    checkpoint_index: int | None
    completed_checkpoint_ids: tuple[str, ...]
    completion_ratio: float
    sampling_requested: bool
    checkpoint_completed_id: str | None
    dwell_elapsed_s: float
    failure_reason: str | None


class PatrolTaskMachine:
    """Advance a strict, ordered patrol workflow from controller observations."""

    def __init__(
        self,
        checkpoints: Iterable[PatrolTaskCheckpoint],
        *,
        arrival_radius_m: float = 0.35,
        dwell_time_s: float = 1.0,
        timeout_s: float = 300.0,
    ) -> None:
        self._checkpoints = tuple(checkpoints)
        if not self._checkpoints:
            raise ValueError("checkpoints must contain at least one checkpoint.")
        if not all(isinstance(item, PatrolTaskCheckpoint) for item in self._checkpoints):
            raise TypeError("checkpoints must contain only PatrolTaskCheckpoint values.")

        checkpoint_ids = [item.checkpoint_id for item in self._checkpoints]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ValueError("checkpoint_id values must be unique.")

        self.arrival_radius_m = _finite_number(arrival_radius_m, "arrival_radius_m")
        if self.arrival_radius_m <= 0.0:
            raise ValueError("arrival_radius_m must be positive.")
        self.dwell_time_s = _finite_number(dwell_time_s, "dwell_time_s")
        if self.dwell_time_s < 0.0:
            raise ValueError("dwell_time_s must be non-negative.")
        self.timeout_s = _finite_number(timeout_s, "timeout_s")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")

        self._state = PatrolTaskState.NAVIGATING
        self._checkpoint_index = 0
        self._completed_checkpoint_ids: list[str] = []
        self._dwell_started_at_s: float | None = None
        self._started_at_s: float | None = None
        self._last_simulation_time_s: float | None = None
        self._last_control_step: int | None = None
        self._failure_reason: str | None = None

    @property
    def state(self) -> PatrolTaskState:
        return self._state

    @property
    def checkpoints(self) -> tuple[PatrolTaskCheckpoint, ...]:
        return self._checkpoints

    @property
    def current_checkpoint(self) -> PatrolTaskCheckpoint | None:
        if self._state is PatrolTaskState.COMPLETED:
            return None
        return self._checkpoints[self._checkpoint_index]

    @property
    def completed_checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(self._completed_checkpoint_ids)

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def update(
        self,
        *,
        position: Sequence[Real],
        collision: bool,
        simulation_time_s: float,
        control_step: int,
    ) -> PatrolTaskUpdate:
        """Consume one controller step and return the resulting task snapshot."""

        current_position = _finite_point(position, "position")
        if not isinstance(collision, bool):
            raise TypeError("collision must be a bool.")
        now = self._accept_clock(simulation_time_s, control_step)

        if self._state in {PatrolTaskState.COMPLETED, PatrolTaskState.FAILED}:
            return self._snapshot()
        if collision:
            return self._fail("collision")
        if self._timed_out(now):
            return self._fail("timeout")
        if self._state is PatrolTaskState.SAMPLING:
            return self._snapshot()

        checkpoint = self._checkpoints[self._checkpoint_index]
        inside_arrival_radius = (
            math.dist(current_position, checkpoint.position) <= self.arrival_radius_m
        )
        if self._state is PatrolTaskState.NAVIGATING:
            if inside_arrival_radius:
                self._state = PatrolTaskState.DWELLING
                self._dwell_started_at_s = now
            return self._snapshot()

        # DWELLING requires continuous occupancy of the arrival radius.  If the
        # robot drifts away, the full dwell interval must be earned again.
        if not inside_arrival_radius:
            self._state = PatrolTaskState.NAVIGATING
            self._dwell_started_at_s = None
            return self._snapshot()

        dwell_elapsed_s = self._current_dwell_elapsed(now)
        if dwell_elapsed_s >= self.dwell_time_s:
            self._state = PatrolTaskState.SAMPLING
            return self._snapshot(sampling_requested=True)
        return self._snapshot()

    def record_sampling_result(
        self,
        *,
        success: bool,
        simulation_time_s: float,
        control_step: int,
        failure_reason: str | None = None,
    ) -> PatrolTaskUpdate:
        """Acknowledge the requested inspection and advance only on success."""

        if not isinstance(success, bool):
            raise TypeError("success must be a bool.")
        if self._state is not PatrolTaskState.SAMPLING:
            raise RuntimeError("A sampling result is valid only while state is SAMPLING.")
        now = self._accept_clock(simulation_time_s, control_step)
        if self._timed_out(now):
            return self._fail("timeout")
        if not success:
            reason = (
                _non_empty_text(failure_reason, "failure_reason")
                if failure_reason is not None
                else "sampling_failed"
            )
            return self._fail(reason)

        completed_id = self._checkpoints[self._checkpoint_index].checkpoint_id
        self._completed_checkpoint_ids.append(completed_id)
        self._checkpoint_index += 1
        self._dwell_started_at_s = None
        if self._checkpoint_index == len(self._checkpoints):
            self._state = PatrolTaskState.COMPLETED
        else:
            self._state = PatrolTaskState.NAVIGATING
        return self._snapshot(checkpoint_completed_id=completed_id)

    def _accept_clock(self, simulation_time_s: float, control_step: int) -> float:
        now = _finite_number(simulation_time_s, "simulation_time_s")
        if now < 0.0:
            raise ValueError("simulation_time_s must be non-negative.")
        if isinstance(control_step, bool) or not isinstance(control_step, int):
            raise TypeError("control_step must be an integer.")
        if control_step < 0:
            raise ValueError("control_step must be non-negative.")
        if self._last_simulation_time_s is not None and now < self._last_simulation_time_s:
            raise ValueError("simulation_time_s must be monotonic.")
        if self._last_control_step is not None and control_step < self._last_control_step:
            raise ValueError("control_step must be monotonic.")
        if self._started_at_s is None:
            self._started_at_s = now
        self._last_simulation_time_s = now
        self._last_control_step = control_step
        return now

    def _timed_out(self, now: float) -> bool:
        assert self._started_at_s is not None
        return now - self._started_at_s >= self.timeout_s

    def _current_dwell_elapsed(self, now: float | None = None) -> float:
        if self._dwell_started_at_s is None:
            return 0.0
        if now is None:
            now = self._last_simulation_time_s
        if now is None:
            return 0.0
        return max(0.0, now - self._dwell_started_at_s)

    def _fail(self, reason: str) -> PatrolTaskUpdate:
        self._state = PatrolTaskState.FAILED
        self._failure_reason = reason
        self._dwell_started_at_s = None
        return self._snapshot()

    def _snapshot(
        self,
        *,
        sampling_requested: bool = False,
        checkpoint_completed_id: str | None = None,
    ) -> PatrolTaskUpdate:
        checkpoint = self.current_checkpoint
        checkpoint_index = None if checkpoint is None else self._checkpoint_index
        return PatrolTaskUpdate(
            state=self._state,
            checkpoint_id=None if checkpoint is None else checkpoint.checkpoint_id,
            checkpoint_index=checkpoint_index,
            completed_checkpoint_ids=tuple(self._completed_checkpoint_ids),
            completion_ratio=len(self._completed_checkpoint_ids) / len(self._checkpoints),
            sampling_requested=sampling_requested,
            checkpoint_completed_id=checkpoint_completed_id,
            dwell_elapsed_s=(
                self._current_dwell_elapsed()
                if self._state is PatrolTaskState.DWELLING
                else 0.0
            ),
            failure_reason=self._failure_reason,
        )


def validate_checkpoint_layout(
    checkpoints: Iterable[PatrolTaskCheckpoint],
    world_positions: Mapping[str, Sequence[Real]],
    *,
    tolerance_m: float = 0.001,
) -> None:
    """Reject drift between task coordinates and checkpoint markers in a world."""

    expected = tuple(checkpoints)
    if not expected or not all(isinstance(item, PatrolTaskCheckpoint) for item in expected):
        raise ValueError("checkpoints must contain PatrolTaskCheckpoint values.")
    tolerance = _finite_number(tolerance_m, "tolerance_m")
    if tolerance <= 0.0:
        raise ValueError("tolerance_m must be positive.")
    expected_ids = {checkpoint.checkpoint_id for checkpoint in expected}
    if set(world_positions) != expected_ids:
        missing = sorted(expected_ids - set(world_positions))
        extra = sorted(set(world_positions) - expected_ids)
        raise ValueError(
            f"World checkpoint ids do not match the task; missing={missing}, extra={extra}."
        )
    for checkpoint in expected:
        actual = _finite_point(
            world_positions[checkpoint.checkpoint_id],
            f"world checkpoint {checkpoint.checkpoint_id!r}",
        )
        error = math.dist(checkpoint.position, actual)
        if error > tolerance:
            raise ValueError(
                f"World checkpoint {checkpoint.checkpoint_id!r} differs from task config "
                f"by {error:.6f} m (tolerance {tolerance:.6f} m)."
            )


def _finite_point(value: Sequence[Real], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain exactly two finite numbers.")
    try:
        x, y = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly two finite numbers.") from error
    return (_finite_number(x, f"{name}[0]"), _finite_number(y, f"{name}[1]"))


def _finite_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()
