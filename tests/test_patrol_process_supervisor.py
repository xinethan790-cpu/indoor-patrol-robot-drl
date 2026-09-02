from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from indoor_patrol_rl.relay import (
    SUPERVISOR_DRAIN_TIMEOUT_EXIT_CODE,
    SUPERVISOR_RELAY_FAILURE_EXIT_CODE,
    PatrolProcessSupervisor,
)


class _Process:
    _next_pid = 2000

    def __init__(self, returncode: int | None = None) -> None:
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        assert self.returncode is not None
        return self.returncode


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _signaler(processes: tuple[_Process, ...], calls: list[tuple[int, int]]) -> Any:
    by_pid = {process.pid: process for process in processes}

    def signal_group(pid: int, signum: int) -> None:
        calls.append((pid, signum))
        process = by_pid[pid]
        if process.returncode is None:
            process.returncode = -signum

    return signal_group


def test_producer_failure_before_first_evidence_stops_waiting_relay(
    tmp_path: Path,
) -> None:
    producer = _Process(9)
    relay = _Process()
    calls: list[tuple[int, int]] = []
    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=tmp_path / "never-created",
        relay_factory=lambda: _Process(),
        signal_process_group=_signaler((producer, relay), calls),
        sleep=lambda _seconds: None,
    )

    outcome = supervisor.run()

    assert outcome.status == "producer_exited_before_evidence"
    assert outcome.exit_code == 9
    assert outcome.terminal_acknowledged is False
    assert calls == [(relay.pid, signal.SIGTERM)]


def test_terminal_ack_can_arrive_during_bounded_drain(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "status.json").write_text("{}", encoding="utf-8")
    (run_directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    producer = _Process(0)
    relay = _Process()
    clock = _Clock()

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if clock.value >= 0.2:
            relay.returncode = 0

    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=run_directory,
        relay_factory=lambda: _Process(),
        terminal_drain_s=1.0,
        poll_interval_s=0.1,
        clock=clock,
        sleep=sleep,
        signal_process_group=lambda _pid, _signal: None,
    )

    outcome = supervisor.run()

    assert outcome.status == "terminal_acknowledged"
    assert outcome.exit_code == 0
    assert outcome.relay_exit_code == 0
    assert outcome.terminal_acknowledged is True
    assert clock.value == 0.2


def test_terminal_drain_timeout_is_nonzero_and_stops_relay(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "status.json").write_text("{}", encoding="utf-8")
    (run_directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    producer = _Process(0)
    relay = _Process()
    clock = _Clock()
    calls: list[tuple[int, int]] = []
    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=run_directory,
        relay_factory=lambda: _Process(),
        terminal_drain_s=0.3,
        poll_interval_s=0.1,
        clock=clock,
        sleep=clock.advance,
        signal_process_group=_signaler((producer, relay), calls),
    )

    outcome = supervisor.run()

    assert outcome.status == "terminal_drain_timeout"
    assert outcome.exit_code == SUPERVISOR_DRAIN_TIMEOUT_EXIT_CODE
    assert outcome.terminal_acknowledged is False
    assert calls == [(relay.pid, signal.SIGTERM)]


def test_signal_request_is_forwarded_to_both_process_groups(tmp_path: Path) -> None:
    producer = _Process()
    relay = _Process()
    clock = _Clock()
    calls: list[tuple[int, int]] = []
    supervisor: PatrolProcessSupervisor

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        supervisor.request_stop(signal.SIGINT)

    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=tmp_path / "run",
        relay_factory=lambda: _Process(),
        clock=clock,
        sleep=sleep,
        signal_process_group=_signaler((producer, relay), calls),
    )

    outcome = supervisor.run()

    assert outcome.status == "signal_shutdown"
    assert outcome.exit_code == 128 + signal.SIGINT
    assert calls == [
        (producer.pid, signal.SIGINT),
        (relay.pid, signal.SIGINT),
    ]


def test_failed_relay_restarts_once_and_reuses_same_run_directory(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "status.json").write_text("{}", encoding="utf-8")
    (run_directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    producer = _Process()
    first_relay = _Process(1)
    resumed_relay = _Process()
    clock = _Clock()
    restarts: list[Path] = []

    def restart() -> _Process:
        restarts.append(run_directory)
        return resumed_relay

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        producer.returncode = 0
        resumed_relay.returncode = 0

    supervisor = PatrolProcessSupervisor(
        producer,
        first_relay,
        run_directory=run_directory,
        relay_factory=restart,
        relay_restart_limit=1,
        clock=clock,
        sleep=sleep,
        signal_process_group=lambda _pid, _signal: None,
    )

    outcome = supervisor.run()

    assert restarts == [run_directory]
    assert outcome.status == "terminal_acknowledged"
    assert outcome.relay_restarts == 1
    assert outcome.terminal_acknowledged is True


def test_relay_failure_without_restart_budget_does_not_change_robot_actions(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "status.json").write_text("{}", encoding="utf-8")
    (run_directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    producer = _Process()
    relay = _Process(2)
    calls: list[tuple[int, int]] = []

    def finish_producer(_seconds: float) -> None:
        producer.returncode = 0

    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=run_directory,
        relay_restart_limit=0,
        signal_process_group=_signaler((producer, relay), calls),
        sleep=finish_producer,
    )

    outcome = supervisor.run()

    assert outcome.status == "relay_failed_during_run"
    assert outcome.exit_code == SUPERVISOR_RELAY_FAILURE_EXIT_CODE
    assert producer.returncode == 0
    assert calls == []


def test_single_partial_live_file_is_not_treated_as_started_evidence(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "partial"
    run_directory.mkdir()
    (run_directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    producer = _Process(1)
    relay = _Process()
    calls: list[tuple[int, int]] = []
    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=run_directory,
        relay_factory=lambda: _Process(),
        signal_process_group=_signaler((producer, relay), calls),
        sleep=lambda _seconds: None,
    )

    outcome = supervisor.run()

    assert outcome.status == "producer_exited_before_evidence"
    assert calls == [(relay.pid, signal.SIGTERM)]


@pytest.mark.parametrize("producer_exit_code", [0, 1, 2, 3, 4])
def test_terminal_ack_before_producer_exit_waits_and_preserves_producer_code(
    tmp_path: Path,
    producer_exit_code: int,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "status.json").write_text("{}", encoding="utf-8")
    (run_directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    producer = _Process()
    relay = _Process(0)

    def finish_producer(_seconds: float) -> None:
        producer.returncode = producer_exit_code

    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=run_directory,
        relay_factory=lambda: _Process(),
        sleep=finish_producer,
        signal_process_group=lambda _pid, _signal: None,
    )

    outcome = supervisor.run()

    assert outcome.status == "terminal_acknowledged"
    assert outcome.exit_code == producer_exit_code
    assert outcome.producer_exit_code == producer_exit_code
    assert outcome.terminal_acknowledged is True


def test_grace_timeout_escalates_to_sigkill_and_reaps_processes(tmp_path: Path) -> None:
    class StubbornProcess(_Process):
        waited = False

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return self.returncode

    producer = StubbornProcess()
    relay = StubbornProcess()
    clock = _Clock()
    calls: list[tuple[int, int]] = []
    by_pid = {producer.pid: producer, relay.pid: relay}

    def signal_group(pid: int, signum: int) -> None:
        calls.append((pid, signum))
        if signum == signal.SIGKILL:
            by_pid[pid].returncode = -signal.SIGKILL

    supervisor = PatrolProcessSupervisor(
        producer,
        relay,
        run_directory=tmp_path / "run",
        relay_factory=lambda: _Process(),
        shutdown_grace_s=0.2,
        poll_interval_s=0.1,
        clock=clock,
        sleep=clock.advance,
        signal_process_group=signal_group,
    )
    supervisor.request_stop(signal.SIGTERM)

    outcome = supervisor.run()

    assert outcome.exit_code == 128 + signal.SIGTERM
    assert calls == [
        (producer.pid, signal.SIGTERM),
        (relay.pid, signal.SIGTERM),
        (producer.pid, signal.SIGKILL),
        (relay.pid, signal.SIGKILL),
    ]
    assert producer.waited is True
    assert relay.waited is True
