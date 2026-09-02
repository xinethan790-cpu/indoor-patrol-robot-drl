"""Lifecycle supervision for one Webots producer and its outbound relay.

The supervisor deliberately knows nothing about relay credentials.  It only
coordinates two already-started process groups and observes the fixed local
evidence filenames.  This keeps network authentication isolated in the relay
child while still giving the producer and uploader one bounded lifecycle.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SUPERVISOR_RELAY_FAILURE_EXIT_CODE = 70
SUPERVISOR_DRAIN_TIMEOUT_EXIT_CODE = 75


class ManagedPatrolProcess(Protocol):
    """Minimal subprocess surface needed by :class:`PatrolProcessSupervisor`."""

    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True, slots=True)
class PatrolSupervisorOutcome:
    """Bounded result of supervising the producer and relay process groups."""

    status: str
    exit_code: int
    producer_exit_code: int | None
    relay_exit_code: int | None
    relay_restarts: int
    terminal_acknowledged: bool


class PatrolProcessSupervisor:
    """Coordinate producer termination, relay recovery, signals, and drain time."""

    def __init__(
        self,
        producer: ManagedPatrolProcess,
        relay: ManagedPatrolProcess,
        *,
        run_directory: str | Path,
        relay_factory: Callable[[], ManagedPatrolProcess] | None = None,
        relay_restart_limit: int = 1,
        terminal_drain_s: float = 30.0,
        shutdown_grace_s: float = 5.0,
        poll_interval_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        signal_process_group: Callable[[int, int], None] | None = None,
        event_handler: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        if isinstance(relay_restart_limit, bool) or not isinstance(relay_restart_limit, int):
            raise TypeError("relay_restart_limit must be an integer.")
        if relay_restart_limit < 0:
            raise ValueError("relay_restart_limit cannot be negative.")
        for name, value, allow_zero in (
            ("terminal_drain_s", terminal_drain_s, True),
            ("shutdown_grace_s", shutdown_grace_s, True),
            ("poll_interval_s", poll_interval_s, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier} and finite.")
        if relay_restart_limit and relay_factory is None:
            raise ValueError("relay_factory is required when relay restarts are enabled.")
        self.producer = producer
        self.relay = relay
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.relay_factory = relay_factory
        self.relay_restart_limit = relay_restart_limit
        self.terminal_drain_s = float(terminal_drain_s)
        self.shutdown_grace_s = float(shutdown_grace_s)
        self.poll_interval_s = float(poll_interval_s)
        self._clock = clock
        self._sleep = sleep
        self._signal_process_group = signal_process_group or _signal_process_group
        self._event_handler = event_handler
        self._requested_signal: int | None = None
        self._relay_restarts = 0
        self._relay_unavailable_code: int | None = None

    def request_stop(self, signum: int) -> None:
        """Request linked shutdown; safe to call from a lightweight signal handler."""

        if isinstance(signum, bool) or not isinstance(signum, int) or signum <= 0:
            raise ValueError("signum must be a positive integer.")
        if self._requested_signal is None:
            self._requested_signal = signum

    def run(self) -> PatrolSupervisorOutcome:
        """Block until terminal acknowledgement, bounded failure, or linked signal."""

        while True:
            if self._requested_signal is not None:
                return self._shutdown_for_signal(self._requested_signal)
            producer_code = self.producer.poll()
            relay_code = self.relay.poll()
            if producer_code is not None:
                return self._finish_after_producer(producer_code, relay_code)
            if (
                self._relay_unavailable_code is None
                and relay_code is not None
                and relay_code != 0
            ):
                if self._runtime_evidence_started() and self._restart_relay():
                    continue
                if self._runtime_evidence_started():
                    # A sidecar failure must not change the robot's actions.  Let
                    # Webots finish locally, then report the cloud failure.
                    self._relay_unavailable_code = relay_code
                    self._announce(
                        "relay_failed_while_producer_active",
                        {"relay_exit_code": relay_code},
                    )
            # A zero relay exit means its CLI observed a terminal ACK.  Keep the
            # Webots process alive until it completes or the user stops it.
            self._sleep(self.poll_interval_s)

    def _finish_after_producer(
        self,
        producer_code: int,
        relay_code: int | None,
    ) -> PatrolSupervisorOutcome:
        if not self._runtime_evidence_started():
            self._announce(
                "producer_exited_before_evidence",
                {"producer_exit_code": producer_code},
            )
            self._stop_process(self.relay, signal.SIGTERM)
            self._wait_then_kill((self.relay,))
            exit_code = producer_code or SUPERVISOR_RELAY_FAILURE_EXIT_CODE
            return PatrolSupervisorOutcome(
                status="producer_exited_before_evidence",
                exit_code=exit_code,
                producer_exit_code=producer_code,
                relay_exit_code=self.relay.poll(),
                relay_restarts=self._relay_restarts,
                terminal_acknowledged=False,
            )
        if self._relay_unavailable_code is not None:
            return PatrolSupervisorOutcome(
                status="relay_failed_during_run",
                exit_code=SUPERVISOR_RELAY_FAILURE_EXIT_CODE,
                producer_exit_code=producer_code,
                relay_exit_code=self._relay_unavailable_code,
                relay_restarts=self._relay_restarts,
                terminal_acknowledged=False,
            )
        deadline = self._clock() + self.terminal_drain_s
        current_relay_code = relay_code
        while True:
            if self._requested_signal is not None:
                return self._shutdown_for_signal(self._requested_signal)
            if current_relay_code == 0:
                return PatrolSupervisorOutcome(
                    status="terminal_acknowledged",
                    exit_code=producer_code,
                    producer_exit_code=producer_code,
                    relay_exit_code=0,
                    relay_restarts=self._relay_restarts,
                    terminal_acknowledged=True,
                )
            if current_relay_code is not None:
                if self._restart_relay():
                    current_relay_code = None
                else:
                    self._announce(
                        "relay_failed_during_drain",
                        {"relay_exit_code": current_relay_code},
                    )
                    return PatrolSupervisorOutcome(
                        status="relay_failed_during_drain",
                        exit_code=SUPERVISOR_RELAY_FAILURE_EXIT_CODE,
                        producer_exit_code=producer_code,
                        relay_exit_code=current_relay_code,
                        relay_restarts=self._relay_restarts,
                        terminal_acknowledged=False,
                    )
            now = self._clock()
            if now >= deadline:
                self._announce("terminal_drain_timeout", {})
                self._stop_process(self.relay, signal.SIGTERM)
                self._wait_then_kill((self.relay,))
                return PatrolSupervisorOutcome(
                    status="terminal_drain_timeout",
                    exit_code=SUPERVISOR_DRAIN_TIMEOUT_EXIT_CODE,
                    producer_exit_code=producer_code,
                    relay_exit_code=self.relay.poll(),
                    relay_restarts=self._relay_restarts,
                    terminal_acknowledged=False,
                )
            self._sleep(min(self.poll_interval_s, max(0.0, deadline - now)))
            current_relay_code = self.relay.poll()

    def _restart_relay(self) -> bool:
        if (
            self.relay_factory is None
            or self._relay_restarts >= self.relay_restart_limit
        ):
            return False
        self._relay_restarts += 1
        self._announce("relay_restarting", {"restart_count": self._relay_restarts})
        try:
            self.relay = self.relay_factory()
        except Exception:
            self._announce("relay_restart_failed", {"restart_count": self._relay_restarts})
            return False
        return True

    def _shutdown_for_signal(self, signum: int) -> PatrolSupervisorOutcome:
        self._announce("signal_shutdown", {"signal": signum})
        self._stop_process(self.producer, signum)
        self._stop_process(self.relay, signum)
        self._wait_then_kill((self.producer, self.relay))
        return PatrolSupervisorOutcome(
            status="signal_shutdown",
            exit_code=128 + signum,
            producer_exit_code=self.producer.poll(),
            relay_exit_code=self.relay.poll(),
            relay_restarts=self._relay_restarts,
            terminal_acknowledged=False,
        )

    def _runtime_evidence_started(self) -> bool:
        return all(
            (self.run_directory / filename).is_file()
            for filename in ("status.json", "events.jsonl")
        )

    def _stop_process(self, process: ManagedPatrolProcess, signum: int) -> None:
        if process.poll() is not None:
            return
        try:
            self._signal_process_group(process.pid, signum)
        except ProcessLookupError:
            return

    def _wait_then_kill(self, processes: tuple[ManagedPatrolProcess, ...]) -> None:
        deadline = self._clock() + self.shutdown_grace_s
        while any(process.poll() is None for process in processes):
            now = self._clock()
            if now >= deadline:
                break
            self._sleep(min(self.poll_interval_s, max(0.0, deadline - now)))
        for process in processes:
            self._stop_process(process, signal.SIGKILL)
        for process in processes:
            try:
                process.wait(timeout=max(0.1, self.shutdown_grace_s))
            except subprocess.TimeoutExpired:
                # The process group has already received SIGKILL.  Keep the
                # supervisor's own exit bounded even if a broken test double or
                # platform process API fails to observe it promptly.
                continue

    def _announce(self, event: str, details: Mapping[str, object]) -> None:
        if self._event_handler is not None:
            self._event_handler(event, details)


def _signal_process_group(pid: int, signum: int) -> None:
    os.killpg(pid, signum)
