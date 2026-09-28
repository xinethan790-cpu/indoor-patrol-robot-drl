#!/usr/bin/env python3
"""Launch one Webots patrol and its authenticated cloud relay as one lifecycle."""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Any options not recognized here are forwarded unchanged to "
            "scripts/run_webots_patrol.py (for example --headless or --scenario stress)."
        ),
    )
    parser.add_argument("--endpoint", required=True, help="Public HTTPS relay upload endpoint.")
    parser.add_argument("--device-id", help="Public relay device ID; environment/default applies.")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--run-id", help="Stable run ID; generated once when omitted.")
    parser.add_argument("--output-dir", help="New result directory for this run.")
    parser.add_argument("--map-config", type=Path)
    parser.add_argument("--relay-startup-wait-s", type=float, default=60.0)
    parser.add_argument("--terminal-drain-s", type=float, default=45.0)
    parser.add_argument("--shutdown-grace-s", type=float, default=5.0)
    parser.add_argument("--supervisor-poll-interval-s", type=float, default=0.1)
    parser.add_argument("--relay-restart-limit", type=int, default=1)
    parser.add_argument("--relay-poll-interval-s", type=float, default=0.75)
    parser.add_argument("--relay-request-timeout-s", type=float, default=5.0)
    parser.add_argument("--relay-backoff-initial-s", type=float, default=1.0)
    parser.add_argument("--relay-backoff-maximum-s", type=float, default=30.0)
    parser.add_argument("--relay-backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--relay-backoff-jitter-ratio", type=float, default=0.15)
    args, webots_arguments = parser.parse_known_args(argv)
    if webots_arguments and webots_arguments[0] == "--":
        webots_arguments = webots_arguments[1:]
    return args, webots_arguments


def _generated_run_id(seed: int, *, now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return f"webots-patrol-seed{seed}-{moment.strftime('%Y%m%dT%H%M%S%fZ')}"


def build_child_commands(
    *,
    repository_root: Path,
    run_id: str,
    seed: int,
    output_directory: Path,
    endpoint: str,
    device_id: str,
    map_config: Path,
    relay_startup_wait_s: float,
    relay_poll_interval_s: float,
    relay_request_timeout_s: float,
    relay_backoff_initial_s: float,
    relay_backoff_maximum_s: float,
    relay_backoff_multiplier: float,
    relay_backoff_jitter_ratio: float,
    webots_arguments: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Build public child arguments; upload secrets are intentionally absent."""

    relay_command = [
        sys.executable,
        str(repository_root / "scripts/publish_patrol_relay.py"),
        "--run-dir",
        str(output_directory),
        "--endpoint",
        endpoint,
        "--device-id",
        device_id,
        "--map-config",
        str(map_config),
        "--expected-run-id",
        run_id,
        "--startup-wait-s",
        f"{relay_startup_wait_s:g}",
        "--poll-interval-s",
        f"{relay_poll_interval_s:g}",
        "--timeout-s",
        f"{relay_request_timeout_s:g}",
        "--backoff-initial-s",
        f"{relay_backoff_initial_s:g}",
        "--backoff-maximum-s",
        f"{relay_backoff_maximum_s:g}",
        "--backoff-multiplier",
        f"{relay_backoff_multiplier:g}",
        "--backoff-jitter-ratio",
        f"{relay_backoff_jitter_ratio:g}",
    ]
    producer_command = [
        sys.executable,
        str(repository_root / "scripts/run_webots_patrol.py"),
        *webots_arguments,
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--output-dir",
        str(output_directory),
        "--auto-quit",
    ]
    return relay_command, producer_command


def child_environments(
    environ: Mapping[str, str],
    *,
    secret_variable: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Give the secret only to the relay, never to Webots or its controller."""

    relay_environment = dict(environ)
    producer_environment = dict(environ)
    for name in tuple(producer_environment):
        if (
            name == secret_variable
            or name == "WEBOTS_AUTO_QUIT"
            or name.startswith("WEBOTS_PATROL_")
        ):
            producer_environment.pop(name, None)
    return relay_environment, producer_environment


def _safe_supervisor_line(event: str, details: Mapping[str, object]) -> str:
    if event == "relay_restarting":
        return f"PATROL_SUPERVISOR relay_restarting count={details['restart_count']}"
    if event == "signal_shutdown":
        return f"PATROL_SUPERVISOR signal_shutdown signal={details['signal']}"
    if event in {
        "producer_exited_before_evidence",
        "relay_failed_while_producer_active",
        "relay_failed_during_drain",
        "terminal_drain_timeout",
        "relay_restart_failed",
    }:
        return f"PATROL_SUPERVISOR {event}"
    return f"PATROL_SUPERVISOR {event}"


def _finite_timing(value: float, name: str, *, allow_zero: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        or (not allow_zero and value == 0.0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier} and finite.")
    return float(value)


def _reject_forwarded_authority(arguments: Sequence[str]) -> None:
    reserved = ("--seed", "--run-id", "--output-dir")
    for argument in arguments:
        if argument in reserved or any(
            argument.startswith(f"{name}=") for name in reserved
        ):
            raise ValueError(
                f"{argument.split('=', 1)[0]} must be supplied to the combined launcher, "
                "not after the forwarding separator."
            )


def _stop_unmanaged_processes(
    processes: Sequence[subprocess.Popen[bytes] | None],
    *,
    grace_s: float,
) -> None:
    has_posix_groups = hasattr(os, "killpg") and hasattr(signal, "SIGKILL")

    def signal_tree(process: subprocess.Popen[bytes], signum: int) -> None:
        if has_posix_groups:
            os.killpg(process.pid, signum)
        else:
            # Windows has no process groups; terminate the whole child tree.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )

    active = [process for process in processes if process is not None and process.poll() is None]
    for process in active:
        try:
            signal_tree(process, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in active:
        try:
            process.wait(timeout=max(0.1, grace_s))
            continue
        except subprocess.TimeoutExpired:
            pass
        try:
            signal_tree(process, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.1, grace_s))
        except subprocess.TimeoutExpired:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args, webots_arguments = parse_args(argv)
    _reject_forwarded_authority(webots_arguments)
    repository_root = Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    from indoor_patrol_rl.dashboard import DashboardMap
    from indoor_patrol_rl.identifiers import validate_patrol_run_id
    from indoor_patrol_rl.relay import (
        DEFAULT_SECRET_ENVIRONMENT_VARIABLE,
        PatrolProcessSupervisor,
        RelayBackoff,
        load_upload_secret,
        resolve_device_id,
        validate_https_endpoint,
    )

    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ValueError("--seed must be a non-negative integer.")
    relay_startup_wait_s = _finite_timing(
        args.relay_startup_wait_s,
        "--relay-startup-wait-s",
        allow_zero=False,
    )
    terminal_drain_s = _finite_timing(
        args.terminal_drain_s,
        "--terminal-drain-s",
        allow_zero=True,
    )
    shutdown_grace_s = _finite_timing(
        args.shutdown_grace_s,
        "--shutdown-grace-s",
        allow_zero=True,
    )
    supervisor_poll_interval_s = _finite_timing(
        args.supervisor_poll_interval_s,
        "--supervisor-poll-interval-s",
        allow_zero=False,
    )
    relay_poll_interval_s = _finite_timing(
        args.relay_poll_interval_s,
        "--relay-poll-interval-s",
        allow_zero=False,
    )
    relay_request_timeout_s = _finite_timing(
        args.relay_request_timeout_s,
        "--relay-request-timeout-s",
        allow_zero=False,
    )
    if (
        isinstance(args.relay_restart_limit, bool)
        or not isinstance(args.relay_restart_limit, int)
        or args.relay_restart_limit < 0
    ):
        raise ValueError("--relay-restart-limit must be a non-negative integer.")
    backoff = RelayBackoff(
        initial_s=args.relay_backoff_initial_s,
        maximum_s=args.relay_backoff_maximum_s,
        multiplier=args.relay_backoff_multiplier,
        jitter_ratio=args.relay_backoff_jitter_ratio,
    )
    endpoint = validate_https_endpoint(args.endpoint)
    device_id = resolve_device_id(args.device_id)
    # Fail before Webots starts if the existing environment/Keychain shell
    # handoff did not provide a usable secret.  Never retain or print its value.
    secret = load_upload_secret()
    del secret

    run_id = validate_patrol_run_id(
        args.run_id or _generated_run_id(args.seed),
        field_name="--run-id",
    )
    output_value = args.output_dir or f"artifacts/results/{run_id}"
    output_directory = Path(output_value).expanduser()
    if not output_directory.is_absolute():
        output_directory = repository_root / output_directory
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"Patrol output directory already exists; choose a new run: {output_directory}"
        )
    map_config = args.map_config or repository_root / "configs/maps/patrol_route.yaml"
    map_config = map_config.expanduser()
    if not map_config.is_absolute():
        map_config = repository_root / map_config
    map_config = map_config.resolve(strict=True)
    DashboardMap.from_yaml(map_config)

    relay_command, producer_command = build_child_commands(
        repository_root=repository_root,
        run_id=run_id,
        seed=args.seed,
        output_directory=output_directory,
        endpoint=endpoint,
        device_id=device_id,
        map_config=map_config,
        relay_startup_wait_s=relay_startup_wait_s,
        relay_poll_interval_s=relay_poll_interval_s,
        relay_request_timeout_s=relay_request_timeout_s,
        relay_backoff_initial_s=backoff.initial_s,
        relay_backoff_maximum_s=backoff.maximum_s,
        relay_backoff_multiplier=backoff.multiplier,
        relay_backoff_jitter_ratio=backoff.jitter_ratio,
        webots_arguments=webots_arguments,
    )
    relay_environment, producer_environment = child_environments(
        os.environ,
        secret_variable=DEFAULT_SECRET_ENVIRONMENT_VARIABLE,
    )
    print(f"PATROL_SUPERVISOR starting run={run_id}", flush=True)

    relay_process: subprocess.Popen[bytes] | None = None
    producer_process: subprocess.Popen[bytes] | None = None
    supervisor: PatrolProcessSupervisor | None = None
    pending_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal pending_signal
        if supervisor is None:
            if pending_signal is None:
                pending_signal = signum
            return
        supervisor.request_stop(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, handle_signal)

    def start_relay() -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            relay_command,
            cwd=repository_root,
            env=relay_environment,
            start_new_session=True,
        )

    try:
        relay_process = start_relay()
        producer_process = subprocess.Popen(
            producer_command,
            cwd=repository_root,
            env=producer_environment,
            start_new_session=True,
        )
        supervisor = PatrolProcessSupervisor(
            producer_process,
            relay_process,
            run_directory=output_directory,
            relay_factory=start_relay,
            relay_restart_limit=args.relay_restart_limit,
            terminal_drain_s=terminal_drain_s,
            shutdown_grace_s=shutdown_grace_s,
            poll_interval_s=supervisor_poll_interval_s,
            event_handler=lambda event, details: print(
                _safe_supervisor_line(event, details),
                flush=True,
            ),
        )
        if pending_signal is not None:
            supervisor.request_stop(pending_signal)
        outcome = supervisor.run()
    except BaseException:
        _stop_unmanaged_processes(
            (producer_process, relay_process),
            grace_s=shutdown_grace_s,
        )
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    print(
        "PATROL_SUPERVISOR finished "
        f"run={run_id} status={outcome.status} "
        f"producer_exit={outcome.producer_exit_code} "
        f"relay_exit={outcome.relay_exit_code} "
        f"terminal_ack={str(outcome.terminal_acknowledged).lower()}",
        flush=True,
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
