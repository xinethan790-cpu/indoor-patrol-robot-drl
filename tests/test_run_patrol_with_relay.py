from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import indoor_patrol_rl.relay as relay_module

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_patrol_with_relay.py"
SPEC = importlib.util.spec_from_file_location("_test_run_patrol_with_relay", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)

SECRET = "8gP1yQ6dT9mK3vN7xR2cF5jH0bL4wZsA"
ENDPOINT = "https://relay.example.test/v1/patrol/dashboard"


class _Process:
    _next_pid = 5000

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


def test_commands_and_logs_never_contain_secret_and_webots_does_not_inherit_it(
    tmp_path: Path,
) -> None:
    relay_command, producer_command = RUNNER.build_child_commands(
        repository_root=REPOSITORY_ROOT,
        run_id="one-command-run",
        seed=7000,
        output_directory=tmp_path / "run",
        endpoint=ENDPOINT,
        device_id="macmini-m4",
        map_config=REPOSITORY_ROOT / "configs/maps/patrol_route.yaml",
        relay_startup_wait_s=60.0,
        relay_poll_interval_s=0.75,
        relay_request_timeout_s=5.0,
        relay_backoff_initial_s=1.0,
        relay_backoff_maximum_s=30.0,
        relay_backoff_multiplier=2.0,
        relay_backoff_jitter_ratio=0.15,
        webots_arguments=("--headless",),
    )
    relay_environment, producer_environment = RUNNER.child_environments(
        {
            "PATH": "/bin",
            "PATROL_RELAY_UPLOAD_SECRET": SECRET,
            "WEBOTS_PATROL_MODEL": "stale-secret-model.zip",
            "WEBOTS_PATROL_MAX_STEPS": "1",
            "WEBOTS_AUTO_QUIT": "1",
        },
        secret_variable="PATROL_RELAY_UPLOAD_SECRET",
    )

    assert SECRET not in " ".join(relay_command + producer_command)
    assert relay_environment["PATROL_RELAY_UPLOAD_SECRET"] == SECRET
    assert "PATROL_RELAY_UPLOAD_SECRET" not in producer_environment
    assert "WEBOTS_PATROL_MODEL" not in producer_environment
    assert "WEBOTS_PATROL_MAX_STEPS" not in producer_environment
    assert "WEBOTS_AUTO_QUIT" not in producer_environment
    assert Path(relay_command[1]).parent.name == "scripts"
    assert Path(relay_command[1]).name == "publish_patrol_relay.py"
    assert Path(producer_command[1]).parent.name == "scripts"
    assert Path(producer_command[1]).name == "run_webots_patrol.py"
    assert "--headless" in producer_command
    assert producer_command[-1] == "--auto-quit"
    assert SECRET not in RUNNER._safe_supervisor_line(
        "relay_restart_failed",
        {"message": SECRET},
    )
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_main_starts_waiting_relay_before_webots_and_fixes_one_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[list[str], dict[str, Any], _Process]] = []

    def popen(command: list[str], **kwargs: Any) -> _Process:
        process = _Process(command)
        launches.append((command, kwargs, process))
        return process

    class FakeSupervisor:
        def __init__(self, producer: _Process, relay: _Process, **kwargs: Any) -> None:
            assert producer is launches[1][2]
            assert relay is launches[0][2]
            assert kwargs["run_directory"] == tmp_path / "run"

        def request_stop(self, _signum: int) -> None:
            raise AssertionError("no signal expected")

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(
                status="terminal_acknowledged",
                exit_code=0,
                producer_exit_code=0,
                relay_exit_code=0,
                terminal_acknowledged=True,
            )

    monkeypatch.setenv("PATROL_RELAY_UPLOAD_SECRET", SECRET)
    monkeypatch.setattr(RUNNER.subprocess, "Popen", popen)
    monkeypatch.setattr(relay_module, "PatrolProcessSupervisor", FakeSupervisor)

    result = RUNNER.main(
        [
            "--endpoint",
            ENDPOINT,
            "--run-id",
            "one-command-run",
            "--output-dir",
            str(tmp_path / "run"),
            "--headless",
            "--scenario",
            "stress",
        ]
    )

    assert result == 0
    assert len(launches) == 2
    relay_command, relay_launch, _relay = launches[0]
    producer_command, producer_launch, _producer = launches[1]
    assert relay_command[1].endswith("publish_patrol_relay.py")
    assert producer_command[1].endswith("run_webots_patrol.py")
    for command in (relay_command, producer_command):
        assert command[command.index("--run-id") + 1] == "one-command-run" if (
            "--run-id" in command
        ) else command[command.index("--expected-run-id") + 1] == "one-command-run"
    assert relay_launch["env"]["PATROL_RELAY_UPLOAD_SECRET"] == SECRET
    assert "PATROL_RELAY_UPLOAD_SECRET" not in producer_launch["env"]
    assert producer_command[2:5] == ["--headless", "--scenario", "stress"]
    assert producer_command[-1] == "--auto-quit"


def test_missing_secret_fails_before_any_child_is_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATROL_RELAY_UPLOAD_SECRET", raising=False)
    monkeypatch.setattr(
        RUNNER.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("children must not start without a secret"),
    )

    with pytest.raises(relay_module.RelayConfigurationError, match="not set") as error:
        RUNNER.main(
            [
                "--endpoint",
                ENDPOINT,
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )

    assert SECRET not in str(error.value)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "forwarded",
    [
        ["--", "--run-id", "other-run"],
        ["--", "--output-dir=other-directory"],
        ["--", "--seed", "9"],
    ],
)
def test_forwarded_arguments_cannot_override_supervisor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forwarded: list[str],
) -> None:
    monkeypatch.setenv("PATROL_RELAY_UPLOAD_SECRET", SECRET)
    monkeypatch.setattr(
        RUNNER.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("invalid authority must fail before launch"),
    )

    with pytest.raises(ValueError, match="combined launcher"):
        RUNNER.main(
            [
                "--endpoint",
                ENDPOINT,
                "--run-id",
                "one-command-run",
                "--output-dir",
                str(tmp_path / "run"),
                *forwarded,
            ]
        )


@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL"),
    reason="graceful-to-kill reaping uses POSIX process-group signals",
)
def test_second_child_start_failure_terminates_and_reaps_first_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubbornRelay(_Process):
        wait_calls = 0

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.returncode is None:
                raise RUNNER.subprocess.TimeoutExpired("relay", timeout)
            return self.returncode

    relay = StubbornRelay(["relay"])
    launches = 0
    signals: list[tuple[int, int]] = []

    def popen(command: list[str], **_kwargs: Any) -> _Process:
        nonlocal launches
        launches += 1
        if launches == 1:
            return relay
        raise OSError("fixture producer start failure")

    def kill_group(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGKILL:
            relay.returncode = -signum

    monkeypatch.setenv("PATROL_RELAY_UPLOAD_SECRET", SECRET)
    monkeypatch.setattr(RUNNER.subprocess, "Popen", popen)
    monkeypatch.setattr(RUNNER.os, "killpg", kill_group)

    with pytest.raises(OSError, match="producer start failure"):
        RUNNER.main(
            [
                "--endpoint",
                ENDPOINT,
                "--run-id",
                "failed-start-run",
                "--output-dir",
                str(tmp_path / "run"),
                "--shutdown-grace-s",
                "0",
            ]
        )

    assert signals == [
        (relay.pid, signal.SIGTERM),
        (relay.pid, signal.SIGKILL),
    ]
    assert relay.returncode == -signal.SIGKILL
    assert relay.wait_calls == 2
