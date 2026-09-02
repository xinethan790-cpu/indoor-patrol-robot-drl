from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from indoor_patrol_rl.dashboard import DashboardEvidenceError
from indoor_patrol_rl.relay import RelayPermanentError

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
PUBLISHER_PATH = REPOSITORY_ROOT / "scripts/publish_patrol_relay.py"
SPEC = importlib.util.spec_from_file_location("_test_publish_patrol_relay", PUBLISHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISHER
SPEC.loader.exec_module(PUBLISHER)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_relay_can_start_before_directory_and_first_status_with_bounded_wait(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    calls = 0
    repository = SimpleNamespace(
        read=lambda: SimpleNamespace(current=SimpleNamespace(run_id="early-run"))
    )

    def factory(path: Path) -> Any:
        nonlocal calls
        calls += 1
        assert path == tmp_path / "run"
        if calls < 3:
            raise DashboardEvidenceError("not ready")
        return repository

    resolved = PUBLISHER.wait_for_patrol_repository(
        tmp_path / "run",
        startup_wait_s=1.0,
        expected_run_id="early-run",
        repository_factory=factory,
        monotonic=clock,
        sleep=clock.sleep,
        poll_interval_s=0.1,
    )

    assert resolved is repository
    assert calls == 3
    assert clock.value == 0.2
    assert not (tmp_path / "run").exists()


def test_relay_startup_wait_times_out_without_spinning_forever(tmp_path: Path) -> None:
    clock = _Clock()

    def unavailable(_path: Path) -> Any:
        raise DashboardEvidenceError("not ready")

    with pytest.raises(TimeoutError, match="timed out after 0.25s"):
        PUBLISHER.wait_for_patrol_repository(
            tmp_path / "run",
            startup_wait_s=0.25,
            expected_run_id="early-run",
            repository_factory=unavailable,
            monotonic=clock,
            sleep=clock.sleep,
            poll_interval_s=0.1,
        )

    assert clock.value == 0.25
    assert not (tmp_path / "run").exists()


def test_zero_startup_wait_preserves_immediate_repository_open(tmp_path: Path) -> None:
    repository = object()
    reads = 0

    def factory(_path: Path) -> object:
        nonlocal reads
        reads += 1
        return repository

    assert PUBLISHER.wait_for_patrol_repository(
        tmp_path / "existing",
        startup_wait_s=0.0,
        expected_run_id=None,
        repository_factory=factory,
    ) is repository
    assert reads == 1


def test_first_snapshot_run_id_mismatch_fails_without_retry(tmp_path: Path) -> None:
    clock = _Clock()
    reads = 0

    def read() -> SimpleNamespace:
        nonlocal reads
        reads += 1
        return SimpleNamespace(current=SimpleNamespace(run_id="different-run"))

    repository = SimpleNamespace(read=read)

    with pytest.raises(RelayPermanentError, match="does not match"):
        PUBLISHER.wait_for_patrol_repository(
            tmp_path / "run",
            startup_wait_s=1.0,
            expected_run_id="expected-run",
            repository_factory=lambda _path: repository,
            monotonic=clock,
            sleep=clock.sleep,
        )

    assert reads == 1
    assert clock.value == 0.0
