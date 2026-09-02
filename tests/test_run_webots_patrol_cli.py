from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_webots_patrol.py"


def _load_runner_module() -> Any:
    module_name = "_test_run_webots_patrol"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _clear_webots_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("WEBOTS_"):
            monkeypatch.delenv(name, raising=False)


def _capture_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], dict[str, Any]]]:
    launches: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(
        RUNNER,
        "discover_webots",
        lambda _explicit: Path("/Applications/Webots.app/Contents/MacOS/webots"),
    )

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "env" in kwargs:
            launches.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    return launches


def test_default_run_id_and_output_are_unique_per_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    moments = iter(
        (
            datetime(2026, 8, 31, 1, 2, 3, 4, tzinfo=UTC),
            datetime(2026, 8, 31, 1, 2, 3, 5, tzinfo=UTC),
        )
    )

    class SequenceClock:
        @classmethod
        def now(cls, timezone: object) -> datetime:
            assert timezone is UTC
            return next(moments)

    monkeypatch.setattr(RUNNER, "datetime", SequenceClock)
    launches = _capture_launches(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--seed", "23"])

    assert RUNNER.main() == 0
    assert RUNNER.main() == 0

    assert len(launches) == 2
    first_environment = launches[0][1]["env"]
    second_environment = launches[1][1]["env"]
    first_id = "webots-patrol-seed23-20260831T010203000004Z"
    second_id = "webots-patrol-seed23-20260831T010203000005Z"
    assert first_environment["WEBOTS_PATROL_RUN_ID"] == first_id
    assert second_environment["WEBOTS_PATROL_RUN_ID"] == second_id
    assert first_environment["WEBOTS_PATROL_OUTPUT_DIR"] == str(
        REPOSITORY_ROOT / "artifacts/results" / first_id
    )
    assert second_environment["WEBOTS_PATROL_OUTPUT_DIR"] == str(
        REPOSITORY_ROOT / "artifacts/results" / second_id
    )
    assert (
        first_environment["WEBOTS_PATROL_OUTPUT_DIR"]
        != second_environment["WEBOTS_PATROL_OUTPUT_DIR"]
    )


def test_explicit_model_requires_unvalidated_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER_PATH), "--model", "artifacts/models/candidate.zip"],
    )

    with pytest.raises(ValueError, match="--allow-unvalidated-model"):
        RUNNER.main()


def test_explicit_model_and_opt_in_are_forwarded_to_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    launches = _capture_launches(monkeypatch)
    output = tmp_path / "exploratory-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            "candidate-check",
            "--output-dir",
            str(output),
            "--model",
            "artifacts/models/candidate.zip",
            "--allow-unvalidated-model",
        ],
    )

    assert RUNNER.main() == 0

    environment = launches[0][1]["env"]
    assert environment["WEBOTS_PATROL_MODEL"] == "artifacts/models/candidate.zip"
    assert environment["WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL"] == "1"
    assert environment["WEBOTS_PATROL_RUN_ID"] == "candidate-check"
    assert environment["WEBOTS_PATROL_OUTPUT_DIR"] == str(output)


def test_existing_output_directory_is_rejected_before_webots_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    output = tmp_path / "existing-run"
    output.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER_PATH), "--output-dir", str(output)],
    )

    with pytest.raises(FileExistsError, match="already exists"):
        RUNNER.main()


def test_timeout_and_inspection_config_are_forwarded_to_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    launches = _capture_launches(monkeypatch)
    output = tmp_path / "configured-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            "configured-run",
            "--output-dir",
            str(output),
            "--inspection-config",
            "configs/inspection/lab_demo.yaml",
            "--timeout-s",
            "75.5",
            "--max-steps",
            "900",
            "--headless",
        ],
    )

    assert RUNNER.main() == 0

    command, launch = launches[0]
    environment = launch["env"]
    assert environment["WEBOTS_PATROL_INSPECTION_CONFIG"] == ("configs/inspection/lab_demo.yaml")
    assert environment["WEBOTS_PATROL_SCENARIO_CONFIG"] == (
        "configs/patrol/scenario_operational_v1.yaml"
    )
    assert environment["WEBOTS_PATROL_TIMEOUT_S"] == "75.5"
    assert environment["WEBOTS_PATROL_MAX_STEPS"] == "900"
    assert environment["WEBOTS_AUTO_QUIT"] == "1"
    assert command[-4:-1] == ["--batch", "--mode=fast", "--no-rendering"]
    assert command[-1] == str(REPOSITORY_ROOT / "worlds/patrol_route_ppo.wbt")


def test_stress_scenario_is_forwarded_to_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    launches = _capture_launches(monkeypatch)
    output = tmp_path / "stress-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            "stress-run",
            "--output-dir",
            str(output),
            "--scenario",
            "stress",
        ],
    )

    assert RUNNER.main() == 0

    environment = launches[0][1]["env"]
    assert environment["WEBOTS_PATROL_SCENARIO_CONFIG"] == (
        "configs/patrol/scenario_stress_v1.yaml"
    )


def test_auto_quit_preserves_rendering_and_exits_after_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    launches = _capture_launches(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            "auto-quit-run",
            "--output-dir",
            str(tmp_path / "auto-quit-run"),
            "--auto-quit",
        ],
    )

    assert RUNNER.main() == 0

    command, launch = launches[0]
    assert launch["env"]["WEBOTS_AUTO_QUIT"] == "1"
    assert "--batch" not in command
    assert "--no-rendering" not in command


def test_frame_capture_uses_rendered_realtime_batch_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    launches = _capture_launches(monkeypatch)
    output = tmp_path / "video-run"
    frames = output / "frames"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            "video-run",
            "--output-dir",
            str(output),
            "--capture-frames-dir",
            str(frames),
            "--capture-fps",
            "12.5",
        ],
    )

    assert RUNNER.main() == 0

    command, launch = launches[0]
    environment = launch["env"]
    assert environment["WEBOTS_PATROL_CAPTURE_FRAMES_DIR"] == str(frames)
    assert environment["WEBOTS_PATROL_CAPTURE_FPS"] == "12.5"
    assert environment["WEBOTS_AUTO_QUIT"] == "1"
    assert command[-3:-1] == ["--batch", "--mode=realtime"]
    assert "--no-rendering" not in command
    assert command[-1] == str(REPOSITORY_ROOT / "worlds/patrol_route_ppo.wbt")


def test_existing_frame_capture_directory_is_rejected_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    frames = tmp_path / "frames"
    frames.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--output-dir",
            str(tmp_path / "result"),
            "--capture-frames-dir",
            str(frames),
        ],
    )

    with pytest.raises(FileExistsError, match="Frame capture directory already exists"):
        RUNNER.main()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--timeout-s", "0"], "--timeout-s must be positive"),
        (["--max-steps", "0"], "--max-steps must be positive"),
        (["--run-id", "nested/run"], "cannot contain path separators"),
        (["--capture-fps", "nan"], "--capture-fps must be positive and finite"),
        (
            ["--headless", "--capture-frames-dir", "new-frames"],
            "mutually exclusive",
        ),
    ],
)
def test_invalid_launch_arguments_are_rejected(
    arguments: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_webots_environment(monkeypatch)
    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), *arguments])

    with pytest.raises(ValueError, match=message):
        RUNNER.main()
