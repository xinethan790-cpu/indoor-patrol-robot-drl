from __future__ import annotations

import json
from pathlib import Path

import pytest

from indoor_patrol_rl.application import (
    PATROL_FRAME_FILENAME_PATTERN,
    PatrolFrameCapture,
    PatrolFrameCaptureConfig,
    SimulationFrameCadence,
)


def _minimal_jpeg(*, width: int = 64, height: int = 32) -> bytes:
    component_table = b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    start_of_frame = (
        b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03"
        + component_table
    )
    return b"\xff\xd8" + start_of_frame + b"\xff\xd9"


def test_simulation_frame_cadence_uses_absolute_simulation_time() -> None:
    cadence = SimulationFrameCadence(10.0)
    physics_times = [index * 0.032 for index in range(11)]

    captured_at = [time_s for time_s in physics_times if cadence.claim(time_s)]

    assert captured_at == pytest.approx([0.0, 0.128, 0.224, 0.32])
    assert cadence.claim_count == 4
    assert cadence.next_deadline_s == pytest.approx(0.4)


def test_simulation_frame_cadence_rejects_time_reversal() -> None:
    cadence = SimulationFrameCadence(10.0)
    assert cadence.claim(0.0)
    assert not cadence.claim(0.032)

    with pytest.raises(ValueError, match="monotonic"):
        cadence.claim(0.01)


def test_frame_capture_writes_numbered_jpegs_and_metadata(tmp_path: Path) -> None:
    frame_directory = tmp_path / "frames"
    calls: list[tuple[Path, int]] = []

    def fake_exporter(filename: str, quality: int) -> None:
        path = Path(filename)
        calls.append((path, quality))
        path.write_bytes(_minimal_jpeg())

    recorder = PatrolFrameCapture(
        PatrolFrameCaptureConfig(directory=frame_directory, fps=10.0),
        run_id="video-demo-seed7000",
        physics_step_s=0.032,
        exporter=fake_exporter,
    )
    for index in range(11):
        recorder.capture_if_due(index * 0.032)
    metadata_path = recorder.finalize(simulation_duration_s=0.32)

    assert recorder.frame_count == 4
    assert [path.name for path, _quality in calls] == [
        PATROL_FRAME_FILENAME_PATTERN % index for index in range(4)
    ]
    assert {quality for _path, quality in calls} == {90}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "filename_pattern": "frame_%08d.jpg",
        "first_frame_simulation_time_s": 0.0,
        "fps": 10.0,
        "frame_count": 4,
        "height_px": 32,
        "image_format": "jpeg",
        "jpeg_quality": 90,
        "last_frame_simulation_time_s": pytest.approx(0.32),
        "run_id": "video-demo-seed7000",
        "schema_version": "1.0",
        "simulation_duration_s": 0.32,
        "width_px": 64,
    }


def test_frame_capture_refuses_an_existing_directory(tmp_path: Path) -> None:
    frame_directory = tmp_path / "frames"
    frame_directory.mkdir()

    with pytest.raises(FileExistsError):
        PatrolFrameCapture(
            PatrolFrameCaptureConfig(directory=frame_directory),
            run_id="existing-directory",
            physics_step_s=0.032,
            exporter=lambda _filename, _quality: None,
        )


def test_frame_capture_fails_closed_when_exporter_writes_nothing(tmp_path: Path) -> None:
    recorder = PatrolFrameCapture(
        PatrolFrameCaptureConfig(directory=tmp_path / "frames"),
        run_id="missing-frame",
        physics_step_s=0.032,
        exporter=lambda _filename, _quality: None,
    )

    with pytest.raises(RuntimeError, match="non-empty frame"):
        recorder.capture_if_due(0.0)


def test_frame_capture_fails_closed_before_finish_when_sequence_is_empty(
    tmp_path: Path,
) -> None:
    recorder = PatrolFrameCapture(
        PatrolFrameCaptureConfig(directory=tmp_path / "frames"),
        run_id="empty-sequence",
        physics_step_s=0.032,
        exporter=lambda _filename, _quality: None,
    )

    with pytest.raises(RuntimeError, match="produced no images"):
        recorder.finalize(simulation_duration_s=0.0)


def test_frame_capture_rate_cannot_exceed_physics_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed the Webots physics rate"):
        PatrolFrameCapture(
            PatrolFrameCaptureConfig(directory=tmp_path / "frames", fps=40.0),
            run_id="too-fast",
            physics_step_s=0.032,
            exporter=lambda _filename, _quality: None,
        )

    assert not (tmp_path / "frames").exists()


@pytest.mark.parametrize("run_id", ["bad/run", "has space", "-leading", "a" * 129])
def test_frame_capture_uses_the_shared_safe_run_id_contract(
    tmp_path: Path,
    run_id: str,
) -> None:
    directory = tmp_path / "frames"
    with pytest.raises(ValueError, match="must match"):
        PatrolFrameCapture(
            PatrolFrameCaptureConfig(directory=directory),
            run_id=run_id,
            physics_step_s=0.032,
            exporter=lambda _filename, _quality: None,
        )
    assert not directory.exists()
