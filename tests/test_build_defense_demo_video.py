from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

Image = pytest.importorskip("PIL.Image", reason="video extras are not installed")

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/build_defense_demo_video.py"


def _load_module() -> Any:
    module_name = "_test_build_defense_demo_video"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


VIDEO = _load_module()
NATIVE_FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
HAS_CHINESE_FONT = any(path.is_file() for path in VIDEO.DEFAULT_FONT_CANDIDATES)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture_files(
    tmp_path: Path, *, frame_indexes: tuple[int, ...] = (0, 1, 2)
) -> tuple[Path, Path, Path]:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in frame_indexes:
        Image.new("RGB", (320, 180), (20 + index, 40, 60)).save(
            frames / f"frame_{index:08d}.jpg"
        )
    _write_json(
        frames / "capture_metadata.json",
        {
            "schema_version": "1.0",
            "run_id": "demo-seed7000",
            "fps": 10.0,
            "frame_count": len(frame_indexes),
            "first_frame_simulation_time_s": 0.0,
            "last_frame_simulation_time_s": 0.2,
            "simulation_duration_s": 0.2,
            "width_px": 320,
            "height_px": 180,
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = run_dir / "summary.json"
    _write_json(
        summary,
        {
            "run_id": "demo-seed7000",
            "outcome": "completed",
            "checkpoints_total": 4,
            "checkpoints_visited": 4,
            "collision_count": 0,
            "actual_path_length_m": 11.7429,
            "elapsed_simulation_time_s": 88.352,
            "patrol_scenario_qualified": False,
            "checkpoint_records": [
                {
                    "checkpoint_id": f"P{index}",
                    "reached": True,
                    "arrival_simulation_time_s": index * 20.0,
                }
                for index in range(1, 5)
            ],
            "ttc_escape_guard_event_records": [
                {"simulation_time_s": 12.0, "active": True},
                {"simulation_time_s": 12.128, "active": True},
                {"simulation_time_s": 30.0, "active": False},
            ],
        },
    )
    inspection = run_dir / "inspection_summary.json"
    _write_json(
        inspection,
        {
            "run_id": "demo-seed7000",
            "sample_count": 36,
            "alarms_raised": 3,
            "alarm_events": [
                {
                    "event_type": "raised",
                    "rule_id": "smoke_detected",
                    "sensor_id": "smoke_obscuration",
                    "simulation_time_s": 61.0,
                }
            ],
        },
    )
    return frames, summary, inspection


def test_default_segment_plan_is_exactly_ninety_seconds(tmp_path: Path) -> None:
    config = VIDEO.VideoBuildConfig(output=tmp_path / "demo.mp4")

    plan = VIDEO.build_segment_plan(config)

    assert plan.total_frames == 900
    assert plan.title_frames == 50
    assert plan.architecture_frames == 70
    assert plan.body_frames == 720
    assert plan.ending_frames == 60
    assert plan.duration_s == pytest.approx(90.0)


def test_frame_sequence_requires_contiguous_frames_and_matching_metadata(tmp_path: Path) -> None:
    frames, _summary, _inspection = _fixture_files(tmp_path, frame_indexes=(0, 2))

    with pytest.raises(ValueError, match="连续无缺失"):
        VIDEO.load_frame_sequence(frames)


def test_run_identity_and_timeline_are_bound_to_capture(tmp_path: Path) -> None:
    frames, summary, inspection = _fixture_files(tmp_path)
    sequence = VIDEO.load_frame_sequence(frames)
    data = VIDEO.load_demo_data(
        summary,
        inspection_summary_path=inspection,
        expected_run_id=str(sequence.metadata["run_id"]),
    )

    assert sequence.frame_count == 3
    assert sequence.simulation_time_for_index(1) == pytest.approx(0.1)
    assert data.run_id == "demo-seed7000"
    assert data.checkpoint_times[0] == ("P1", 20.0)
    assert data.guard_active_ranges == ((12.0, 12.128),)
    assert VIDEO._active_alarm_labels(data, 62.0) == ("烟雾异常",)


def test_run_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    _frames, summary, inspection = _fixture_files(tmp_path)

    with pytest.raises(ValueError, match="run_id 不一致"):
        VIDEO.load_demo_data(
            summary,
            inspection_summary_path=inspection,
            expected_run_id="different-run",
        )


def test_source_resampling_preserves_first_and_last_frame() -> None:
    assert VIDEO.source_frame_indices(3, 7) == (0, 0, 1, 1, 1, 2, 2)
    assert VIDEO.source_frame_indices(1, 4) == (0, 0, 0, 0)


def test_cards_and_simulation_overlay_render_as_full_hd_rgb(tmp_path: Path) -> None:
    _frames, summary, inspection = _fixture_files(tmp_path)
    data = VIDEO.load_demo_data(summary, inspection_summary_path=inspection)
    fonts = VIDEO.FontBook()
    source = Image.new("RGB", (640, 360), (80, 100, 120))

    rendered = (
        VIDEO.render_title_card(data, fonts),
        VIDEO.render_architecture_card(fonts),
        VIDEO.render_simulation_frame(source, data=data, fonts=fonts, simulation_time_s=62.0),
        VIDEO.render_ending_card(data, fonts),
    )

    assert all(image.size == (1920, 1080) and image.mode == "RGB" for image in rendered)
    assert rendered[0].getpixel((200, 300)) != rendered[1].getpixel((200, 300))
    assert VIDEO.simulation_subtitle(data, 62.0)[0] == "环境异常告警"


def test_ffmpeg_command_encodes_silent_h264_full_hd(tmp_path: Path) -> None:
    command = VIDEO.ffmpeg_command(
        ffmpeg=Path("/opt/homebrew/bin/ffmpeg"),
        destination=tmp_path / "demo.mp4",
        fps=10.0,
    )

    assert "1920x1080" in command
    assert "libx264" in command
    assert "yuv420p" in command
    assert "-an" in command
    assert command[-1] == str(tmp_path / "demo.mp4")


@pytest.mark.skipif(
    not NATIVE_FFMPEG.is_file() or not HAS_CHINESE_FONT,
    reason="native FFmpeg and a Chinese font are required for the encoding smoke test",
)
def test_native_ffmpeg_smoke_writes_video_and_audit_manifest(tmp_path: Path) -> None:
    frames, summary, inspection = _fixture_files(tmp_path)
    sequence = VIDEO.load_frame_sequence(frames)
    data = VIDEO.load_demo_data(summary, inspection_summary_path=inspection)

    output, manifest_path = VIDEO.build_video(
        sequence=sequence,
        data=data,
        config=VIDEO.VideoBuildConfig(
            output=tmp_path / "defense-demo.mp4",
            ffmpeg=NATIVE_FFMPEG,
            fps=10.0,
            duration_s=1.0,
            title_duration_s=0.2,
            architecture_duration_s=0.2,
            ending_duration_s=0.2,
        ),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert output.stat().st_size > 0
    assert manifest["codec"] == "H.264/libx264"
    assert manifest["audio"] is False
    assert manifest["encoded_frame_count"] == 10
    assert manifest["output_sha256"] == VIDEO._sha256(output)
