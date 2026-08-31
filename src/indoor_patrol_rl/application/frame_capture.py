"""Deterministic Webots screenshot capture without an in-process video encoder.

The recorder intentionally works with a small exporter callable instead of
importing Webots.  This keeps cadence, filesystem, and metadata behavior
testable on machines where the Webots Python controller module is unavailable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATROL_FRAME_CAPTURE_SCHEMA_VERSION = "1.0"
PATROL_FRAME_FILENAME_PATTERN = "frame_%08d.jpg"
PATROL_FRAME_METADATA_FILENAME = "capture_metadata.json"

FrameExporter = Callable[[str, int], Any]


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _non_negative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return result


@dataclass(frozen=True, slots=True)
class PatrolFrameCaptureConfig:
    """Validated settings for one new, exclusive screenshot sequence."""

    directory: Path
    fps: float = 10.0
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        directory = Path(self.directory).expanduser()
        if not directory.is_absolute():
            raise ValueError("Frame capture directory must be absolute.")
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "fps", _positive_finite(self.fps, "fps"))
        if (
            isinstance(self.jpeg_quality, bool)
            or not isinstance(self.jpeg_quality, int)
            or not 0 <= self.jpeg_quality <= 100
        ):
            raise ValueError("jpeg_quality must be an integer between 0 and 100.")

    @property
    def frame_period_s(self) -> float:
        return 1.0 / self.fps


class SimulationFrameCadence:
    """Choose the first physics sample at or after each simulation-time deadline."""

    _TIME_TOLERANCE_S = 1e-9

    def __init__(self, fps: float) -> None:
        self.fps = _positive_finite(fps, "fps")
        self._next_deadline_index = 0
        self._last_simulation_time_s: float | None = None
        self._claim_count = 0

    @property
    def claim_count(self) -> int:
        return self._claim_count

    @property
    def next_deadline_s(self) -> float:
        return self._next_deadline_index / self.fps

    def claim(self, simulation_time_s: float) -> bool:
        """Return true once when a frame is due, advancing from an absolute clock."""

        now = _non_negative_finite(simulation_time_s, "simulation_time_s")
        if (
            self._last_simulation_time_s is not None
            and now + self._TIME_TOLERANCE_S < self._last_simulation_time_s
        ):
            raise ValueError("simulation_time_s must be monotonic.")
        self._last_simulation_time_s = now
        if now + self._TIME_TOLERANCE_S < self.next_deadline_s:
            return False

        # Normally only one deadline is crossed because capture FPS is bounded
        # by the physics rate.  Advancing past every elapsed deadline prevents a
        # delayed simulation step from producing duplicate copies of one view.
        while now + self._TIME_TOLERANCE_S >= self.next_deadline_s:
            self._next_deadline_index += 1
        self._claim_count += 1
        return True


class PatrolFrameCapture:
    """Export an auditable, continuously numbered JPEG frame sequence."""

    def __init__(
        self,
        config: PatrolFrameCaptureConfig,
        *,
        run_id: str,
        physics_step_s: float,
        exporter: FrameExporter,
    ) -> None:
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id must be non-empty.")
        if any(character in normalized_run_id for character in "/\\"):
            raise ValueError("run_id cannot contain path separators.")
        step_s = _positive_finite(physics_step_s, "physics_step_s")
        maximum_capture_fps = 1.0 / step_s
        if config.fps > maximum_capture_fps + 1e-9:
            raise ValueError(
                "fps cannot exceed the Webots physics rate "
                f"({maximum_capture_fps:g} frames/s)."
            )
        if not callable(exporter):
            raise TypeError("exporter must be callable.")

        # The directory itself is the reservation.  Even an empty existing
        # directory is refused so separate runs can never silently intermingle.
        config.directory.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.run_id = normalized_run_id
        self.physics_step_s = step_s
        self._exporter = exporter
        self._cadence = SimulationFrameCadence(config.fps)
        self._frame_paths: list[Path] = []
        self._frame_times_s: list[float] = []
        self._finalized = False

    @property
    def frame_count(self) -> int:
        return len(self._frame_paths)

    @property
    def metadata_path(self) -> Path:
        return self.config.directory / PATROL_FRAME_METADATA_FILENAME

    def capture_if_due(self, simulation_time_s: float) -> bool:
        """Export one screenshot if the absolute simulation-time cadence is due."""

        if self._finalized:
            raise RuntimeError("Frame capture has already been finalized.")
        now = _non_negative_finite(simulation_time_s, "simulation_time_s")
        if not self._cadence.claim(now):
            return False

        path = self.config.directory / (PATROL_FRAME_FILENAME_PATTERN % self.frame_count)
        if path.exists():
            raise FileExistsError(f"Frame output already exists: {path}")
        self._exporter(str(path), self.config.jpeg_quality)
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Webots did not export a non-empty frame: {path}")
        self._frame_paths.append(path)
        self._frame_times_s.append(now)
        return True

    def finalize(self, *, simulation_duration_s: float) -> Path:
        """Validate the sequence and write exclusive metadata before simulation exit."""

        if self._finalized:
            raise RuntimeError("Frame capture has already been finalized.")
        duration = _non_negative_finite(simulation_duration_s, "simulation_duration_s")
        if not self._frame_paths:
            raise RuntimeError("Frame capture produced no images; refusing successful finish.")
        if duration + 1e-9 < self._frame_times_s[-1]:
            raise ValueError("simulation_duration_s cannot precede the final frame.")

        discovered = sorted(self.config.directory.glob("frame_*.jpg"))
        if discovered != self._frame_paths:
            raise RuntimeError("Captured frame sequence is missing files or contains extras.")
        if any(not path.is_file() or path.stat().st_size <= 0 for path in discovered):
            raise RuntimeError("Captured frame sequence contains an empty or invalid file.")
        if self.metadata_path.exists():
            raise FileExistsError(f"Capture metadata already exists: {self.metadata_path}")

        resolution = _jpeg_resolution(discovered[0])
        width_px = None if resolution is None else resolution[0]
        height_px = None if resolution is None else resolution[1]
        metadata = {
            "schema_version": PATROL_FRAME_CAPTURE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "fps": self.config.fps,
            "frame_count": self.frame_count,
            "simulation_duration_s": duration,
            "first_frame_simulation_time_s": self._frame_times_s[0],
            "last_frame_simulation_time_s": self._frame_times_s[-1],
            "width_px": width_px,
            "height_px": height_px,
            "image_format": "jpeg",
            "jpeg_quality": self.config.jpeg_quality,
            "filename_pattern": PATROL_FRAME_FILENAME_PATTERN,
        }
        with self.metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self._finalized = True
        return self.metadata_path


def _jpeg_resolution(path: Path) -> tuple[int, int] | None:
    """Read JPEG dimensions from a start-of-frame marker, when one is present."""

    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    cursor = 2
    while cursor + 3 < len(data):
        if data[cursor] != 0xFF:
            cursor += 1
            continue
        while cursor < len(data) and data[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(data):
            break
        marker = data[cursor]
        cursor += 1
        if marker in {0x00, 0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if cursor + 1 >= len(data):
            break
        segment_length = int.from_bytes(data[cursor : cursor + 2], "big")
        if segment_length < 2 or cursor + segment_length > len(data):
            return None
        if marker in start_of_frame_markers and segment_length >= 7:
            height = int.from_bytes(data[cursor + 3 : cursor + 5], "big")
            width = int.from_bytes(data[cursor + 5 : cursor + 7], "big")
            return (width, height) if min(width, height) > 0 else None
        cursor += segment_length
    return None
