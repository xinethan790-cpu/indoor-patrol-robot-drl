#!/usr/bin/env python3
"""Build a silent, Chinese-captioned graduation-project demo from Webots frames.

The script deliberately renders every title, caption, progress indicator, and
metric with Pillow before handing raw RGB frames to FFmpeg.  It therefore does
not depend on FFmpeg's optional ``drawtext`` or ``subtitles`` filters, which are
not present in every Homebrew build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as error:  # pragma: no cover - exercised only on an incomplete install
    raise SystemExit(
        "视频生成需要 Pillow。请先执行：python -m pip install -e '.[video]'"
    ) from error


def _default_ffmpeg() -> Path:
    discovered = shutil.which("ffmpeg")
    if discovered:
        return Path(discovered)
    return DEFAULT_FFMPEG


VIDEO_SCHEMA_VERSION = "defense-demo-video/v1"
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
DEFAULT_OUTPUT_FPS = 10.0
DEFAULT_DURATION_S = 90.0
DEFAULT_FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FRAME_PATTERN = re.compile(r"frame_(\d{8})\.jpg")
_WINDOWS_FONT_DIRECTORY = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
DEFAULT_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    _WINDOWS_FONT_DIRECTORY / "msyh.ttc",
    _WINDOWS_FONT_DIRECTORY / "msyhbd.ttc",
    _WINDOWS_FONT_DIRECTORY / "simhei.ttf",
    _WINDOWS_FONT_DIRECTORY / "simsun.ttc",
)


@dataclass(frozen=True, slots=True)
class FrameSequence:
    directory: Path
    paths: tuple[Path, ...]
    metadata_path: Path
    metadata: Mapping[str, Any]
    fps: float
    first_time_s: float
    last_time_s: float

    @property
    def frame_count(self) -> int:
        return len(self.paths)

    def simulation_time_for_index(self, index: int) -> float:
        if not 0 <= index < self.frame_count:
            raise IndexError(index)
        if self.frame_count == 1:
            return self.first_time_s
        fraction = index / (self.frame_count - 1)
        return self.first_time_s + fraction * (self.last_time_s - self.first_time_s)


@dataclass(frozen=True, slots=True)
class DemoData:
    summary_path: Path
    summary: Mapping[str, Any]
    inspection_summary_path: Path | None
    inspection: Mapping[str, Any]
    run_id: str
    checkpoint_times: tuple[tuple[str, float], ...]
    alarm_events: tuple[Mapping[str, Any], ...]
    guard_active_ranges: tuple[tuple[float, float], ...]

    @property
    def checkpoint_total(self) -> int:
        return _as_non_negative_int(self.summary.get("checkpoints_total", 0), "checkpoints_total")

    @property
    def checkpoint_visited(self) -> int:
        return _as_non_negative_int(
            self.summary.get("checkpoints_visited", 0), "checkpoints_visited"
        )


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    fps: float
    total_frames: int
    title_frames: int
    architecture_frames: int
    body_frames: int
    ending_frames: int

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps


@dataclass(frozen=True, slots=True)
class VideoBuildConfig:
    output: Path
    ffmpeg: Path = field(default_factory=_default_ffmpeg)
    font: Path | None = None
    fps: float = DEFAULT_OUTPUT_FPS
    duration_s: float = DEFAULT_DURATION_S
    title_duration_s: float = 5.0
    architecture_duration_s: float = 7.0
    ending_duration_s: float = 6.0
    overwrite: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.fps, "fps"),
            (self.duration_s, "duration_s"),
            (self.title_duration_s, "title_duration_s"),
            (self.architecture_duration_s, "architecture_duration_s"),
            (self.ending_duration_s, "ending_duration_s"),
        ):
            _positive_finite(value, name)
        if not str(self.output).lower().endswith(".mp4"):
            raise ValueError("output must use the .mp4 extension.")


class FontBook:
    """Cache a Chinese-capable font at the sizes needed by the renderer."""

    def __init__(self, explicit_path: Path | None = None) -> None:
        candidates = (explicit_path,) if explicit_path is not None else DEFAULT_FONT_CANDIDATES
        self.path = next((path for path in candidates if path is not None and path.is_file()), None)
        if self.path is None:
            searched = ", ".join(str(path) for path in candidates if path is not None)
            raise FileNotFoundError(f"未找到可显示中文的字体。已检查：{searched}")
        self._cache: dict[int, ImageFont.FreeTypeFont] = {}

    def get(self, size: int) -> ImageFont.FreeTypeFont:
        if size <= 0:
            raise ValueError("font size must be positive.")
        if size not in self._cache:
            self._cache[size] = ImageFont.truetype(str(self.path), size=size)
        return self._cache[size]


class RawVideoEncoder:
    """Stream RGB frames to a native FFmpeg process."""

    def __init__(self, ffmpeg: Path, destination: Path, fps: float) -> None:
        if not ffmpeg.is_file():
            raise FileNotFoundError(f"FFmpeg 不存在：{ffmpeg}")
        self.command = ffmpeg_command(ffmpeg=ffmpeg, destination=destination, fps=fps)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None or self._process.stderr is None:
            self._process.kill()
            raise RuntimeError("无法建立 FFmpeg 输入输出管道。")
        self._stdin: BinaryIO = self._process.stdin
        self._stderr: BinaryIO = self._process.stderr
        self.frame_count = 0
        self._finished = False

    def write(self, image: Image.Image) -> None:
        if self._finished:
            raise RuntimeError("Encoder has already finished.")
        if image.mode != "RGB" or image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
            raise ValueError("Every encoded image must be a 1920x1080 RGB frame.")
        try:
            self._stdin.write(image.tobytes())
        except BrokenPipeError as error:
            details = self._stderr.read().decode("utf-8", errors="replace").strip()
            self._process.wait()
            raise RuntimeError(f"FFmpeg 提前退出：{details or '没有错误详情'}") from error
        self.frame_count += 1

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("Encoder has already finished.")
        self._stdin.close()
        details = self._stderr.read().decode("utf-8", errors="replace").strip()
        return_code = self._process.wait()
        self._finished = True
        if return_code != 0:
            raise RuntimeError(f"FFmpeg 编码失败（退出码 {return_code}）：{details}")

    def abort(self) -> None:
        if self._finished:
            return
        try:
            self._stdin.close()
        except (BrokenPipeError, OSError):
            pass
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._finished = True


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _as_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _as_finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取{label}：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是 JSON 对象：{path}")
    return value


def load_frame_sequence(frames_dir: Path, metadata_path: Path | None = None) -> FrameSequence:
    """Load and fail-closed on a missing, extra, or mismatched screenshot sequence."""

    directory = frames_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"帧目录不存在：{directory}")
    metadata_file = (
        metadata_path.expanduser().resolve()
        if metadata_path is not None
        else directory / "capture_metadata.json"
    )
    metadata = _load_json_object(metadata_file, "采集元数据")

    numbered: list[tuple[int, Path]] = []
    unexpected: list[Path] = []
    for path in directory.glob("frame_*.jpg"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            unexpected.append(path)
        else:
            numbered.append((int(match.group(1)), path))
    if unexpected:
        raise ValueError(f"帧目录含有命名不规范的 JPEG：{unexpected[0].name}")
    numbered.sort(key=lambda pair: pair[0])
    if not numbered:
        raise ValueError(f"帧目录中没有 frame_XXXXXXXX.jpg：{directory}")
    expected_indexes = list(range(len(numbered)))
    actual_indexes = [index for index, _path in numbered]
    if actual_indexes != expected_indexes:
        raise ValueError("Webots 帧序列必须从 00000000 开始且连续无缺失。")
    paths = tuple(path for _index, path in numbered)
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise ValueError("Webots 帧序列含有空文件。")

    recorded_count = _as_non_negative_int(metadata.get("frame_count"), "frame_count")
    if recorded_count != len(paths):
        raise ValueError(
            f"采集元数据记录 {recorded_count} 帧，但目录中实际为 {len(paths)} 帧。"
        )
    fps = _positive_finite(metadata.get("fps"), "capture fps")
    first_time_s = _as_finite_float(
        metadata.get("first_frame_simulation_time_s", 0.0),
        "first_frame_simulation_time_s",
    )
    last_time_s = _as_finite_float(
        metadata.get("last_frame_simulation_time_s"), "last_frame_simulation_time_s"
    )
    if first_time_s < 0.0 or last_time_s < first_time_s:
        raise ValueError("采集元数据中的仿真时间范围无效。")
    return FrameSequence(
        directory=directory,
        paths=paths,
        metadata_path=metadata_file,
        metadata=metadata,
        fps=fps,
        first_time_s=first_time_s,
        last_time_s=last_time_s,
    )


def load_demo_data(
    summary_path: Path,
    *,
    inspection_summary_path: Path | None = None,
    expected_run_id: str | None = None,
) -> DemoData:
    summary_file = summary_path.expanduser().resolve()
    summary = _load_json_object(summary_file, "巡检运行摘要")
    run_id = str(summary.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("巡检运行摘要缺少 run_id。")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(
            f"帧与摘要的 run_id 不一致：{expected_run_id!r} != {run_id!r}。"
        )

    inferred_inspection = summary_file.with_name("inspection_summary.json")
    inspection_file = inspection_summary_path or (
        inferred_inspection if inferred_inspection.is_file() else None
    )
    if inspection_file is None:
        inspection: Mapping[str, Any] = {}
        resolved_inspection_file = None
    else:
        resolved_inspection_file = inspection_file.expanduser().resolve()
        inspection = _load_json_object(resolved_inspection_file, "环境巡检摘要")
        inspection_run_id = str(inspection.get("run_id", "")).strip()
        if inspection_run_id and inspection_run_id != run_id:
            raise ValueError("环境巡检摘要与运行摘要的 run_id 不一致。")

    checkpoint_times: list[tuple[str, float]] = []
    records = summary.get("checkpoint_records", [])
    if not isinstance(records, list):
        raise ValueError("checkpoint_records must be a list.")
    for record in records:
        if not isinstance(record, dict) or not record.get("reached"):
            continue
        checkpoint_id = str(record.get("checkpoint_id", "巡检点"))
        reached_at = _as_finite_float(
            record.get("arrival_simulation_time_s"), "arrival_simulation_time_s"
        )
        checkpoint_times.append((checkpoint_id, reached_at))
    checkpoint_times.sort(key=lambda item: item[1])

    raw_alarm_events = inspection.get("alarm_events", [])
    if not isinstance(raw_alarm_events, list):
        raise ValueError("alarm_events must be a list.")
    alarm_events = tuple(event for event in raw_alarm_events if isinstance(event, dict))

    raw_guard_events = summary.get("ttc_escape_guard_event_records", [])
    if not isinstance(raw_guard_events, list):
        raise ValueError("ttc_escape_guard_event_records must be a list.")
    active_times = sorted(
        _as_finite_float(event.get("simulation_time_s"), "guard simulation_time_s")
        for event in raw_guard_events
        if isinstance(event, dict) and event.get("active") is True
    )
    guard_ranges: list[tuple[float, float]] = []
    for active_time in active_times:
        if guard_ranges and active_time - guard_ranges[-1][1] <= 0.35:
            guard_ranges[-1] = (guard_ranges[-1][0], active_time)
        else:
            guard_ranges.append((active_time, active_time))

    return DemoData(
        summary_path=summary_file,
        summary=summary,
        inspection_summary_path=resolved_inspection_file,
        inspection=inspection,
        run_id=run_id,
        checkpoint_times=tuple(checkpoint_times),
        alarm_events=alarm_events,
        guard_active_ranges=tuple(guard_ranges),
    )


def build_segment_plan(config: VideoBuildConfig) -> SegmentPlan:
    total = round(config.duration_s * config.fps)
    title = round(config.title_duration_s * config.fps)
    architecture = round(config.architecture_duration_s * config.fps)
    ending = round(config.ending_duration_s * config.fps)
    body = total - title - architecture - ending
    if min(total, title, architecture, ending, body) <= 0:
        raise ValueError("视频总时长必须大于三个信息卡时长之和，并为每段保留至少一帧。")
    return SegmentPlan(
        fps=config.fps,
        total_frames=total,
        title_frames=title,
        architecture_frames=architecture,
        body_frames=body,
        ending_frames=ending,
    )


def source_frame_indices(source_count: int, output_count: int) -> tuple[int, ...]:
    if source_count <= 0 or output_count <= 0:
        raise ValueError("source_count and output_count must be positive.")
    if output_count == 1 or source_count == 1:
        return (0,) * output_count
    return tuple(
        round(index * (source_count - 1) / (output_count - 1))
        for index in range(output_count)
    )


def ffmpeg_command(*, ffmpeg: Path, destination: Path, fps: float) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _gradient_background() -> Image.Image:
    colors: list[tuple[int, int, int]] = []
    for y in range(OUTPUT_HEIGHT):
        fraction = y / max(OUTPUT_HEIGHT - 1, 1)
        colors.append(
            (
                round(11 + 10 * fraction),
                round(24 + 13 * fraction),
                round(45 + 21 * fraction),
            )
        )
    strip = Image.new("RGB", (1, OUTPUT_HEIGHT))
    strip.putdata(colors)
    return strip.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT))


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text((OUTPUT_WIDTH // 2, y), text, font=font, fill=fill, anchor="ma")


def _draw_chip(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    accent: tuple[int, int, int] = (54, 211, 153),
) -> int:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0] + 54
    draw.rounded_rectangle((x, y, x + width, y + 58), radius=29, fill=(25, 47, 73))
    draw.ellipse((x + 19, y + 22, x + 31, y + 34), fill=accent)
    draw.text((x + 42, y + 29), text, font=font, fill=(220, 231, 242), anchor="lm")
    return width


def render_title_card(data: DemoData, fonts: FontBook) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((84, 74, 1836, 1006), radius=42, outline=(48, 76, 105), width=2)
    draw.ellipse((148, 140, 188, 180), fill=(54, 211, 153))
    draw.text(
        (211, 160),
        "物联网工程 · 毕业设计",
        font=fonts.get(32),
        fill=(139, 156, 176),
        anchor="lm",
    )
    _draw_centered(
        draw,
        "基于深度强化学习的室内巡检机器人",
        y=330,
        font=fonts.get(70),
        fill=(245, 249, 253),
    )
    _draw_centered(
        draw,
        "动态避障仿真系统设计与实现",
        y=438,
        font=fonts.get(70),
        fill=(98, 221, 181),
    )
    _draw_centered(
        draw,
        "从路径规划、智能决策到环境感知与告警闭环",
        y=573,
        font=fonts.get(34),
        fill=(172, 188, 207),
    )
    chips = ("Webots R2025a", "A* 全局规划", "PPO 局部避障", "TTC 安全恢复")
    chip_font = fonts.get(26)
    widths = [draw.textbbox((0, 0), chip, font=chip_font)[2] + 54 for chip in chips]
    gap = 22
    x = (OUTPUT_WIDTH - sum(widths) - gap * (len(chips) - 1)) // 2
    for chip, width in zip(chips, widths, strict=True):
        _draw_chip(draw, chip, x=x, y=700, font=chip_font)
        x += width + gap
    draw.text(
        (OUTPUT_WIDTH // 2, 920),
        f"演示运行：{data.run_id}",
        font=fonts.get(25),
        fill=(102, 123, 147),
        anchor="mm",
    )
    return image


def _architecture_box(
    draw: ImageDraw.ImageDraw,
    fonts: FontBook,
    *,
    box: tuple[int, int, int, int],
    number: str,
    title: str,
    lines: Sequence[str],
    accent: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=30, fill=(21, 42, 66), outline=(49, 77, 104), width=2)
    draw.rounded_rectangle((left + 24, top + 24, left + 82, top + 82), radius=17, fill=accent)
    draw.text((left + 53, top + 53), number, font=fonts.get(26), fill=(9, 25, 43), anchor="mm")
    draw.text((left + 104, top + 54), title, font=fonts.get(36), fill=(240, 246, 252), anchor="lm")
    for index, line in enumerate(lines):
        draw.ellipse(
            (left + 35, top + 118 + index * 52, left + 45, top + 128 + index * 52),
            fill=accent,
        )
        draw.text(
            (left + 62, top + 123 + index * 52),
            line,
            font=fonts.get(27),
            fill=(168, 186, 205),
            anchor="lm",
        )


def render_architecture_card(fonts: FontBook) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    draw.text((110, 86), "系统架构与技术路线", font=fonts.get(54), fill=(244, 249, 253))
    draw.text(
        (110, 160),
        "轻硬件、重软件：在可复现实验中完成“规划—决策—安全—业务”闭环",
        font=fonts.get(29),
        fill=(147, 166, 187),
    )
    boxes = (
        (
            (110, 260, 880, 560),
            "1",
            "任务与规划层",
            ("四点巡检任务状态机", "栅格地图与 A* 全局路径"),
            (96, 165, 250),
        ),
        (
            (1040, 260, 1810, 560),
            "2",
            "智能决策层",
            ("激光雷达状态编码", "PPO 输出线速度与角速度"),
            (54, 211, 153),
        ),
        (
            (110, 640, 880, 940),
            "4",
            "巡检应用层",
            ("温度 / 烟雾 / 一氧化碳", "异常告警、记录与指标看板"),
            (248, 113, 113),
        ),
        (
            (1040, 640, 1810, 940),
            "3",
            "安全执行层",
            ("安全盾约束危险动作", "TTC 风险检测与脱困恢复"),
            (251, 191, 36),
        ),
    )
    for box, number, title, lines, accent in boxes:
        _architecture_box(
            draw,
            fonts,
            box=box,
            number=number,
            title=title,
            lines=lines,
            accent=accent,
        )
    draw.line((880, 410, 1040, 410), fill=(81, 111, 139), width=5)
    draw.polygon(((1022, 396), (1040, 410), (1022, 424)), fill=(81, 111, 139))
    draw.line((1425, 560, 1425, 640), fill=(81, 111, 139), width=5)
    draw.polygon(((1411, 622), (1425, 640), (1439, 622)), fill=(81, 111, 139))
    draw.line((1040, 790, 880, 790), fill=(81, 111, 139), width=5)
    draw.polygon(((898, 776), (880, 790), (898, 804)), fill=(81, 111, 139))
    return image


def _completed_checkpoints(data: DemoData, simulation_time_s: float) -> int:
    return sum(time_s <= simulation_time_s for _checkpoint, time_s in data.checkpoint_times)


def _active_alarm_labels(data: DemoData, simulation_time_s: float) -> tuple[str, ...]:
    active: dict[str, str] = {}
    for event in data.alarm_events:
        event_time = _as_finite_float(event.get("simulation_time_s", 0.0), "alarm time")
        if event_time > simulation_time_s:
            continue
        rule_id = str(event.get("rule_id", "unknown"))
        event_type = event.get("event_type")
        if event_type == "raised":
            active[rule_id] = str(event.get("sensor_id", rule_id))
        elif event_type == "cleared":
            active.pop(rule_id, None)
    labels = {
        "temperature": "温度过高",
        "smoke_obscuration": "烟雾异常",
        "carbon_monoxide": "一氧化碳超标",
    }
    return tuple(labels.get(sensor_id, sensor_id) for sensor_id in active.values())


def simulation_subtitle(data: DemoData, simulation_time_s: float) -> tuple[str, str]:
    active_alarms = _active_alarm_labels(data, simulation_time_s)
    if active_alarms:
        return ("环境异常告警", "、".join(active_alarms) + "；告警事件同步写入审计记录")

    for checkpoint_id, reached_at in data.checkpoint_times:
        if reached_at <= simulation_time_s <= reached_at + 3.0:
            return (
                f"抵达巡检点 {checkpoint_id}",
                "模拟采集温度、烟雾遮蔽率与一氧化碳浓度",
            )
    for started_at, ended_at in data.guard_active_ranges:
        if started_at - 0.25 <= simulation_time_s <= ended_at + 1.0:
            return (
                "动态风险安全恢复",
                "TTC 守护器覆盖危险动作，机器人减速、脱困并继续任务",
            )

    completed = _completed_checkpoints(data, simulation_time_s)
    total = max(data.checkpoint_total, 1)
    if completed == 0:
        return ("全局路线规划", "A* 在栅格地图上生成从机器人到 P1 的可行路径")
    if completed < total:
        return (
            "深度强化学习局部避障",
            "PPO 根据激光雷达观测实时输出运动控制，并跟踪下一巡检点",
        )
    return ("巡检任务完成", "四个巡检点均已访问，正在汇总安全、路径与传感器数据")


def _contain_on_canvas(source: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    normalized = ImageOps.exif_transpose(source).convert("RGB")
    contained = ImageOps.contain(normalized, target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (7, 14, 25))
    x = (target_size[0] - contained.width) // 2
    y = (target_size[1] - contained.height) // 2
    canvas.paste(contained, (x, y))
    return canvas


def render_simulation_frame(
    source: Image.Image,
    *,
    data: DemoData,
    fonts: FontBook,
    simulation_time_s: float,
) -> Image.Image:
    image = _contain_on_canvas(source, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, OUTPUT_WIDTH, 96), fill=(13, 29, 49))
    draw.ellipse((44, 36, 64, 56), fill=(54, 211, 153))
    draw.text(
        (82, 48),
        "室内智能巡检 · Webots 真实仿真",
        font=fonts.get(31),
        fill=(237, 244, 250),
        anchor="lm",
    )
    draw.rounded_rectangle((1587, 24, 1874, 72), radius=24, fill=(28, 52, 77))
    draw.text(
        (1730, 48),
        f"仿真时间 {simulation_time_s:05.1f} s",
        font=fonts.get(24),
        fill=(176, 195, 214),
        anchor="mm",
    )

    completed = _completed_checkpoints(data, simulation_time_s)
    total = max(data.checkpoint_total, 1)
    progress = min(max(completed / total, 0.0), 1.0)
    title, subtitle = simulation_subtitle(data, simulation_time_s)
    draw.rectangle((0, 892, OUTPUT_WIDTH, OUTPUT_HEIGHT), fill=(13, 29, 49))
    draw.text((64, 938), title, font=fonts.get(36), fill=(99, 221, 181), anchor="lm")
    draw.text((64, 997), subtitle, font=fonts.get(27), fill=(211, 224, 236), anchor="lm")
    draw.text(
        (1848, 937),
        f"巡检进度  {completed}/{total}",
        font=fonts.get(28),
        fill=(222, 233, 243),
        anchor="rm",
    )
    draw.rounded_rectangle((1485, 982, 1848, 1000), radius=9, fill=(37, 59, 82))
    if progress > 0:
        draw.rounded_rectangle(
            (1485, 982, round(1485 + 363 * progress), 1000),
            radius=9,
            fill=(54, 211, 153),
        )
    return image


def _metric_value(summary: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    value = summary.get(key, default)
    return default if value is None else value


def _draw_metric(
    draw: ImageDraw.ImageDraw,
    fonts: FontBook,
    *,
    center_x: int,
    top: int,
    value: str,
    label: str,
    accent: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(
        (center_x - 250, top, center_x + 250, top + 205),
        radius=28,
        fill=(21, 43, 67),
        outline=(49, 77, 104),
        width=2,
    )
    draw.text((center_x, top + 72), value, font=fonts.get(54), fill=accent, anchor="mm")
    draw.text((center_x, top + 151), label, font=fonts.get(27), fill=(166, 185, 204), anchor="mm")


def render_ending_card(data: DemoData, fonts: FontBook) -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image)
    completed = data.summary.get("outcome") == "completed"
    draw.ellipse((110, 92, 158, 140), fill=(54, 211, 153) if completed else (248, 113, 113))
    draw.text(
        (184, 116),
        "本次巡检运行完成" if completed else "本次巡检运行结果",
        font=fonts.get(52),
        fill=(244, 249, 253),
        anchor="lm",
    )
    draw.text(
        (110, 186),
        "指标均由运行摘要自动读取，视频画面与实验记录使用同一 run_id",
        font=fonts.get(28),
        fill=(144, 165, 187),
    )
    visited = data.checkpoint_visited
    total = data.checkpoint_total
    collision_count = _as_non_negative_int(
        _metric_value(data.summary, "collision_count"), "collision_count"
    )
    path_length = _as_finite_float(
        _metric_value(data.summary, "actual_path_length_m"), "actual_path_length_m"
    )
    elapsed = _as_finite_float(
        _metric_value(data.summary, "elapsed_simulation_time_s"),
        "elapsed_simulation_time_s",
    )
    sample_count = _as_non_negative_int(
        _metric_value(data.inspection, "sample_count"), "sample_count"
    )
    alarms = _as_non_negative_int(
        _metric_value(data.inspection, "alarms_raised"), "alarms_raised"
    )
    metrics = (
        (380, 298, f"{visited} / {total}", "巡检点完成", (96, 165, 250)),
        (960, 298, str(collision_count), "碰撞次数", (54, 211, 153)),
        (1540, 298, f"{path_length:.2f} m", "实际行驶路径", (251, 191, 36)),
        (380, 570, f"{elapsed:.1f} s", "仿真任务用时", (96, 165, 250)),
        (960, 570, str(sample_count), "环境传感器样本", (54, 211, 153)),
        (1540, 570, str(alarms), "异常告警触发", (248, 113, 113)),
    )
    for center_x, top, value, label, accent in metrics:
        _draw_metric(
            draw,
            fonts,
            center_x=center_x,
            top=top,
            value=value,
            label=label,
            accent=accent,
        )
    qualification = data.summary.get("patrol_scenario_qualified")
    note = (
        "说明：以上为应用层场景演示结果；不替代正式多随机种子泛化结论。"
        if qualification is False
        else "结果来源：本次 Webots 室内巡检仿真运行。"
    )
    draw.text((OUTPUT_WIDTH // 2, 912), note, font=fonts.get(25), fill=(123, 144, 165), anchor="mm")
    draw.text(
        (OUTPUT_WIDTH // 2, 974),
        "A* 路径规划  ·  PPO 深度强化学习  ·  动态避障  ·  环境感知与告警",
        font=fonts.get(29),
        fill=(202, 217, 231),
        anchor="mm",
    )
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_version(ffmpeg: Path) -> str:
    completed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()[0].strip()


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_video(
    *,
    sequence: FrameSequence,
    data: DemoData,
    config: VideoBuildConfig,
) -> tuple[Path, Path]:
    """Render the four-part defense video and an auditable sidecar manifest."""

    output = config.output.expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if not config.overwrite:
        for target in (output, manifest_path):
            if target.exists():
                raise FileExistsError(f"输出已存在；如需替换请显式使用 --overwrite：{target}")
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_segment_plan(config)
    fonts = FontBook(config.font.expanduser().resolve() if config.font else None)
    frame_indexes = source_frame_indices(sequence.frame_count, plan.body_frames)

    video_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-", suffix=".tmp.mp4", dir=output.parent, delete=False
    )
    temporary_video = Path(video_handle.name)
    video_handle.close()
    manifest_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-", suffix=".tmp.json", dir=output.parent, delete=False
    )
    temporary_manifest = Path(manifest_handle.name)
    manifest_handle.close()
    temporary_manifest.unlink()

    encoder: RawVideoEncoder | None = None
    try:
        encoder = RawVideoEncoder(config.ffmpeg.expanduser().resolve(), temporary_video, config.fps)
        title = render_title_card(data, fonts)
        for _index in range(plan.title_frames):
            encoder.write(title)
        architecture = render_architecture_card(fonts)
        for _index in range(plan.architecture_frames):
            encoder.write(architecture)
        for source_index in frame_indexes:
            with Image.open(sequence.paths[source_index]) as source:
                simulation_time_s = sequence.simulation_time_for_index(source_index)
                encoder.write(
                    render_simulation_frame(
                        source,
                        data=data,
                        fonts=fonts,
                        simulation_time_s=simulation_time_s,
                    )
                )
        ending = render_ending_card(data, fonts)
        for _index in range(plan.ending_frames):
            encoder.write(ending)
        encoder.finish()
        if encoder.frame_count != plan.total_frames:
            raise RuntimeError(
                f"编码帧数不一致：{encoder.frame_count} != {plan.total_frames}。"
            )
        if temporary_video.stat().st_size <= 0:
            raise RuntimeError("FFmpeg 未生成非空 MP4。")

        manifest = {
            "schema_version": VIDEO_SCHEMA_VERSION,
            "run_id": data.run_id,
            "output_file": output.name,
            "output_sha256": _sha256(temporary_video),
            "codec": "H.264/libx264",
            "pixel_format": "yuv420p",
            "audio": False,
            "width_px": OUTPUT_WIDTH,
            "height_px": OUTPUT_HEIGHT,
            "fps": config.fps,
            "duration_s": plan.duration_s,
            "encoded_frame_count": plan.total_frames,
            "segments": {
                "title_frames": plan.title_frames,
                "architecture_frames": plan.architecture_frames,
                "simulation_frames": plan.body_frames,
                "ending_frames": plan.ending_frames,
            },
            "source": {
                "frame_directory": str(sequence.directory),
                "frame_count": sequence.frame_count,
                "capture_fps": sequence.fps,
                "capture_metadata": str(sequence.metadata_path),
                "capture_metadata_sha256": _sha256(sequence.metadata_path),
                "summary": str(data.summary_path),
                "summary_sha256": _sha256(data.summary_path),
                "inspection_summary": (
                    str(data.inspection_summary_path)
                    if data.inspection_summary_path is not None
                    else None
                ),
                "inspection_summary_sha256": (
                    _sha256(data.inspection_summary_path)
                    if data.inspection_summary_path is not None
                    else None
                ),
            },
            "font": str(fonts.path),
            "ffmpeg": _ffmpeg_version(config.ffmpeg.expanduser().resolve()),
        }
        _write_manifest(temporary_manifest, manifest)
        os.replace(temporary_video, output)
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        if encoder is not None:
            encoder.abort()
        temporary_video.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    return output, manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 Webots 连续截图与同次运行摘要合成为约 90 秒的中文毕设汇报视频。"
    )
    parser.add_argument("--frames-dir", type=Path, required=True, help="Webots 连续 JPEG 帧目录。")
    parser.add_argument("--summary", type=Path, required=True, help="同次巡检运行的 summary.json。")
    parser.add_argument(
        "--capture-metadata",
        type=Path,
        help="采集元数据；默认读取帧目录内的 capture_metadata.json。",
    )
    parser.add_argument(
        "--inspection-summary",
        type=Path,
        help="环境采样与告警摘要；默认读取 summary.json 同目录文件。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/videos/indoor-patrol-robot-defense-demo-v1.mp4"),
        help="输出 MP4 路径。",
    )
    parser.add_argument("--font", type=Path, help="中文 TrueType/OpenType 字体路径。")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG, help="FFmpeg 可执行文件。")
    parser.add_argument("--fps", type=float, default=DEFAULT_OUTPUT_FPS, help="输出帧率。")
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S, help="视频总时长（秒）。"
    )
    parser.add_argument("--title-duration", type=float, default=5.0, help="标题卡时长（秒）。")
    parser.add_argument(
        "--architecture-duration", type=float, default=7.0, help="系统架构卡时长（秒）。"
    )
    parser.add_argument("--ending-duration", type=float, default=6.0, help="结尾指标卡时长（秒）。")
    parser.add_argument("--overwrite", action="store_true", help="显式允许替换已有 MP4 和清单。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    sequence = load_frame_sequence(args.frames_dir, args.capture_metadata)
    expected_run_id = str(sequence.metadata.get("run_id", "")).strip() or None
    data = load_demo_data(
        args.summary,
        inspection_summary_path=args.inspection_summary,
        expected_run_id=expected_run_id,
    )
    output, manifest = build_video(
        sequence=sequence,
        data=data,
        config=VideoBuildConfig(
            output=args.output,
            ffmpeg=args.ffmpeg,
            font=args.font,
            fps=args.fps,
            duration_s=args.duration,
            title_duration_s=args.title_duration,
            architecture_duration_s=args.architecture_duration,
            ending_duration_s=args.ending_duration,
            overwrite=args.overwrite,
        ),
    )
    print(f"视频已生成：{output}")
    print(f"可复现清单：{manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
