"""Deterministic, dependency-free benchmark charts and Markdown reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

_PLANNER_COLORS = (
    "#2563EB",
    "#F97316",
    "#16A34A",
    "#9333EA",
    "#0891B2",
    "#DC2626",
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def load_report_inputs(summary_path: Path, episodes_path: Path) -> dict[str, Any]:
    """Load a benchmark summary and verify it against the raw episode table."""

    with summary_path.open(encoding="utf-8") as stream:
        summary = dict(_require_mapping(json.load(stream), "summary"))

    required = {
        "schema_version",
        "split",
        "episode_count",
        "planners",
        "conditions",
        "by_planner_condition",
        "by_planner",
    }
    missing = required - summary.keys()
    if missing:
        raise ValueError(f"summary is missing fields: {sorted(missing)}")

    planners = tuple(str(value) for value in summary["planners"])
    conditions = tuple(str(value) for value in summary["conditions"])
    if not planners or len(planners) != len(set(planners)):
        raise ValueError("summary.planners must contain unique planner names.")
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("summary.conditions must contain unique condition ids.")

    with episodes_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"planner", "condition_id"}.issubset(rows[0]):
        raise ValueError("episodes.csv must contain planner and condition_id columns.")
    if len(rows) != int(summary["episode_count"]):
        raise ValueError(
            "episodes.csv row count does not match summary.episode_count: "
            f"{len(rows)} != {summary['episode_count']}"
        )

    raw_counts = Counter((row["planner"], row["condition_id"]) for row in rows)
    summary_counts: Counter[tuple[str, str]] = Counter()
    for raw_group in summary["by_planner_condition"]:
        group = _require_mapping(raw_group, "by_planner_condition item")
        key = (str(group["planner"]), str(group["condition_id"]))
        if key in summary_counts:
            raise ValueError(f"summary contains a duplicate planner-condition group: {key}")
        summary_counts[key] = int(group["episodes"])
    if raw_counts != summary_counts:
        raise ValueError("Planner-condition episode counts differ between CSV and summary.")
    return summary


def _group_lookup(summary: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(group["planner"]), str(group["condition_id"])): group
        for group in summary["by_planner_condition"]
    }


def _planner_colors(planners: Sequence[str]) -> dict[str, str]:
    return {
        planner: _PLANNER_COLORS[index % len(_PLANNER_COLORS)]
        for index, planner in enumerate(planners)
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _rate(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _svg_header(title: str, description: str, width: int, height: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        ),
        f"  <title>{escape(title)}</title>",
        f"  <desc>{escape(description)}</desc>",
        "  <style>",
        (
            "    text { font-family: Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; "
            "fill: #172033; }"
        ),
        "    .title { font-size: 24px; font-weight: 700; }",
        "    .subtitle { font-size: 13px; fill: #526079; }",
        "    .axis { stroke: #64748B; stroke-width: 1; }",
        "    .grid { stroke: #D7DEE9; stroke-width: 1; }",
        "    .tick { font-size: 12px; fill: #526079; }",
        "    .panel-title { font-size: 16px; font-weight: 700; }",
        "    .legend { font-size: 13px; }",
        "    .value { font-size: 9px; font-weight: 700; }",
        "  </style>",
        f'  <rect width="{width}" height="{height}" fill="#FFFFFF"/>',
        (
            f'  <text class="title" x="{width / 2:.1f}" y="34" '
            f'text-anchor="middle">{escape(title)}</text>'
        ),
        (
            f'  <text class="subtitle" x="{width / 2:.1f}" y="56" text-anchor="middle">'
            f"{escape(description)}</text>"
        ),
    ]


def _legend(
    planners: Sequence[str], colors: Mapping[str, str], *, center_x: float, y: float
) -> list[str]:
    item_width = 118.0
    x = center_x - item_width * len(planners) / 2
    lines: list[str] = []
    for planner in planners:
        label = planner.upper() if planner.lower() == "ppo" else planner.title()
        lines.append(
            f'  <rect x="{x:.1f}" y="{y - 11:.1f}" width="14" height="14" '
            f'rx="2" fill="{colors[planner]}"/>'
        )
        lines.append(
            f'  <text class="legend" x="{x + 21:.1f}" y="{y:.1f}">{escape(label)}</text>'
        )
        x += item_width
    return lines


def _nice_upper(values: Sequence[float], *, minimum: float) -> float:
    maximum = max(values, default=minimum)
    padded = max(minimum, maximum * 1.08)
    raw_tick = padded / 5.0
    magnitude = 10 ** math.floor(math.log10(raw_tick)) if raw_tick > 0 else 1.0
    normalized = raw_tick / magnitude
    if normalized <= 1.0:
        tick = magnitude
    elif normalized <= 2.0:
        tick = 2.0 * magnitude
    elif normalized <= 2.5:
        tick = 2.5 * magnitude
    elif normalized <= 5.0:
        tick = 5.0 * magnitude
    else:
        tick = 10.0 * magnitude
    return math.ceil(padded / tick) * tick


def _bar_panel(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    planners: Sequence[str],
    conditions: Sequence[str],
    colors: Mapping[str, str],
    values: Mapping[tuple[str, str], float | None],
    y_max: float,
    value_formatter: str,
) -> list[str]:
    left = x + 62
    top = y + 35
    plot_width = width - 82
    plot_height = height - 95
    bottom = top + plot_height
    lines = [
        f'  <text class="panel-title" x="{x + width / 2:.1f}" y="{y + 16:.1f}" '
        f'text-anchor="middle">{escape(title)}</text>',
        f'  <line class="axis" x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}"/>',
        (
            f'  <line class="axis" x1="{left:.1f}" y1="{bottom:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{bottom:.1f}"/>'
        ),
        (
            f'  <text class="tick" transform="translate({x + 16:.1f} {top + plot_height / 2:.1f}) '
            f'rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'
        ),
    ]
    tick_step = y_max / 5
    if math.isclose(tick_step, round(tick_step)):
        tick_digits = 0
    elif tick_step >= 0.1:
        tick_digits = 1
    else:
        tick_digits = 2
    for tick in range(6):
        value = y_max * tick / 5
        tick_y = bottom - plot_height * tick / 5
        lines.append(
            f'  <line class="grid" x1="{left:.1f}" y1="{tick_y:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{tick_y:.1f}"/>'
        )
        label = f"{value:.{tick_digits}f}"
        lines.append(
            f'  <text class="tick" x="{left - 9:.1f}" y="{tick_y + 4:.1f}" '
            f'text-anchor="end">{label}</text>'
        )

    group_width = plot_width / len(conditions)
    gap = 6.0
    bar_width = min(
        28.0,
        max(
            7.0,
            (group_width * 0.72 - gap * (len(planners) - 1)) / len(planners),
        ),
    )
    for condition_index, condition in enumerate(conditions):
        group_center = left + group_width * (condition_index + 0.5)
        total_width = len(planners) * bar_width + (len(planners) - 1) * gap
        first_x = group_center - total_width / 2
        for planner_index, planner in enumerate(planners):
            value = values.get((planner, condition))
            if value is None or not math.isfinite(value):
                continue
            bar_height = max(0.0, min(plot_height, plot_height * value / y_max))
            bar_x = first_x + planner_index * (bar_width + gap)
            bar_y = bottom - bar_height
            label = value_formatter.format(value)
            lines.extend(
                [
                    (
                        f'  <rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                        f'height="{bar_height:.1f}" rx="2" fill="{colors[planner]}">'
                        f"<title>{escape(planner)} / {escape(condition)}: "
                        f"{escape(label)}</title></rect>"
                    ),
                    (
                        f'  <text class="value" x="{bar_x + bar_width / 2:.1f}" '
                        f'y="{max(top + 9, bar_y - 5):.1f}" text-anchor="middle">'
                        f"{escape(label)}</text>"
                    ),
                ]
            )
        lines.append(
            f'  <text class="tick" x="{group_center:.1f}" y="{bottom + 22:.1f}" '
            f'text-anchor="middle">{escape(condition)}</text>'
        )
    return lines


def render_outcome_rates_svg(summary: Mapping[str, Any]) -> str:
    """Render grouped success and collision rates by benchmark condition."""

    planners = tuple(str(value) for value in summary["planners"])
    conditions = tuple(str(value) for value in summary["conditions"])
    colors = _planner_colors(planners)
    groups = _group_lookup(summary)
    success = {
        key: 100.0 * float(value["success_rate"]) for key, value in groups.items()
    }
    collision = {
        key: 100.0 * float(value["collision_rate"]) for key, value in groups.items()
    }
    lines = _svg_header(
        "成功率与碰撞率对比",
        (
            f"{summary['split']} split · {summary['episode_count']} episodes · "
            "higher success / lower collision is better"
        ),
        1200,
        570,
    )
    lines.extend(_legend(planners, colors, center_x=600, y=84))
    common = {
        "y": 105.0,
        "width": 550.0,
        "height": 430.0,
        "y_label": "Rate (%)",
        "planners": planners,
        "conditions": conditions,
        "colors": colors,
        "y_max": 100.0,
        "value_formatter": "{:.1f}%",
    }
    lines.extend(_bar_panel(x=25.0, title="成功率", values=success, **common))
    lines.extend(_bar_panel(x=625.0, title="碰撞率", values=collision, **common))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_efficiency_svg(summary: Mapping[str, Any]) -> str:
    """Render successful-run completion time and path-length efficiency."""

    planners = tuple(str(value) for value in summary["planners"])
    conditions = tuple(str(value) for value in summary["conditions"])
    colors = _planner_colors(planners)
    groups = _group_lookup(summary)
    completion = {
        key: (
            None
            if value["completion_time_s_mean"] is None
            else float(value["completion_time_s_mean"])
        )
        for key, value in groups.items()
    }
    path_ratio = {
        key: (
            None
            if value["path_length_ratio_mean"] is None
            else float(value["path_length_ratio_mean"])
        )
        for key, value in groups.items()
    }
    time_upper = _nice_upper(
        [value for value in completion.values() if value is not None], minimum=5.0
    )
    path_upper = _nice_upper(
        [value for value in path_ratio.values() if value is not None], minimum=1.0
    )
    lines = _svg_header(
        "成功回合的时间与路径效率",
        "Means include successful episodes only · lower is better",
        1200,
        570,
    )
    lines.extend(_legend(planners, colors, center_x=600, y=84))
    common = {
        "y": 105.0,
        "width": 550.0,
        "height": 430.0,
        "planners": planners,
        "conditions": conditions,
        "colors": colors,
    }
    lines.extend(
        _bar_panel(
            x=25.0,
            title="平均完成时间",
            y_label="Time (s)",
            values=completion,
            y_max=time_upper,
            value_formatter="{:.2f}",
            **common,
        )
    )
    lines.extend(
        _bar_panel(
            x=625.0,
            title="平均路径长度比",
            y_label="Actual / reference path",
            values=path_ratio,
            y_max=path_upper,
            value_formatter="{:.3f}",
            **common,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overall_highlights(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    if not groups:
        return []
    best_success = max(groups, key=lambda group: float(group["success_rate"]))
    lowest_collision = min(groups, key=lambda group: float(group["collision_rate"]))
    with_time = [group for group in groups if group["completion_time_s_mean"] is not None]
    with_path = [group for group in groups if group["path_length_ratio_mean"] is not None]
    highlights = [
        (
            f"- 最高总体成功率：**{best_success['planner']}**"
            f"（{_rate(best_success['success_rate'])}）。"
        ),
        (
            f"- 最低总体碰撞率：**{lowest_collision['planner']}**"
            f"（{_rate(lowest_collision['collision_rate'])}）。"
        ),
    ]
    if with_time:
        fastest = min(with_time, key=lambda group: float(group["completion_time_s_mean"]))
        highlights.append(
            f"- 成功回合平均完成时间最短：**{fastest['planner']}**"
            f"（{_fmt(fastest['completion_time_s_mean'], 2)} s）。"
        )
    if with_path:
        shortest = min(with_path, key=lambda group: float(group["path_length_ratio_mean"]))
        highlights.append(
            f"- 成功回合平均路径长度比最低：**{shortest['planner']}**"
            f"（{_fmt(shortest['path_length_ratio_mean'])}）。"
        )
    return highlights


def render_markdown_report(
    summary: Mapping[str, Any],
    *,
    summary_sha256: str,
    episodes_sha256: str,
    input_dir: Path,
) -> str:
    """Render a compact thesis-ready metric report from the machine summary."""

    overall = [
        _require_mapping(group, "by_planner item") for group in summary["by_planner"]
    ]
    by_condition = [
        _require_mapping(group, "by_planner_condition item")
        for group in summary["by_planner_condition"]
    ]
    lines = [
        "# 运动学验证集基准报告",
        "",
        f"- 数据划分：`{summary['split']}`（验证集结果不可代替最终 Webots 物理仿真测试）",
        f"- 总回合数：{summary['episode_count']}",
        f"- 规划器：{', '.join(f'`{name}`' for name in summary['planners'])}",
        f"- 工况：{', '.join(f'`{name}`' for name in summary['conditions'])}",
        (
            "- 生成命令：`.venv/bin/python scripts/plot_benchmark.py "
            f"--input-dir {input_dir.as_posix()}`"
        ),
        "",
        "## 总体指标",
        "",
        (
            "| 规划器 | 回合 | 成功率 | 碰撞率 | 超时率 | 完成时间¹ (s) | "
            "路径长度比¹ | 最小间隙² (m) | 近碰撞占比² | 动作变化平方² | 推理 P95² (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in overall:
        lines.append(
            f"| {group['planner']} | {group['episodes']} | {_rate(group['success_rate'])} | "
            f"{_rate(group['collision_rate'])} | {_rate(group['timeout_rate'])} | "
            f"{_fmt(group['completion_time_s_mean'], 2)} | "
            f"{_fmt(group['path_length_ratio_mean'])} | "
            f"{_fmt(group['minimum_clearance_m_mean'])} | "
            f"{_rate(group['near_collision_ratio_mean'])} | "
            f"{_fmt(group['action_delta_sq_mean'], 4)} | "
            f"{_fmt(group['inference_latency_p95_ms'], 3)} |"
        )
    lines.extend(
        [
            "",
            (
                "¹ 仅统计成功到达的回合；² 统计该规划器的全部回合。"
                "路径长度比 = 实际路径长度 / 起终点直线参考长度。"
            ),
            "",
            "## 分工况指标",
            "",
            (
                "| 工况 | 规划器 | 回合 | 成功率 | 碰撞率 | 超时率 | "
                "完成时间¹ (s) | 路径长度比¹ | 最小间隙² (m) |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    condition_order = {name: index for index, name in enumerate(summary["conditions"])}
    planner_order = {name: index for index, name in enumerate(summary["planners"])}
    by_condition.sort(
        key=lambda group: (
            condition_order[str(group["condition_id"])],
            planner_order[str(group["planner"])],
        )
    )
    for group in by_condition:
        lines.append(
            f"| {group['condition_id']} | {group['planner']} | {group['episodes']} | "
            f"{_rate(group['success_rate'])} | {_rate(group['collision_rate'])} | "
            f"{_rate(group['timeout_rate'])} | {_fmt(group['completion_time_s_mean'], 2)} | "
            f"{_fmt(group['path_length_ratio_mean'])} | {_fmt(group['minimum_clearance_m_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 可复述结论",
            "",
            *_overall_highlights(overall),
            (
                "- 上述结论只描述本次固定验证种子下的运动学后端表现；"
                "模型选择后还需在未参与调参的测试种子和 Webots 物理仿真中复核。"
            ),
            "",
            "## 图表",
            "",
            "![成功率与碰撞率](outcome_rates.svg)",
            "",
            "![时间与路径效率](efficiency.svg)",
            "",
            "## 数据完整性",
            "",
            f"- `summary.json` SHA-256：`{summary_sha256}`",
            f"- `episodes.csv` SHA-256：`{episodes_sha256}`",
            "- 报告生成器在绘图前校验 CSV 行数以及每个“规划器 × 工况”的回合数是否与 JSON 一致。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_benchmark_report(
    *, summary_path: Path, episodes_path: Path, output_dir: Path
) -> tuple[Path, ...]:
    """Validate inputs and write two SVG charts plus a Markdown metric report."""

    summary = load_report_inputs(summary_path, episodes_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcome_path = output_dir / "outcome_rates.svg"
    efficiency_path = output_dir / "efficiency.svg"
    report_path = output_dir / "benchmark_report.md"
    outcome_path.write_text(
        render_outcome_rates_svg(summary), encoding="utf-8", newline="\n"
    )
    efficiency_path.write_text(
        render_efficiency_svg(summary), encoding="utf-8", newline="\n"
    )
    report_path.write_text(
        render_markdown_report(
            summary,
            summary_sha256=_sha256(summary_path),
            episodes_sha256=_sha256(episodes_path),
            input_dir=summary_path.parent,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return outcome_path, efficiency_path, report_path
