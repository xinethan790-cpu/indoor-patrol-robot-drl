#!/usr/bin/env python3
"""Generate the public, diagnostic-only Webots risk report and SVG figures.

The generator is intentionally dependency-free: it uses only the Python
standard library, never launches Webots, and fails closed unless every input
matches the published, locked v1 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

LOCKED_V1_REPORT_SHA256 = "ff560bb2c4d3ef8df3e180c6d117e6ce324b306493fb900af7fdd65ac68c2e50"
LOCKED_METRICS_JSON_SHA256 = "7671e26fed44813f9594c6445e9ff0fb3a3e6c489b8b2a7db151f65c16e3fb73"
LOCKED_METRICS_CSV_SHA256 = "0487ebc927d63b1de3f53035f4e94b3a19cd26cb4ab9f51d3237af92dcdc056b"
LOCKED_SENSITIVITY_REPORT_SHA256 = (
    "88cedc14df9b72b53ecf2948e68a891a95dc3b38aeaf2fea8a45d6d653e3acbd"
)
LOCKED_SENSITIVITY_EPISODES_SHA256 = (
    "6444d16b72cd47430b90a07ad20803abbfb1ee061986d987542b67fefba758ec"
)
EXPECTED_SEEDS = tuple(range(20494, 20614))
EXEMPLAR_SEED = 20500
OUTPUT_FILENAMES = (
    "diagnostic_report.md",
    "outcomes_by_axis.svg",
    "mechanism_evidence_v1_vs_sensitivity.svg",
    "risk_timeline_seed20500.svg",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array.")
    return value


def _require_file_sha(path: Path, expected: str, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise PermissionError(
            f"{label} is not the locked published artifact: expected {expected}, actual {actual}."
        )
    return actual


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse {label}: {error}") from error


def _validate_v1_report(report: Mapping[str, Any]) -> None:
    scope = _mapping(report.get("scope"), "v1.scope")
    decision = _mapping(report.get("mechanism_decision"), "v1.mechanism_decision")
    geometry = _mapping(decision.get("geometry"), "v1.geometry")
    response = _mapping(decision.get("response"), "v1.response")
    inputs = _mapping(report.get("inputs"), "v1.inputs")
    artifacts = _mapping(report.get("artifacts"), "v1.artifacts")
    metrics_json = _mapping(artifacts.get("risk_episode_metrics_json"), "v1.artifacts.metrics_json")
    metrics_csv = _mapping(artifacts.get("risk_episode_metrics_csv"), "v1.artifacts.metrics_csv")

    if report.get("schema_version") != 1 or report.get("episode_count") != 120:
        raise PermissionError("v1 report must be the 120-episode schema-v1 diagnostic report.")
    if report.get("scenario_seeds") != list(EXPECTED_SEEDS):
        raise PermissionError("v1 report does not contain the exact locked diagnostic seeds.")
    expected_scope = {
        "split": "diagnostic",
        "formal_mechanism_inference": True,
        "checkpoint_selection": "forbidden",
        "validation_or_test_access": "forbidden",
    }
    if any(scope.get(key) != value for key, value in expected_scope.items()):
        raise PermissionError("v1 report scope is not diagnostic-only and selection-safe.")
    if decision.get("decision") != "indeterminate":
        raise PermissionError("The locked v1 formal mechanism decision must remain indeterminate.")
    if (
        geometry.get("decision") != "indeterminate"
        or geometry.get("lidar_only_episodes") != 47
        or geometry.get("classifiable_lidar_only_episodes") != 19
        or geometry.get("ambiguous_lidar_only_episodes") != 28
    ):
        raise PermissionError("Unexpected v1 formal geometry evidence.")
    insufficient = _mapping(
        response.get("short_horizon_response_insufficient"), "v1.response.insufficient"
    )
    timely = _mapping(response.get("timely_deceleration_response"), "v1.response.timely")
    if (
        response.get("decision") != "indeterminate"
        or response.get("physical_episodes") != 22
        or response.get("reaction_evaluable_physical_episodes") != 22
        or insufficient.get("successes") != 22
        or timely.get("successes") != 0
    ):
        raise PermissionError("Unexpected v1 formal response evidence.")
    if metrics_json.get("sha256") != LOCKED_METRICS_JSON_SHA256:
        raise PermissionError("v1 report does not bind the locked metrics JSON.")
    if metrics_csv.get("sha256") != LOCKED_METRICS_CSV_SHA256:
        raise PermissionError("v1 report does not bind the locked metrics CSV.")
    collection = _mapping(inputs.get("collection_report"), "v1.inputs.collection_report")
    if collection.get("status") != "collection_complete":
        raise PermissionError("v1 report does not bind a completed diagnostic collection.")


def _validate_sensitivity_report(report: Mapping[str, Any]) -> None:
    scope = _mapping(report.get("scope"), "sensitivity.scope")
    decision = _mapping(report.get("post_hoc_sensitivity_decision"), "sensitivity.decision")
    threshold_result = _mapping(
        decision.get("unchanged_v1_threshold_application"),
        "sensitivity.unchanged_v1_threshold_application",
    )
    geometry = _mapping(threshold_result.get("geometry"), "sensitivity.geometry")
    response = _mapping(threshold_result.get("response"), "sensitivity.response")
    inputs = _mapping(report.get("inputs"), "sensitivity.inputs")
    v1_input = _mapping(inputs.get("v1_risk_analysis_report"), "sensitivity.inputs.v1")
    json_input = _mapping(inputs.get("v1_risk_episode_metrics_json"), "sensitivity.inputs.json")
    csv_input = _mapping(inputs.get("v1_risk_episode_metrics_csv"), "sensitivity.inputs.csv")

    expected_scope = {
        "split": "diagnostic",
        "episode_count": 120,
        "post_hoc_sensitivity": True,
        "formal_mechanism_inference": False,
        "does_not_replace_v1": True,
        "checkpoint_selection": "forbidden",
        "validation_or_test_access": "forbidden",
        "webots_execution": "forbidden_and_not_used",
    }
    if report.get("audit_type") != "post_hoc_sensitivity" or any(
        scope.get(key) != value for key, value in expected_scope.items()
    ):
        raise PermissionError("LiDAR-origin report is not the locked nonformal diagnostic audit.")
    if scope.get("scenario_seeds") != list(EXPECTED_SEEDS):
        raise PermissionError("Sensitivity report does not contain the exact diagnostic seeds.")
    if (
        decision.get("not_a_formal_mechanism_decision") is not True
        or threshold_result.get("decision") != "dynamic_short_horizon_response_insufficient"
        or geometry.get("physical_imminent_dynamic", {}).get("successes") != 43
        or geometry.get("ambiguous_lidar_only_episodes") != 4
        or response.get("short_horizon_response_insufficient", {}).get("successes") != 46
        or response.get("timely_deceleration_response", {}).get("successes") != 0
    ):
        raise PermissionError("Unexpected post-hoc LiDAR-origin sensitivity evidence.")
    if (
        v1_input.get("sha256") != LOCKED_V1_REPORT_SHA256
        or v1_input.get("formal_decision_unchanged") != "indeterminate"
        or json_input.get("sha256") != LOCKED_METRICS_JSON_SHA256
        or csv_input.get("sha256") != LOCKED_METRICS_CSV_SHA256
    ):
        raise PermissionError("Sensitivity report does not bind the locked v1 evidence.")


def _load_and_crosscheck_metrics(json_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    metrics_raw = _list(_load_json(json_path, "risk metrics JSON"), "risk metrics JSON")
    metrics = [_mapping(value, f"metrics[{index}]") for index, value in enumerate(metrics_raw)]
    if [value.get("scenario_seed") for value in metrics] != list(EXPECTED_SEEDS):
        raise PermissionError("Metrics JSON must contain the exact ordered diagnostic seeds.")
    required = {
        "episode_id",
        "scenario_seed",
        "route_axis",
        "outcome",
        "mechanism_classification",
        "risk_onset_step",
        "risk_onset_time_s",
        "risk_onset_to_terminal_s",
        "risk_curve",
    }
    for index, value in enumerate(metrics):
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"metrics[{index}] is missing {missing}.")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(metrics):
        raise ValueError("Metrics CSV/JSON row counts do not match.")
    comparison_fields = ("episode_id", "route_axis", "outcome", "mechanism_classification")
    for index, (row, metric) in enumerate(zip(rows, metrics, strict=True)):
        if int(row.get("scenario_seed", -1)) != metric["scenario_seed"] or any(
            row.get(field) != str(metric[field]) for field in comparison_fields
        ):
            raise ValueError(f"Metrics CSV/JSON mismatch at row {index}.")
    return metrics


def load_locked_inputs(
    *,
    v1_report_path: Path,
    metrics_json_path: Path,
    metrics_csv_path: Path,
    sensitivity_report_path: Path,
) -> dict[str, Any]:
    """Load and semantically cross-check the four immutable report inputs."""

    hashes = {
        "v1_report": _require_file_sha(v1_report_path, LOCKED_V1_REPORT_SHA256, "v1 risk report"),
        "metrics_json": _require_file_sha(
            metrics_json_path, LOCKED_METRICS_JSON_SHA256, "v1 metrics JSON"
        ),
        "metrics_csv": _require_file_sha(
            metrics_csv_path, LOCKED_METRICS_CSV_SHA256, "v1 metrics CSV"
        ),
        "sensitivity_report": _require_file_sha(
            sensitivity_report_path,
            LOCKED_SENSITIVITY_REPORT_SHA256,
            "LiDAR-origin sensitivity report",
        ),
    }
    v1_report = _mapping(_load_json(v1_report_path, "v1 report"), "v1 report")
    sensitivity = _mapping(
        _load_json(sensitivity_report_path, "sensitivity report"), "sensitivity report"
    )
    _validate_v1_report(v1_report)
    _validate_sensitivity_report(sensitivity)
    sensitivity_episodes_path = sensitivity_report_path.with_name(
        "lidar_origin_sensitivity_per_episode.json"
    )
    hashes["sensitivity_episodes"] = _require_file_sha(
        sensitivity_episodes_path,
        LOCKED_SENSITIVITY_EPISODES_SHA256,
        "LiDAR-origin sensitivity per-episode JSON",
    )
    sensitivity_artifact = _mapping(
        sensitivity.get("artifacts", {}).get("per_episode_json"),
        "sensitivity.artifacts.per_episode_json",
    )
    if sensitivity_artifact.get("sha256") != LOCKED_SENSITIVITY_EPISODES_SHA256:
        raise PermissionError("Sensitivity report does not bind its locked per-episode JSON.")
    sensitivity_rows = [
        _mapping(value, f"sensitivity_rows[{index}]")
        for index, value in enumerate(
            _list(
                _load_json(sensitivity_episodes_path, "sensitivity per-episode JSON"),
                "sensitivity per-episode JSON",
            )
        )
    ]
    if len(sensitivity_rows) != 47:
        raise PermissionError("Sensitivity per-episode evidence must contain 47 LiDAR-only rows.")
    seed_20523 = next(
        (value for value in sensitivity_rows if value.get("scenario_seed") == 20523), None
    )
    if seed_20523 is None or (
        seed_20523.get("gps_unique_attributed_obstacle_index") != 0
        or seed_20523.get("post_hoc_sensitivity_mechanism_classification") != "ambiguous"
        or seed_20523.get("post_hoc_sensitivity_mechanism_reason")
        != "geometry_falls_inside_proxy_tolerance_band"
    ):
        raise PermissionError("Unexpected locked seed-20523 tolerance-band evidence.")
    unresolved_seeds = {
        value.get("scenario_seed")
        for value in sensitivity_rows
        if value.get("post_hoc_sensitivity_classification") == "unresolved_under_both_origins"
    }
    if unresolved_seeds != {20503, 20598, 20607}:
        raise PermissionError("Unexpected under-both-origins unresolved sensitivity seeds.")
    metrics = _load_and_crosscheck_metrics(metrics_json_path, metrics_csv_path)

    axis_counts = Counter((value["route_axis"], value["outcome"]) for value in metrics)
    expected_axis_counts = {
        ("horizontal", "success"): 29,
        ("horizontal", "collision"): 31,
        ("vertical", "success"): 41,
        ("vertical", "collision"): 19,
    }
    if dict(axis_counts) != expected_axis_counts:
        raise PermissionError(f"Unexpected route-axis outcomes: {dict(axis_counts)}")

    physical_seeds = [
        int(value["scenario_seed"])
        for value in metrics
        if value["mechanism_classification"] == "physical_imminent_dynamic"
    ]
    if not physical_seeds or min(physical_seeds) != EXEMPLAR_SEED:
        raise PermissionError("Seed 20500 is no longer the fixed lowest physical v1 exemplar.")
    exemplar = next(value for value in metrics if value["scenario_seed"] == EXEMPLAR_SEED)
    curve = [
        _mapping(value, "exemplar.risk_curve[]")
        for value in _list(exemplar["risk_curve"], "exemplar.risk_curve")
    ]
    steps = [value.get("step") for value in curve]
    if (
        steps != list(range(46, 62))
        or exemplar["outcome"] != "collision"
        or exemplar["mechanism_classification"] != "physical_imminent_dynamic"
        or exemplar["risk_onset_step"] != 52
    ):
        raise PermissionError("Unexpected locked seed-20500 exemplar trajectory.")
    if any(
        not isinstance(value.get("post_elapsed_s"), (int, float))
        or not isinstance(value.get("dynamic_gap_m"), (int, float))
        or not isinstance(value.get("interval_commanded_linear_mps"), (int, float))
        for value in curve
    ):
        raise ValueError("Seed-20500 risk timeline contains non-numeric plotting values.")

    return {
        "v1_report": v1_report,
        "sensitivity": sensitivity,
        "sensitivity_episodes_path": sensitivity_episodes_path,
        "metrics": metrics,
        "axis_counts": expected_axis_counts,
        "exemplar": exemplar,
        "hashes": hashes,
    }


def _svg_document(*, width: int, height: int, title: str, description: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"  <title>{escape(title)}</title>\n"
        f"  <desc>{escape(description)}</desc>\n"
        "  <rect width=" + f'"{width}" height="{height}" fill="#f8fafc"/>\n' + body + "\n</svg>\n"
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172033",
) -> str:
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escape(value)}</text>'
    )


def _vertical_text(x: float, y: float, value: str, *, size: int = 14) -> str:
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="middle" fill="#526078" '
        f'transform="rotate(-90 {x:.1f} {y:.1f})">{escape(value)}</text>'
    )


def _line_path(points: Sequence[tuple[float, float]]) -> str:
    if not points:
        raise ValueError("Cannot draw an empty line.")
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y) in enumerate(points)
    )


def render_outcomes_by_axis(axis_counts: Mapping[tuple[str, str], int]) -> str:
    width, height = 960, 560
    left, right, top, bottom = 110.0, 70.0, 120.0, 95.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = 60.0
    lines = [
        _text(50, 48, "Diagnostic outcomes by route axis", size=28, weight=700),
        _text(
            50,
            78,
            "120 diagnostic episodes · 60 per axis · no validation/test access",
            size=15,
            fill="#526078",
        ),
    ]
    for tick in range(0, 61, 10):
        y = top + plot_height - tick / max_count * plot_height
        lines.append(
            f'  <line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" '
            'y2="' + f'{y:.1f}" stroke="#d9e0ea" stroke-width="1"/>'
        )
        lines.append(_text(left - 15, y + 5, str(tick), size=13, anchor="end", fill="#526078"))
    lines.append(_vertical_text(28, top + plot_height / 2, "episodes"))
    colors = {"success": "#167d64", "collision": "#c94b55"}
    labels = {"success": "Success", "collision": "Collision"}
    group_centers = [left + plot_width * 0.29, left + plot_width * 0.71]
    bar_width, gap = 105.0, 22.0
    for center, axis in zip(group_centers, ("horizontal", "vertical"), strict=True):
        for index, outcome in enumerate(("success", "collision")):
            value = axis_counts[(axis, outcome)]
            x = center + (index - 0.5) * (bar_width + gap)
            bar_height = value / max_count * plot_height
            y = top + plot_height - bar_height
            lines.append(
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="6" fill="{colors[outcome]}"/>'
            )
            lines.append(
                _text(
                    x + bar_width / 2,
                    y - 10,
                    f"{value} ({value / 60:.1%})",
                    size=15,
                    weight=700,
                    anchor="middle",
                )
            )
            lines.append(
                _text(
                    x + bar_width / 2,
                    top + plot_height + 28,
                    labels[outcome],
                    size=14,
                    anchor="middle",
                    fill="#526078",
                )
            )
        lines.append(
            _text(center, height - 30, axis.capitalize(), size=18, weight=700, anchor="middle")
        )
    lines.extend(
        [
            f'  <rect x="{width - 305}" y="36" width="18" height="18" rx="3" fill="#167d64"/>',
            _text(width - 278, 51, "Success", size=14),
            f'  <rect x="{width - 190}" y="36" width="18" height="18" rx="3" fill="#c94b55"/>',
            _text(width - 163, 51, "Collision", size=14),
        ]
    )
    return _svg_document(
        width=width,
        height=height,
        title="Diagnostic outcomes by route axis",
        description=(
            "Horizontal: 29 successes and 31 collisions. Vertical: 41 successes and 19 collisions."
        ),
        body="\n".join(lines),
    )


def render_mechanism_evidence() -> str:
    width, height = 1020, 650
    left, right = 260.0, 70.0
    scale_width = width - left - right
    lines = [
        _text(
            50, 48, "Mechanism evidence: formal v1 vs LiDAR-origin sensitivity", size=27, weight=700
        ),
        _text(
            50,
            79,
            "The sensitivity audit is post-hoc and nonformal; it does not replace v1.",
            size=15,
            fill="#526078",
        ),
    ]
    for tick in range(0, 51, 10):
        x = left + tick / 50 * scale_width
        lines.append(
            f'  <line x1="{x:.1f}" y1="125" x2="{x:.1f}" y2="565" '
            'stroke="#d9e0ea" stroke-width="1"/>'
        )
        lines.append(_text(x, 590, str(tick), size=13, anchor="middle", fill="#526078"))

    geometry = (("Formal v1", 19, 28), ("Post-hoc sensitivity", 43, 4))
    response = (("Formal v1", 22, 0), ("Post-hoc sensitivity", 46, 0))
    lines.append(_text(50, 132, "LiDAR-only geometry (n=47)", size=18, weight=700))
    for row, (label, physical, ambiguous) in enumerate(geometry):
        y = 158 + row * 64
        physical_width = physical / 50 * scale_width
        ambiguous_width = ambiguous / 50 * scale_width
        lines.append(_text(left - 18, y + 25, label, size=14, anchor="end"))
        lines.append(
            f'  <rect x="{left:.1f}" y="{y:.1f}" width="{physical_width:.1f}" height="34" '
            'rx="4" fill="#2a6fbb"/>'
        )
        lines.append(
            f'  <rect x="{left + physical_width:.1f}" y="{y:.1f}" '
            f'width="{ambiguous_width:.1f}" height="34" rx="4" fill="#d89b2b"/>'
        )
        lines.append(
            _text(
                left + physical_width / 2,
                y + 23,
                str(physical),
                size=14,
                weight=700,
                anchor="middle",
                fill="#ffffff",
            )
        )
        lines.append(
            _text(
                left + physical_width + ambiguous_width / 2,
                y + 23,
                str(ambiguous),
                size=14,
                weight=700,
                anchor="middle",
                fill="#172033",
            )
        )

    lines.append(_text(50, 330, "Reaction-evaluable physical candidates", size=18, weight=700))
    for row, (label, insufficient, timely) in enumerate(response):
        y = 356 + row * 64
        insufficient_width = insufficient / 50 * scale_width
        timely_width = timely / 50 * scale_width
        lines.append(_text(left - 18, y + 25, label, size=14, anchor="end"))
        lines.append(
            f'  <rect x="{left:.1f}" y="{y:.1f}" width="{insufficient_width:.1f}" '
            'height="34" rx="4" fill="#c94b55"/>'
        )
        if timely_width:
            lines.append(
                f'  <rect x="{left + insufficient_width:.1f}" y="{y:.1f}" '
                f'width="{timely_width:.1f}" height="34" rx="4" fill="#167d64"/>'
            )
        lines.append(
            _text(
                left + insufficient_width / 2,
                y + 23,
                str(insufficient),
                size=14,
                weight=700,
                anchor="middle",
                fill="#ffffff",
            )
        )
        lines.append(
            _text(
                left + insufficient_width + 12, y + 23, f"timely={timely}", size=13, fill="#526078"
            )
        )

    legend_y = 620
    legend = (
        ("#2a6fbb", "physical dynamic"),
        ("#d89b2b", "ambiguous"),
        ("#c94b55", "response insufficient"),
        ("#167d64", "timely response"),
    )
    x = 120.0
    for color, label in legend:
        lines.append(
            f'  <rect x="{x:.1f}" y="{legend_y - 15}" width="16" height="16" '
            f'rx="3" fill="{color}"/>'
        )
        lines.append(_text(x + 23, legend_y, label, size=13))
        x += 205
    return _svg_document(
        width=width,
        height=height,
        title="Formal v1 and LiDAR-origin sensitivity mechanism evidence",
        description=(
            "Formal v1 has 19 classifiable physical and 28 ambiguous LiDAR-only cases, and "
            "remains indeterminate. The nonformal sensitivity has 43 physical and 4 ambiguous "
            "LiDAR-only cases. Response-insufficient counts are 22 and 46 respectively."
        ),
        body="\n".join(lines),
    )


def render_risk_timeline(exemplar: Mapping[str, Any]) -> str:
    curve = [_mapping(value, "risk_curve[]") for value in exemplar["risk_curve"]]
    width, height = 1060, 700
    left, right = 105.0, 75.0
    plot_width = width - left - right
    times = [float(value["post_elapsed_s"]) for value in curve]
    start, end = min(times), max(times)
    onset = float(exemplar["risk_onset_time_s"])

    def x_value(time_s: float) -> float:
        return left + (time_s - start) / (end - start) * plot_width

    gap_top, gap_height = 145.0, 270.0
    gap_min, gap_max = -0.05, 0.55

    def gap_y(value: float) -> float:
        clipped = min(gap_max, max(gap_min, value))
        return gap_top + (gap_max - clipped) / (gap_max - gap_min) * gap_height

    speed_top, speed_height = 485.0, 125.0
    speed_max = 0.22

    def speed_y(value: float) -> float:
        return speed_top + (speed_max - value) / speed_max * speed_height

    onset_x = x_value(onset)
    end_x = x_value(end)
    lines = [
        _text(50, 46, "Seed 20500 risk timeline (fixed exemplar)", size=28, weight=700),
        _text(
            50,
            78,
            "Lowest physical-imminent v1 seed · fixed example, not population inference",
            size=15,
            fill="#526078",
        ),
        f'  <rect x="{onset_x:.1f}" y="{gap_top:.1f}" width="{end_x - onset_x:.1f}" '
        f'height="{speed_top + speed_height - gap_top:.1f}" fill="#fee8ea" opacity="0.70"/>',
    ]
    for value in (-0.05, 0.0, 0.15, 0.30, 0.45, 0.55):
        y = gap_y(value)
        lines.append(
            f'  <line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" '
            f'y2="{y:.1f}" stroke="{("#a92f3a" if value == 0 else "#d9e0ea")}" '
            f'stroke-width="{(2 if value == 0 else 1)}"/>'
        )
        lines.append(_text(left - 14, y + 5, f"{value:.2f}", size=12, anchor="end", fill="#526078"))
    lines.append(_vertical_text(24, gap_top + gap_height / 2, "dynamic gap (m)"))
    gap_points = [
        (x_value(float(value["post_elapsed_s"])), gap_y(float(value["dynamic_gap_m"])))
        for value in curve
    ]
    lines.append(
        f'  <path d="{_line_path(gap_points)}" fill="none" stroke="#2a6fbb" '
        'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for x, y in gap_points:
        lines.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2a6fbb"/>')

    for value in (0.0, 0.1, 0.2):
        y = speed_y(value)
        lines.append(
            f'  <line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" '
            f'y2="{y:.1f}" stroke="#d9e0ea" stroke-width="1"/>'
        )
        lines.append(_text(left - 14, y + 5, f"{value:.1f}", size=12, anchor="end", fill="#526078"))
    lines.append(_vertical_text(24, speed_top + speed_height / 2, "command (m/s)"))
    speed_points = [
        (
            x_value(float(value["post_elapsed_s"])),
            speed_y(float(value["interval_commanded_linear_mps"])),
        )
        for value in curve
    ]
    lines.append(
        f'  <path d="{_line_path(speed_points)}" fill="none" stroke="#7b4ab5" '
        'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for x, y in speed_points:
        lines.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#7b4ab5"/>')

    baseline_values = [
        float(value["interval_commanded_linear_mps"])
        for value in curve
        if 50 <= int(value["step"]) <= 52
    ]
    baseline_values.sort()
    reaction_threshold = baseline_values[1] * 0.70
    threshold_y = speed_y(reaction_threshold)
    response_end = min(end, onset + 0.512)
    lines.append(
        f'  <line x1="{onset_x:.1f}" y1="{threshold_y:.1f}" '
        f'x2="{x_value(response_end):.1f}" y2="{threshold_y:.1f}" '
        'stroke="#c94b55" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    lines.append(
        _text(
            x_value(response_end) + 8,
            threshold_y + 5,
            "30% decel threshold",
            size=12,
            fill="#a92f3a",
        )
    )
    lines.append(
        f'  <line x1="{onset_x:.1f}" y1="{gap_top - 8:.1f}" x2="{onset_x:.1f}" '
        f'y2="{speed_top + speed_height:.1f}" stroke="#c94b55" stroke-width="2" '
        'stroke-dasharray="6 5"/>'
    )
    lines.append(
        _text(
            onset_x + 7,
            gap_top - 17,
            "risk onset: step 52, 6.656 s",
            size=13,
            weight=700,
            fill="#a92f3a",
        )
    )
    lines.append(
        _text(
            end_x - 5,
            gap_top - 17,
            "LiDAR collision: step 61, 7.808 s",
            size=13,
            weight=700,
            anchor="end",
            fill="#a92f3a",
        )
    )
    for time_s in (5.888, 6.272, 6.656, 7.04, 7.424, 7.808):
        x = x_value(time_s)
        lines.append(_text(x, 652, f"{time_s:.3f}", size=12, anchor="middle", fill="#526078"))
    lines.append(
        _text(
            left + plot_width / 2,
            680,
            "post elapsed time (s)",
            size=14,
            anchor="middle",
            fill="#526078",
        )
    )
    lines.append(
        _text(785, 110, "shaded = persistent dynamic-risk interval", size=13, fill="#a92f3a")
    )
    return _svg_document(
        width=width,
        height=height,
        title="Risk timeline for fixed exemplar seed 20500",
        description=(
            "Dynamic gap decreases from 0.511 metres to negative overlap. Persistent risk begins "
            "at 6.656 seconds and LiDAR collision occurs at 7.808 seconds. The commanded speed "
            "does not cross the 30 percent deceleration threshold within the 0.512 second "
            "response window."
        ),
        body="\n".join(lines),
    )


def _relative_link(output_directory: Path, target: Path) -> str:
    relative = os.path.relpath(target.resolve(), start=output_directory.resolve())
    return Path(relative).as_posix()


def _fmt_hash_row(label: str, path_link: str, digest: str) -> str:
    return f"| {label} | [`{Path(path_link).name}`]({path_link}) | `{digest}` |"


def render_markdown(
    *,
    loaded: Mapping[str, Any],
    output_directory: Path,
    v1_report_path: Path,
    metrics_json_path: Path,
    metrics_csv_path: Path,
    sensitivity_report_path: Path,
) -> str:
    v1 = _mapping(loaded["v1_report"], "loaded.v1")
    sensitivity = _mapping(loaded["sensitivity"], "loaded.sensitivity")
    exemplar = _mapping(loaded["exemplar"], "loaded.exemplar")
    hashes = _mapping(loaded["hashes"], "loaded.hashes")
    axis_counts = loaded["axis_counts"]
    v1_geometry = v1["mechanism_decision"]["geometry"]
    v1_response = v1["mechanism_decision"]["response"]
    sensitivity_counts = sensitivity["attribution_response_counts"]
    sensitivity_corrected = sensitivity["corrected_physical_candidate_set"]
    collection = v1["inputs"]["collection_report"]
    context = v1["inputs"]["episode_context"]
    episodes = v1["inputs"]["episodes_csv"]
    model_sha = v1["scope"]["model_sha256"]

    links = {
        "v1": _relative_link(output_directory, v1_report_path),
        "json": _relative_link(output_directory, metrics_json_path),
        "csv": _relative_link(output_directory, metrics_csv_path),
        "sensitivity": _relative_link(output_directory, sensitivity_report_path),
        "sensitivity_episodes": _relative_link(
            output_directory, loaded["sensitivity_episodes_path"]
        ),
    }
    hash_rows = [
        _fmt_hash_row("formal v1 analysis report", links["v1"], hashes["v1_report"]),
        _fmt_hash_row("per-episode metrics JSON", links["json"], hashes["metrics_json"]),
        _fmt_hash_row("per-episode metrics CSV", links["csv"], hashes["metrics_csv"]),
        _fmt_hash_row(
            "LiDAR-origin sensitivity report",
            links["sensitivity"],
            hashes["sensitivity_report"],
        ),
        _fmt_hash_row(
            "LiDAR-origin per-episode JSON",
            links["sensitivity_episodes"],
            hashes["sensitivity_episodes"],
        ),
    ]
    hash_table = "\n".join(hash_rows)
    horizontal_success = axis_counts[("horizontal", "success")]
    horizontal_collision = axis_counts[("horizontal", "collision")]
    vertical_success = axis_counts[("vertical", "success")]
    vertical_collision = axis_counts[("vertical", "collision")]
    onset = float(exemplar["risk_onset_time_s"])
    terminal = onset + float(exemplar["risk_onset_to_terminal_s"])
    v1_insufficient = v1_response["short_horizon_response_insufficient"]["successes"]

    return rf"""# Webots 动态避障风险诊断（公开版）

> **结论边界：**本页只报告固定 `20494～20613` 的 120 个 **diagnostic** 回合。正式 v1
> 机制判定是 **`indeterminate`（证据不足，无法定论）**。LiDAR 原点敏感性分析是
> **post-hoc / nonformal** 审计，只用于解释测量原点带来的归因变化，**不替代 v1 正式判定**。
> 本阶段未访问 validation seeds 或 held-out test seeds，也没有据此选择 checkpoint。

## 1. 目标与范围

该诊断回答的是：“当前 recovery-v1 策略的碰撞，更像是 LiDAR 阈值代理误触发，还是动态障碍
近碰与短时响应不足？”它不是最终性能评测。数据固定为 120 回合，水平/垂直路线各 60 回合，
模型 SHA-256 为 `{model_sha}`。采集状态为 `{collection["status"]}`；checkpoint selection、
validation/test access 均被协议禁止。

![按路线方向统计的诊断结果](./outcomes_by_axis.svg)

- 水平路线：成功 {horizontal_success}/60（{horizontal_success / 60:.1%}），碰撞
  {horizontal_collision}/60（{horizontal_collision / 60:.1%}）。
- 垂直路线：成功 {vertical_success}/60（{vertical_success / 60:.1%}），碰撞
  {vertical_collision}/60（{vertical_collision / 60:.1%}）。
- 合计：成功 70/120（58.3%），碰撞 50/120（41.7%）。这些比例仅描述 diagnostic 分布。

## 2. 正式 v1 判定：indeterminate

正式 `relative_motion_cpa_v1` 分析在 47 个 LiDAR-only 碰撞中，仅能把
{v1_geometry["classifiable_lidar_only_episodes"]} 个唯一归因为 physical dynamic geometry，另有
{v1_geometry["ambiguous_lidar_only_episodes"]} 个无法唯一归因（歧义率
{v1_geometry["ambiguous_fraction"]:.1%}）。协议要求至少 24 个可分类失败且歧义率不高于 25%；
因此几何层不能通过门槛，正式结论保持 `indeterminate`。

响应层有 {v1_response["reaction_evaluable_physical_episodes"]} 个可评估 physical candidates：
0 个在 0.512 s 窗口内达到 30% 减速，{v1_insufficient}
个呈现 short-horizon response-insufficient signature。但样本数仍低于预注册门槛 24，不能单独把
总体机制定论为“响应不足”。

![正式 v1 与敏感性审计的机制证据](./mechanism_evidence_v1_vs_sensitivity.svg)

## 3. LiDAR 原点敏感性：只能作为后验解释

R2025a TurtleBot3 Burger 的 extension slot 在机器人坐标系中有 `[-0.03, 0.0] m` 平移。
敏感性审计用同一 extension slot 的 GPS 记录位置作为 LiDAR 原点代理，重新计算终止帧射线与动态
障碍圆的交点残差；它**没有新记录真实 LiDAR node pose**。这是对固定数据的后验重算，未运行
Webots，formal mechanism inference 明确为 `false`。

- v1 的 28 个 ambiguous LiDAR-only 回合中，
  {sensitivity_counts["resolved_by_gps_lidar_origin_proxy"]} 个因 GPS/LiDAR 代理原点获得唯一归因，
  另 {sensitivity_counts["unresolved_under_both_origins"]} 个在两种原点下都未解析；
- 代理原点下最终有 43 个 physical、4 个 ambiguous、0 个 proxy-compatible。第 4 个
  ambiguous 是 seed `20523`：它虽获得唯一归因，但几何量落在 proxy tolerance band 内，
  所以没有被改写为 physical；
- 加上 3 个 contact-confirmed，共 {sensitivity_corrected["reaction_evaluable"]} 个响应可评估候选；
  0 个 timely deceleration，{sensitivity_corrected["short_horizon_response_insufficient"]} 个为
  response-insufficient signature。

沿用 v1 阈值会得到
`dynamic_short_horizon_response_insufficient_under_sensitivity_only_rule`，但这仍是
**sensitivity-only outcome，不是正式机制结论**。下一步若要升级证据，应在新协议中直接记录 LiDAR
节点世界坐标并预先锁定判定规则，而不是把本次后验结果改名为正式结论。

## 4. 固定轨迹示例：seed 20500

seed `20500` 是 v1 中被归为 `physical_imminent_dynamic` 的**最小 scenario seed**，因此按固定规则
选为可视化示例；它不代表平均回合，也不用于总体推断。该水平路线回合以 LiDAR-only collision
结束。持续动态风险从 step 52（{onset:.3f} s）开始，step 61（{terminal:.3f} s）触发终止，
间隔 {float(exemplar["risk_onset_to_terminal_s"]):.3f} s。正式响应窗口内未检测到达到阈值的减速。

![seed 20500 风险时间线](./risk_timeline_seed20500.svg)

图中上半部分是动态障碍表面间隙（`dynamic_gap_m`），下半部分是该控制区间的线速度命令。
红色阴影仅标出该回合的 persistent dynamic-risk 区间；虚线阈值只对应 v1 的 30% 减速规则。

## 5. 证据哈希

生成器在出图前逐一复核以下直接输入，并交叉核对 CSV/JSON 的 120 个 episode、seed、路线方向、
结果和机制分类。任何字节变化都会拒绝生成。

| 直接输入 | 相对链接 | SHA-256 |
|---|---|---|
{hash_table}

上游绑定：collection report `{collection["sha256"]}`；episode context `{context["sha256"]}`；
episodes CSV `{episodes["sha256"]}`；model `{model_sha}`。

## 6. 复现命令

以下命令均在仓库根目录运行。前两条分别重算正式分析和后验敏感性；使用新的输出目录，避免覆盖
锁定证据。第三条只读取锁定产物并生成本页与 SVG，**不会启动 Webots**。

```bash
COLLECTION=artifacts/results/webots_recovery_v1_risk_trajectory_v1/collection_report.json
V1=artifacts/results/webots_recovery_v1_risk_analysis_v1/risk_analysis_report.json
python scripts/analyze_webots_risk_trajectory.py \
  --collection-report "$COLLECTION" \
  --output-dir artifacts/results/reproduced_risk_analysis
python scripts/audit_webots_lidar_origin_sensitivity.py \
  --collection-report "$COLLECTION" \
  --v1-risk-analysis-report "$V1" \
  --output-dir artifacts/results/reproduced_lidar_origin_sensitivity
python scripts/generate_webots_risk_diagnostic_report.py
```

正式 v1 的公开数据入口为 [`risk_analysis_report.json`]({links["v1"]})；后验敏感性入口为
[`lidar_origin_sensitivity_report.json`]({links["sensitivity"]})。本页只做可审计展示，不修改任何实验产物。
"""


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def generate_report(
    *,
    v1_report_path: Path,
    metrics_json_path: Path,
    metrics_csv_path: Path,
    sensitivity_report_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Verify locked evidence and atomically generate the public report files."""

    paths = (v1_report_path, metrics_json_path, metrics_csv_path, sensitivity_report_path)
    paths = (*paths, sensitivity_report_path.with_name("lidar_origin_sensitivity_per_episode.json"))
    before = [_sha256_file(path) if path.is_file() else None for path in paths]
    loaded = load_locked_inputs(
        v1_report_path=v1_report_path,
        metrics_json_path=metrics_json_path,
        metrics_csv_path=metrics_csv_path,
        sensitivity_report_path=sensitivity_report_path,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(
        loaded=loaded,
        output_directory=output_directory,
        v1_report_path=v1_report_path,
        metrics_json_path=metrics_json_path,
        metrics_csv_path=metrics_csv_path,
        sensitivity_report_path=sensitivity_report_path,
    )
    outputs = {
        "diagnostic_report.md": markdown,
        "outcomes_by_axis.svg": render_outcomes_by_axis(loaded["axis_counts"]),
        "mechanism_evidence_v1_vs_sensitivity.svg": render_mechanism_evidence(),
        "risk_timeline_seed20500.svg": render_risk_timeline(loaded["exemplar"]),
    }
    if tuple(outputs) != OUTPUT_FILENAMES:
        raise RuntimeError("Internal output manifest mismatch.")
    after_parse = [_sha256_file(path) for path in paths]
    if before != after_parse:
        raise RuntimeError("Locked inputs changed while the report was being generated.")
    for filename, content in outputs.items():
        _atomic_write(output_directory / filename, content)
    if [_sha256_file(path) for path in paths] != before:
        raise RuntimeError("Locked inputs changed while report outputs were being written.")
    return {
        "output_directory": output_directory,
        "files": {name: _sha256_file(output_directory / name) for name in outputs},
        "formal_v1_decision": "indeterminate",
        "sensitivity_formal": False,
        "split": "diagnostic",
        "validation_or_test_access": "forbidden",
        "exemplar_seed": EXEMPLAR_SEED,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    analysis = repository_root / "artifacts/results/webots_recovery_v1_risk_analysis_v1"
    sensitivity = (
        repository_root / "artifacts/results/webots_recovery_v1_lidar_origin_sensitivity_v1"
    )
    parser.add_argument(
        "--v1-report",
        type=Path,
        default=analysis / "risk_analysis_report.json",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=analysis / "risk_episode_metrics.json",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=analysis / "risk_episode_metrics.csv",
    )
    parser.add_argument(
        "--sensitivity-report",
        type=Path,
        default=sensitivity / "lidar_origin_sensitivity_report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "reports/webots_risk_diagnostic",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = generate_report(
            v1_report_path=args.v1_report,
            metrics_json_path=args.metrics_json,
            metrics_csv_path=args.metrics_csv,
            sensitivity_report_path=args.sensitivity_report,
            output_directory=args.output_dir,
        )
    except Exception as error:
        print(f"WEBOTS_RISK_REPORT_FAILED: {type(error).__name__}: {error}")
        return 2
    print(
        "WEBOTS_RISK_REPORT_OK "
        f"split={summary['split']} "
        f"formal_v1_decision={summary['formal_v1_decision']} "
        f"output={summary['output_directory']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
