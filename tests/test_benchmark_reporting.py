from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from indoor_patrol_rl.evaluation.reporting import generate_benchmark_report


def _write_inputs(directory: Path, *, summary_episode_count: int = 2) -> tuple[Path, Path]:
    summary = {
        "schema_version": "1.0",
        "split": "validation",
        "episode_count": summary_episode_count,
        "planners": ["ppo"],
        "conditions": ["G0"],
        "by_planner_condition": [
            {
                "planner": "ppo",
                "condition_id": "G0",
                "episodes": 2,
                "success_rate": 0.5,
                "collision_rate": 0.5,
                "timeout_rate": 0.0,
                "completion_time_s_mean": 7.0,
                "path_length_ratio_mean": 1.02,
                "minimum_clearance_m_mean": 0.2,
            }
        ],
        "by_planner": [
            {
                "planner": "ppo",
                "episodes": 2,
                "success_rate": 0.5,
                "collision_rate": 0.5,
                "timeout_rate": 0.0,
                "completion_time_s_mean": 7.0,
                "path_length_ratio_mean": 1.02,
                "minimum_clearance_m_mean": 0.2,
                "near_collision_ratio_mean": 0.1,
                "action_delta_sq_mean": 0.03,
                "inference_latency_p95_ms": 0.2,
            }
        ],
    }
    summary_path = directory / "summary.json"
    episodes_path = directory / "episodes.csv"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with episodes_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["planner", "condition_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"planner": "ppo", "condition_id": "G0"},
                {"planner": "ppo", "condition_id": "G0"},
            ]
        )
    return summary_path, episodes_path


def test_generate_benchmark_report_writes_deterministic_artifacts(tmp_path: Path) -> None:
    summary_path, episodes_path = _write_inputs(tmp_path)

    written = generate_benchmark_report(
        summary_path=summary_path,
        episodes_path=episodes_path,
        output_dir=tmp_path / "report",
    )
    first_contents = [path.read_text(encoding="utf-8") for path in written]
    generate_benchmark_report(
        summary_path=summary_path,
        episodes_path=episodes_path,
        output_dir=tmp_path / "report",
    )

    assert [path.name for path in written] == [
        "outcome_rates.svg",
        "efficiency.svg",
        "benchmark_report.md",
    ]
    assert first_contents == [path.read_text(encoding="utf-8") for path in written]
    assert all(path.stat().st_size > 500 for path in written)
    assert "成功率" in first_contents[0]
    assert "平均完成时间" in first_contents[1]
    assert "SHA-256" in first_contents[2]
    assert f"--input-dir {summary_path.parent.as_posix()}" in first_contents[2]


def test_generate_benchmark_report_rejects_csv_summary_count_mismatch(
    tmp_path: Path,
) -> None:
    summary_path, episodes_path = _write_inputs(tmp_path, summary_episode_count=3)

    with pytest.raises(ValueError, match="row count"):
        generate_benchmark_report(
            summary_path=summary_path,
            episodes_path=episodes_path,
            output_dir=tmp_path / "report",
        )
