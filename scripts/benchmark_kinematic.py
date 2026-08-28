#!/usr/bin/env python3
"""Compare reactive, DWA, and PPO policies on identical kinematic scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_patrol_rl.evaluation import (
    EpisodeRecord,
    load_benchmark_config,
    require_test_confirmation,
    run_kinematic_benchmark,
    write_benchmark_outputs,
)


def _progress(record: EpisodeRecord, completed: int, total: int) -> None:
    print(
        f"[{completed:>3}/{total}] {record.condition_id} seed={record.scenario_seed} "
        f"planner={record.planner:<8} outcome={record.outcome}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fair, seed-matched benchmark in the pure Python backend."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/eval/benchmark.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required to consume the held-out test seeds.",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        help="Optional planner subset; defaults to the list in the YAML config.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Limit seeds per condition for a smoke run.",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        help="Override the PPO model path in the YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to artifacts/results/kinematic_<split>.",
    )
    args = parser.parse_args()

    try:
        require_test_confirmation(args.split, confirmed=args.confirm_test)
    except PermissionError as error:
        parser.error(str(error))

    config = load_benchmark_config(args.config)
    output_dir = args.output_dir or Path("artifacts/results") / f"kinematic_{args.split}"
    records = run_kinematic_benchmark(
        config,
        split=args.split,
        planners=args.planners,
        episode_limit=args.episodes,
        ppo_model=args.ppo_model,
        progress=_progress,
    )
    summary = write_benchmark_outputs(output_dir, records)
    print(f"Saved {len(records)} episodes to {output_dir / 'episodes.csv'}")
    print(f"Saved {len(summary['by_planner_condition'])} groups to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
