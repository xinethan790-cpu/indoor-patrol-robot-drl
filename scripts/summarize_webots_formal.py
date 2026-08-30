#!/usr/bin/env python3
"""Validate and aggregate the complete held-out Webots result matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_patrol_rl.evaluation.formal_reporting import (
    FormalResultInput,
    build_formal_report,
    write_formal_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one final lock and require its exact 400-episode G0-G3 Webots "
            "test matrix before reporting PPO statistics across training seeds "
            "11, 22, and 33."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        required=True,
        metavar=("EPISODES_CSV", "SUMMARY_JSON"),
        help=(
            "One single-planner, single-condition result pair. Repeat exactly once "
            "for every formal planner/condition combination."
        ),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        required=True,
        help="Verified final lock whose provenance every input must match.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to verify every final-lock artifact and result input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/webots_formal_test/summary.json"),
        help="Destination JSON path.",
    )
    args = parser.parse_args()

    inputs = [
        FormalResultInput(episodes_path=Path(episodes), summary_path=Path(summary))
        for episodes, summary in args.input
    ]
    try:
        report = build_formal_report(
            inputs,
            repository_root=args.repository_root,
            final_lock_path=args.lock,
        )
        output_path = write_formal_report(report, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        "Formal Webots matrix passed: "
        f"{report['matrix_validation']['episode_count']} episodes; saved {output_path}"
    )


if __name__ == "__main__":
    main()
