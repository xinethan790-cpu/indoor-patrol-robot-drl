#!/usr/bin/env python3
"""Generate deterministic SVG charts and a Markdown report for a benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_patrol_rl.evaluation.reporting import generate_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark outputs and generate dependency-free SVG charts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/results/kinematic_validation_seed11"),
        help="Directory containing summary.json and episodes.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to --input-dir.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    written = generate_benchmark_report(
        summary_path=args.input_dir / "summary.json",
        episodes_path=args.input_dir / "episodes.csv",
        output_dir=output_dir,
    )
    for path in written:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
