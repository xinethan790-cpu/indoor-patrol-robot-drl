#!/usr/bin/env python3
"""Summarize one completed Webots training Monitor log without rerunning simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from indoor_patrol_rl.evaluation.training_diagnostics import training_diagnostic_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic JSON report split by route, obstacle speed, "
            "and collision-detection source."
        )
    )
    parser.add_argument("monitor", type=Path, help="Stable-Baselines3 Monitor CSV.")
    parser.add_argument("--speed-min", type=float, required=True)
    parser.add_argument("--speed-max", type=float, required=True)
    parser.add_argument("--speed-bins", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = training_diagnostic_report(
            args.monitor,
            speed_minimum=args.speed_min,
            speed_maximum=args.speed_max,
            speed_bin_count=args.speed_bins,
            window_size=args.window_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Webots training diagnostic saved: {args.output}")


if __name__ == "__main__":
    main()
