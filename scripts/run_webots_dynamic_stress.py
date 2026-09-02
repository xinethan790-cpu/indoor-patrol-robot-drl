#!/usr/bin/env python3
"""Run the fixed RB2B validation-only dynamic stress matrix."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from indoor_patrol_rl.evaluation.dynamic_stress import (
    dynamic_stress_exit_code,
    load_dynamic_stress_config,
    run_dynamic_stress,
)
from indoor_patrol_rl.training.webots_curriculum import resolve_webots_executable

DEFAULT_CONFIG = Path("configs/eval/webots_rb2b_posthoc_dynamic_stress_v1.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-registered BASE1/DENS3/FAST3 development stress matrix. "
            "The command is validation-only and has no formal-test switch."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--webots", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_dynamic_stress_config(
            args.config,
            repository_root=args.repository_root,
        )
        report = run_dynamic_stress(
            config,
            webots_executable=resolve_webots_executable(args.webots),
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    for condition in report["conditions"]:
        gate = condition["gate"]
        print(
            f"{condition['condition_id']}: "
            f"success={gate['success_rate']:.3f} "
            f"collision={gate['collision_rate']:.3f} "
            f"gate={'passed' if gate['passed'] else 'failed'}"
        )
    print(f"decision: {report['decision']['outcome']}")
    return dynamic_stress_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
