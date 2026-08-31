#!/usr/bin/env python3
"""Analyze one pre-registered fixed Webots safety-system validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from indoor_patrol_rl.evaluation.fixed_safety_validation import (
    analyze_fixed_safety_validation,
)

DEFAULT_CONFIG = Path("configs/eval/webots_r2a_fixed_safety_validation_v1.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one fixed model-plus-shield Webots run, recompute its locked "
            "acceptance gate, and emit a deployment profile only on pass."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--deployment-directory", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing report artifacts; the default is fail-closed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = analyze_fixed_safety_validation(
            args.config,
            repository_root=args.repository_root,
            output_directory=args.output_directory,
            deployment_directory=args.deployment_directory,
            markdown_path=args.markdown_report,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    passed = bool(report["decision"]["passed"])
    print(
        f"Fixed safety validation {'passed' if passed else 'failed'}: "
        f"{report['decision']['outcome']}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
