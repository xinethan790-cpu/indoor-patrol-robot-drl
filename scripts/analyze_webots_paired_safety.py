#!/usr/bin/env python3
"""Analyze one pre-registered paired Webots safety diagnostic offline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from indoor_patrol_rl.evaluation.paired_safety import analyze_paired_safety

DEFAULT_CONFIG = Path("configs/eval/webots_r2a_front_brake_paired_diagnostic_v1.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the source/result hash chain for a pre-registered two-arm "
            "validation diagnostic, recompute paired safety statistics, and apply its gate."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Paired diagnostic YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve and verify all bound inputs.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Override the configured directory for JSON, YAML, and SHA-256 outputs.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Override the default reports/<diagnostic>/diagnostic_report.md path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing report artifacts. The default is to fail closed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = analyze_paired_safety(
            args.config,
            repository_root=args.repository_root,
            output_directory=args.output_directory,
            markdown_path=args.markdown_report,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    passed = bool(report["decision"]["passed"])
    print(
        f"Paired safety diagnostic {'passed' if passed else 'failed'}: "
        f"{report['decision']['outcome']}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
