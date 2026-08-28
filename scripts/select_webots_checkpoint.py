#!/usr/bin/env python3
"""Select a Webots PPO checkpoint using validation metrics only."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from indoor_patrol_rl.evaluation.checkpoint_selection import (
    CheckpointSelectionGate,
    build_checkpoint_selection,
    write_checkpoint_selection,
)


def _expand_globs(repository_root: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        expanded = pattern if Path(pattern).is_absolute() else str(repository_root / pattern)
        paths.update(Path(value).absolute() for value in glob.glob(expanded, recursive=True))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate and rank Webots checkpoint summaries. "
            "This command never reads the held-out test split."
        )
    )
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=[],
        help="Summary glob relative to the repository; repeat to scan multiple patterns.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--condition", default="W1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min-success-rate", type=float, default=0.75)
    parser.add_argument("--max-collision-rate", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for selection.json and selection.yaml.",
    )
    parser.add_argument(
        "--copy-selected",
        action="store_true",
        help="Copy only a gate-passing formal selection to a stable model alias.",
    )
    parser.add_argument(
        "--selected-alias",
        type=Path,
        help="Alias destination used with --copy-selected.",
    )
    args = parser.parse_args()
    if args.selected_alias is not None and not args.copy_selected:
        parser.error("--selected-alias requires --copy-selected.")

    repository_root = Path(__file__).absolute().parents[1]
    condition = args.condition.upper()
    patterns = args.summary_glob or [
        (
            f"artifacts/results/webots_validation_{condition.casefold()}_"
            f"checkpoint*_seed{args.seed}/summary.json"
        ),
        (
            f"artifacts/results/webots_validation_{condition.casefold()}_"
            f"final_step*_seed{args.seed}/summary.json"
        ),
    ]
    output_directory = args.output_dir or (
        repository_root
        / "artifacts/results"
        / f"webots_validation_{condition.casefold()}_selection_seed{args.seed}"
    )
    if not output_directory.is_absolute():
        output_directory = repository_root / output_directory
    gate = CheckpointSelectionGate(
        seed=args.seed,
        condition=condition,
        episodes=args.episodes,
        min_success_rate=args.min_success_rate,
        max_collision_rate=args.max_collision_rate,
    )
    try:
        summaries = _expand_globs(repository_root, patterns)
        selection = build_checkpoint_selection(
            summaries,
            repository_root=repository_root,
            gate=gate,
            source_globs=patterns,
        )
        json_path, yaml_path = write_checkpoint_selection(
            selection,
            output_directory=output_directory,
            repository_root=repository_root,
            copy_selected=args.copy_selected,
            alias_path=args.selected_alias,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    winner = selection["ranking_winner"]
    formal = selection["formal_selected"]
    print(f"selection status: {selection['selection_status']}")
    if winner is not None:
        print(
            "ranking winner: "
            f"step={winner['step']} gate_passed={winner['gate_passed']} "
            f"model={winner['model_path']}"
        )
    print(f"formal selected: {formal['model_path'] if formal is not None else 'none'}")
    print(f"wrote: {json_path}")
    print(f"wrote: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
