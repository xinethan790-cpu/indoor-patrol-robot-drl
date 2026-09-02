#!/usr/bin/env python3
"""Run the restartable W0-W2 Webots PPO curriculum and validation gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indoor_patrol_rl.training import (
    WarmStartPreflightError,
    WebotsCurriculumRunner,
    apply_preferred_resume_overrides,
    load_webots_curriculum,
    resolve_stage_index,
    resolve_webots_executable,
    verify_warm_start_requirement,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/webots_curriculum.yaml"),
    )
    parser.add_argument("--webots", type=Path, help="Explicit Webots executable or .app path.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--start-stage", help="Stage id (W0-W2) or zero-based index.")
    parser.add_argument("--stop-after-stage", help="Stage id (W0-W2) or zero-based index.")
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--max-collision-rate", type=float)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-seed-start", type=int)
    parser.add_argument(
        "--preferred-resume",
        action="append",
        default=[],
        metavar="STAGE=MODEL",
        help="Override one stage's selected starting checkpoint; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest base path; both .yaml and .json are written.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).absolute().parents[1]
    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    config = load_webots_curriculum(config_path)
    config = apply_preferred_resume_overrides(config, args.preferred_resume)
    try:
        verify_warm_start_requirement(
            repository_root=repository_root,
            config=config,
        )
    except WarmStartPreflightError as error:
        parser.error(str(error))
    start_stage = resolve_stage_index(config.stages, args.start_stage, 0)
    stop_after_stage = resolve_stage_index(
        config.stages, args.stop_after_stage, len(config.stages) - 1
    )
    if stop_after_stage < start_stage:
        parser.error("--stop-after-stage cannot precede --start-stage.")
    webots = resolve_webots_executable(args.webots)
    variant_suffix = f"_{config.model_variant}" if config.model_variant else ""
    manifest = args.manifest or (
        repository_root
        / "artifacts/results"
        / f"webots_curriculum{variant_suffix}_seed{args.seed}"
        / "run_manifest.yaml"
    )
    if not manifest.is_absolute():
        manifest = repository_root / manifest
    runner = WebotsCurriculumRunner(
        repository_root=repository_root,
        config=config,
        webots_executable=webots,
        seed=args.seed,
        manifest_path=manifest,
        dry_run=args.dry_run,
        success_rate_override=args.min_success_rate,
        collision_rate_override=args.max_collision_rate,
        episodes_override=args.eval_episodes,
        seed_start_override=args.eval_seed_start,
    )
    return runner.run(start_stage=start_stage, stop_after_stage=stop_after_stage)


if __name__ == "__main__":
    sys.exit(main())
