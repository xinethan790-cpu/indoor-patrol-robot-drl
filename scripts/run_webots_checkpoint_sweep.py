#!/usr/bin/env python3
"""Run a restartable validation-only sweep over Webots PPO checkpoints."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from indoor_patrol_rl.evaluation.checkpoint_sweep import (
    WebotsCheckpointSweepRequest,
    WebotsSafetyShieldProfile,
    load_safety_shield_profile,
    run_checkpoint_sweep,
    sweep_exit_code,
)
from indoor_patrol_rl.training.webots_curriculum import resolve_webots_executable


def _default_output_directory(
    repository_root: Path,
    *,
    condition: str,
    seed: int,
    model_variant: str | None,
    shield_profile: WebotsSafetyShieldProfile | None,
) -> Path:
    variant_suffix = f"_{model_variant}" if model_variant else ""
    shield_suffix = (
        f"_shield-{shield_profile.result_token}" if shield_profile is not None else ""
    )
    return repository_root / "artifacts/results" / (
        f"webots_validation_{condition.casefold()}_checkpoint_sweep_"
        f"seed{seed}{variant_suffix}{shield_suffix}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate immutable Webots PPO checkpoints on fixed validation seeds. "
            "This command never selects or copies a model."
        )
    )
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-variant")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--obstacle-count", type=int, default=1)
    parser.add_argument("--obstacle-speed-min", type=float, required=True)
    parser.add_argument("--obstacle-speed-max", type=float, required=True)
    parser.add_argument("--horizontal-route-half-length", type=float, default=1.10)
    parser.add_argument("--vertical-route-half-length", type=float, default=1.10)
    parser.add_argument("--port-base", type=int, default=13400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--webots", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--shield-profile",
        type=Path,
        help=(
            "Repository-local YAML file containing one complete validation-only "
            "safety-shield profile."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = Path(__file__).absolute().parents[1].resolve()
    condition = args.condition.upper()
    try:
        shield_profile = (
            load_safety_shield_profile(
                args.shield_profile,
                repository_root=repository_root,
            )
            if args.shield_profile is not None
            else None
        )
        output_directory = args.output_dir or _default_output_directory(
            repository_root,
            condition=condition,
            seed=args.seed,
            model_variant=args.model_variant,
            shield_profile=shield_profile,
        )
        if not output_directory.is_absolute():
            output_directory = repository_root / output_directory
        request = WebotsCheckpointSweepRequest(
            repository_root=repository_root,
            webots_executable=resolve_webots_executable(args.webots),
            seed=args.seed,
            condition=condition,
            model_variant=args.model_variant,
            steps=tuple(args.steps),
            episodes=args.episodes,
            seed_start=args.seed_start,
            obstacle_count=args.obstacle_count,
            obstacle_speed_min=args.obstacle_speed_min,
            obstacle_speed_max=args.obstacle_speed_max,
            horizontal_route_half_length=args.horizontal_route_half_length,
            vertical_route_half_length=args.vertical_route_half_length,
            port_base=args.port_base,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            shield_profile=shield_profile,
        )
        manifest = run_checkpoint_sweep(
            request,
            output_directory=output_directory,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for run in manifest["runs"]:
        metrics = run.get("metrics")
        metric_text = ""
        if isinstance(metrics, dict):
            metric_text = (
                f" success={metrics['success_rate']:.3f}"
                f" collision={metrics['collision_rate']:.3f}"
            )
        print(f"step={run['step']} status={run['status']}{metric_text}")
    print(f"sweep status: {manifest['status']}")
    print(f"wrote: {manifest['manifest_paths']['json']}")
    print(f"wrote: {manifest['manifest_paths']['yaml']}")
    return sweep_exit_code(manifest)


if __name__ == "__main__":
    sys.exit(main())
