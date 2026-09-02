#!/usr/bin/env python3
"""Check formal Webots warm-start evidence without starting Webots."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_patrol_rl.training import (
    WarmStartPreflightError,
    load_webots_curriculum,
    verify_warm_start_requirement,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed readiness check for formal Webots warm-start evidence."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/webots_curriculum.yaml"),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).absolute().parents[1]
    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    try:
        config = load_webots_curriculum(config_path)
        verification = verify_warm_start_requirement(
            repository_root=repository_root,
            config=config,
        )
    except (OSError, ValueError, WarmStartPreflightError) as error:
        print(f"NOT READY\n{error}")
        return 2
    if verification is None:
        print("READY: this curriculum declares no formal warm-start requirement")
        return 0
    print(
        "READY: formal Webots warm-start model, registry, summary, and episodes CSV "
        "are mutually consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
