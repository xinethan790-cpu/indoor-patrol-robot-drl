#!/usr/bin/env python3
"""Launch the four-point A* + waypoint + PPO Webots patrol demo."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7000, help="Dynamic-obstacle seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional control-step limit; the controller default is 3600.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional PPO .zip path. By default the controller prefers the Webots "
            "seed-11 latest model and falls back to the stage-2 kinematic warm start."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Result directory, relative to the repository unless absolute.",
    )
    parser.add_argument("--port", type=int, default=1262, help="Webots TCP port.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in batch fast mode, write results, and exit automatically.",
    )
    parser.add_argument(
        "--webots",
        type=Path,
        help="Webots executable; normally discovered automatically on macOS.",
    )
    return parser.parse_args()


def discover_webots(explicit: Path | None) -> Path:
    if explicit is not None:
        executable = explicit.expanduser().resolve()
    else:
        macos_default = Path("/Applications/Webots.app/Contents/MacOS/webots")
        command = shutil.which("webots")
        executable = macos_default if macos_default.is_file() else Path(command or "")
    if not executable.is_file():
        raise FileNotFoundError(
            "Webots executable was not found. Install Webots R2025a or pass --webots."
        )
    return executable


def main() -> int:
    args = parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")

    repository_root = Path(__file__).absolute().parents[1]
    webots = discover_webots(args.webots)
    subprocess.run(
        [sys.executable, str(repository_root / "scripts/configure_webots_python.py")],
        cwd=repository_root,
        check=True,
    )
    environment = os.environ.copy()
    environment["WEBOTS_PATROL_SEED"] = str(args.seed)
    if args.max_steps is not None:
        environment["WEBOTS_PATROL_MAX_STEPS"] = str(args.max_steps)
    if args.model:
        environment["WEBOTS_PATROL_MODEL"] = args.model
    if args.output_dir:
        environment["WEBOTS_PATROL_OUTPUT_DIR"] = args.output_dir
    if args.headless:
        environment["WEBOTS_AUTO_QUIT"] = "1"

    command = [str(webots), f"--port={args.port}", "--stdout", "--stderr"]
    if args.headless:
        command.extend(["--batch", "--mode=fast", "--no-rendering"])
    command.append(str(repository_root / "worlds/patrol_route_ppo.wbt"))
    completed = subprocess.run(command, cwd=repository_root, env=environment, check=False)
    meanings = {
        0: "four-point patrol completed",
        2: "patrol stopped after a measured collision",
        3: "patrol reached the configured control-step timeout",
    }
    print(f"Webots exit {completed.returncode}: {meanings.get(completed.returncode, 'error')}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
