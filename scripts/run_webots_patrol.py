#!/usr/bin/env python3
"""Launch the four-point A* + waypoint + PPO Webots patrol demo."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCENARIO_CONFIGS = {
    "operational": "configs/patrol/scenario_operational_v1.yaml",
    "stress": "configs/patrol/scenario_stress_v1.yaml",
}


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
        help=("Optional PPO .zip path for an explicitly unvalidated exploratory run."),
    )
    parser.add_argument(
        "--allow-unvalidated-model",
        action="store_true",
        help="Acknowledge that --model bypasses formal model selection.",
    )
    parser.add_argument("--run-id", help="Stable run identity; defaults to seed plus UTC time.")
    parser.add_argument(
        "--output-dir",
        help="New result directory; defaults to artifacts/results/<run-id>.",
    )
    parser.add_argument(
        "--inspection-config",
        default="configs/inspection/environmental_patrol_v1.yaml",
        help="Environmental sensing and alarm scenario YAML.",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_CONFIGS),
        default="operational",
        help=(
            "Patrol obstacle profile: operational is the default application run; "
            "stress keeps the faster out-of-domain diagnostic challenge."
        ),
    )
    parser.add_argument("--timeout-s", type=float, help="Optional whole-task simulation timeout.")
    parser.add_argument("--port", type=int, default=1262, help="Webots TCP port.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in batch fast mode, write results, and exit automatically.",
    )
    parser.add_argument(
        "--auto-quit",
        action="store_true",
        help="Exit Webots after the terminal patrol result while keeping normal rendering.",
    )
    parser.add_argument(
        "--capture-frames-dir",
        type=Path,
        help=(
            "New directory for rendered JPEG frames. Enables batch realtime rendering "
            "and automatic exit; it is intentionally separate from --headless fast mode."
        ),
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=10.0,
        help="Simulation-time frame rate used with --capture-frames-dir (default: 10).",
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
    if args.timeout_s is not None and args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be positive.")
    if not math.isfinite(args.capture_fps) or args.capture_fps <= 0.0:
        raise ValueError("--capture-fps must be positive and finite.")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.model and not args.allow_unvalidated_model:
        raise ValueError("--model requires --allow-unvalidated-model.")
    if args.headless and args.capture_frames_dir is not None:
        raise ValueError(
            "--headless fast mode and --capture-frames-dir realtime rendering "
            "are mutually exclusive."
        )

    repository_root = Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    from indoor_patrol_rl.identifiers import validate_patrol_run_id

    run_id = args.run_id or (
        f"webots-patrol-seed{args.seed}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    try:
        run_id = validate_patrol_run_id(run_id, field_name="--run-id")
    except ValueError as error:
        raise ValueError(
            "--run-id cannot contain path separators or unsupported characters; "
            "it must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$."
        ) from error
    output_value = args.output_dir or f"artifacts/results/{run_id}"
    output_directory = Path(output_value).expanduser()
    if not output_directory.is_absolute():
        output_directory = repository_root / output_directory
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(
            f"Patrol output directory already exists; choose a new run: {output_directory}"
        )
    capture_frames_directory: Path | None = None
    if args.capture_frames_dir is not None:
        capture_frames_directory = args.capture_frames_dir.expanduser()
        if not capture_frames_directory.is_absolute():
            capture_frames_directory = repository_root / capture_frames_directory
        capture_frames_directory = capture_frames_directory.resolve()
        if capture_frames_directory.exists():
            raise FileExistsError(
                "Frame capture directory already exists; choose a new directory: "
                f"{capture_frames_directory}"
            )
        if capture_frames_directory == output_directory:
            raise ValueError("--capture-frames-dir must differ from --output-dir.")
        if output_directory.is_relative_to(capture_frames_directory):
            raise ValueError("--capture-frames-dir cannot contain --output-dir.")
    webots = discover_webots(args.webots)
    subprocess.run(
        [sys.executable, str(repository_root / "scripts/configure_webots_python.py")],
        cwd=repository_root,
        check=True,
    )
    environment = os.environ.copy()
    environment["WEBOTS_PATROL_SEED"] = str(args.seed)
    environment["WEBOTS_PATROL_RUN_ID"] = run_id
    environment["WEBOTS_PATROL_OUTPUT_DIR"] = str(output_directory)
    environment["WEBOTS_PATROL_INSPECTION_CONFIG"] = args.inspection_config
    environment["WEBOTS_PATROL_SCENARIO_CONFIG"] = SCENARIO_CONFIGS[args.scenario]
    if args.max_steps is not None:
        environment["WEBOTS_PATROL_MAX_STEPS"] = str(args.max_steps)
    if args.model:
        environment["WEBOTS_PATROL_MODEL"] = args.model
    if args.allow_unvalidated_model:
        environment["WEBOTS_PATROL_ALLOW_UNVALIDATED_MODEL"] = "1"
    if args.timeout_s is not None:
        environment["WEBOTS_PATROL_TIMEOUT_S"] = str(args.timeout_s)
    if capture_frames_directory is not None:
        environment["WEBOTS_PATROL_CAPTURE_FRAMES_DIR"] = str(capture_frames_directory)
        environment["WEBOTS_PATROL_CAPTURE_FPS"] = f"{args.capture_fps:g}"
        environment["WEBOTS_AUTO_QUIT"] = "1"
    elif args.headless or args.auto_quit:
        environment["WEBOTS_AUTO_QUIT"] = "1"

    command = [str(webots), f"--port={args.port}", "--stdout", "--stderr"]
    if capture_frames_directory is not None:
        command.extend(["--batch", "--mode=realtime"])
    elif args.headless:
        command.extend(["--batch", "--mode=fast", "--no-rendering"])
    command.append(str(repository_root / "worlds/patrol_route_ppo.wbt"))
    completed = subprocess.run(command, cwd=repository_root, env=environment, check=False)
    meanings = {
        0: "four-point patrol completed",
        2: "patrol stopped after a physical contact collision",
        3: "patrol reached the configured control-step timeout",
        4: "patrol-specific TTC escape guard exhausted its bounded recovery window",
    }
    print(f"Webots exit {completed.returncode}: {meanings.get(completed.returncode, 'error')}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
