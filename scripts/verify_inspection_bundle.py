#!/usr/bin/env python3
"""Verify an environmental inspection result bundle and its patrol binding."""

from __future__ import annotations

import argparse
from pathlib import Path

from indoor_patrol_rl.inspection import verify_inspection_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Directory containing inspection_manifest.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = verify_inspection_bundle(args.bundle)
    files = manifest["files"]
    print(
        "Inspection bundle verified: "
        f"run={manifest['run_id']} scenario={manifest['scenario_id']} files={len(files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
