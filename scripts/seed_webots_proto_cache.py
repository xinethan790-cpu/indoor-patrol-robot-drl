#!/usr/bin/env python3
"""Pre-seed the Webots on-demand PROTO asset cache from the local installation.

Webots R2023+ distributions ship ``projects/`` as manifest skeletons and fetch
PROTO files, meshes and textures on demand from ``raw.githubusercontent.com``.
On networks where that host is unreachable the simulation cannot load its
worlds.  When the Webots installation does contain the ``projects/`` content
(see ``docs/windows-setup.md`` for how to restore it), this script copies every
asset of the subtrees referenced by the repository worlds into the local
on-demand cache, keyed exactly as Webots keys it (SHA-1 of the asset URL), so
no download is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

WEBOTS_RAW_BASE = "https://raw.githubusercontent.com/cyberbotics/webots"

# Subtrees referenced (directly or transitively) by the worlds in worlds/.
SUBTREES = (
    "projects/devices/robotis",
    "projects/robots/robotis",
    "projects/objects/backgrounds",
    "projects/objects/floors",
    "projects/objects/solids",
    "projects/objects/factory",
    "projects/objects/geometries",
    "projects/appearances",
    "projects/humans",
    "projects/objects/living_room_furniture",
    "projects/objects/plants",
    "projects/objects/cabinet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    webots_home_env = os.environ.get("WEBOTS_HOME", "").strip()
    parser.add_argument(
        "--webots-home",
        type=Path,
        default=Path(webots_home_env) if webots_home_env else None,
        help="Webots installation directory (defaults to WEBOTS_HOME).",
    )
    parser.add_argument(
        "--webots-version",
        default="R2025a",
        help="Webots release tag used in the worlds' EXTERNPROTO URLs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Asset cache directory (defaults to the per-user Webots cache).",
    )
    return parser.parse_args()


def default_cache_dir() -> Path:
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise SystemExit("LOCALAPPDATA is not set; pass --cache-dir explicitly.")
        return Path(local) / "Cyberbotics" / "Webots" / "cache" / "assets"
    return Path.home() / ".cache" / "Cyberbotics" / "Webots" / "cache" / "assets"


def main() -> int:
    args = parse_args()
    if args.webots_home is None:
        raise SystemExit("Set WEBOTS_HOME or pass --webots-home.")
    webots_home = args.webots_home.expanduser().resolve()
    cache_dir = (args.cache_dir or default_cache_dir()).expanduser().resolve()
    base_url = f"{WEBOTS_RAW_BASE}/{args.webots_version}/"

    files: list[Path] = []
    for subtree in SUBTREES:
        root = webots_home / subtree
        if not root.is_dir():
            print(f"warning: missing subtree {root} (incomplete installation?)", file=sys.stderr)
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise SystemExit(
            f"No asset files found under {webots_home / 'projects'} — the installation "
            "is missing its PROTO content; see docs/windows-setup.md."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for asset in files:
        relative = asset.relative_to(webots_home).as_posix()
        key = hashlib.sha1((base_url + relative).encode()).hexdigest()
        destination = cache_dir / key
        if destination.exists():
            continue
        shutil.copy(asset, destination)
        seeded += 1
    total = len(list(cache_dir.iterdir()))
    print(f"Seeded {seeded} cache entries from {len(files)} assets; cache now holds {total}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
