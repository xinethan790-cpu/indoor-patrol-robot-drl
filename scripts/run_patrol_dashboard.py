#!/usr/bin/env python3
"""启动只读的本地室内巡检看板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="明确指定一个巡检运行结果目录（不会自动选择最近结果）。",
    )
    parser.add_argument(
        "--map-config",
        type=Path,
        help="演示二维地图 YAML；默认使用 configs/maps/patrol_route.yaml。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="本机端口，默认 8765；服务始终只监听 127.0.0.1。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    from indoor_patrol_rl.dashboard import serve_dashboard

    run_directory = args.run_dir.expanduser()
    if not run_directory.is_absolute():
        run_directory = repository_root / run_directory
    map_config = args.map_config or repository_root / "configs/maps/patrol_route.yaml"
    if not map_config.is_absolute():
        map_config = repository_root / map_config
    serve_dashboard(
        run_directory=run_directory,
        map_config=map_config,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
