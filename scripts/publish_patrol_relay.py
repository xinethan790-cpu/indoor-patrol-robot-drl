#!/usr/bin/env python3
"""Upload one validated patrol run to a remote HTTPS relay until terminal ack."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Explicit patrol result directory containing status.json and events.jsonl.",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Public HTTPS snapshot upload endpoint; redirects are refused.",
    )
    parser.add_argument(
        "--device-id",
        help=(
            "Public relay device identifier; defaults to PATROL_RELAY_DEVICE_ID or "
            "macmini-m4."
        ),
    )
    parser.add_argument(
        "--map-config",
        type=Path,
        help="Dashboard map YAML; defaults to configs/maps/patrol_route.yaml.",
    )
    parser.add_argument("--poll-interval-s", type=float, default=0.75)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--backoff-initial-s", type=float, default=1.0)
    parser.add_argument("--backoff-maximum-s", type=float, default=30.0)
    parser.add_argument("--backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--backoff-jitter-ratio", type=float, default=0.15)
    return parser.parse_args()


def _safe_event_line(event: str, details: Mapping[str, object]) -> str:
    if event in {"prepared", "acknowledged"}:
        return (
            f"PATROL_RELAY {event} run={details['run_id']} "
            f"sequence={details['sequence']} terminal={details['terminal']}"
        )
    if event == "retrying":
        return (
            f"PATROL_RELAY retrying failures={details['failure_count']} "
            f"delay_s={float(details['delay_s']):.3f}"
        )
    return f"PATROL_RELAY {event}"


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(repository_root / "src"))
    from indoor_patrol_rl.dashboard import DashboardMap, PatrolDashboardRepository
    from indoor_patrol_rl.relay import (
        HttpRelayTransport,
        PatrolRelayPublisher,
        RelayBackoff,
        load_upload_secret,
        resolve_device_id,
    )

    run_directory = args.run_dir.expanduser()
    if not run_directory.is_absolute():
        run_directory = repository_root / run_directory
    map_config = args.map_config or repository_root / "configs/maps/patrol_route.yaml"
    if not map_config.is_absolute():
        map_config = repository_root / map_config
    secret = load_upload_secret()
    device_id = resolve_device_id(args.device_id)
    transport = HttpRelayTransport(
        args.endpoint,
        device_id=device_id,
        secret=secret,
        timeout_s=args.timeout_s,
    )
    publisher = PatrolRelayPublisher(
        PatrolDashboardRepository(run_directory),
        DashboardMap.from_yaml(map_config),
        transport,
        device_id=device_id,
        poll_interval_s=args.poll_interval_s,
        backoff=RelayBackoff(
            initial_s=args.backoff_initial_s,
            maximum_s=args.backoff_maximum_s,
            multiplier=args.backoff_multiplier,
            jitter_ratio=args.backoff_jitter_ratio,
        ),
        event_handler=lambda event, details: print(
            _safe_event_line(event, details),
            flush=True,
        ),
    )
    acknowledgement = publisher.run_until_terminal_acknowledged()
    print(
        "PATROL_RELAY finished "
        f"run={acknowledgement.run_id} sequence={acknowledgement.sequence} terminal=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
