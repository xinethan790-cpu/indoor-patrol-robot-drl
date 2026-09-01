#!/usr/bin/env python3
"""Build one deterministic, signed relay request for offline contract tests.

The preferred mode projects the verified patrol evidence through the production
repository and dashboard projector.  CI may instead provide a tracked payload
on standard input because the protected Webots result directory is deliberately
not committed.  Both modes use the production canonical JSON and HMAC helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indoor_patrol_rl.dashboard import DashboardMap, PatrolDashboardRepository
from indoor_patrol_rl.dashboard.projector import project_dashboard
from indoor_patrol_rl.relay import (
    MAX_REQUEST_BODY_BYTES,
    build_signature_headers,
    canonical_dashboard_body,
    load_upload_secret,
    resolve_device_id,
)

CONTRACT_SCHEMA_VERSION = "patrol-relay-contract-request/v1"
EXPECTED_RUN_ID = "graduation-demo-seed7000-v1"
EXPECTED_TERMINAL_SEQUENCE = 822


def parse_args() -> argparse.Namespace:
    root = Path(__file__).absolute().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        help=(
            "Verified result directory to read through PatrolDashboardRepository. "
            "When omitted, one patrol-dashboard/v1 object is read from stdin."
        ),
    )
    parser.add_argument(
        "--map-config",
        type=Path,
        default=root / "configs/maps/patrol_route.yaml",
        help="Patrol map used when --run-directory is provided.",
    )
    parser.add_argument("--device-id", default="macmini-m4")
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--expected-run-id", default=EXPECTED_RUN_ID)
    parser.add_argument(
        "--expected-sequence",
        type=int,
        default=EXPECTED_TERMINAL_SEQUENCE,
    )
    return parser.parse_args()


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def read_stdin_payload() -> Mapping[str, Any]:
    try:
        raw = json.loads(sys.stdin.read(), parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"stdin is not finite JSON: {error}") from error
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError("stdin payload must be a JSON object")
    return raw


def project_verified_run(
    run_directory: Path,
    map_config: Path,
    *,
    generated_at: datetime,
) -> Mapping[str, Any]:
    repository = PatrolDashboardRepository(run_directory)
    evidence = repository.read()
    dashboard_map = DashboardMap.from_yaml(map_config)
    return project_dashboard(evidence, dashboard_map, generated_at=generated_at)


def require_expected_terminal_payload(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_sequence: int,
) -> None:
    if payload.get("schema_version") != "patrol-dashboard/v1":
        raise ValueError("payload must use patrol-dashboard/v1")
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("payload.run must be an object")
    if run.get("run_id") != expected_run_id:
        raise ValueError("payload.run.run_id does not match the expected graduation run")
    if run.get("sequence") != expected_sequence:
        raise ValueError("payload.run.sequence does not match the expected terminal sequence")
    if run.get("terminal") is not True:
        raise ValueError("payload must be the terminal graduation snapshot")


def main() -> int:
    args = parse_args()
    if args.timestamp < 0:
        raise SystemExit("--timestamp must be non-negative")
    if args.expected_sequence < 0:
        raise SystemExit("--expected-sequence must be non-negative")

    if args.run_directory is None:
        payload = read_stdin_payload()
        source_label = "tracked_verified_fixture_projection"
    else:
        generated_at = datetime.fromtimestamp(args.timestamp, tz=UTC)
        payload = project_verified_run(
            args.run_directory,
            args.map_config,
            generated_at=generated_at,
        )
        source_label = "verified_repository_projection"

    require_expected_terminal_payload(
        payload,
        expected_run_id=args.expected_run_id,
        expected_sequence=args.expected_sequence,
    )
    body = canonical_dashboard_body(payload)
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise ValueError(f"canonical payload exceeds {MAX_REQUEST_BODY_BYTES} bytes")

    device_id = resolve_device_id(args.device_id)
    secret = load_upload_secret()
    headers = build_signature_headers(
        device_id=device_id,
        timestamp=args.timestamp,
        nonce=args.nonce,
        body=body,
        secret=secret,
    )
    body_sha256 = hashlib.sha256(body).hexdigest()
    if headers["X-Patrol-Body-SHA256"] != body_sha256:
        raise AssertionError("production signature helper returned a different body digest")

    output = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "payload_source": source_label,
        "body_bytes": len(body),
        "body_sha256": body_sha256,
        "event": {
            "httpMethod": "POST",
            "headers": {
                "Content-Type": "application/json; charset=utf-8",
                **headers,
            },
            "body": body.decode("utf-8"),
        },
    }
    sys.stdout.write(
        json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
