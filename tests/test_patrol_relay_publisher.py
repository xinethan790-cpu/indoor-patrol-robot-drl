from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from indoor_patrol_rl.application import (
    PatrolLiveOutcome,
    PatrolLiveReadings,
    PatrolLiveSnapshot,
    PatrolLiveStatusWriter,
    PatrolRuntimeIdentity,
    PatrolTaskState,
)
from indoor_patrol_rl.dashboard import DashboardMap, PatrolDashboardRepository
from indoor_patrol_rl.relay import (
    DEFAULT_DEVICE_ID,
    HttpRelayTransport,
    PatrolRelayPublisher,
    RelayAcknowledgement,
    RelayBackoff,
    RelayConfigurationError,
    RelayPermanentError,
    RelayRetryableError,
    RelayStateError,
    build_signature_headers,
    canonical_dashboard_body,
    load_upload_secret,
    resolve_device_id,
    validate_https_endpoint,
)

SECRET = b"s" * 32
ENDPOINT = "https://relay.example.test/v1/patrol/dashboard"
REPOSITORY_ROOT = Path(__file__).absolute().parents[1]


@dataclass
class _Current:
    run_id: str
    sequence: int
    terminal: bool = False

    @property
    def state(self) -> SimpleNamespace:
        return SimpleNamespace(value="completed" if self.terminal else "navigating")


class _Repository:
    def __init__(self, current: _Current, run_directory: Path) -> None:
        self.current = current
        self.run_directory = run_directory
        self.read_count = 0

    def read(self) -> SimpleNamespace:
        self.read_count += 1
        return SimpleNamespace(current=self.current)


def _project(evidence: Any, _dashboard_map: Any) -> dict[str, object]:
    current = evidence.current
    return {
        "schema_version": "patrol-dashboard/v1",
        "generated_at_utc": "2026-09-01T00:00:00Z",
        "run": {
            "run_id": current.run_id,
            "sequence": current.sequence,
            "terminal": current.terminal,
        },
        "map": {"trajectory": []},
        "task": {},
        "sensors": {},
        "alarms": {},
        "result": {"available": current.terminal},
    }


def _ack(snapshot: Any) -> RelayAcknowledgement:
    return RelayAcknowledgement(
        device_id=snapshot.device_id,
        run_id=snapshot.run_id,
        sequence=snapshot.sequence,
        terminal=snapshot.terminal,
    )


class _Transport:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.uploads: list[Any] = []

    def upload(self, snapshot: Any) -> RelayAcknowledgement:
        self.uploads.append(snapshot)
        if len(self.uploads) <= self.failures:
            raise RelayRetryableError("temporary outage")
        return _ack(snapshot)


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self._body[:limit]


class _Opener:
    def __init__(self, response: _Response | Exception) -> None:
        self.response = response
        self.requests: list[tuple[Any, float]] = []

    def open(self, request: Any, *, timeout: float) -> _Response:
        self.requests.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _snapshot_body(
    *,
    run_id: str = "patrol-run-001",
    sequence: int = 7,
    terminal: bool = False,
) -> tuple[bytes, dict[str, object]]:
    payload = {
        "schema_version": "patrol-dashboard/v1",
        "run": {"run_id": run_id, "sequence": sequence, "terminal": terminal},
    }
    return canonical_dashboard_body(payload), payload


def _relay_snapshot(
    *,
    run_id: str = "patrol-run-001",
    sequence: int = 7,
    terminal: bool = False,
) -> Any:
    from indoor_patrol_rl.relay import RelaySnapshot

    body, _payload = _snapshot_body(run_id=run_id, sequence=sequence, terminal=terminal)
    return RelaySnapshot(
        device_id="macmini-m4",
        run_id=run_id,
        sequence=sequence,
        terminal=terminal,
        body=body,
        body_sha256=hashlib.sha256(body).hexdigest(),
    )


def test_hmac_headers_use_device_timestamp_nonce_and_body_hash() -> None:
    body = b'{"schema_version":"patrol-dashboard/v1"}'
    headers = build_signature_headers(
        device_id="macmini-m4",
        timestamp=1_788_192_000,
        nonce="0123456789abcdef0123456789abcdef",
        body=body,
        secret=SECRET,
    )

    body_hash = hashlib.sha256(body).hexdigest()
    canonical = (
        "macmini-m4\n1788192000\n0123456789abcdef0123456789abcdef\n" + body_hash
    ).encode()
    assert headers == {
        "X-Patrol-Signature-Version": "patrol-hmac-sha256-v1",
        "X-Patrol-Device-Id": "macmini-m4",
        "X-Patrol-Timestamp": "1788192000",
        "X-Patrol-Nonce": "0123456789abcdef0123456789abcdef",
        "X-Patrol-Body-SHA256": body_hash,
        "X-Patrol-Signature": hmac.new(SECRET, canonical, hashlib.sha256).hexdigest(),
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://relay.example.test/upload",
        "https://relay.example.test",
        "https://user:password@relay.example.test/upload",
        "https://relay.example.test/upload?device=one",
        "https://relay.example.test/upload#fragment",
    ],
)
def test_endpoint_rejects_non_https_or_ambiguous_urls(endpoint: str) -> None:
    with pytest.raises(RelayConfigurationError):
        validate_https_endpoint(endpoint)


def test_secret_and_device_identity_resolve_only_from_safe_inputs() -> None:
    assert load_upload_secret({"PATROL_RELAY_UPLOAD_SECRET": "x" * 32}) == b"x" * 32
    assert resolve_device_id(environ={}) == DEFAULT_DEVICE_ID
    assert resolve_device_id(
        environ={"PATROL_RELAY_DEVICE_ID": "macmini-m4-lab"}
    ) == "macmini-m4-lab"
    assert resolve_device_id("explicit-device", {"PATROL_RELAY_DEVICE_ID": "ignored"}) == (
        "explicit-device"
    )
    with pytest.raises(RelayConfigurationError, match="not set"):
        load_upload_secret({})
    with pytest.raises(RelayConfigurationError, match="at least 32"):
        load_upload_secret({"PATROL_RELAY_UPLOAD_SECRET": "short"})


def test_http_transport_posts_exact_body_and_requires_matching_ack() -> None:
    snapshot = _relay_snapshot(terminal=True)
    response_body = json.dumps(
        {
            "accepted": True,
            "device_id": snapshot.device_id,
            "run_id": snapshot.run_id,
            "sequence": snapshot.sequence,
            "terminal": snapshot.terminal,
        }
    ).encode()
    opener = _Opener(_Response(200, response_body))
    transport = HttpRelayTransport(
        ENDPOINT,
        device_id="macmini-m4",
        secret=SECRET,
        timeout_s=4.5,
        clock=lambda: 1_788_192_000.9,
        nonce_factory=lambda: "nonce-0123456789",
        opener=opener,
    )

    acknowledgement = transport.upload(snapshot)

    assert acknowledgement == _ack(snapshot)
    request, timeout = opener.requests[0]
    assert request.full_url == ENDPOINT
    assert request.method == "POST"
    assert request.data == snapshot.body
    assert timeout == 4.5
    normalized_headers = {key.casefold(): value for key, value in request.header_items()}
    assert normalized_headers["content-type"] == "application/json; charset=utf-8"
    assert normalized_headers["x-patrol-device-id"] == "macmini-m4"
    assert normalized_headers["x-patrol-timestamp"] == "1788192000"
    assert normalized_headers["x-patrol-nonce"] == "nonce-0123456789"
    assert normalized_headers["x-patrol-body-sha256"] == snapshot.body_sha256


def test_http_transport_classifies_network_and_http_failures() -> None:
    snapshot = _relay_snapshot()
    transport = HttpRelayTransport(
        ENDPOINT,
        device_id="macmini-m4",
        secret=SECRET,
        opener=_Opener(urllib.error.URLError("offline")),
    )
    with pytest.raises(RelayRetryableError, match="offline"):
        transport.upload(snapshot)

    retryable = HttpRelayTransport(
        ENDPOINT,
        device_id="macmini-m4",
        secret=SECRET,
        opener=_Opener(_Response(503, b"{}")),
    )
    with pytest.raises(RelayRetryableError, match="503"):
        retryable.upload(snapshot)

    permanent = HttpRelayTransport(
        ENDPOINT,
        device_id="macmini-m4",
        secret=SECRET,
        opener=_Opener(_Response(401, b"{}")),
    )
    with pytest.raises(RelayPermanentError, match="401"):
        permanent.upload(snapshot)


def test_http_transport_rejects_malformed_or_mismatched_ack() -> None:
    snapshot = _relay_snapshot(sequence=1)
    wrong_sequence = json.dumps(
        {
            "accepted": True,
            "device_id": snapshot.device_id,
            "run_id": snapshot.run_id,
            "sequence": 2,
            "terminal": False,
        }
    ).encode()
    transport = HttpRelayTransport(
        ENDPOINT,
        device_id="macmini-m4",
        secret=SECRET,
        opener=_Opener(_Response(200, wrong_sequence)),
    )
    with pytest.raises(RelayPermanentError, match="sequence"):
        transport.upload(snapshot)


def test_publisher_uploads_each_run_sequence_once_and_rejects_rollback(
    tmp_path: Path,
) -> None:
    repository = _Repository(_Current("run-a", 5), tmp_path)
    transport = _Transport()
    publisher = PatrolRelayPublisher(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        transport,
        device_id="macmini-m4",
        projector=_project,
    )

    assert publisher.publish_latest().status == "acknowledged"
    assert publisher.publish_latest().status == "duplicate"
    assert len(transport.uploads) == 1

    repository.current = _Current("run-b", 5)
    assert publisher.publish_latest().status == "acknowledged"
    assert len(transport.uploads) == 2

    repository.current = _Current("run-b", 4)
    with pytest.raises(RelayStateError, match="backward"):
        publisher.publish_latest()


def test_publisher_reuses_real_repository_and_dashboard_projection(tmp_path: Path) -> None:
    run_directory = tmp_path / "live-run"
    identity = PatrolRuntimeIdentity(
        model_id="models/demo.zip",
        model_sha256="a" * 64,
        safety_shield_enabled=True,
        safety_shield_id="shield-v1",
        safety_shield_sha256="b" * 64,
    )
    snapshot = PatrolLiveSnapshot(
        run_id="relay-integration-run",
        sequence=0,
        state=PatrolTaskState.NAVIGATING,
        outcome=PatrolLiveOutcome.RUNNING,
        seed=7000,
        control_step=0,
        simulation_time_s=0.0,
        position=(-1.6, 0.0),
        current_checkpoint_id="P1",
        current_checkpoint_ordinal=1,
        completed_checkpoint_count=0,
        checkpoint_count=4,
        progress_ratio=0.0,
        local_goal=(-1.0, -0.6),
        remaining_distance_m=4.0,
        latest_readings=PatrolLiveReadings(),
        active_alerts=(),
        runtime_identity=identity,
    )
    PatrolLiveStatusWriter(run_directory, run_id=snapshot.run_id).publish(snapshot)
    transport = _Transport()
    publisher = PatrolRelayPublisher(
        PatrolDashboardRepository(run_directory),
        DashboardMap.from_yaml(REPOSITORY_ROOT / "configs/maps/patrol_route.yaml"),
        transport,
        device_id="macmini-m4",
    )

    result = publisher.publish_latest()

    assert result.status == "acknowledged"
    assert len(transport.uploads) == 1
    uploaded = json.loads(transport.uploads[0].body)
    assert uploaded["schema_version"] == "patrol-dashboard/v1"
    assert uploaded["run"]["run_id"] == snapshot.run_id
    assert uploaded["run"]["sequence"] == 0
    assert uploaded["map"]["robot"] == {"x_m": -1.6, "y_m": 0.0}
    assert uploaded["source"]["map_qualification"] == "demonstration_only"


def test_retry_keeps_identical_body_and_terminal_requires_ack(tmp_path: Path) -> None:
    repository = _Repository(_Current("terminal-run", 12, terminal=True), tmp_path)
    transport = _Transport(failures=2)
    sleeps: list[float] = []
    events: list[tuple[str, dict[str, object]]] = []
    publisher = PatrolRelayPublisher(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        transport,
        device_id="macmini-m4",
        poll_interval_s=0.25,
        backoff=RelayBackoff(
            initial_s=1.0,
            maximum_s=8.0,
            multiplier=2.0,
            jitter_ratio=0.0,
        ),
        projector=_project,
        sleep=sleeps.append,
        random_value=lambda: 0.5,
        event_handler=lambda event, details: events.append((event, dict(details))),
    )

    acknowledgement = publisher.run_until_terminal_acknowledged()

    assert acknowledgement == RelayAcknowledgement("macmini-m4", "terminal-run", 12, True)
    assert sleeps == [1.0, 2.0]
    assert repository.read_count == 1
    assert len(transport.uploads) == 3
    assert len({item.body for item in transport.uploads}) == 1
    assert len({item.body_sha256 for item in transport.uploads}) == 1
    assert [event for event, _details in events] == [
        "prepared",
        "retrying",
        "retrying",
        "acknowledged",
    ]


def test_permanent_transport_failure_is_not_retried(tmp_path: Path) -> None:
    class RejectingTransport:
        def upload(self, _snapshot: Any) -> RelayAcknowledgement:
            raise RelayPermanentError("bad signature")

    publisher = PatrolRelayPublisher(
        _Repository(_Current("run-a", 0), tmp_path),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        RejectingTransport(),
        device_id="macmini-m4",
        projector=_project,
        sleep=lambda _delay: pytest.fail("permanent failures must not sleep or retry"),
    )

    with pytest.raises(RelayPermanentError, match="bad signature"):
        publisher.run_until_terminal_acknowledged()


def test_backoff_is_bounded_and_deterministic_without_jitter() -> None:
    backoff = RelayBackoff(
        initial_s=1.0,
        maximum_s=5.0,
        multiplier=2.0,
        jitter_ratio=0.0,
    )
    assert [backoff.delay(index, random_value=0.5) for index in range(1, 6)] == [
        1.0,
        2.0,
        4.0,
        5.0,
        5.0,
    ]
    assert backoff.delay(100_000, random_value=0.5) == 5.0
