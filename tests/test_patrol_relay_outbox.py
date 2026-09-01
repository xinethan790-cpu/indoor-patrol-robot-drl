from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import indoor_patrol_rl.relay.outbox as outbox_module
from indoor_patrol_rl.application import PatrolRunIdError
from indoor_patrol_rl.dashboard import DashboardEvidenceError
from indoor_patrol_rl.relay import (
    RELAY_OUTBOX_DIRECTORY_NAME,
    RELAY_OUTBOX_STATE_FILENAME,
    PatrolRelayPublisher,
    RelayAcknowledgement,
    RelayOutbox,
    RelayPermanentError,
    RelayRetryableError,
    RelayStateError,
    canonical_dashboard_body,
)


@dataclass
class _Current:
    run_id: str
    sequence: int
    terminal: bool = False

    @property
    def state(self) -> SimpleNamespace:
        return SimpleNamespace(value="completed" if self.terminal else "navigating")


class _Repository:
    def __init__(self, run_directory: Path, current: _Current) -> None:
        self.run_directory = run_directory
        self.current = current
        self.read_count = 0

    def read(self) -> SimpleNamespace:
        self.read_count += 1
        return SimpleNamespace(current=self.current)


def _payload(current: _Current, generated_at: str) -> dict[str, object]:
    return {
        "schema_version": "patrol-dashboard/v1",
        "generated_at_utc": generated_at,
        "run": {
            "run_id": current.run_id,
            "sequence": current.sequence,
            "terminal": current.terminal,
        },
    }


def _projector(generated_at: str):
    def project(evidence: Any, _dashboard_map: Any) -> dict[str, object]:
        return _payload(evidence.current, generated_at)

    return project


def _ack(snapshot: Any) -> RelayAcknowledgement:
    return RelayAcknowledgement(
        snapshot.device_id,
        snapshot.run_id,
        snapshot.sequence,
        snapshot.terminal,
    )


class _CloudStore:
    def __init__(self) -> None:
        self.accepted_body: bytes | None = None
        self.accept_count = 0

    def accept(self, snapshot: Any) -> RelayAcknowledgement:
        self.accept_count += 1
        if self.accepted_body is None:
            self.accepted_body = snapshot.body
        elif snapshot.body != self.accepted_body:
            raise AssertionError("same run/sequence was retried with a different body")
        return _ack(snapshot)


class _LostAcknowledgementTransport:
    def __init__(self, cloud: _CloudStore) -> None:
        self.cloud = cloud

    def upload(self, snapshot: Any) -> RelayAcknowledgement:
        self.cloud.accept(snapshot)
        raise RelayRetryableError("connection closed after cloud commit")


class _CloudTransport:
    def __init__(self, cloud: _CloudStore) -> None:
        self.cloud = cloud
        self.upload_count = 0

    def upload(self, snapshot: Any) -> RelayAcknowledgement:
        self.upload_count += 1
        return self.cloud.accept(snapshot)


def _pending_body(
    *,
    run_id: str = "restart-run",
    sequence: int = 8,
    terminal: bool = False,
) -> bytes:
    return canonical_dashboard_body(
        {
            "schema_version": "patrol-dashboard/v1",
            "run": {
                "run_id": run_id,
                "sequence": sequence,
                "terminal": terminal,
            },
        }
    )


def test_restart_reuses_exact_body_after_cloud_commit_with_lost_ack(tmp_path: Path) -> None:
    current = _Current("restart-run", 8)
    cloud = _CloudStore()
    first_repository = _Repository(tmp_path, current)
    first = PatrolRelayPublisher(
        first_repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        _LostAcknowledgementTransport(cloud),
        device_id="macmini-m4",
        projector=_projector("2026-09-01T08:00:00Z"),
    )

    with pytest.raises(RelayRetryableError, match="cloud commit"):
        first.publish_latest()

    first_body = cloud.accepted_body
    assert first_body is not None
    state_path = (
        tmp_path / RELAY_OUTBOX_DIRECTORY_NAME / RELAY_OUTBOX_STATE_FILENAME
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["pending"]["body_sha256"] == hashlib.sha256(first_body).hexdigest()

    second_repository = _Repository(tmp_path, current)
    transport = _CloudTransport(cloud)
    second = PatrolRelayPublisher(
        second_repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        transport,
        device_id="macmini-m4",
        projector=lambda *_args: pytest.fail("a recovered body must not be reprojected"),
    )

    result = second.publish_latest()

    assert result.status == "acknowledged"
    assert result.snapshot is not None and result.snapshot.body == first_body
    assert second_repository.read_count == 0
    assert cloud.accept_count == 2


def test_terminal_ack_restart_finishes_without_evidence_read_or_network(tmp_path: Path) -> None:
    current = _Current("terminal-restart", 21, terminal=True)
    cloud = _CloudStore()
    first = PatrolRelayPublisher(
        _Repository(tmp_path, current),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        _CloudTransport(cloud),
        device_id="macmini-m4",
        projector=_projector("2026-09-01T08:01:00Z"),
    )
    expected = first.run_until_terminal_acknowledged()

    class _NoNetwork:
        def upload(self, _snapshot: Any) -> RelayAcknowledgement:
            pytest.fail("persisted terminal acknowledgement must avoid network access")

    repository = _Repository(tmp_path, current)
    resumed = PatrolRelayPublisher(
        repository,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        _NoNetwork(),
        device_id="macmini-m4",
        projector=lambda *_args: pytest.fail("terminal restart must not reproject"),
    )

    assert resumed.run_until_terminal_acknowledged() == expected
    assert repository.read_count == 0
    state_text = (
        tmp_path / RELAY_OUTBOX_DIRECTORY_NAME / RELAY_OUTBOX_STATE_FILENAME
    ).read_text(encoding="utf-8")
    assert "secret" not in state_text.casefold()
    assert "signature" not in state_text.casefold()


def test_failed_ack_state_replace_leaves_exact_pending_body_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _Current("ack-window", 3)
    cloud = _CloudStore()
    publisher = PatrolRelayPublisher(
        _Repository(tmp_path, current),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        _CloudTransport(cloud),
        device_id="macmini-m4",
        projector=_projector("2026-09-01T08:02:00Z"),
    )
    real_replace = os.replace
    replace_count = 0

    def fail_second_replace(source: str | Path, target: str | Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated crash before ack state replace")
        real_replace(source, target)

    monkeypatch.setattr(outbox_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="before ack"):
        publisher.publish_latest()
    assert cloud.accepted_body is not None

    monkeypatch.setattr(outbox_module.os, "replace", real_replace)
    recovered = RelayOutbox(tmp_path, device_id="macmini-m4")
    assert recovered.pending is not None
    assert recovered.pending.body == cloud.accepted_body
    assert recovered.last_acknowledgement is None
    assert not tuple((tmp_path / RELAY_OUTBOX_DIRECTORY_NAME).glob(".state-*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update({"signature": "must-not-be-persisted"}),
        lambda state: state.update({"device_id": "another-device"}),
        lambda state: state["pending"].update({"body_sha256": "0" * 64}),
        lambda state: state["pending"].update({"sequence": 999}),
        lambda state: state["pending"].update({"run_id": "bad/run"}),
    ],
)
def test_outbox_rejects_tampered_json_identity_and_hash(
    tmp_path: Path,
    mutation: Any,
) -> None:
    body = _pending_body()
    outbox = RelayOutbox(tmp_path, device_id="macmini-m4")
    outbox.save_pending(
        run_id="restart-run",
        sequence=8,
        terminal=False,
        body=body,
        body_sha256=hashlib.sha256(body).hexdigest(),
    )
    state_path = outbox.state_path
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mutation(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_path.chmod(0o600)

    with pytest.raises(RelayStateError):
        RelayOutbox(tmp_path, device_id="macmini-m4")


def test_outbox_rejects_symlink_directory_and_state(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    run_with_link = tmp_path / "linked-run"
    run_with_link.mkdir()
    (run_with_link / RELAY_OUTBOX_DIRECTORY_NAME).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RelayStateError, match="symbolic link"):
        RelayOutbox(run_with_link, device_id="macmini-m4")

    run_with_state_link = tmp_path / "state-linked-run"
    run_with_state_link.mkdir()
    outbox_directory = run_with_state_link / RELAY_OUTBOX_DIRECTORY_NAME
    outbox_directory.mkdir(mode=0o700)
    outside_state = tmp_path / "outside-state.json"
    outside_state.write_text("{}", encoding="utf-8")
    (outbox_directory / RELAY_OUTBOX_STATE_FILENAME).symlink_to(outside_state)
    with pytest.raises(RelayStateError, match="non-symlink"):
        RelayOutbox(run_with_state_link, device_id="macmini-m4")


def test_outbox_rejects_malformed_and_duplicate_key_json(tmp_path: Path) -> None:
    directory = tmp_path / RELAY_OUTBOX_DIRECTORY_NAME
    directory.mkdir(mode=0o700)
    state_path = directory / RELAY_OUTBOX_STATE_FILENAME
    state_path.write_text(
        '{"schema_version":"patrol-relay-outbox/v1",'
        '"device_id":"macmini-m4","device_id":"macmini-m4",'
        '"last_ack":null,"pending":null}',
        encoding="utf-8",
    )
    state_path.chmod(0o600)
    with pytest.raises(RelayStateError, match="duplicate JSON key"):
        RelayOutbox(tmp_path, device_id="macmini-m4")

    state_path.write_text("{not-json", encoding="utf-8")
    state_path.chmod(0o600)
    with pytest.raises(RelayStateError, match="strict UTF-8 JSON"):
        RelayOutbox(tmp_path, device_id="macmini-m4")


@pytest.mark.parametrize("run_id", ["bad/run", " space", "-leading", "a" * 129])
def test_invalid_local_run_id_fails_before_projection_and_network(
    tmp_path: Path,
    run_id: str,
) -> None:
    class _NoNetwork:
        upload_count = 0

        def upload(self, _snapshot: Any) -> RelayAcknowledgement:
            self.upload_count += 1
            pytest.fail("invalid local evidence must not reach the network")

    transport = _NoNetwork()
    publisher = PatrolRelayPublisher(
        _Repository(tmp_path, _Current(run_id, 0)),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        transport,
        device_id="macmini-m4",
        projector=lambda *_args: pytest.fail("invalid run_id must fail before projection"),
    )

    with pytest.raises(RelayPermanentError, match="run_id"):
        publisher.publish_latest()
    assert transport.upload_count == 0


def test_non_safe_local_sequence_fails_before_projection_and_network(tmp_path: Path) -> None:
    class _NoNetwork:
        upload_count = 0

        def upload(self, _snapshot: Any) -> RelayAcknowledgement:
            self.upload_count += 1
            pytest.fail("invalid local sequence must not reach the network")

    transport = _NoNetwork()
    publisher = PatrolRelayPublisher(
        _Repository(tmp_path, _Current("safe-run", 1 << 53)),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        transport,
        device_id="macmini-m4",
        projector=lambda *_args: pytest.fail("invalid sequence must fail before projection"),
    )

    with pytest.raises(RelayPermanentError, match="safe integer"):
        publisher.publish_latest()
    assert transport.upload_count == 0


def test_invalid_run_id_wrapped_by_real_repository_is_a_permanent_failure(
    tmp_path: Path,
) -> None:
    class _InvalidRepository:
        run_directory = tmp_path

        def read(self) -> Any:
            try:
                raise PatrolRunIdError("run_id must match the shared contract")
            except PatrolRunIdError as error:
                raise DashboardEvidenceError("巡检证据读取失败") from error

    class _NoNetwork:
        def upload(self, _snapshot: Any) -> RelayAcknowledgement:
            pytest.fail("invalid repository evidence must not reach the network")

    publisher = PatrolRelayPublisher(
        _InvalidRepository(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        _NoNetwork(),
        device_id="macmini-m4",
        projector=lambda *_args: pytest.fail("invalid evidence must not be projected"),
    )

    with pytest.raises(RelayPermanentError, match="invalid run_id"):
        publisher.publish_latest()
