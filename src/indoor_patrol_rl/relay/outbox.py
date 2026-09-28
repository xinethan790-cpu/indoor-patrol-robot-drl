"""Crash-safe local outbox for patrol relay request bodies and acknowledgements."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indoor_patrol_rl.dashboard.projector import DASHBOARD_SCHEMA_VERSION
from indoor_patrol_rl.identifiers import validate_patrol_run_id

RELAY_OUTBOX_SCHEMA_VERSION = "patrol-relay-outbox/v1"
RELAY_OUTBOX_DIRECTORY_NAME = ".patrol-relay"
RELAY_OUTBOX_STATE_FILENAME = "state.json"
MAX_PERSISTED_BODY_BYTES = 128 * 1024
MAX_OUTBOX_STATE_BYTES = 256 * 1024
MAX_RELAY_SEQUENCE = (1 << 53) - 1

_DEVICE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", flags=re.ASCII)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)
_TOP_LEVEL_KEYS = {"schema_version", "device_id", "last_ack", "pending"}
_ACK_KEYS = {"run_id", "sequence", "terminal", "body_sha256"}
_PENDING_KEYS = {*_ACK_KEYS, "body_base64"}


class RelayStateError(RuntimeError):
    """Persisted relay progress is missing, unsafe, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class RelayOutboxPending:
    """One exact request body durably retained until its matching ack is saved."""

    run_id: str
    sequence: int
    terminal: bool
    body: bytes
    body_sha256: str


@dataclass(frozen=True, slots=True)
class RelayOutboxAcknowledgement:
    """The latest acknowledgement durably committed for this run directory."""

    run_id: str
    sequence: int
    terminal: bool
    body_sha256: str


class RelayOutbox:
    """Atomically persist one pending request and the latest accepted sequence."""

    def __init__(self, run_directory: str | Path, *, device_id: str) -> None:
        self.device_id = _validate_device_id(device_id)
        self.run_directory = _resolve_run_directory(run_directory)
        self.directory = self.run_directory / RELAY_OUTBOX_DIRECTORY_NAME
        self.state_path = self.directory / RELAY_OUTBOX_STATE_FILENAME
        self._ensure_private_directory()
        last_ack, pending = self._load_state()
        self._last_ack = last_ack
        self._pending = pending

    @property
    def pending(self) -> RelayOutboxPending | None:
        return self._pending

    @property
    def last_acknowledgement(self) -> RelayOutboxAcknowledgement | None:
        return self._last_ack

    def save_pending(
        self,
        *,
        run_id: str,
        sequence: int,
        terminal: bool,
        body: bytes,
        body_sha256: str,
    ) -> RelayOutboxPending:
        """Durably store an exact body before it becomes eligible for upload."""

        pending = _validated_pending(
            run_id=run_id,
            sequence=sequence,
            terminal=terminal,
            body=body,
            body_sha256=body_sha256,
        )
        if self._pending is not None:
            if pending == self._pending:
                return self._pending
            raise RelayStateError("Relay outbox already contains a different pending request.")
        if self._last_ack is not None:
            if self._last_ack.terminal:
                raise RelayStateError("A terminal relay acknowledgement cannot have a successor.")
            if (
                self._last_ack.run_id == pending.run_id
                and pending.sequence <= self._last_ack.sequence
            ):
                raise RelayStateError(
                    "Pending relay sequence must be newer than the persisted acknowledgement."
                )
        self._write_state(last_ack=self._last_ack, pending=pending)
        self._pending = pending
        return pending

    def acknowledge_pending(
        self,
        *,
        run_id: str,
        sequence: int,
        terminal: bool,
        body_sha256: str,
    ) -> RelayOutboxAcknowledgement:
        """Atomically replace the pending body with its matching accepted identity."""

        acknowledgement = _validated_acknowledgement(
            run_id=run_id,
            sequence=sequence,
            terminal=terminal,
            body_sha256=body_sha256,
        )
        pending = self._pending
        if pending is None:
            raise RelayStateError("Cannot persist an acknowledgement without a pending request.")
        if (
            acknowledgement.run_id != pending.run_id
            or acknowledgement.sequence != pending.sequence
            or acknowledgement.terminal is not pending.terminal
            or acknowledgement.body_sha256 != pending.body_sha256
        ):
            raise RelayStateError("Relay acknowledgement does not match the persisted request.")
        self._write_state(last_ack=acknowledgement, pending=None)
        self._last_ack = acknowledgement
        self._pending = None
        return acknowledgement

    def _ensure_private_directory(self) -> None:
        if self.directory.is_symlink():
            raise RelayStateError("Relay outbox directory cannot be a symbolic link.")
        created = False
        try:
            self.directory.mkdir(mode=0o700)
            created = True
        except FileExistsError:
            pass
        self._validate_directory()
        if created:
            _fsync_directory(self.run_directory)

    def _validate_directory(self) -> None:
        if self.directory.is_symlink():
            raise RelayStateError("Relay outbox directory cannot be a symbolic link.")
        try:
            resolved = self.directory.resolve(strict=True)
            metadata = self.directory.stat()
        except OSError as error:
            raise RelayStateError(f"Relay outbox directory is not accessible: {error}") from error
        if resolved.parent != self.run_directory or not stat.S_ISDIR(metadata.st_mode):
            raise RelayStateError("Relay outbox directory escaped the explicit run directory.")
        # POSIX mode bits are not meaningful on Windows/NTFS, where actual access
        # is governed by ACLs; Python's st_mode there does not reflect them.
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RelayStateError("Relay outbox directory permissions must be owner-only.")

    def _load_state(
        self,
    ) -> tuple[RelayOutboxAcknowledgement | None, RelayOutboxPending | None]:
        raw = self._read_state_bytes()
        if raw is None:
            return None, None
        try:
            root = _strict_json_object(raw, "relay outbox state")
            _exact_keys(root, _TOP_LEVEL_KEYS, "relay outbox state")
            if root["schema_version"] != RELAY_OUTBOX_SCHEMA_VERSION:
                raise RelayStateError("Relay outbox schema version is unsupported.")
            if _validate_device_id(root["device_id"]) != self.device_id:
                raise RelayStateError(
                    "Relay outbox device_id does not match the configured uploader."
                )
            last_ack = _decode_optional_acknowledgement(root["last_ack"])
            pending = _decode_optional_pending(root["pending"])
            _validate_state_relationship(last_ack, pending)
        except RelayStateError:
            raise
        except (TypeError, ValueError) as error:
            raise RelayStateError(f"Relay outbox state is invalid: {error}") from error
        return last_ack, pending

    def _read_state_bytes(self) -> bytes | None:
        self._validate_directory()
        try:
            metadata = self.state_path.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise RelayStateError(f"Relay outbox state is not accessible: {error}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RelayStateError("Relay outbox state must be a regular non-symlink file.")
        # POSIX mode bits are not meaningful on Windows/NTFS; see _validate_directory.
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RelayStateError("Relay outbox state permissions must be owner-only.")
        if metadata.st_size > MAX_OUTBOX_STATE_BYTES:
            raise RelayStateError("Relay outbox state exceeds its size limit.")

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.state_path, flags)
        except OSError as error:
            raise RelayStateError(
                f"Relay outbox state could not be opened safely: {error}"
            ) from error
        try:
            opened_metadata = os.fstat(descriptor)
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise RelayStateError("Relay outbox state changed type while being opened.")
            if (
                opened_metadata.st_dev != metadata.st_dev
                or opened_metadata.st_ino != metadata.st_ino
            ):
                raise RelayStateError("Relay outbox state changed while being opened.")
            chunks: list[bytes] = []
            remaining = MAX_OUTBOX_STATE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        if len(raw) > MAX_OUTBOX_STATE_BYTES:
            raise RelayStateError("Relay outbox state exceeds its size limit.")
        return raw

    def _write_state(
        self,
        *,
        last_ack: RelayOutboxAcknowledgement | None,
        pending: RelayOutboxPending | None,
    ) -> None:
        _validate_state_relationship(last_ack, pending)
        value = {
            "schema_version": RELAY_OUTBOX_SCHEMA_VERSION,
            "device_id": self.device_id,
            "last_ack": _encode_acknowledgement(last_ack),
            "pending": _encode_pending(pending),
        }
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        if len(encoded) > MAX_OUTBOX_STATE_BYTES:
            raise RelayStateError("Relay outbox state exceeds its size limit.")

        self._validate_directory()
        if self.state_path.is_symlink():
            raise RelayStateError("Relay outbox state cannot be a symbolic link.")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".state-",
            suffix=".tmp",
            dir=self.directory,
        )
        temporary_path = Path(temporary_name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._validate_directory()
            if self.state_path.is_symlink():
                raise RelayStateError("Relay outbox state cannot be a symbolic link.")
            os.replace(temporary_path, self.state_path)
            _fsync_directory(self.directory)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise


def _resolve_run_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise RelayStateError(f"Relay run directory is not accessible: {error}") from error
    if not resolved.is_dir():
        raise RelayStateError("Relay run directory must be an existing directory.")
    return resolved


def _validate_device_id(value: object) -> str:
    if not isinstance(value, str) or _DEVICE_ID_PATTERN.fullmatch(value) is None:
        raise RelayStateError("Relay outbox device_id is invalid.")
    return value


def _validated_sequence(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_RELAY_SEQUENCE
    ):
        raise RelayStateError("Relay outbox sequence must be a non-negative safe integer.")
    return value


def _validated_terminal(value: object) -> bool:
    if not isinstance(value, bool):
        raise RelayStateError("Relay outbox terminal must be a boolean.")
    return value


def _validated_hash(value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RelayStateError("Relay outbox body_sha256 must be a lowercase SHA-256 digest.")
    return value


def _validated_run_id(value: object) -> str:
    try:
        return validate_patrol_run_id(value)
    except ValueError as error:
        raise RelayStateError(f"Relay outbox run_id is invalid: {error}") from error


def _validated_acknowledgement(
    *,
    run_id: object,
    sequence: object,
    terminal: object,
    body_sha256: object,
) -> RelayOutboxAcknowledgement:
    return RelayOutboxAcknowledgement(
        run_id=_validated_run_id(run_id),
        sequence=_validated_sequence(sequence),
        terminal=_validated_terminal(terminal),
        body_sha256=_validated_hash(body_sha256),
    )


def _validated_pending(
    *,
    run_id: object,
    sequence: object,
    terminal: object,
    body: object,
    body_sha256: object,
) -> RelayOutboxPending:
    acknowledgement = _validated_acknowledgement(
        run_id=run_id,
        sequence=sequence,
        terminal=terminal,
        body_sha256=body_sha256,
    )
    if not isinstance(body, bytes) or not body or len(body) > MAX_PERSISTED_BODY_BYTES:
        raise RelayStateError(
            f"Relay outbox body must contain 1-{MAX_PERSISTED_BODY_BYTES} bytes."
        )
    actual_hash = hashlib.sha256(body).hexdigest()
    if not hmac.compare_digest(actual_hash, acknowledgement.body_sha256):
        raise RelayStateError("Relay outbox body does not match body_sha256.")
    payload = _strict_json_object(body, "relay pending body")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if canonical != body:
        raise RelayStateError("Relay outbox pending body is not canonical JSON.")
    if payload.get("schema_version") != DASHBOARD_SCHEMA_VERSION:
        raise RelayStateError("Relay outbox pending body has an unsupported schema version.")
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise RelayStateError("Relay outbox pending body has no run object.")
    expected = {
        "run_id": acknowledgement.run_id,
        "sequence": acknowledgement.sequence,
        "terminal": acknowledgement.terminal,
    }
    for key, expected_value in expected.items():
        actual = run.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise RelayStateError(
                f"Relay outbox pending body {key} does not match persisted metadata."
            )
    return RelayOutboxPending(
        run_id=acknowledgement.run_id,
        sequence=acknowledgement.sequence,
        terminal=acknowledgement.terminal,
        body=body,
        body_sha256=acknowledgement.body_sha256,
    )


def _decode_optional_acknowledgement(value: object) -> RelayOutboxAcknowledgement | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RelayStateError("Relay outbox last_ack must be an object or null.")
    _exact_keys(value, _ACK_KEYS, "relay outbox last_ack")
    return _validated_acknowledgement(**value)


def _decode_optional_pending(value: object) -> RelayOutboxPending | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RelayStateError("Relay outbox pending must be an object or null.")
    _exact_keys(value, _PENDING_KEYS, "relay outbox pending")
    body_base64 = value["body_base64"]
    if not isinstance(body_base64, str):
        raise RelayStateError("Relay outbox body_base64 must be a string.")
    try:
        encoded = body_base64.encode("ascii")
        body = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise RelayStateError("Relay outbox body_base64 is invalid.") from error
    if base64.b64encode(body) != encoded:
        raise RelayStateError("Relay outbox body_base64 is not canonical.")
    return _validated_pending(
        run_id=value["run_id"],
        sequence=value["sequence"],
        terminal=value["terminal"],
        body=body,
        body_sha256=value["body_sha256"],
    )


def _validate_state_relationship(
    last_ack: RelayOutboxAcknowledgement | None,
    pending: RelayOutboxPending | None,
) -> None:
    if last_ack is None or pending is None:
        return
    if last_ack.terminal:
        raise RelayStateError("Relay outbox cannot continue an acknowledged terminal run.")
    if last_ack.run_id != pending.run_id:
        return
    if pending.sequence <= last_ack.sequence:
        raise RelayStateError(
            "Relay outbox pending sequence must follow its persisted acknowledgement."
        )


def _encode_acknowledgement(
    value: RelayOutboxAcknowledgement | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "run_id": value.run_id,
        "sequence": value.sequence,
        "terminal": value.terminal,
        "body_sha256": value.body_sha256,
    }


def _encode_pending(value: RelayOutboxPending | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "run_id": value.run_id,
        "sequence": value.sequence,
        "terminal": value.terminal,
        "body_sha256": value.body_sha256,
        "body_base64": base64.b64encode(value.body).decode("ascii"),
    }


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelayStateError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise RelayStateError(f"{label} must be a JSON object.")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RelayStateError(
            f"{label} keys mismatch; missing={missing}, unexpected={unexpected}."
        )


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        # Windows cannot open directories as file descriptors, so the
        # directory-entry flush is POSIX-only; rename durability there relies
        # on the NTFS journal.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RelayStateError(f"Relay outbox directory could not be synced: {error}") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise RelayStateError(f"Relay outbox directory could not be synced: {error}") from error
    finally:
        os.close(descriptor)


__all__ = [
    "MAX_OUTBOX_STATE_BYTES",
    "MAX_PERSISTED_BODY_BYTES",
    "MAX_RELAY_SEQUENCE",
    "RELAY_OUTBOX_DIRECTORY_NAME",
    "RELAY_OUTBOX_SCHEMA_VERSION",
    "RELAY_OUTBOX_STATE_FILENAME",
    "RelayOutbox",
    "RelayOutboxAcknowledgement",
    "RelayOutboxPending",
    "RelayStateError",
]
