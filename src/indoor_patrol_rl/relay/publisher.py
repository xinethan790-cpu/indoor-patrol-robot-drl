"""Push validated dashboard snapshots to an authenticated HTTPS relay.

This module is intentionally a sidecar rather than part of the Webots control
loop.  Network failures can therefore delay the dashboard without delaying or
changing robot actions.  The sidecar never accepts inbound connections and it
never reads files other than those approved by :class:`PatrolDashboardRepository`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random as random_module
import re
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Protocol
from urllib.parse import urlsplit

from indoor_patrol_rl.dashboard.projector import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardMap,
    project_dashboard,
)
from indoor_patrol_rl.dashboard.repository import (
    DashboardEvidence,
    DashboardEvidenceError,
    PatrolDashboardRepository,
)
from indoor_patrol_rl.identifiers import PatrolRunIdError, validate_patrol_run_id
from indoor_patrol_rl.relay.outbox import MAX_RELAY_SEQUENCE, RelayOutbox, RelayStateError

SIGNATURE_VERSION = "patrol-hmac-sha256-v1"
DEFAULT_SECRET_ENVIRONMENT_VARIABLE = "PATROL_RELAY_UPLOAD_SECRET"
DEFAULT_DEVICE_ENVIRONMENT_VARIABLE = "PATROL_RELAY_DEVICE_ID"
DEFAULT_DEVICE_ID = "macmini-m4"
DEFAULT_RESPONSE_LIMIT_BYTES = 16 * 1024
MAX_REQUEST_BODY_BYTES = 128 * 1024
MIN_UPLOAD_SECRET_BYTES = 32
_DEVICE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_NONCE_PATTERN = re.compile(r"[A-Za-z0-9._~-]{16,128}")
_SECRET_TEMPLATE_PATTERN = re.compile(r"(?:\{\{[^{}\r\n]+\}\}|\$\{[^{}\r\n]+\})")
_EXAMPLE_SECRET_PATTERN = re.compile(
    r"(?:change.?me|replace.?me|example|sample|default|placeholder|your.?secret|"
    r"your.?key|test.?secret|demo.?secret|password|not.?for.?production)",
    re.IGNORECASE,
)
_RETRYABLE_HTTP_STATUSES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_EARLY,
    HTTPStatus.TOO_MANY_REQUESTS,
}


class RelayConfigurationError(ValueError):
    """The relay endpoint, identity, secret, or timing configuration is unsafe."""


class RelayRetryableError(RuntimeError):
    """A transient relay failure that may be retried with backoff."""


class RelayPermanentError(RuntimeError):
    """A relay rejection or malformed acknowledgement that must fail closed."""


@dataclass(frozen=True, slots=True)
class RelayBackoff:
    """Bounded exponential retry timing with optional symmetric jitter."""

    initial_s: float = 1.0
    maximum_s: float = 30.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.15

    def __post_init__(self) -> None:
        values = (self.initial_s, self.maximum_s, self.multiplier, self.jitter_ratio)
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in values
        ):
            raise RelayConfigurationError("Relay backoff values must be finite numbers.")
        if self.initial_s < 0.0:
            raise RelayConfigurationError("Relay backoff initial_s cannot be negative.")
        if self.maximum_s < self.initial_s:
            raise RelayConfigurationError("Relay backoff maximum_s cannot be below initial_s.")
        if self.multiplier < 1.0:
            raise RelayConfigurationError("Relay backoff multiplier must be at least one.")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise RelayConfigurationError("Relay backoff jitter_ratio must be within [0, 1].")

    def delay(self, failure_count: int, *, random_value: float) -> float:
        """Return a retry delay for a one-based consecutive failure count."""

        if isinstance(failure_count, bool) or not isinstance(failure_count, int):
            raise TypeError("failure_count must be an integer.")
        if failure_count <= 0:
            raise ValueError("failure_count must be positive.")
        if (
            isinstance(random_value, bool)
            or not isinstance(random_value, (int, float))
            or not math.isfinite(random_value)
        ):
            raise TypeError("random_value must be a finite number.")
        if not 0.0 <= random_value <= 1.0:
            raise ValueError("random_value must be within [0, 1].")
        try:
            scaled = self.initial_s * (self.multiplier ** (failure_count - 1))
        except OverflowError:
            scaled = math.inf
        base = min(self.maximum_s, scaled)
        jitter = base * self.jitter_ratio * ((2.0 * random_value) - 1.0)
        return max(0.0, base + jitter)


@dataclass(frozen=True, slots=True)
class RelaySnapshot:
    """One immutable JSON request retained byte-for-byte across retries."""

    device_id: str
    run_id: str
    sequence: int
    terminal: bool
    body: bytes
    body_sha256: str


@dataclass(frozen=True, slots=True)
class RelayAcknowledgement:
    """A relay acknowledgement bound to exactly one uploaded snapshot."""

    device_id: str
    run_id: str
    sequence: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Outcome of one non-sleeping publisher attempt."""

    status: str
    snapshot: RelaySnapshot | None = None
    acknowledgement: RelayAcknowledgement | None = None


class RelayTransport(Protocol):
    """Transport boundary used by the publisher and deterministic tests."""

    def upload(self, snapshot: RelaySnapshot) -> RelayAcknowledgement: ...


class DashboardProjector(Protocol):
    """Structural type of the existing dashboard projection function."""

    def __call__(
        self,
        evidence: DashboardEvidence,
        dashboard_map: DashboardMap,
    ) -> dict[str, object]: ...


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Disable redirects so an HTTPS request can never be downgraded."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


def validate_https_endpoint(value: str) -> str:
    """Return a normalized HTTPS endpoint or reject unsafe URL features."""

    if not isinstance(value, str) or not value.strip():
        raise RelayConfigurationError("Relay endpoint must be a non-empty HTTPS URL.")
    endpoint = value.strip()
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RelayConfigurationError("Relay endpoint must use HTTPS and include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise RelayConfigurationError("Relay endpoint cannot contain URL credentials.")
    if parsed.query or parsed.fragment:
        raise RelayConfigurationError("Relay endpoint cannot contain a query or fragment.")
    if not parsed.path or parsed.path == "/":
        raise RelayConfigurationError("Relay endpoint must include an explicit upload path.")
    try:
        _ = parsed.port
    except ValueError as error:
        raise RelayConfigurationError("Relay endpoint contains an invalid port.") from error
    return endpoint


def validate_device_id(value: str) -> str:
    """Validate the public device identifier used in the signature."""

    if not isinstance(value, str):
        raise RelayConfigurationError("device_id must be a string.")
    device_id = value.strip()
    if _DEVICE_ID_PATTERN.fullmatch(device_id) is None:
        raise RelayConfigurationError(
            "device_id must be 1-64 ASCII letters, digits, dots, underscores, colons, or hyphens."
        )
    return device_id


def load_upload_secret(
    environ: Mapping[str, str] | None = None,
    *,
    variable_name: str = DEFAULT_SECRET_ENVIRONMENT_VARIABLE,
) -> bytes:
    """Load the uploader secret exclusively from an environment mapping."""

    source = os.environ if environ is None else environ
    raw = source.get(variable_name)
    if raw is None or raw == "":
        raise RelayConfigurationError(
            f"Required relay secret environment variable {variable_name} is not set."
        )
    return validate_upload_secret(raw, source_label=f"Relay secret in {variable_name}")


def validate_upload_secret(
    value: str | bytes,
    *,
    source_label: str = "Relay secret",
) -> bytes:
    """Validate one secret without including its value in errors or logs."""

    if isinstance(value, bytes):
        try:
            raw = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RelayConfigurationError(
                f"{source_label} must be valid UTF-8."
            ) from error
    elif isinstance(value, str):
        raw = value
    else:
        raise RelayConfigurationError(f"{source_label} must be text or UTF-8 bytes.")
    if raw == "":
        raise RelayConfigurationError(f"{source_label} is not set.")
    normalized = raw.strip()
    if raw != normalized:
        raise RelayConfigurationError(
            f"{source_label} must not contain leading or trailing whitespace."
        )
    secret = normalized.encode("utf-8")
    if _SECRET_TEMPLATE_PATTERN.search(normalized) is not None:
        raise RelayConfigurationError(f"{source_label} contains a template expression.")
    if len(secret) < MIN_UPLOAD_SECRET_BYTES:
        raise RelayConfigurationError(
            f"{source_label} must contain at least {MIN_UPLOAD_SECRET_BYTES} UTF-8 bytes."
        )
    if (
        _EXAMPLE_SECRET_PATTERN.search(normalized) is not None
        or len(set(normalized)) < 8
    ):
        raise RelayConfigurationError(
            f"{source_label} matches a sample, default, or low-diversity value."
        )
    return secret


def resolve_device_id(
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    variable_name: str = DEFAULT_DEVICE_ENVIRONMENT_VARIABLE,
) -> str:
    """Resolve a public device ID from CLI, environment, or the safe default."""

    source = os.environ if environ is None else environ
    value = explicit if explicit is not None else source.get(variable_name, DEFAULT_DEVICE_ID)
    return validate_device_id(value)


def canonical_dashboard_body(payload: Mapping[str, object]) -> bytes:
    """Encode a stable, finite JSON request body for hashing and retry."""

    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RelayPermanentError(f"Dashboard payload is not finite JSON: {error}") from error
    return text.encode("utf-8")


def build_signature_headers(
    *,
    device_id: str,
    timestamp: int,
    nonce: str,
    body: bytes,
    secret: bytes,
) -> dict[str, str]:
    """Build the relay's canonical HMAC-SHA256 authentication headers."""

    safe_device_id = validate_device_id(device_id)
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise RelayConfigurationError("Relay timestamp must be a non-negative integer.")
    if not isinstance(nonce, str) or _NONCE_PATTERN.fullmatch(nonce) is None:
        raise RelayConfigurationError(
            "Relay nonce must be 16-128 characters from [A-Za-z0-9._~-]."
        )
    if not isinstance(body, bytes):
        raise TypeError("body must be bytes.")
    safe_secret = validate_upload_secret(secret)
    body_sha256 = hashlib.sha256(body).hexdigest()
    timestamp_text = str(timestamp)
    canonical = "\n".join((safe_device_id, timestamp_text, nonce, body_sha256)).encode("utf-8")
    signature = hmac.new(safe_secret, canonical, hashlib.sha256).hexdigest()
    return {
        "X-Patrol-Signature-Version": SIGNATURE_VERSION,
        "X-Patrol-Device-Id": safe_device_id,
        "X-Patrol-Timestamp": timestamp_text,
        "X-Patrol-Nonce": nonce,
        "X-Patrol-Body-SHA256": body_sha256,
        "X-Patrol-Signature": signature,
    }


class HttpRelayTransport:
    """Authenticated, no-redirect HTTPS POST transport."""

    def __init__(
        self,
        endpoint: str,
        *,
        device_id: str,
        secret: bytes,
        timeout_s: float = 5.0,
        response_limit_bytes: int = DEFAULT_RESPONSE_LIMIT_BYTES,
        clock: Callable[[], float] = time.time,
        nonce_factory: Callable[[], str] | None = None,
        opener: Any | None = None,
    ) -> None:
        self.endpoint = validate_https_endpoint(endpoint)
        self.device_id = validate_device_id(device_id)
        self._secret = validate_upload_secret(secret)
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise RelayConfigurationError("Relay timeout_s must be positive and finite.")
        self.timeout_s = float(timeout_s)
        if (
            isinstance(response_limit_bytes, bool)
            or not isinstance(response_limit_bytes, int)
            or response_limit_bytes <= 0
        ):
            raise RelayConfigurationError("response_limit_bytes must be a positive integer.")
        self.response_limit_bytes = response_limit_bytes
        self._clock = clock
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(16))
        self._opener = opener or urllib.request.build_opener(_RejectRedirectHandler())

    def upload(self, snapshot: RelaySnapshot) -> RelayAcknowledgement:
        """Upload one immutable body and require a matching JSON acknowledgement."""

        if snapshot.device_id != self.device_id:
            raise RelayPermanentError("Snapshot device_id does not match relay transport.")
        if hashlib.sha256(snapshot.body).hexdigest() != snapshot.body_sha256:
            raise RelayPermanentError("Snapshot body hash changed before upload.")
        clock_value = self._clock()
        if (
            isinstance(clock_value, bool)
            or not isinstance(clock_value, (int, float))
            or not math.isfinite(clock_value)
            or clock_value < 0.0
        ):
            raise RelayPermanentError("Relay clock did not return a non-negative epoch time.")
        timestamp = int(clock_value)
        nonce = self._nonce_factory()
        signature_headers = build_signature_headers(
            device_id=self.device_id,
            timestamp=timestamp,
            nonce=nonce,
            body=snapshot.body,
            secret=self._secret,
        )
        request = urllib.request.Request(
            self.endpoint,
            data=snapshot.body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                **signature_headers,
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                status = int(response.status)
                raw = response.read(self.response_limit_bytes + 1)
        except urllib.error.HTTPError as error:
            self._raise_http_error(error.code)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RelayRetryableError(f"Relay request failed: {error}") from error
        if len(raw) > self.response_limit_bytes:
            raise RelayPermanentError("Relay acknowledgement exceeds the response size limit.")
        if not 200 <= status < 300:
            self._raise_http_error(status)
        return _parse_acknowledgement(raw, snapshot)

    @staticmethod
    def _raise_http_error(status: int) -> None:
        try:
            status_value = HTTPStatus(status)
        except ValueError:
            status_value = None
        if status_value in _RETRYABLE_HTTP_STATUSES or 500 <= status <= 599:
            raise RelayRetryableError(f"Relay returned retryable HTTP {status}.")
        raise RelayPermanentError(f"Relay rejected the upload with HTTP {status}.")


class PatrolRelayPublisher:
    """Coalesce local evidence into authenticated, acknowledged relay updates."""

    def __init__(
        self,
        repository: PatrolDashboardRepository,
        dashboard_map: DashboardMap,
        transport: RelayTransport,
        *,
        device_id: str,
        poll_interval_s: float = 0.75,
        backoff: RelayBackoff | None = None,
        projector: DashboardProjector = project_dashboard,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random_module.random,
        event_handler: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        self.repository = repository
        self.dashboard_map = dashboard_map
        self.transport = transport
        self.device_id = validate_device_id(device_id)
        if (
            isinstance(poll_interval_s, bool)
            or not isinstance(poll_interval_s, (int, float))
            or not math.isfinite(poll_interval_s)
            or poll_interval_s <= 0.0
        ):
            raise RelayConfigurationError("poll_interval_s must be positive and finite.")
        self.poll_interval_s = float(poll_interval_s)
        self.backoff = backoff or RelayBackoff()
        self._projector = projector
        self._sleep = sleep
        self._random_value = random_value
        self._event_handler = event_handler
        self._outbox = RelayOutbox(repository.run_directory, device_id=self.device_id)
        persisted_pending = self._outbox.pending
        self._pending = (
            None
            if persisted_pending is None
            else RelaySnapshot(
                device_id=self.device_id,
                run_id=persisted_pending.run_id,
                sequence=persisted_pending.sequence,
                terminal=persisted_pending.terminal,
                body=persisted_pending.body,
                body_sha256=persisted_pending.body_sha256,
            )
        )
        persisted_acknowledgement = self._outbox.last_acknowledgement
        self._last_acknowledgement = (
            None
            if persisted_acknowledgement is None
            else RelayAcknowledgement(
                device_id=self.device_id,
                run_id=persisted_acknowledgement.run_id,
                sequence=persisted_acknowledgement.sequence,
                terminal=persisted_acknowledgement.terminal,
            )
        )

    @property
    def pending(self) -> RelaySnapshot | None:
        return self._pending

    @property
    def last_acknowledgement(self) -> RelayAcknowledgement | None:
        return self._last_acknowledgement

    def publish_latest(self) -> PublishResult:
        """Prepare and upload at most one snapshot without sleeping or retrying."""

        if self._pending is None:
            preparation = self._prepare_latest()
            if preparation is not None:
                return preparation
        assert self._pending is not None
        snapshot = self._pending
        acknowledgement = self.transport.upload(snapshot)
        if acknowledgement != RelayAcknowledgement(
            device_id=snapshot.device_id,
            run_id=snapshot.run_id,
            sequence=snapshot.sequence,
            terminal=snapshot.terminal,
        ):
            raise RelayPermanentError("Relay transport returned a mismatched acknowledgement.")
        self._outbox.acknowledge_pending(
            run_id=acknowledgement.run_id,
            sequence=acknowledgement.sequence,
            terminal=acknowledgement.terminal,
            body_sha256=snapshot.body_sha256,
        )
        self._last_acknowledgement = acknowledgement
        self._pending = None
        self._announce(
            "acknowledged",
            {
                "run_id": acknowledgement.run_id,
                "sequence": acknowledgement.sequence,
                "terminal": acknowledgement.terminal,
            },
        )
        return PublishResult("acknowledged", snapshot, acknowledgement)

    def run_until_terminal_acknowledged(self) -> RelayAcknowledgement:
        """Retry transient failures and return only after a terminal ack."""

        if (
            self._pending is None
            and self._last_acknowledgement is not None
            and self._last_acknowledgement.terminal
        ):
            return self._last_acknowledgement
        failures = 0
        while True:
            try:
                result = self.publish_latest()
            except DashboardEvidenceError as error:
                failures = 0
                self._announce("evidence_unavailable", {"message": str(error)})
                self._sleep(self.poll_interval_s)
                continue
            except RelayRetryableError as error:
                failures += 1
                delay = self.backoff.delay(
                    failures,
                    random_value=self._random_value(),
                )
                self._announce(
                    "retrying",
                    {"failure_count": failures, "delay_s": delay, "message": str(error)},
                )
                self._sleep(delay)
                continue
            failures = 0
            if result.acknowledgement is not None and result.acknowledgement.terminal:
                return result.acknowledgement
            self._sleep(self.poll_interval_s)

    def _prepare_latest(self) -> PublishResult | None:
        try:
            evidence = self.repository.read()
        except DashboardEvidenceError as error:
            if _exception_chain_contains(error, PatrolRunIdError):
                raise RelayPermanentError(
                    "Local patrol evidence contains an invalid run_id."
                ) from error
            raise
        current = evidence.current
        try:
            run_id = validate_patrol_run_id(current.run_id)
        except ValueError as error:
            raise RelayPermanentError(f"Local patrol run_id is invalid: {error}") from error
        sequence = current.sequence
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or not 0 <= sequence <= MAX_RELAY_SEQUENCE
        ):
            raise RelayPermanentError(
                "Local patrol sequence must be a non-negative safe integer."
            )
        previous = self._last_acknowledgement
        if previous is not None and previous.run_id == run_id:
            if sequence < previous.sequence:
                raise RelayStateError(
                    f"Local relay sequence moved backward: {sequence} < {previous.sequence}."
                )
            if sequence == previous.sequence:
                return PublishResult("duplicate")
        terminal = current.state.value in {"completed", "failed"}
        payload = self._projector(evidence, self.dashboard_map)
        _validate_projected_identity(
            payload,
            run_id=run_id,
            sequence=sequence,
            terminal=terminal,
        )
        body = canonical_dashboard_body(payload)
        if len(body) > MAX_REQUEST_BODY_BYTES:
            raise RelayPermanentError(
                f"Dashboard payload exceeds {MAX_REQUEST_BODY_BYTES} bytes."
            )
        snapshot = RelaySnapshot(
            device_id=self.device_id,
            run_id=run_id,
            sequence=sequence,
            terminal=terminal,
            body=body,
            body_sha256=hashlib.sha256(body).hexdigest(),
        )
        self._outbox.save_pending(
            run_id=snapshot.run_id,
            sequence=snapshot.sequence,
            terminal=snapshot.terminal,
            body=snapshot.body,
            body_sha256=snapshot.body_sha256,
        )
        self._pending = snapshot
        self._announce(
            "prepared",
            {"run_id": run_id, "sequence": sequence, "terminal": snapshot.terminal},
        )
        return None

    def _announce(self, event: str, details: Mapping[str, object]) -> None:
        if self._event_handler is not None:
            self._event_handler(event, details)


def _validate_projected_identity(
    payload: Mapping[str, object],
    *,
    run_id: str,
    sequence: int,
    terminal: bool,
) -> None:
    if payload.get("schema_version") != DASHBOARD_SCHEMA_VERSION:
        raise RelayPermanentError("Projected dashboard schema version is unsupported.")
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise RelayPermanentError("Projected dashboard payload has no run object.")
    if run.get("run_id") != run_id or run.get("sequence") != sequence:
        raise RelayPermanentError("Projected dashboard identity differs from local evidence.")
    if run.get("terminal") is not terminal:
        raise RelayPermanentError("Projected dashboard terminal state differs from evidence.")


def _parse_acknowledgement(
    raw: bytes,
    snapshot: RelaySnapshot,
) -> RelayAcknowledgement:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelayPermanentError("Relay acknowledgement is not valid UTF-8 JSON.") from error
    if not isinstance(value, Mapping):
        raise RelayPermanentError("Relay acknowledgement must be a JSON object.")
    if value.get("accepted") is not True:
        raise RelayPermanentError("Relay acknowledgement did not confirm acceptance.")
    expected = {
        "device_id": snapshot.device_id,
        "run_id": snapshot.run_id,
        "sequence": snapshot.sequence,
        "terminal": snapshot.terminal,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise RelayPermanentError(f"Relay acknowledgement {key} does not match upload.")
    return RelayAcknowledgement(**expected)


def _exception_chain_contains(error: BaseException, expected_type: type[BaseException]) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, expected_type):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
