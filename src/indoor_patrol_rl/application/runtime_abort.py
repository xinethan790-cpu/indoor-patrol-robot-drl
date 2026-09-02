"""Minimal, bounded evidence for an unexpected patrol-controller abort.

This record is deliberately not a substitute for the normal inspection bundle.
It exists only so an already-started cloud run can receive an honest terminal
state after an unexpected controller exception.  Raw exception messages are
never persisted because they may contain filesystem paths, credentials, or
other deployment details.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indoor_patrol_rl.identifiers import validate_patrol_run_id

RUNTIME_ABORT_FILENAME = "runtime_abort.json"
RUNTIME_ABORT_SCHEMA_VERSION = "patrol-runtime-abort/v1"
RUNTIME_ABORT_REASON_CODE = "unhandled_controller_exception"
RUNTIME_ABORT_EVIDENCE_SCOPE = "runtime_abort_minimal_evidence"
MAX_RUNTIME_ABORT_FAILURE_REASON_LENGTH = 128
MAX_RUNTIME_ABORT_EXCEPTION_TYPE_LENGTH = 64

_EXCEPTION_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def sanitized_exception_type(error: BaseException) -> str:
    """Return a bounded class name without retaining the exception message."""

    candidate = type(error).__name__
    if _EXCEPTION_TYPE_PATTERN.fullmatch(candidate) is None:
        return "Exception"
    return candidate


def runtime_abort_failure_reason(exception_type: str) -> str:
    """Return the only permitted user-visible abort reason representation."""

    if _EXCEPTION_TYPE_PATTERN.fullmatch(exception_type) is None:
        raise ValueError("exception_type must be a bounded Python identifier.")
    reason = f"unhandled controller exception ({exception_type})"
    if len(reason) > MAX_RUNTIME_ABORT_FAILURE_REASON_LENGTH:
        raise ValueError("runtime abort failure reason exceeds its size limit.")
    return reason


@dataclass(frozen=True, slots=True)
class PatrolRuntimeAbort:
    """A minimal terminal record bound to one failed live-status snapshot."""

    run_id: str
    sequence: int
    seed: int
    control_step: int
    simulation_time_s: float
    checkpoints_visited: int
    checkpoints_total: int
    collision_count: int
    sample_count: int
    exception_type: str
    failure_reason: str
    reason_code: str = RUNTIME_ABORT_REASON_CODE
    evidence_scope: str = RUNTIME_ABORT_EVIDENCE_SCOPE
    raw_exception_message_included: bool = False
    schema_version: str = field(default=RUNTIME_ABORT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", validate_patrol_run_id(self.run_id))
        for name in (
            "sequence",
            "seed",
            "control_step",
            "checkpoints_visited",
            "checkpoints_total",
            "collision_count",
            "sample_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.checkpoints_total <= 0:
            raise ValueError("checkpoints_total must be positive.")
        if self.checkpoints_visited > self.checkpoints_total:
            raise ValueError("checkpoints_visited cannot exceed checkpoints_total.")
        if (
            isinstance(self.simulation_time_s, bool)
            or not isinstance(self.simulation_time_s, (int, float))
            or not math.isfinite(float(self.simulation_time_s))
            or self.simulation_time_s < 0
        ):
            raise ValueError("simulation_time_s must be a non-negative finite number.")
        object.__setattr__(self, "simulation_time_s", float(self.simulation_time_s))
        if _EXCEPTION_TYPE_PATTERN.fullmatch(self.exception_type) is None:
            raise ValueError("exception_type must be a bounded Python identifier.")
        if self.failure_reason != runtime_abort_failure_reason(self.exception_type):
            raise ValueError("failure_reason must be the redacted runtime-abort reason.")
        if self.reason_code != RUNTIME_ABORT_REASON_CODE:
            raise ValueError("runtime abort reason_code is unsupported.")
        if self.evidence_scope != RUNTIME_ABORT_EVIDENCE_SCOPE:
            raise ValueError("runtime abort evidence_scope is unsupported.")
        if self.raw_exception_message_included is not False:
            raise ValueError("runtime abort evidence cannot include a raw exception message.")

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        run_id: str,
        sequence: int,
        seed: int,
        control_step: int,
        simulation_time_s: float,
        checkpoints_visited: int,
        checkpoints_total: int,
        collision_count: int,
        sample_count: int,
    ) -> PatrolRuntimeAbort:
        exception_type = sanitized_exception_type(error)
        return cls(
            run_id=run_id,
            sequence=sequence,
            seed=seed,
            control_step=control_step,
            simulation_time_s=simulation_time_s,
            checkpoints_visited=checkpoints_visited,
            checkpoints_total=checkpoints_total,
            collision_count=collision_count,
            sample_count=sample_count,
            exception_type=exception_type,
            failure_reason=runtime_abort_failure_reason(exception_type),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PatrolRuntimeAbort:
        if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
            raise ValueError("runtime abort evidence must be an object.")
        expected = {
            "schema_version",
            "run_id",
            "sequence",
            "seed",
            "control_step",
            "simulation_time_s",
            "checkpoints_visited",
            "checkpoints_total",
            "collision_count",
            "sample_count",
            "exception_type",
            "failure_reason",
            "reason_code",
            "evidence_scope",
            "raw_exception_message_included",
        }
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise ValueError(
                f"runtime abort evidence keys are invalid; missing={missing}, extra={extra}."
            )
        if raw["schema_version"] != RUNTIME_ABORT_SCHEMA_VERSION:
            raise ValueError("runtime abort schema_version is unsupported.")
        return cls(
            run_id=raw["run_id"],
            sequence=raw["sequence"],
            seed=raw["seed"],
            control_step=raw["control_step"],
            simulation_time_s=raw["simulation_time_s"],
            checkpoints_visited=raw["checkpoints_visited"],
            checkpoints_total=raw["checkpoints_total"],
            collision_count=raw["collision_count"],
            sample_count=raw["sample_count"],
            exception_type=raw["exception_type"],
            failure_reason=raw["failure_reason"],
            reason_code=raw["reason_code"],
            evidence_scope=raw["evidence_scope"],
            raw_exception_message_included=raw["raw_exception_message_included"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "seed": self.seed,
            "control_step": self.control_step,
            "simulation_time_s": self.simulation_time_s,
            "checkpoints_visited": self.checkpoints_visited,
            "checkpoints_total": self.checkpoints_total,
            "collision_count": self.collision_count,
            "sample_count": self.sample_count,
            "exception_type": self.exception_type,
            "failure_reason": self.failure_reason,
            "reason_code": self.reason_code,
            "evidence_scope": self.evidence_scope,
            "raw_exception_message_included": self.raw_exception_message_included,
        }


def write_patrol_runtime_abort(
    output_directory: str | Path,
    evidence: PatrolRuntimeAbort,
) -> Path:
    """Atomically persist one minimal abort record with owner-only permissions."""

    if not isinstance(evidence, PatrolRuntimeAbort):
        raise TypeError("evidence must be a PatrolRuntimeAbort instance.")
    directory = Path(output_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("runtime abort output directory must be a non-symlink directory.")
    path = directory / RUNTIME_ABORT_FILENAME
    if path.is_symlink():
        raise ValueError("runtime abort evidence path cannot be a symbolic link.")
    encoded = (
        json.dumps(
            evidence.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".runtime-abort-",
        suffix=".tmp",
        dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
