"""Diagnostic-only per-episode Webots context artifact.

This deliberately lives beside, rather than inside, :class:`EpisodeRecord` so
the stable ``episodes.csv`` v1.0 contract remains unchanged.  The JSON Lines
artifact is an additional debugging aid and is never part of validation or
held-out test output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EPISODE_CONTEXT_FILENAME = "episode_context.jsonl"
EPISODE_CONTEXT_SCHEMA_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = dict(record)
    if result.get("schema_version") != EPISODE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"Episode context row {index} must use schema version "
            f"{EPISODE_CONTEXT_SCHEMA_VERSION}."
        )
    identity = result.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"Episode context row {index} identity must be an object.")
    if identity.get("split") != "diagnostic":
        raise ValueError(
            f"Episode context row {index} must belong to the diagnostic split."
        )
    episode_id = identity.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(
            f"Episode context row {index} identity.episode_id must be non-empty."
        )
    return result


def write_episode_context_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Atomically write diagnostic context and return its provenance binding."""

    if not records:
        raise ValueError("At least one diagnostic episode context record is required.")
    validated = [_validated_record(record, index) for index, record in enumerate(records)]
    episode_ids = [record["identity"]["episode_id"] for record in validated]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Diagnostic episode context episode_id values must be unique.")

    root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        portable_path = resolved_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Episode context artifact must remain inside the repository.") from error

    lines: list[str] = []
    for index, record in enumerate(validated):
        try:
            lines.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Episode context row {index} is not finite JSON data: {error}"
            ) from error

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_name(f".{resolved_path.name}.tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(resolved_path)
    return {
        "path": portable_path,
        "schema_version": EPISODE_CONTEXT_SCHEMA_VERSION,
        "count": len(validated),
        "sha256": _sha256_file(resolved_path),
    }


__all__ = [
    "EPISODE_CONTEXT_FILENAME",
    "EPISODE_CONTEXT_SCHEMA_VERSION",
    "write_episode_context_jsonl",
]
