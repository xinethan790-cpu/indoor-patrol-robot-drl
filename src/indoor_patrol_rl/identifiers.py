"""Shared public identifier contracts used by patrol evidence and relays."""

from __future__ import annotations

import re

PATROL_RUN_ID_REGEX = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
PATROL_RUN_ID_PATTERN = re.compile(PATROL_RUN_ID_REGEX, flags=re.ASCII)


class PatrolRunIdError(ValueError):
    """A Python patrol run ID violates the shared local/cloud contract."""


def validate_patrol_run_id(value: object, *, field_name: str = "run_id") -> str:
    """Return a run ID that is safe in local evidence and relay state."""

    if not isinstance(value, str) or PATROL_RUN_ID_PATTERN.fullmatch(value) is None:
        raise PatrolRunIdError(
            f"{field_name} must match {PATROL_RUN_ID_REGEX}: 1-128 ASCII letters, "
            "digits, dots, underscores, colons, or hyphens, beginning with a letter "
            "or digit."
        )
    return value


__all__ = [
    "PATROL_RUN_ID_PATTERN",
    "PATROL_RUN_ID_REGEX",
    "PatrolRunIdError",
    "validate_patrol_run_id",
]
