from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indoor_patrol_rl.evaluation.shield_evidence import (
    SHIELD_EVIDENCE_FILENAME,
    SHIELD_EVIDENCE_SCHEMA_VERSION,
    build_shield_evidence_record,
    validate_shield_evidence_record,
    write_shield_evidence_jsonl,
)


def _info(
    step: int,
    *,
    enabled: bool,
    policy: tuple[float, float] = (0.5, 0.1),
    shielded: tuple[float, float] = (0.5, 0.1),
) -> dict[str, Any]:
    intervention = float(np.linalg.norm(np.asarray(shielded) - np.asarray(policy)))
    return {
        "step_count": step,
        "policy_action": np.asarray(policy, dtype=np.float32),
        "shielded_action": np.asarray(shielded, dtype=np.float32),
        "executed_action": np.asarray((0.2, 0.05), dtype=np.float32),
        "shield_enabled": enabled,
        "shield_active": enabled and intervention > 1e-6,
        "shield_intervention": intervention,
        "shield_front_clearance": 0.35,
        "shield_left_clearance": 1.1,
        "shield_right_clearance": 1.2,
        "shield_speed_fraction_limit": 0.4,
        "shield_normalized_speed_limit": -0.2,
    }


def _record(*, enabled: bool = True) -> dict[str, Any]:
    steps = [
        _info(1, enabled=enabled),
        _info(
            2,
            enabled=enabled,
            shielded=(-0.2, 0.1) if enabled else (0.5, 0.1),
        ),
    ]
    terminal = {
        "step_count": 2,
        "shield_intervention_epsilon": 1e-6,
        "episode_shield_intervention_ratio": 0.5 if enabled else 0.0,
        "episode_shield_intervention_mean": 0.35 if enabled else 0.0,
    }
    return build_shield_evidence_record(
        identity={
            "episode_id": "diagnostic-DB1-25728-candidate",
            "split": "diagnostic",
            "condition_id": "DB1",
            "scenario_seed": 25_728,
            "planner": "candidate",
        },
        steps=steps,
        intervention_epsilon=1e-6,
        terminal_info=terminal,
    )


def test_build_record_preserves_direct_actions_and_episode_intervention() -> None:
    record = _record(enabled=True)

    assert record["schema_version"] == SHIELD_EVIDENCE_SCHEMA_VERSION
    assert record["steps"][1]["policy_action"] == pytest.approx([0.5, 0.1])
    assert record["steps"][1]["shielded_action"] == pytest.approx([-0.2, 0.1])
    assert record["steps"][1]["shield_active"] is True
    assert record["summary"] == {
        "shield_enabled": True,
        "step_count": 2,
        "intervention_epsilon": 1e-6,
        "active_step_count": 1,
        "active_step_ratio": 0.5,
        "intervention_sum": pytest.approx(0.7),
        "intervention_mean": pytest.approx(0.35),
    }


def test_disabled_shield_requires_zero_intervention_and_preserves_action() -> None:
    record = _record(enabled=False)
    assert record["summary"]["active_step_count"] == 0
    assert record["summary"]["active_step_ratio"] == 0.0
    assert all(step["policy_action"] == step["shielded_action"] for step in record["steps"])

    invalid = copy.deepcopy(record)
    invalid["steps"][0]["shielded_action"][0] = -0.5
    invalid["steps"][0]["shield_intervention"] = 1.0
    with pytest.raises(ValueError, match="disabled shield must preserve"):
        validate_shield_evidence_record(invalid)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record["steps"][0].__setitem__("unknown", 1),
        lambda record: record["steps"][1].__setitem__("step", 3),
        lambda record: record["steps"][0].__setitem__("shield_intervention", float("nan")),
        lambda record: record["summary"].__setitem__("active_step_ratio", 0.25),
    ],
)
def test_validator_fails_closed_on_schema_and_summary_drift(mutation: Any) -> None:
    record = _record()
    mutation(record)
    with pytest.raises(ValueError):
        validate_shield_evidence_record(record)


def test_builder_rejects_terminal_counter_disagreement() -> None:
    with pytest.raises(ValueError, match="ratio disagrees"):
        build_shield_evidence_record(
            identity=_record()["identity"],
            steps=[_info(1, enabled=True)],
            intervention_epsilon=1e-6,
            terminal_info={
                "step_count": 1,
                "shield_intervention_epsilon": 1e-6,
                "episode_shield_intervention_ratio": 0.5,
                "episode_shield_intervention_mean": 0.0,
            },
        )


def test_builder_rejects_terminal_intervention_epsilon_drift() -> None:
    terminal = {
        "step_count": 1,
        "shield_intervention_epsilon": 2e-6,
        "episode_shield_intervention_ratio": 0.0,
        "episode_shield_intervention_mean": 0.0,
    }
    with pytest.raises(ValueError, match="shield_intervention_epsilon disagrees"):
        build_shield_evidence_record(
            identity=_record()["identity"],
            steps=[_info(1, enabled=True)],
            intervention_epsilon=1e-6,
            terminal_info=terminal,
        )


def test_writer_is_atomic_bound_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "artifacts/results/run" / SHIELD_EVIDENCE_FILENAME
    record = _record()

    binding = write_shield_evidence_jsonl(output, [record], repository_root=tmp_path)

    assert binding == {
        "path": "artifacts/results/run/shield_evidence.jsonl",
        "schema_version": "1.0",
        "count": 1,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == record
    assert not output.with_name(f".{output.name}.tmp").exists()
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_shield_evidence_jsonl(output, [record], repository_root=tmp_path)
    assert output.read_bytes() == original


def test_writer_rejects_non_diagnostic_and_duplicate_identity(tmp_path: Path) -> None:
    record = _record()
    invalid = copy.deepcopy(record)
    invalid["identity"]["split"] = "validation"
    with pytest.raises(PermissionError, match="diagnostic"):
        write_shield_evidence_jsonl(
            tmp_path / SHIELD_EVIDENCE_FILENAME,
            [invalid],
            repository_root=tmp_path,
        )
    with pytest.raises(ValueError, match="must be unique"):
        write_shield_evidence_jsonl(
            tmp_path / SHIELD_EVIDENCE_FILENAME,
            [record, record],
            repository_root=tmp_path,
        )
