#!/usr/bin/env python3
"""One-time, no-rerun finalization of the completed Webots risk collection.

The normal collector completed all 120 Webots episodes and validated its result
artifacts, but its final tracked-worktree guard observed a concurrent, offline-
analyzer-only commit before the standard report could be written.  This script
does not start Webots and never writes inside the collection result directory.
It accepts exactly that predeclared Git transition, reruns the original strict
artifact parser, and atomically seals the normal report plus a separate audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

EXIT_FINALIZATION_FAILED = 2
_ORIGINAL_REVISION = "3ca553dfd35ca2f2c66728e7046abbfc588c62c9"
_INTERMEDIATE_REVISION = "65940692be2b78b8b32d2f5fc9059b5efb30b1ff"
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONFIG = "configs/eval/webots_recovery_v1_risk_trajectory_v1.yaml"
_NORMAL_RUNNER = "scripts/run_webots_risk_trajectory.py"
_CONTROLLER = "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
_WORLD = "worlds/webots_ppo_evaluate.wbt"
_MODEL = (
    "artifacts/models/webots/seed_11/w1_recovery_v1/"
    "ppo_webots_s11_step000040960.zip"
)
_RESULT_DIRECTORY = (
    "artifacts/results/webots_diagnostic_recovery_v1_risk_trajectory_v1_seed11"
)
_REPORT_DIRECTORY = "artifacts/results/webots_recovery_v1_risk_trajectory_v1"
_STAGING_DIRECTORY = (
    "artifacts/results/.webots_recovery_v1_risk_trajectory_v1.finalize-staging"
)
_RESULT_FILENAMES = frozenset(
    {"summary.json", "episodes.csv", "episode_context.jsonl"}
)
_EXPECTED_SOURCE_SHA256 = {
    _CONFIG: "fcbae35b8d5f66130a86d5fe2f65cb57f05dd502515cf36bdbd19df0d6b207d3",
    _NORMAL_RUNNER: "b2f745e5bdb2ec94f8ff130dc0fe9ffb6626257d2137ee8614a5b2a956b8d7ad",
    _CONTROLLER: "fa4871d52c1173cf67fe89b97696f63d10673755643c4b0b03306cfc0386a1fd",
    _WORLD: "e0ae7ac8aa2a7981f94673d656d76940166cb12ceab8dd2b17a17329985d5425",
}
_EXPECTED_MODEL_SHA256 = (
    "c5e7c3de42c7664477097b2734975660fec840ad5b7f45425098158c27cb7275"
)
_EXPECTED_WEBOTS_SHA256 = (
    "34e3824b3bb1d64cea8ed58ec4089b607fb83c906d36290df8e2d4a8ee2227c0"
)
_EXPECTED_RESULT_SHA256 = {
    "summary.json": "c6a6d8b1844fb4f6720c6caa3a1b9e6c509bb8c0abed27f84f6afe467a974519",
    "episodes.csv": "1665d9441ef6c28ab8a256aab22f54b4c75b7ac304d37bb902b37d4865c1c5a1",
    "episode_context.jsonl": (
        "fe68fe80c94be7c1c8ccedb75e54435b897c1e16c9305b8ac7c75e412814ed3f"
    ),
}
_RUNTIME_CHANGE_STATS = {
    "scripts/analyze_webots_risk_trajectory.py": (15, 0),
    "tests/test_analyze_webots_risk_trajectory.py": (5, 0),
}
_INTERMEDIATE_FILE_SHA256 = {
    "scripts/analyze_webots_risk_trajectory.py": (
        "f9f1c448461d91fad34417417374e17c141d9772e52c67dcbc8afff635ae1297"
    ),
    "tests/test_analyze_webots_risk_trajectory.py": (
        "039414a3f792557c28874c0b73cd48fcd8cd8d3fbac2c6b9449df6936042ea8a"
    ),
}
_FINALIZER_PATHS = frozenset(
    {
        "scripts/finalize_webots_risk_trajectory.py",
        "tests/test_finalize_webots_risk_trajectory.py",
    }
)
_ALLOWED_DIFF_PATHS = frozenset(_RUNTIME_CHANGE_STATS) | _FINALIZER_PATHS
_INTERMEDIATE_DIFF_STATUS = {path: "M" for path in _RUNTIME_CHANGE_STATS}
_FINALIZER_DIFF_STATUS = {path: "A" for path in _FINALIZER_PATHS}
_COLLECTION_EXPERIMENT_ID = "webots_recovery_v1_risk_trajectory_v1"
_CONDITION_ID = "DR1"
_PLANNER_LABEL = "ppo_recovery_v1_risk_trace"
_MODEL_VARIANT = "w1_recovery_v1"
_TRAINING_SEED = 11
_SEED_START = 20_494
_EPISODES = 120
_PORT = 13_640
_GUARD_FAILURE_REASON = "Git HEAD or tracked worktree changed during collection."

GitRunner = Callable[..., subprocess.CompletedProcess[Any]]
NormalRunnerLoader = Callable[[Path], ModuleType]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(
    repository_root: Path,
    *arguments: str,
    binary: bool = False,
    git_runner: GitRunner = subprocess.run,
) -> str | bytes:
    completed = git_runner(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        stdout = completed.stdout
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        detail = str(stderr or stdout or "").strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {detail or completed.returncode}"
        )
    return completed.stdout


def _git_revision(
    repository_root: Path, *, git_runner: GitRunner = subprocess.run
) -> str:
    revision = str(
        _git(
            repository_root,
            "rev-parse",
            "--verify",
            "HEAD",
            git_runner=git_runner,
        )
    ).strip().casefold()
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("Git HEAD is not a full lowercase commit revision.")
    return revision


def _require_clean_repository(
    repository_root: Path, *, git_runner: GitRunner = subprocess.run
) -> None:
    status = str(
        _git(
            repository_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
            git_runner=git_runner,
        )
    ).strip()
    if status:
        raise RuntimeError("Finalization requires a clean tracked worktree: " + status)
    untracked = str(
        _git(
            repository_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            git_runner=git_runner,
        )
    ).strip()
    if untracked:
        raise RuntimeError(
            "Finalization forbids non-ignored untracked files: "
            + " | ".join(untracked.splitlines())
        )


def _parse_name_status(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in value.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] not in {"A", "M"} or parts[1] in result:
            raise ValueError(f"Unsupported original-to-HEAD diff record: {line!r}.")
        result[parts[1]] = parts[0]
    return result


def _parse_numstat(value: str) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for line in value.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Unsupported original-to-HEAD numstat record: {line!r}.")
        path = parts[2]
        if path in result:
            raise ValueError(f"Duplicate original-to-HEAD numstat path: {path}.")
        result[path] = (int(parts[0]), int(parts[1]))
    return result


def _verify_git_transition(
    repository_root: Path, *, git_runner: GitRunner = subprocess.run
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Accept exactly the two offline changes and this finalizer's two files."""

    current_revision = _git_revision(repository_root, git_runner=git_runner)
    _require_clean_repository(repository_root, git_runner=git_runner)

    def only_parent(revision: str, label: str) -> str:
        tokens = str(
            _git(
                repository_root,
                "rev-list",
                "--parents",
                "-n",
                "1",
                revision,
                git_runner=git_runner,
            )
        ).strip().split()
        if len(tokens) != 2 or tokens[0] != revision:
            raise PermissionError(f"{label} must be a non-merge commit with one parent.")
        return tokens[1]

    if only_parent(_INTERMEDIATE_REVISION, "Intermediate review commit") != (
        _ORIGINAL_REVISION
    ):
        raise PermissionError("Intermediate review commit must directly follow original.")
    if only_parent(current_revision, "Finalizer commit") != _INTERMEDIATE_REVISION:
        raise PermissionError(
            "Finalizer HEAD must directly follow the fixed intermediate review commit."
        )

    intermediate_status = _parse_name_status(
        str(
            _git(
                repository_root,
                "diff",
                "--name-status",
                "--no-renames",
                f"{_ORIGINAL_REVISION}..{_INTERMEDIATE_REVISION}",
                "--",
                git_runner=git_runner,
            )
        ).strip()
    )
    if intermediate_status != _INTERMEDIATE_DIFF_STATUS:
        raise PermissionError(
            "Original-to-intermediate diff must contain exactly the two offline "
            f"analyzer modifications; observed {intermediate_status}."
        )
    intermediate_numstat = _parse_numstat(
        str(
            _git(
                repository_root,
                "diff",
                "--numstat",
                "--no-renames",
                f"{_ORIGINAL_REVISION}..{_INTERMEDIATE_REVISION}",
                "--",
                git_runner=git_runner,
            )
        ).strip()
    )
    if intermediate_numstat != _RUNTIME_CHANGE_STATS:
        raise PermissionError(
            "Concurrent offline analyzer changes must remain exactly 20 added lines: "
            f"observed {intermediate_numstat}."
        )
    for path, expected_sha256 in _INTERMEDIATE_FILE_SHA256.items():
        actual_sha256 = hashlib.sha256(
            _git_blob(
                repository_root,
                _INTERMEDIATE_REVISION,
                path,
                git_runner=git_runner,
            )
        ).hexdigest()
        if actual_sha256 != expected_sha256:
            raise PermissionError(
                f"Intermediate review blob SHA-256 changed for {path}: {actual_sha256}."
            )

    finalizer_status = _parse_name_status(
        str(
            _git(
                repository_root,
                "diff",
                "--name-status",
                "--no-renames",
                f"{_INTERMEDIATE_REVISION}..{current_revision}",
                "--",
                git_runner=git_runner,
            )
        ).strip()
    )
    if finalizer_status != _FINALIZER_DIFF_STATUS:
        raise PermissionError(
            "Intermediate-to-HEAD diff must add exactly this finalizer and its test; "
            f"observed {finalizer_status}."
        )
    finalizer_numstat = _parse_numstat(
        str(
            _git(
                repository_root,
                "diff",
                "--numstat",
                "--no-renames",
                f"{_INTERMEDIATE_REVISION}..{current_revision}",
                "--",
                git_runner=git_runner,
            )
        ).strip()
    )
    if set(finalizer_numstat) != _FINALIZER_PATHS:
        raise PermissionError("Intermediate-to-HEAD numstat escaped the finalizer allowlist.")
    combined = {**intermediate_numstat, **finalizer_numstat}
    if set(combined) != _ALLOWED_DIFF_PATHS:
        raise PermissionError("Two-commit transition escaped the exact allowlist.")
    return current_revision, combined


def _git_blob(
    repository_root: Path,
    revision: str,
    path: str,
    *,
    git_runner: GitRunner = subprocess.run,
) -> bytes:
    value = _git(
        repository_root,
        "cat-file",
        "blob",
        f"{revision}:{path}",
        binary=True,
        git_runner=git_runner,
    )
    assert isinstance(value, bytes)
    return value


def _verify_original_and_current_sources(
    repository_root: Path,
    webots_executable: Path,
    *,
    git_runner: GitRunner = subprocess.run,
) -> dict[str, Any]:
    root = repository_root.resolve()
    bindings: dict[str, Any] = {"git_revision": _ORIGINAL_REVISION}
    labels = {
        _CONFIG: "config",
        _NORMAL_RUNNER: "runner",
        _CONTROLLER: "controller",
        _WORLD: "world",
    }
    for path, label in labels.items():
        expected = _EXPECTED_SOURCE_SHA256[path]
        blob_sha256 = hashlib.sha256(
            _git_blob(root, _ORIGINAL_REVISION, path, git_runner=git_runner)
        ).hexdigest()
        if blob_sha256 != expected:
            raise ValueError(
                f"Original Git blob {path} SHA-256 mismatch: {blob_sha256}."
            )
        current_path = root / path
        if not current_path.is_file() or current_path.is_symlink():
            raise FileNotFoundError(f"Protected current source is not a regular file: {path}")
        current_sha256 = _sha256_file(current_path)
        if current_sha256 != expected:
            raise PermissionError(
                f"Protected collection source changed after the original run: {path}."
            )
        bindings[label] = {"path": path, "sha256": expected}

    model_path = root / _MODEL
    if not model_path.is_file() or model_path.is_symlink():
        raise FileNotFoundError(f"Locked recovery-v1 model is missing: {model_path}")
    if _sha256_file(model_path) != _EXPECTED_MODEL_SHA256:
        raise PermissionError("Locked recovery-v1 model SHA-256 changed.")
    bindings["model"] = {"path": _MODEL, "sha256": _EXPECTED_MODEL_SHA256}

    executable = webots_executable.resolve()
    if not executable.is_file() or executable.is_symlink():
        raise FileNotFoundError(f"Bound Webots executable is missing: {executable}")
    if _sha256_file(executable) != _EXPECTED_WEBOTS_SHA256:
        raise PermissionError("Webots executable does not match the original dry-run binding.")
    bindings["webots_executable"] = {
        "path": str(executable),
        "sha256": _EXPECTED_WEBOTS_SHA256,
    }
    return bindings


def _load_normal_runner(repository_root: Path) -> ModuleType:
    path = repository_root.resolve() / _NORMAL_RUNNER
    name = "_locked_original_webots_risk_trajectory_runner"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the locked normal runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_loaded_config(config: Any) -> None:
    expected_seeds = tuple(range(_SEED_START, _SEED_START + _EPISODES))
    checks = {
        "source_sha256": _EXPECTED_SOURCE_SHA256[_CONFIG],
        "experiment_id": _COLLECTION_EXPERIMENT_ID,
        "condition_id": _CONDITION_ID,
        "training_seed": _TRAINING_SEED,
        "scenario_seed_start": _SEED_START,
        "episodes": _EPISODES,
        "context_schema_version": "1.1",
    }
    for name, expected in checks.items():
        if getattr(config, name, None) != expected:
            raise PermissionError(f"Locked normal config {name} changed.")
    if tuple(config.seeds) != expected_seeds:
        raise PermissionError("Locked normal config seed set changed.")
    model = config.model
    for name, expected in {
        "portable_path": _MODEL,
        "sha256": _EXPECTED_MODEL_SHA256,
        "variant": _MODEL_VARIANT,
        "run_name": Path(_RESULT_DIRECTORY).name,
        "planner_label": _PLANNER_LABEL,
    }.items():
        if getattr(model, name, None) != expected:
            raise PermissionError(f"Locked normal model binding {name} changed.")
    scenario = config.scenario
    if (
        scenario.obstacle_count != 1
        or tuple(scenario.obstacle_speed_range) != (0.15, 0.30)
        or scenario.horizontal_route_half_length != 1.10
        or scenario.vertical_route_half_length != 1.10
        or scenario.safety_shield is not False
        or scenario.deterministic_policy is not True
    ):
        raise PermissionError("Locked normal scenario changed.")


def _execution_environment(config: Any) -> dict[str, str]:
    return {
        "WEBOTS_PORT": str(_PORT),
        "WEBOTS_MODEL_SEED": str(_TRAINING_SEED),
        "WEBOTS_EVAL_MODEL": _MODEL,
        "WEBOTS_EVAL_MODEL_SHA256": _EXPECTED_MODEL_SHA256,
        "WEBOTS_MODEL_VARIANT": _MODEL_VARIANT,
        "WEBOTS_EVAL_SPLIT": "diagnostic",
        "WEBOTS_EVAL_SEED_START": str(_SEED_START),
        "WEBOTS_EVAL_EPISODES": str(_EPISODES),
        "WEBOTS_EVAL_CONDITION": _CONDITION_ID,
        "WEBOTS_EVAL_PLANNER": "ppo",
        "WEBOTS_EVAL_PLANNER_LABEL": _PLANNER_LABEL,
        "WEBOTS_EVAL_RUN_NAME": Path(_RESULT_DIRECTORY).name,
        "WEBOTS_OBSTACLE_COUNT": "1",
        "WEBOTS_OBSTACLE_SPEED_MIN": "0.15",
        "WEBOTS_OBSTACLE_SPEED_MAX": "0.3",
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": "1.1",
        "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": "1.1",
        "WEBOTS_SAFETY_SHIELD": "0",
        "WEBOTS_EVAL_DETERMINISTIC_POLICY": "1",
        "WEBOTS_EVAL_EPISODE_CONTEXT": "1",
        "WEBOTS_RISK_TRAJECTORY_ID": _COLLECTION_EXPERIMENT_ID,
        "WEBOTS_RISK_TRAJECTORY_CONFIG": _CONFIG,
        "WEBOTS_RISK_TRAJECTORY_CONFIG_SHA256": config.source_sha256,
    }


def _execution_command(
    repository_root: Path, webots_executable: Path
) -> tuple[str, ...]:
    return (
        str(webots_executable.resolve()),
        f"--port={_PORT}",
        "--batch",
        "--mode=fast",
        "--no-rendering",
        "--stdout",
        "--stderr",
        str((repository_root.resolve() / _WORLD).resolve()),
    )


def _verify_result_directory(repository_root: Path) -> dict[str, dict[str, str]]:
    root = repository_root.resolve()
    result = root / _RESULT_DIRECTORY
    if not result.is_dir() or result.is_symlink():
        raise FileNotFoundError(f"Completed result directory is missing: {result}")
    entries = list(result.iterdir())
    names = {entry.name for entry in entries}
    if names != _RESULT_FILENAMES or any(
        not entry.is_file() or entry.is_symlink() for entry in entries
    ):
        raise PermissionError(
            "Completed result directory must contain exactly summary.json, "
            "episodes.csv, and episode_context.jsonl."
        )
    bindings: dict[str, dict[str, str]] = {}
    for name in sorted(_RESULT_FILENAMES):
        path = result / name
        actual = _sha256_file(path)
        expected = _EXPECTED_RESULT_SHA256[name]
        if actual != expected:
            raise PermissionError(
                f"Completed result artifact {name} changed: expected {expected}, "
                f"actual {actual}."
            )
        bindings[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": actual,
        }
    return bindings


def _require_destinations_absent(repository_root: Path) -> tuple[Path, Path]:
    root = repository_root.resolve()
    report = root / _REPORT_DIRECTORY
    staging = root / _STAGING_DIRECTORY
    existing = [path for path in (report, staging) if path.exists()]
    if existing:
        raise FileExistsError(
            "One-time finalizer refuses to overwrite report or staging paths: "
            + ", ".join(str(path) for path in existing)
        )
    return report, staging


def _standard_report(
    *,
    config: Any,
    parsed: Any,
    run: Any,
    source_bindings: Mapping[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    seeds = list(range(_SEED_START, _SEED_START + _EPISODES))
    return {
        "schema_version": 1,
        "experiment_id": _COLLECTION_EXPERIMENT_ID,
        "status": "collection_complete",
        "generated_at_utc": generated_at_utc,
        "split": "diagnostic",
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "performance_gate_applied": False,
        "source_bindings": dict(source_bindings),
        "scenario": {
            "condition_id": _CONDITION_ID,
            "scenario_seed_start": _SEED_START,
            "episodes": _EPISODES,
            "scenario_seeds": seeds,
            "axis_counts": {"horizontal": 60, "vertical": 60},
            "obstacle_count": 1,
            "obstacle_speed_range": [0.15, 0.30],
            "horizontal_route_half_length": 1.10,
            "vertical_route_half_length": 1.10,
            "safety_shield": False,
            "deterministic_policy": True,
        },
        "collection": {
            "expected_episode_count": _EPISODES,
            "episodes_csv_count": len(parsed.rows_by_seed),
            "episode_context_count": len(parsed.contexts_by_seed),
            "risk_trace_count": len(parsed.contexts_by_seed),
            "episode_context_schema_version": config.context_schema_version,
            "expected_route_axis_counts": {"horizontal": 60, "vertical": 60},
            "adaptive_extension": "forbidden",
            "complete": True,
        },
        "model": {
            "path": _MODEL,
            "sha256": _EXPECTED_MODEL_SHA256,
            "variant": _MODEL_VARIANT,
            "training_seed": _TRAINING_SEED,
        },
        "execution": {
            "command": list(run.command),
            "environment": dict(run.environment),
            "port": run.port,
            "exit_code": 0,
        },
        "descriptive_metrics_not_used_as_a_gate": dict(parsed.metrics),
        "artifacts": dict(parsed.artifacts),
        "access": {"validation_or_test": "forbidden"},
        "written_reports": {
            "json": f"{_REPORT_DIRECTORY}/collection_report.json",
            "yaml": f"{_REPORT_DIRECTORY}/collection_report.yaml",
            "sha256_sidecar": f"{_REPORT_DIRECTORY}/collection_report.sha256",
        },
    }


def _runtime_change_records() -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "classification": "offline_analysis_only",
            "added_lines": stats[0],
            "removed_lines": stats[1],
            "intermediate_blob_sha256": _INTERMEDIATE_FILE_SHA256[path],
        }
        for path, stats in sorted(_RUNTIME_CHANGE_STATS.items())
    ]


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _verify_staged_reports(
    staging_directory: Path,
    *,
    standard_report: Mapping[str, Any],
    finalize_audit: Mapping[str, Any],
) -> None:
    """Re-read every staged byte and verify exact objects and sidecars."""

    expected_names = {
        "collection_report.json",
        "collection_report.yaml",
        "collection_report.sha256",
        "finalize_audit.json",
        "finalize_audit.sha256",
    }
    entries = list(staging_directory.iterdir())
    if {entry.name for entry in entries} != expected_names or any(
        not entry.is_file() or entry.is_symlink() for entry in entries
    ):
        raise RuntimeError("Finalization staging directory contains unexpected files.")

    standard_json = staging_directory / "collection_report.json"
    standard_yaml = staging_directory / "collection_report.yaml"
    standard_sidecar = staging_directory / "collection_report.sha256"
    audit_json = staging_directory / "finalize_audit.json"
    audit_sidecar = staging_directory / "finalize_audit.sha256"
    reread_standard_json = json.loads(standard_json.read_text(encoding="utf-8"))
    if reread_standard_json != dict(standard_report):
        raise RuntimeError("Staged collection_report.json differs from its memory object.")
    reread_standard_yaml = yaml.safe_load(standard_yaml.read_text(encoding="utf-8"))
    if reread_standard_yaml != dict(standard_report):
        raise RuntimeError("Staged collection_report.yaml differs from its memory object.")
    expected_standard_sidecar = (
        f"{_sha256_file(standard_json)}  collection_report.json\n"
        f"{_sha256_file(standard_yaml)}  collection_report.yaml\n"
    )
    if standard_sidecar.read_text(encoding="utf-8") != expected_standard_sidecar:
        raise RuntimeError("Staged collection_report.sha256 has non-exact bindings.")

    reread_audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if reread_audit != dict(finalize_audit):
        raise RuntimeError("Staged finalize_audit.json differs from its memory object.")
    expected_audit_sidecar = f"{_sha256_file(audit_json)}  finalize_audit.json\n"
    if audit_sidecar.read_text(encoding="utf-8") != expected_audit_sidecar:
        raise RuntimeError("Staged finalize_audit.sha256 has a non-exact binding.")

    # A second enumeration is deliberately the final staging operation before rename.
    final_entries = list(staging_directory.iterdir())
    if {entry.name for entry in final_entries} != expected_names or any(
        not entry.is_file() or entry.is_symlink() for entry in final_entries
    ):
        raise RuntimeError("Staged report set changed during read-back verification.")


def finalize_collection(
    *,
    repository_root: Path,
    webots_executable: Path,
    git_runner: GitRunner = subprocess.run,
    normal_runner_loader: NormalRunnerLoader = _load_normal_runner,
) -> Mapping[str, Any]:
    """Validate and seal the completed artifacts without starting Webots."""

    root = repository_root.resolve()
    report_directory, staging_directory = _require_destinations_absent(root)
    current_revision, diff_numstat = _verify_git_transition(
        root, git_runner=git_runner
    )
    source_bindings = _verify_original_and_current_sources(
        root,
        webots_executable,
        git_runner=git_runner,
    )
    result_bindings = _verify_result_directory(root)

    normal_runner = normal_runner_loader(root)
    config = normal_runner.load_risk_trajectory_config(
        root / _CONFIG,
        repository_root=root,
    )
    _verify_loaded_config(config)
    environment = _execution_environment(config)
    command = _execution_command(root, webots_executable)
    run = normal_runner.RiskTrajectoryRun(
        port=_PORT,
        command=command,
        environment=environment,
        result_directory=root / _RESULT_DIRECTORY,
        report_directory=report_directory,
    )
    parsed = normal_runner._parse_result(
        run,
        config=config,
        repository_root=root,
    )
    if len(parsed.rows_by_seed) != _EPISODES or len(parsed.contexts_by_seed) != _EPISODES:
        raise RuntimeError("Strict normal parser did not return all 120 episodes and contexts.")
    axis_counts = Counter(
        context["route_axis"] for context in parsed.contexts_by_seed.values()
    )
    if axis_counts != {"horizontal": 60, "vertical": 60}:
        raise RuntimeError(f"Strict normal parser did not return H/V=60/60: {axis_counts}.")
    if dict(parsed.artifacts) != {
        "result_directory": _RESULT_DIRECTORY,
        "summary_json": result_bindings["summary.json"],
        "episodes_csv": result_bindings["episodes.csv"],
        "episode_context_jsonl": result_bindings["episode_context.jsonl"],
    }:
        raise RuntimeError("Strict parser artifact bindings disagree with sealed results.")

    # Repeat every trust check immediately before the first write.
    if _git_revision(root, git_runner=git_runner) != current_revision:
        raise RuntimeError("Git HEAD changed during finalizer preflight.")
    _require_clean_repository(root, git_runner=git_runner)
    _verify_original_and_current_sources(
        root,
        webots_executable,
        git_runner=git_runner,
    )
    if _verify_result_directory(root) != result_bindings:
        raise RuntimeError("Completed result artifacts changed during finalizer preflight.")
    _require_destinations_absent(root)

    generated_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    standard_report = _standard_report(
        config=config,
        parsed=parsed,
        run=run,
        source_bindings=source_bindings,
        generated_at_utc=generated_at_utc,
    )
    normal_runner._write_report(
        standard_report,
        output_directory=staging_directory,
    )
    standard_paths = {
        "json": staging_directory / "collection_report.json",
        "yaml": staging_directory / "collection_report.yaml",
        "sha256_sidecar": staging_directory / "collection_report.sha256",
    }
    if set(path.name for path in staging_directory.iterdir()) != {
        "collection_report.json",
        "collection_report.yaml",
        "collection_report.sha256",
    }:
        raise RuntimeError("Locked normal report writer emitted unexpected files.")
    standard_bindings = {
        name: {
            "path": f"{_REPORT_DIRECTORY}/{path.name}",
            "sha256": _sha256_file(path),
        }
        for name, path in standard_paths.items()
    }
    finalizer_file_sha256 = {
        path: hashlib.sha256(
            _git_blob(root, current_revision, path, git_runner=git_runner)
        ).hexdigest()
        for path in sorted(_FINALIZER_PATHS)
    }
    finalize_audit: dict[str, Any] = {
        "schema_version": 1,
        "audit_type": "one_time_webots_risk_trajectory_finalization",
        "status": "sealed_without_webots_rerun",
        "generated_at_utc": generated_at_utc,
        "original_collection_revision": _ORIGINAL_REVISION,
        "intermediate_review_revision": _INTERMEDIATE_REVISION,
        "current_finalizer_revision": current_revision,
        "original_revision_is_ancestor": True,
        "tracked_worktree_clean": True,
        "nonignored_untracked_files": [],
        "original_guard_failure": {
            "stage": "post_collection_clean_tree_guard",
            "reason": _GUARD_FAILURE_REASON,
            "webots_subprocess_exit_code": 0,
            "standard_report_was_written": False,
            "runtime_changes": _runtime_change_records(),
            "runtime_change_total_added_lines": 20,
            "runtime_change_total_removed_lines": 0,
        },
        "allowed_original_to_finalizer_diff": {
            "paths": sorted(_ALLOWED_DIFF_PATHS),
            "numstat": {
                path: {"added_lines": stats[0], "removed_lines": stats[1]}
                for path, stats in sorted(diff_numstat.items())
            },
        },
        "reviewed_commit_chain": {
            "original": {"revision": _ORIGINAL_REVISION},
            "intermediate": {
                "revision": _INTERMEDIATE_REVISION,
                "parent": _ORIGINAL_REVISION,
                "changed_file_sha256": dict(sorted(_INTERMEDIATE_FILE_SHA256.items())),
            },
            "finalizer": {
                "revision": current_revision,
                "parent": _INTERMEDIATE_REVISION,
                "added_file_sha256": finalizer_file_sha256,
            },
        },
        "no_webots_rerun": True,
        "validation_split_was_run": False,
        "test_split_was_run": False,
        "validation_or_test_access": "forbidden",
        "source_bindings_from_original_dry_run": dict(source_bindings),
        "result_artifacts": result_bindings,
        "standard_report": {
            "status": "collection_complete",
            "bindings": standard_bindings,
            "collection_report_sha256": standard_bindings["json"]["sha256"],
        },
        "writes": {
            "result_directory_modified": False,
            "webots_started": False,
            "standard_report_directory": _REPORT_DIRECTORY,
            "finalize_audit_json": f"{_REPORT_DIRECTORY}/finalize_audit.json",
            "finalize_audit_sha256": f"{_REPORT_DIRECTORY}/finalize_audit.sha256",
        },
    }
    audit_json = staging_directory / "finalize_audit.json"
    audit_sidecar = staging_directory / "finalize_audit.sha256"
    _write_new_json(audit_json, finalize_audit)
    audit_sidecar.write_text(
        f"{_sha256_file(audit_json)}  {audit_json.name}\n",
        encoding="utf-8",
    )
    expected_staging_names = {
        "collection_report.json",
        "collection_report.yaml",
        "collection_report.sha256",
        "finalize_audit.json",
        "finalize_audit.sha256",
    }
    if set(path.name for path in staging_directory.iterdir()) != expected_staging_names:
        raise RuntimeError("Finalization staging directory contains unexpected files.")

    # Seal only if tracked sources and the collection are still byte-identical.
    if _git_revision(root, git_runner=git_runner) != current_revision:
        raise RuntimeError("Git HEAD changed while finalization reports were staged.")
    _require_clean_repository(root, git_runner=git_runner)
    if _verify_result_directory(root) != result_bindings:
        raise RuntimeError("Completed result artifacts changed while reports were staged.")
    if report_directory.exists():
        raise FileExistsError("Standard report destination appeared during finalization.")
    _verify_staged_reports(
        staging_directory,
        standard_report=standard_report,
        finalize_audit=finalize_audit,
    )
    staging_directory.replace(report_directory)
    return {
        "status": "sealed_without_webots_rerun",
        "standard_report": f"{_REPORT_DIRECTORY}/collection_report.json",
        "standard_report_sha256": standard_bindings["json"]["sha256"],
        "finalize_audit": f"{_REPORT_DIRECTORY}/finalize_audit.json",
        "finalize_audit_sha256": _sha256_file(report_directory / "finalize_audit.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webots", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).absolute().parents[1]
    try:
        result = finalize_collection(
            repository_root=root,
            webots_executable=args.webots,
        )
    except Exception as error:
        print(
            f"WEBOTS_RISK_TRAJECTORY_FINALIZE_FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_FINALIZATION_FAILED
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("WEBOTS_RISK_TRAJECTORY_FINALIZED_WITHOUT_RERUN", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
