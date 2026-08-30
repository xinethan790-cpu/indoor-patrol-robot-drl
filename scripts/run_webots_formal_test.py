#!/usr/bin/env python3
"""Run the one-shot, locked 400-episode Webots test matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indoor_patrol_rl.evaluation.formal_protocol import (
    FinalLock,
    begin_test_attempt,
    file_sha256,
    invalidate_test_attempt,
    load_final_lock,
    mark_test_success,
)
from indoor_patrol_rl.evaluation.formal_reporting import (
    FormalResultInput,
    build_formal_report,
    write_formal_report,
)
from indoor_patrol_rl.evaluation.webots_formal_config import (
    WebotsFormalConfig,
    load_webots_formal_config,
    resolve_formal_test_request,
)
from indoor_patrol_rl.training.webots_curriculum import (
    choose_available_port,
    resolve_webots_executable,
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
PortAllocator = Callable[[int, set[int]], int]
PortVerifier = Callable[[int], None]
LockLoader = Callable[..., FinalLock]
SourceVerifier = Callable[[Path], str]
WindowsMetadataProbe = Callable[[Path], Mapping[str, Any]]
RuntimeVerifier = Callable[[str], None]
BeginAttempt = Callable[..., Mapping[str, Any]]
MarkSuccess = Callable[..., Mapping[str, Any]]
InvalidateAttempt = Callable[..., Mapping[str, Any]]
ReportBuilder = Callable[..., dict[str, Any]]
ReportWriter = Callable[[Mapping[str, Any], Path], Path]

_RUN_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_WEBOTS_VERSION_PATTERN = re.compile(r"(?<![A-Za-z0-9])R\d{4}[a-z](?![A-Za-z0-9])")
_DEFAULT_TIMEOUT_SECONDS = 3_600.0
_VERSION_PROBE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class WebotsExecutableEvidence:
    """Auditable identity collected from the executable before test consumption."""

    executable_path: str
    executable_sha256: str
    expected_version: str
    detected_version: str
    version_probe_stdout: str
    version_probe_stderr: str
    windows_file_version: str | None
    windows_product_version: str | None
    signature_status: str
    signature_status_message: str | None
    signer_subject: str | None
    signer_thumbprint: str | None
    probed_at: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "executable_path": self.executable_path,
            "executable_sha256": self.executable_sha256,
            "expected_version": self.expected_version,
            "detected_version": self.detected_version,
            "version_probe_stdout": self.version_probe_stdout,
            "version_probe_stderr": self.version_probe_stderr,
            "windows_file_version": self.windows_file_version,
            "windows_product_version": self.windows_product_version,
            "signature_status": self.signature_status,
            "signature_status_message": self.signature_status_message,
            "signer_subject": self.signer_subject,
            "signer_thumbprint": self.signer_thumbprint,
            "probed_at": self.probed_at,
        }


@dataclass(frozen=True)
class FormalWebotsRun:
    """One of the twenty fixed planner-condition simulator invocations."""

    condition_id: str
    planner: str
    planner_label: str
    training_seed: int | None
    model_path: str | None
    model_sha256: str | None
    port: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    staging_directory: Path
    result_directory: Path


@dataclass(frozen=True)
class FormalTestExecution:
    """Result of a dry plan or a completed formal execution."""

    dry_run: bool
    runs: tuple[FormalWebotsRun, ...]
    report_path: Path | None = None
    result_manifest_path: Path | None = None
    ledger: Mapping[str, Any] | None = None
    executable_evidence: WebotsExecutableEvidence | None = None


def _resolve_inside_repository(repository_root: Path, value: str | Path, label: str) -> Path:
    root = repository_root.resolve()
    configured = Path(value)
    candidate = configured if configured.is_absolute() else root / configured
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository: {value}") from error
    return resolved


def _portable(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Formal artifact must stay inside the repository: {path}") from error


def _clean_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("WEBOTS_") and name != "WEBOTS_HOME":
            environment.pop(name)
    environment.update(overrides)
    return environment


def _validate_timeout_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a finite number greater than zero.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("timeout_seconds must be finite and greater than zero.")
    return result


def _require_port_available(port: int) -> None:
    """Fail if the freshly selected Webots port cannot be bound right now."""

    if type(port) is not int or not 1 <= port <= 65_535:
        raise ValueError(f"Invalid Webots port: {port!r}.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise RuntimeError(
            f"Webots port {port} is unavailable immediately before process launch."
        ) from error


def _probe_windows_executable_metadata(executable: Path) -> Mapping[str, Any]:
    """Read Windows version-resource and Authenticode state without enforcing signing."""

    if os.name != "nt":
        return {
            "windows_file_version": None,
            "windows_product_version": None,
            "signature_status": "NotApplicable",
            "signature_status_message": None,
            "signer_subject": None,
            "signer_thumbprint": None,
        }
    script = """
$ErrorActionPreference = 'Stop'
$item = Get-Item -LiteralPath $env:CODEX_WEBOTS_EXECUTABLE_PROBE
$signature = Get-AuthenticodeSignature -LiteralPath $env:CODEX_WEBOTS_EXECUTABLE_PROBE
$subject = $null
$thumbprint = $null
if ($null -ne $signature.SignerCertificate) {
  $subject = $signature.SignerCertificate.Subject
  $thumbprint = $signature.SignerCertificate.Thumbprint
}
[ordered]@{
  windows_file_version = $item.VersionInfo.FileVersion
  windows_product_version = $item.VersionInfo.ProductVersion
  signature_status = [string]$signature.Status
  signature_status_message = $signature.StatusMessage
  signer_subject = $subject
  signer_thumbprint = $thumbprint
} | ConvertTo-Json -Compress
"""
    environment = os.environ.copy()
    environment["CODEX_WEBOTS_EXECUTABLE_PROBE"] = str(executable)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Could not inspect the Webots Windows file version and Authenticode status: "
            + (detail or f"exit code {completed.returncode}")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Windows returned invalid executable metadata JSON.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("signature_status"), str):
        raise RuntimeError("Windows executable metadata omitted the signature status.")
    return payload


def probe_webots_executable(
    executable: Path,
    *,
    expected_version: str,
    timeout_seconds: float,
    process_runner: ProcessRunner = subprocess.run,
    windows_metadata_probe: WindowsMetadataProbe = _probe_windows_executable_metadata,
) -> WebotsExecutableEvidence:
    """Execute ``webots --version`` and capture immutable executable evidence."""

    path = executable.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Webots executable does not exist: {path}")
    before_sha256 = file_sha256(path)
    completed = process_runner(
        [str(path), "--version"],
        cwd=path.parent,
        env=_clean_environment({}),
        check=False,
        capture_output=True,
        text=True,
        timeout=min(timeout_seconds, _VERSION_PROBE_TIMEOUT_SECONDS),
    )
    stdout = _subprocess_output_text(completed.stdout)
    stderr = _subprocess_output_text(completed.stderr)
    if completed.returncode != 0:
        detail = (stderr or stdout).strip()
        raise RuntimeError(
            "Webots executable version probe failed: "
            + (detail or f"exit code {completed.returncode}")
        )
    versions = tuple(dict.fromkeys(_WEBOTS_VERSION_PATTERN.findall(f"{stdout}\n{stderr}")))
    if versions != (expected_version,):
        detected = ", ".join(versions) or "no Webots release token"
        raise ValueError(
            f"Webots executable version {detected} does not match locked {expected_version}."
        )
    metadata = windows_metadata_probe(path)
    signature_status = metadata.get("signature_status")
    if not isinstance(signature_status, str) or not signature_status:
        raise ValueError("Webots executable metadata must record a signature status.")
    after_sha256 = file_sha256(path)
    if after_sha256 != before_sha256:
        raise RuntimeError("Webots executable changed while its identity was being probed.")
    return WebotsExecutableEvidence(
        executable_path=str(path),
        executable_sha256=before_sha256,
        expected_version=expected_version,
        detected_version=expected_version,
        version_probe_stdout=stdout,
        version_probe_stderr=stderr,
        windows_file_version=metadata.get("windows_file_version"),
        windows_product_version=metadata.get("windows_product_version"),
        signature_status=signature_status,
        signature_status_message=metadata.get("signature_status_message"),
        signer_subject=metadata.get("signer_subject"),
        signer_thumbprint=metadata.get("signer_thumbprint"),
        probed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _verify_webots_executable(evidence: WebotsExecutableEvidence) -> None:
    path = Path(evidence.executable_path)
    if not path.is_file() or file_sha256(path) != evidence.executable_sha256:
        raise RuntimeError("Webots executable fingerprint drifted after preflight.")


def _write_webots_executable_evidence(
    output_root: Path, evidence: WebotsExecutableEvidence
) -> Path:
    path = output_root / "webots_executable.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(evidence.to_mapping(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def require_clean_locked_source(repository_root: Path) -> str:
    """Return the current full Git revision only for a clean evaluation checkout."""

    def git(arguments: Sequence[str]) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: {detail or completed.returncode}"
            )
        return completed.stdout.strip()

    revision = git(("rev-parse", "--verify", "HEAD")).casefold()
    if _GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("Git HEAD is not a full lowercase 40- or 64-character revision.")
    dirty = git(("status", "--porcelain", "--untracked-files=all"))
    if dirty:
        preview = " | ".join(dirty.splitlines()[:5])
        raise RuntimeError(
            "Formal test requires the clean source checkout recorded by its lock: " + preview
        )
    return revision


def _allocate_port(preferred: int, reserved: set[int]) -> int:
    return choose_available_port(preferred, reserved)


def _sequential_port(preferred: int, reserved: set[int]) -> int:
    port = preferred
    while port in reserved:
        port += 1
    if port > 65_535:
        raise RuntimeError("Formal Webots port range is exhausted.")
    reserved.add(port)
    return port


def _safe_run_name(campaign_id: str, condition_id: str, planner_label: str) -> str:
    value = f"formal_stage_{campaign_id}_{condition_id}_{planner_label}"
    normalized = _RUN_NAME_PATTERN.sub("_", value)
    if Path(normalized).name != normalized:
        raise ValueError("Generated formal Webots run name is not path-safe.")
    return normalized


def _with_runtime_port(run: FormalWebotsRun, port: int) -> FormalWebotsRun:
    if type(port) is not int or not 1 <= port <= 65_535:
        raise ValueError(f"Port allocator returned invalid Webots port: {port!r}.")
    command = (run.command[0], f"--port={port}", *run.command[2:])
    environment = {**run.environment, "WEBOTS_PORT": str(port)}
    return replace(run, port=port, command=command, environment=environment)


def _verify_lock_and_config(
    final_lock: FinalLock,
    config: WebotsFormalConfig,
    repository_root: Path,
) -> None:
    config_path = _resolve_inside_repository(
        repository_root, final_lock.evaluation_config_path, "locked formal config"
    )
    if config.source_path != config_path:
        raise ValueError("Loaded Webots formal config differs from the final lock path.")
    if config.source_sha256 != final_lock.evaluation_config_sha256:
        raise ValueError("Loaded Webots formal config differs from the final lock digest.")
    final_lock.split_spec.require_training_seeds(config.ppo_training_seeds)
    final_lock.split_spec.require_evaluation_seeds("test", config.scenario_seeds, confirm_test=True)
    if config.required_runs != 20 or config.required_episode_count != 400:
        raise ValueError("Formal Webots config must require exactly 20 runs and 400 episodes.")


def build_formal_run_plan(
    *,
    repository_root: Path,
    final_lock: FinalLock,
    config: WebotsFormalConfig,
    webots_executable: Path,
    port_base: int = 13_490,
    port_allocator: PortAllocator = _allocate_port,
) -> tuple[FormalWebotsRun, ...]:
    """Resolve all twenty requests without touching the ledger or simulator."""

    root = repository_root.resolve()
    _verify_lock_and_config(final_lock, config, root)
    world = _resolve_inside_repository(root, config.world, "formal Webots world")
    if not world.is_file():
        raise FileNotFoundError(f"Formal Webots world does not exist: {world}")
    output_root = _resolve_inside_repository(root, final_lock.output_directory, "output")
    model_by_seed = {item.training_seed: item for item in final_lock.model_manifests}
    planner_requests = (("reactive", None), ("dwa", None)) + tuple(
        ("ppo", seed) for seed in config.ppo_training_seeds
    )
    reserved_ports: set[int] = set()
    plans: list[FormalWebotsRun] = []
    for condition in config.conditions:
        for planner, training_seed in planner_requests:
            request = resolve_formal_test_request(
                config,
                condition_id=condition.condition_id,
                planner=planner,
                training_seed=training_seed,
                seed_start=config.scenario_seed_start,
                episode_count=config.scenario_seed_count,
                confirmed=True,
            )
            model = model_by_seed.get(training_seed) if training_seed is not None else None
            if training_seed is not None and model is None:
                raise ValueError(f"Final lock has no model manifest for PPO seed {training_seed}.")
            port = port_allocator(port_base, reserved_ports)
            if type(port) is not int or not 1 <= port <= 65_535:
                raise ValueError(f"Port allocator returned invalid Webots port: {port!r}.")
            run_name = _safe_run_name(
                final_lock.campaign_id, condition.condition_id, request.planner_label
            )
            staging = root / "artifacts" / "results" / run_name
            result = output_root / "runs" / condition.condition_id / request.planner_label
            environment = {
                "WEBOTS_FORMAL_CAMPAIGN_ID": final_lock.campaign_id,
                "WEBOTS_FINAL_LOCK_SHA256": final_lock.lock_sha256,
                "WEBOTS_FORMAL_CONFIG": final_lock.evaluation_config_path,
                "WEBOTS_EVAL_CONFIG_SHA256": config.source_sha256,
                "WEBOTS_EVAL_SPLIT": "test",
                "WEBOTS_CONFIRM_TEST": "1",
                "WEBOTS_EVAL_SEED_START": str(config.scenario_seed_start),
                "WEBOTS_EVAL_EPISODES": str(config.scenario_seed_count),
                "WEBOTS_EVAL_CONDITION": condition.condition_id,
                "WEBOTS_EVAL_PLANNER": request.planner,
                "WEBOTS_EVAL_PLANNER_LABEL": request.planner_label,
                "WEBOTS_EVAL_RUN_NAME": run_name,
                "WEBOTS_STATIC_LAYOUT": condition.static_layout,
                "WEBOTS_OBSTACLE_COUNT": str(condition.obstacle_count),
                "WEBOTS_OBSTACLE_SPEED_MIN": str(condition.obstacle_speed_range[0]),
                "WEBOTS_OBSTACLE_SPEED_MAX": str(condition.obstacle_speed_range[1]),
                "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH": str(condition.horizontal_route_half_length),
                "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH": str(condition.vertical_route_half_length),
                "WEBOTS_ACTION_REPEAT": str(config.control.action_repeat),
                "WEBOTS_MAX_EPISODE_STEPS": str(config.control.max_episode_steps),
                "WEBOTS_MAX_LINEAR_SPEED": str(config.control.max_linear_speed),
                "WEBOTS_MAX_ANGULAR_SPEED": str(config.control.max_angular_speed),
                "WEBOTS_MAX_LINEAR_ACCELERATION": str(config.control.max_linear_acceleration),
                "WEBOTS_MAX_ANGULAR_ACCELERATION": str(config.control.max_angular_acceleration),
            }
            if model is not None:
                environment.update(
                    {
                        "WEBOTS_MODEL_SEED": str(training_seed),
                        "WEBOTS_EVAL_MODEL": model.model_path,
                        "WEBOTS_EVAL_MODEL_SHA256": model.model_sha256,
                    }
                )
            command = (
                str(webots_executable),
                f"--port={port}",
                "--batch",
                "--mode=fast",
                "--no-rendering",
                "--stdout",
                "--stderr",
                str(world),
            )
            plans.append(
                FormalWebotsRun(
                    condition_id=condition.condition_id,
                    planner=request.planner,
                    planner_label=request.planner_label,
                    training_seed=training_seed,
                    model_path=model.model_path if model is not None else None,
                    model_sha256=model.model_sha256 if model is not None else None,
                    port=port,
                    command=command,
                    environment=environment,
                    staging_directory=staging,
                    result_directory=result,
                )
            )
    if len(plans) != config.required_runs:
        raise ValueError(f"Formal run plan must contain {config.required_runs} requests.")
    keys = {(item.condition_id, item.planner_label) for item in plans}
    if len(keys) != config.required_runs:
        raise ValueError("Formal run plan contains duplicate planner-condition requests.")
    if len({item.port for item in plans}) != config.required_runs:
        raise ValueError("Formal run plan contains duplicate Webots ports.")
    return tuple(plans)


def _augment_summary(
    run: FormalWebotsRun,
    *,
    repository_root: Path,
    final_lock: FinalLock,
    config: WebotsFormalConfig,
) -> tuple[Path, Path]:
    episodes_path = run.staging_directory / "episodes.csv"
    summary_path = run.staging_directory / "summary.json"
    if not episodes_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Webots run {run.condition_id}/{run.planner_label} produced no result pair."
        )
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise ValueError("Webots summary.json must contain an object.")
    expected = {
        "split": "test",
        "episode_count": config.scenario_seed_count,
        "planners": [run.planner_label],
        "conditions": [run.condition_id],
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(
                f"Webots summary {name} for {run.condition_id}/{run.planner_label} "
                "does not match the formal request."
            )
    episodes_sha256 = file_sha256(episodes_path)
    existing = summary.get("evaluation_provenance")
    if not isinstance(existing, dict):
        raise ValueError("Webots summary evaluation_provenance must be an object.")
    world_path = _resolve_inside_repository(repository_root, config.world, "formal Webots world")
    expected_provenance: dict[str, Any] = {
        "schema_version": 1,
        "split": "test",
        "condition_id": run.condition_id,
        "planner": run.planner,
        "planner_label": run.planner_label,
        "training_seed": run.training_seed,
        "model_path": run.model_path,
        "model_sha256": run.model_sha256,
        "scenario_seed_start": config.scenario_seed_start,
        "scenario_seed_count": config.scenario_seed_count,
        "scenario_seeds": list(config.scenario_seeds),
        "obstacle_count": config.condition(run.condition_id).obstacle_count,
        "obstacle_speed_range": list(config.condition(run.condition_id).obstacle_speed_range),
        "horizontal_route_half_length": config.condition(
            run.condition_id
        ).horizontal_route_half_length,
        "vertical_route_half_length": config.condition(run.condition_id).vertical_route_half_length,
        "static_layout": config.condition(run.condition_id).static_layout,
        "safety_shield_enabled": False,
        "experiment_id": config.experiment_id,
        "webots_version": config.webots_version,
        "formal_config_path": final_lock.evaluation_config_path,
        "formal_config_sha256": config.source_sha256,
        "world_path": config.world,
        "world_sha256": file_sha256(world_path),
        "condition_semantic_sha256": config.condition(run.condition_id).semantic_sha256,
        "episodes_csv_sha256": episodes_sha256,
    }
    for name, value in expected_provenance.items():
        if name not in existing:
            raise ValueError(f"Webots summary provenance is missing {name}.")
        if existing[name] != value:
            raise ValueError(f"Webots summary provenance conflicts on {name}.")

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "split": "test",
        "condition_id": run.condition_id,
        "scenario_seed_start": config.scenario_seed_start,
        "scenario_seed_count": config.scenario_seed_count,
        "scenario_seeds": list(config.scenario_seeds),
        "formal_campaign_id": final_lock.campaign_id,
        "final_lock_sha256": final_lock.lock_sha256,
        "test_consumption_key": final_lock.test_consumption_key,
        "source_revision": final_lock.source_revision,
        "source_bundle_sha256": final_lock.source_bundle_sha256,
        "evaluator_sha256": final_lock.evaluator_sha256,
        "controller_sha256": final_lock.controller_sha256,
        "evaluation_config_sha256": config.source_sha256,
    }
    if run.training_seed is not None:
        provenance.update(
            {
                "training_seed": run.training_seed,
                "model_path": run.model_path,
                "model_sha256": run.model_sha256,
                "model_variant": None,
            }
        )
    for name, value in provenance.items():
        if name in existing and existing[name] != value:
            raise ValueError(f"Webots summary provenance conflicts on {name}.")
    summary["evaluation_provenance"] = {**existing, **provenance}
    summary["episodes_sha256"] = episodes_sha256
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return episodes_path, summary_path


def _move_result_pair(run: FormalWebotsRun) -> FormalResultInput:
    if run.result_directory.exists():
        raise FileExistsError(f"Formal result directory already exists: {run.result_directory}")
    run.result_directory.parent.mkdir(parents=True, exist_ok=True)
    run.staging_directory.rename(run.result_directory)
    return FormalResultInput(
        episodes_path=run.result_directory / "episodes.csv",
        summary_path=run.result_directory / "summary.json",
    )


def _subprocess_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_one(
    run: FormalWebotsRun,
    *,
    repository_root: Path,
    final_lock: FinalLock,
    config: WebotsFormalConfig,
    output_root: Path,
    process_runner: ProcessRunner,
    timeout_seconds: float,
    executable_evidence: WebotsExecutableEvidence,
    runtime_verifier: RuntimeVerifier,
    port_allocator: PortAllocator,
    reserved_ports: set[int],
    port_verifier: PortVerifier,
) -> tuple[FormalResultInput, FormalWebotsRun]:
    if run.staging_directory.exists():
        raise FileExistsError(f"Formal staging directory already exists: {run.staging_directory}")
    if run.result_directory.exists():
        raise FileExistsError(f"Formal result directory already exists: {run.result_directory}")
    runtime_verifier(f"before {run.condition_id}/{run.planner_label}")
    run = _with_runtime_port(run, port_allocator(run.port, reserved_ports))
    logs = output_root / "logs" / run.condition_id / run.planner_label
    logs.mkdir(parents=True, exist_ok=False)
    invocation = {
        "schema_version": 1,
        "condition_id": run.condition_id,
        "planner": run.planner,
        "planner_label": run.planner_label,
        "training_seed": run.training_seed,
        "port": run.port,
        "command": list(run.command),
        "formal_environment": dict(sorted(run.environment.items())),
        "staging_directory": _portable(run.staging_directory, repository_root),
        "result_directory": _portable(run.result_directory, repository_root),
        "timeout_seconds": timeout_seconds,
        "webots_executable": executable_evidence.to_mapping(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (logs / "invocation.json").write_text(
        json.dumps(invocation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed: subprocess.CompletedProcess[str] | None = None
    process_error: BaseException | None = None
    process_started = False
    try:
        port_verifier(run.port)
        process_started = True
        completed = process_runner(
            list(run.command),
            cwd=repository_root,
            env=_clean_environment(run.environment),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except BaseException as error:
        process_error = error
        (logs / "stdout.log").write_text(
            _subprocess_output_text(getattr(error, "stdout", None)),
            encoding="utf-8",
        )
        (logs / "stderr.log").write_text(
            _subprocess_output_text(getattr(error, "stderr", None)),
            encoding="utf-8",
        )
        (logs / "invocation_error.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if completed is not None:
        (logs / "stdout.log").write_text(
            _subprocess_output_text(completed.stdout), encoding="utf-8"
        )
        (logs / "stderr.log").write_text(
            _subprocess_output_text(completed.stderr), encoding="utf-8"
        )
        (logs / "exit_status.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "returncode": completed.returncode,
                    "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if process_started:
        try:
            runtime_verifier(f"after {run.condition_id}/{run.planner_label}")
        except BaseException as integrity_error:
            (logs / "post_process_integrity_error.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error_type": type(integrity_error).__name__,
                        "message": str(integrity_error),
                        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if process_error is not None:
                raise integrity_error from process_error
            raise
    if process_error is not None:
        raise process_error
    if completed is None:  # pragma: no cover - defensive guard for injected runners
        raise RuntimeError("Webots process runner returned no completion record.")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Webots failed for {run.condition_id}/{run.planner_label} "
            f"with exit code {completed.returncode}."
        )
    _augment_summary(
        run,
        repository_root=repository_root,
        final_lock=final_lock,
        config=config,
    )
    return _move_result_pair(run), run


def _write_result_manifest(
    *,
    repository_root: Path,
    final_lock: FinalLock,
    config: WebotsFormalConfig,
    report_path: Path,
    inputs: Sequence[FormalResultInput],
    executable_evidence: WebotsExecutableEvidence,
    executable_evidence_path: Path,
) -> Path:
    result_path = report_path.parent / "result_manifest.json"
    artifacts = [
        {
            "episodes_path": _portable(item.episodes_path, repository_root),
            "episodes_sha256": file_sha256(item.episodes_path),
            "summary_path": _portable(item.summary_path, repository_root),
            "summary_sha256": file_sha256(item.summary_path),
        }
        for item in inputs
    ]
    payload = {
        "schema_version": 1,
        "campaign_id": final_lock.campaign_id,
        "final_lock_sha256": final_lock.lock_sha256,
        "test_consumption_key": final_lock.test_consumption_key,
        "source_revision": final_lock.source_revision,
        "world_path": final_lock.world_path,
        "world_sha256": final_lock.world_sha256,
        "evaluator_sha256": final_lock.evaluator_sha256,
        "controller_sha256": final_lock.controller_sha256,
        "source_bundle_sha256": final_lock.source_bundle_sha256,
        "status": "completed",
        "split": "test",
        "training_seeds": list(final_lock.split_spec.training_seeds),
        "scenario_seeds": list(final_lock.split_spec.test_scenario_seeds),
        "models": [
            {
                "training_seed": item.training_seed,
                "model_path": item.model_path,
                "model_sha256": item.model_sha256,
            }
            for item in final_lock.model_manifests
        ],
        "evaluation_config_path": final_lock.evaluation_config_path,
        "evaluation_config_sha256": config.source_sha256,
        "formal_report_path": _portable(report_path, repository_root),
        "formal_report_sha256": file_sha256(report_path),
        "webots_executable": executable_evidence.to_mapping(),
        "webots_executable_evidence_path": _portable(executable_evidence_path, repository_root),
        "webots_executable_evidence_sha256": file_sha256(executable_evidence_path),
        "run_count": len(inputs),
        "episode_count": config.required_episode_count,
        "run_artifacts": artifacts,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result_path


def execute_formal_test(
    *,
    repository_root: Path,
    final_lock_path: Path,
    webots_executable: Path,
    confirm_test: bool,
    dry_run: bool = False,
    port_base: int = 13_490,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    process_runner: ProcessRunner = subprocess.run,
    executable_probe_runner: ProcessRunner = subprocess.run,
    windows_metadata_probe: WindowsMetadataProbe = _probe_windows_executable_metadata,
    port_allocator: PortAllocator = _allocate_port,
    port_verifier: PortVerifier = _require_port_available,
    lock_loader: LockLoader = load_final_lock,
    source_verifier: SourceVerifier = require_clean_locked_source,
    begin_attempt: BeginAttempt = begin_test_attempt,
    mark_success: MarkSuccess = mark_test_success,
    invalidate_attempt: InvalidateAttempt = invalidate_test_attempt,
    report_builder: ReportBuilder = build_formal_report,
    report_writer: ReportWriter = write_formal_report,
) -> FormalTestExecution:
    """Execute the locked matrix, invalidating the ledger on every post-start error."""

    if confirm_test is not True:
        raise PermissionError("Formal Webots test requires explicit --confirm-test.")
    validated_timeout = _validate_timeout_seconds(timeout_seconds)
    root = repository_root.resolve()
    lock_path = _resolve_inside_repository(root, final_lock_path, "final lock")
    final_lock = lock_loader(lock_path, repository_root=root, verify_files=True)
    source_revision = source_verifier(root)
    if source_revision != final_lock.source_revision:
        raise ValueError(
            "Current Git HEAD differs from the clean source revision recorded by the final lock."
        )
    config_path = _resolve_inside_repository(
        root, final_lock.evaluation_config_path, "formal config"
    )
    config = load_webots_formal_config(config_path)
    planned_executable = webots_executable.expanduser().resolve()
    runs = build_formal_run_plan(
        repository_root=root,
        final_lock=final_lock,
        config=config,
        webots_executable=planned_executable,
        port_base=port_base,
        port_allocator=_sequential_port,
    )
    ledger_path = _resolve_inside_repository(root, final_lock.test_ledger_path, "test ledger")
    output_root = _resolve_inside_repository(root, final_lock.output_directory, "formal output")
    preexisting = [
        path
        for path in (
            ledger_path,
            output_root,
            *(run.staging_directory for run in runs),
        )
        if path.exists()
    ]
    if preexisting:
        preview = ", ".join(str(path) for path in preexisting[:3])
        raise FileExistsError(
            "Formal test preflight refuses existing ledger, output, or staging paths: " + preview
        )
    executable_evidence = probe_webots_executable(
        planned_executable,
        expected_version=config.webots_version,
        timeout_seconds=validated_timeout,
        process_runner=executable_probe_runner,
        windows_metadata_probe=windows_metadata_probe,
    )

    def verify_runtime_integrity(stage: str) -> None:
        current_lock = lock_loader(lock_path, repository_root=root, verify_files=True)
        if current_lock.lock_sha256 != final_lock.lock_sha256:
            raise RuntimeError(f"Final lock drifted {stage}.")
        current_revision = source_verifier(root)
        if current_revision != final_lock.source_revision:
            raise RuntimeError(f"Source revision or clean-tree state drifted {stage}.")
        _verify_webots_executable(executable_evidence)

    if dry_run:
        verify_runtime_integrity("after dry-run executable probe")
        return FormalTestExecution(
            dry_run=True,
            runs=runs,
            executable_evidence=executable_evidence,
        )

    verify_runtime_integrity("before ledger creation")
    attempt_started = False
    try:
        begin_attempt(
            repository_root=root,
            final_lock_path=lock_path,
            confirm_test=True,
            runtime_evidence=executable_evidence.to_mapping(),
        )
        attempt_started = True
        output_root.mkdir(parents=True, exist_ok=False)
        executable_evidence_path = _write_webots_executable_evidence(
            output_root, executable_evidence
        )
        verify_runtime_integrity("before formal execution")
        inputs: list[FormalResultInput] = []
        executed_runs: list[FormalWebotsRun] = []
        reserved_ports: set[int] = set()
        for run in runs:
            result_input, executed_run = _run_one(
                run,
                repository_root=root,
                final_lock=final_lock,
                config=config,
                output_root=output_root,
                process_runner=process_runner,
                timeout_seconds=validated_timeout,
                executable_evidence=executable_evidence,
                runtime_verifier=verify_runtime_integrity,
                port_allocator=port_allocator,
                reserved_ports=reserved_ports,
                port_verifier=port_verifier,
            )
            inputs.append(result_input)
            executed_runs.append(executed_run)
        report = report_builder(
            inputs,
            repository_root=root,
            final_lock_path=lock_path,
        )
        matrix = report.get("matrix_validation")
        if not isinstance(matrix, Mapping) or matrix.get("episode_count") != (
            config.required_episode_count
        ):
            raise ValueError("Formal report did not validate the exact 400-episode matrix.")
        report_path = output_root / "formal_report.json"
        if report_path.exists():
            raise FileExistsError(f"Formal report already exists: {report_path}")
        report_writer(report, report_path)
        result_path = _write_result_manifest(
            repository_root=root,
            final_lock=final_lock,
            config=config,
            report_path=report_path,
            inputs=inputs,
            executable_evidence=executable_evidence,
            executable_evidence_path=executable_evidence_path,
        )
        verify_runtime_integrity("immediately before success transition")
        ledger = mark_success(
            repository_root=root,
            final_lock_path=lock_path,
            result_manifest_path=result_path,
        )
        return FormalTestExecution(
            dry_run=False,
            runs=tuple(executed_runs),
            report_path=report_path,
            result_manifest_path=result_path,
            ledger=ledger,
            executable_evidence=executable_evidence,
        )
    except BaseException as error:
        if attempt_started:
            reason = f"{type(error).__name__}: {error}".strip()
            try:
                invalidate_attempt(
                    repository_root=root,
                    final_lock_path=lock_path,
                    reason=reason or "Unknown formal test failure.",
                )
            except Exception as invalidation_error:
                raise RuntimeError(
                    "Formal test failed and its ledger could not be invalidated: "
                    f"{invalidation_error}"
                ) from error
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume one locked held-out test attempt and run the exact 400-episode "
            "G0-G3 Webots matrix."
        )
    )
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).absolute().parents[1],
    )
    parser.add_argument("--webots", type=Path)
    parser.add_argument("--port-base", type=int, default=13_490)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="Finite positive timeout for each Webots process (default: 3600 seconds).",
    )
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print all twenty commands without creating a ledger or output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        webots = resolve_webots_executable(args.webots)
        execution = execute_formal_test(
            repository_root=args.repository_root,
            final_lock_path=args.lock,
            webots_executable=webots,
            confirm_test=args.confirm_test,
            dry_run=args.dry_run,
            port_base=args.port_base,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, PermissionError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if execution.dry_run:
        for run in execution.runs:
            print(f"PLAN {run.condition_id}/{run.planner_label}: " + " ".join(run.command))
        print("Dry-run only: ledger and held-out test were not consumed.")
    else:
        print(f"Formal Webots test completed: {execution.result_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
