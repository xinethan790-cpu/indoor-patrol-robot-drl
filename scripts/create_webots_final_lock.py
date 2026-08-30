#!/usr/bin/env python3
"""Create the immutable three-seed lock for the held-out Webots test."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indoor_patrol_rl.evaluation.formal_protocol import (
    FinalLock,
    FormalModelManifest,
    SplitSpec,
    build_formal_source_bundle,
    file_sha256,
    protected_test_consumption_key,
    protected_test_ledger_path,
    source_bundle_sha256,
    write_final_lock,
    write_or_validate_model_provenance,
)
from indoor_patrol_rl.evaluation.webots_formal_config import load_webots_formal_config

GitRunner = Callable[..., subprocess.CompletedProcess[str]]
_GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


@dataclass(frozen=True)
class ModelEvidenceInput:
    """Six CLI values that bind one independent training seed to its evidence."""

    training_seed: int
    model_path: Path
    metadata_path: Path
    curriculum_manifest_path: Path
    selection_path: Path
    validation_episodes_path: Path


def _inside_repository(repository_root: Path, value: Path, label: str) -> tuple[Path, str]:
    root = repository_root.resolve()
    candidate = value if value.is_absolute() else root / value
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository: {value}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved, relative.as_posix()


def _repository_destination(repository_root: Path, value: Path, label: str) -> Path:
    root = repository_root.resolve()
    candidate = value if value.is_absolute() else root / value
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository: {value}") from error
    return resolved


def _portable_destination(repository_root: Path, value: Path, label: str) -> str:
    return _repository_destination(repository_root, value, label).relative_to(
        repository_root.resolve()
    ).as_posix()


def _git_command(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    git_runner: GitRunner,
) -> str:
    completed = git_runner(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail or completed.returncode}")
    return completed.stdout.strip()


def require_clean_head(
    repository_root: Path,
    *,
    git_runner: GitRunner = subprocess.run,
) -> str:
    """Return HEAD only when every tracked and untracked source file is clean."""

    revision = _git_command(
        repository_root,
        ("rev-parse", "--verify", "HEAD"),
        git_runner=git_runner,
    ).casefold()
    if _GIT_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("git rev-parse HEAD did not return a full 40- or 64-character revision.")
    dirty = _git_command(
        repository_root,
        ("status", "--porcelain", "--untracked-files=all"),
        git_runner=git_runner,
    )
    if dirty:
        preview = " | ".join(dirty.splitlines()[:5])
        raise RuntimeError(
            "Formal lock creation requires a clean Git HEAD; commit or remove: " + preview
        )
    return revision


def parse_model_evidence(entries: Sequence[Sequence[Any]]) -> tuple[ModelEvidenceInput, ...]:
    """Parse repeatable six-value evidence entries and enforce the three seeds."""

    parsed: list[ModelEvidenceInput] = []
    for index, entry in enumerate(entries):
        if len(entry) != 6:
            raise ValueError(
                f"model evidence {index + 1} must contain SEED MODEL METADATA CURRICULUM "
                "SELECTION VALIDATION_EPISODES."
            )
        try:
            seed = int(entry[0])
        except (TypeError, ValueError) as error:
            raise ValueError(f"model evidence {index + 1} SEED must be an integer.") from error
        parsed.append(
            ModelEvidenceInput(
                training_seed=seed,
                model_path=Path(entry[1]),
                metadata_path=Path(entry[2]),
                curriculum_manifest_path=Path(entry[3]),
                selection_path=Path(entry[4]),
                validation_episodes_path=Path(entry[5]),
            )
        )
    parsed.sort(key=lambda item: item.training_seed)
    SplitSpec().require_training_seeds([item.training_seed for item in parsed])
    return tuple(parsed)


def create_webots_final_lock(
    *,
    repository_root: Path,
    config_path: Path,
    evidence_entries: Sequence[Sequence[Any]],
    campaign_id: str | None = None,
    output_path: Path | None = None,
    test_output_directory: Path | None = None,
    ledger_path: Path | None = None,
    git_runner: GitRunner = subprocess.run,
) -> tuple[FinalLock, Path]:
    """Build, validate, hash, and exclusively write a formal Webots lock."""

    root = repository_root.resolve()
    config_file, config_portable = _inside_repository(root, config_path, "formal config")
    config = load_webots_formal_config(config_file)
    revision = require_clean_head(root, git_runner=git_runner)
    evidence = parse_model_evidence(evidence_entries)
    manifests: list[FormalModelManifest] = []
    for item in evidence:
        model, model_portable = _inside_repository(root, item.model_path, "model")
        metadata, metadata_portable = _inside_repository(
            root, item.metadata_path, "training metadata"
        )
        curriculum, curriculum_portable = _inside_repository(
            root, item.curriculum_manifest_path, "curriculum manifest"
        )
        selection, selection_portable = _inside_repository(
            root, item.selection_path, "selection manifest"
        )
        episodes, episodes_portable = _inside_repository(
            root, item.validation_episodes_path, "validation episodes"
        )
        provenance, provenance_payload = write_or_validate_model_provenance(
            repository_root=root,
            training_seed=item.training_seed,
            model_path=model,
            metadata_path=metadata,
            curriculum_manifest_path=curriculum,
        )
        provenance_portable = provenance.relative_to(root).as_posix()
        manifests.append(
            FormalModelManifest(
                training_seed=item.training_seed,
                model_path=model_portable,
                model_sha256=file_sha256(model),
                training_metadata_path=metadata_portable,
                training_metadata_sha256=file_sha256(metadata),
                model_provenance_path=provenance_portable,
                model_provenance_sha256=file_sha256(provenance),
                curriculum_manifest_path=curriculum_portable,
                curriculum_manifest_sha256=file_sha256(curriculum),
                selection_manifest_path=selection_portable,
                selection_manifest_sha256=file_sha256(selection),
                validation_episodes_path=episodes_portable,
                validation_episodes_sha256=file_sha256(episodes),
                training_stage="W2",
                model_variant=provenance_payload["model_variant"],
                training_protocol_sha256=provenance_payload[
                    "training_protocol_sha256"
                ],
                algorithm_config_sha256=provenance_payload["algorithm_config_sha256"],
            )
        )

    campaign = campaign_id or config.experiment_id
    output = output_path or Path(f"artifacts/results/formal_locks/{campaign}.json")
    results = test_output_directory or Path(f"artifacts/results/formal_runs/{campaign}")
    split_spec = SplitSpec()
    global_ledger = Path(protected_test_ledger_path(split_spec))
    ledger = ledger_path or global_ledger
    source_bundle = build_formal_source_bundle(root)
    source_by_path = {item.path: item for item in source_bundle}
    world, world_portable = _inside_repository(root, Path(config.world), "formal world")
    evaluator = source_by_path["scripts/run_webots_formal_test.py"]
    controller = source_by_path[
        "controllers/webots_ppo_evaluate/webots_ppo_evaluate.py"
    ]
    final_lock = FinalLock(
        campaign_id=campaign,
        source_revision=revision,
        source_dirty=False,
        evaluation_config_path=config_portable,
        evaluation_config_sha256=config.source_sha256,
        world_path=world_portable,
        world_sha256=file_sha256(world),
        evaluator_path=evaluator.path,
        evaluator_sha256=evaluator.sha256,
        controller_path=controller.path,
        controller_sha256=controller.sha256,
        source_bundle=source_bundle,
        source_bundle_sha256=source_bundle_sha256(source_bundle),
        test_consumption_key=protected_test_consumption_key(split_spec),
        output_directory=_portable_destination(root, results, "test output directory"),
        test_ledger_path=_portable_destination(root, ledger, "test ledger"),
        model_manifests=tuple(manifests),
        split_spec=split_spec,
    )
    destination = _repository_destination(root, output, "final lock output")
    write_final_lock(destination, final_lock, repository_root=root)
    return final_lock, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-overwritable formal lock after validating three independent "
            "Webots models and a clean Git HEAD."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).absolute().parents[1],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/webots_formal.yaml"),
    )
    parser.add_argument(
        "--model-evidence",
        action="append",
        nargs=6,
        required=True,
        metavar=(
            "SEED",
            "MODEL",
            "METADATA",
            "CURRICULUM",
            "SELECTION",
            "VALIDATION_EPISODES",
        ),
        help=(
            "Repeat once for each independently trained seed 11, 22, and 33. "
            "MODEL must already have its training-time .formal-provenance.json sidecar."
        ),
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-output-directory", type=Path)
    parser.add_argument(
        "--ledger",
        type=Path,
        help=(
            "Compatibility option; only the canonical protected-test global ledger "
            "path is accepted."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        final_lock, destination = create_webots_final_lock(
            repository_root=args.repository_root,
            config_path=args.config,
            evidence_entries=args.model_evidence,
            campaign_id=args.campaign_id,
            output_path=args.output,
            test_output_directory=args.test_output_directory,
            ledger_path=args.ledger,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Created formal lock: {destination}")
    print(f"Final lock SHA-256: {final_lock.lock_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
