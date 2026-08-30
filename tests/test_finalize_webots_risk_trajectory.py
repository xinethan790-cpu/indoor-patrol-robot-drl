from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/finalize_webots_risk_trajectory.py"


def _load_finalizer() -> Any:
    name = "_test_finalize_webots_risk_trajectory"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINALIZER = _load_finalizer()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_blob_sha256(root: Path, revision: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


class _FakeNormalRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.parse_calls = 0
        self.write_calls = 0
        self.tamper_standard_sidecar = False
        model = SimpleNamespace(
            portable_path=FINALIZER._MODEL,
            path=root / FINALIZER._MODEL,
            sha256=FINALIZER._EXPECTED_MODEL_SHA256,
            variant="w1_recovery_v1",
            run_name=Path(FINALIZER._RESULT_DIRECTORY).name,
            planner_label="ppo_recovery_v1_risk_trace",
        )
        scenario = SimpleNamespace(
            obstacle_count=1,
            obstacle_speed_range=(0.15, 0.30),
            horizontal_route_half_length=1.10,
            vertical_route_half_length=1.10,
            safety_shield=False,
            deterministic_policy=True,
        )
        self.config = SimpleNamespace(
            source_sha256=FINALIZER._EXPECTED_SOURCE_SHA256[FINALIZER._CONFIG],
            experiment_id="webots_recovery_v1_risk_trajectory_v1",
            condition_id="DR1",
            training_seed=11,
            scenario_seed_start=20_494,
            episodes=120,
            context_schema_version="1.1",
            seeds=tuple(range(20_494, 20_614)),
            model=model,
            scenario=scenario,
        )

    @staticmethod
    def RiskTrajectoryRun(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def load_risk_trajectory_config(
        self, _path: Path, *, repository_root: Path
    ) -> SimpleNamespace:
        assert repository_root == self.root
        return self.config

    def _parse_result(
        self,
        run: SimpleNamespace,
        *,
        config: SimpleNamespace,
        repository_root: Path,
    ) -> SimpleNamespace:
        assert config is self.config
        assert repository_root == self.root
        assert run.result_directory == self.root / FINALIZER._RESULT_DIRECTORY
        self.parse_calls += 1
        result = self.root / FINALIZER._RESULT_DIRECTORY
        artifacts = {
            "result_directory": FINALIZER._RESULT_DIRECTORY,
            "summary_json": {
                "path": f"{FINALIZER._RESULT_DIRECTORY}/summary.json",
                "sha256": _sha256(result / "summary.json"),
            },
            "episodes_csv": {
                "path": f"{FINALIZER._RESULT_DIRECTORY}/episodes.csv",
                "sha256": _sha256(result / "episodes.csv"),
            },
            "episode_context_jsonl": {
                "path": f"{FINALIZER._RESULT_DIRECTORY}/episode_context.jsonl",
                "sha256": _sha256(result / "episode_context.jsonl"),
            },
        }
        seeds = tuple(range(20_494, 20_614))
        contexts = {
            seed: {"route_axis": "horizontal" if index < 60 else "vertical"}
            for index, seed in enumerate(seeds)
        }
        return SimpleNamespace(
            rows_by_seed={seed: {"outcome": "collision"} for seed in seeds},
            contexts_by_seed=contexts,
            metrics={
                "overall": {
                    "episodes": 120,
                    "success_count": 0,
                    "collision_count": 120,
                    "timeout_count": 0,
                    "error_count": 0,
                },
                "by_route_axis": {
                    "horizontal": {"episodes": 60},
                    "vertical": {"episodes": 60},
                },
            },
            artifacts=artifacts,
        )

    def _write_report(
        self, report: dict[str, Any], *, output_directory: Path
    ) -> None:
        self.write_calls += 1
        output_directory.mkdir(parents=True, exist_ok=False)
        json_path = output_directory / "collection_report.json"
        yaml_path = output_directory / "collection_report.yaml"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        yaml_path.write_text(
            yaml.safe_dump(report, sort_keys=False),
            encoding="utf-8",
        )
        (output_directory / "collection_report.sha256").write_text(
            f"{_sha256(json_path)}  collection_report.json\n"
            f"{_sha256(yaml_path)}  collection_report.yaml\n",
            encoding="utf-8",
        )
        if self.tamper_standard_sidecar:
            (output_directory / "collection_report.sha256").write_text(
                "0" * 64 + "  collection_report.json\n",
                encoding="utf-8",
            )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, _FakeNormalRunner]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")

    source_payloads = {
        ".gitignore": b"/artifacts/\n/tools/\n",
        FINALIZER._CONFIG: b"locked config\n",
        FINALIZER._NORMAL_RUNNER: b"locked normal runner\n",
        FINALIZER._CONTROLLER: b"locked controller\n",
        FINALIZER._WORLD: b"locked world\n",
        "scripts/analyze_webots_risk_trajectory.py": b"base analyzer\n",
        "tests/test_analyze_webots_risk_trajectory.py": b"base analyzer test\n",
    }
    for portable, payload in source_payloads.items():
        path = root / portable
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "original collection revision")
    original_revision = _git(root, "rev-parse", "HEAD")

    analyzer = root / "scripts/analyze_webots_risk_trajectory.py"
    analyzer.write_text(
        analyzer.read_text(encoding="utf-8")
        + "".join(f"offline analyzer line {index}\n" for index in range(15)),
        encoding="utf-8",
    )
    analyzer_test = root / "tests/test_analyze_webots_risk_trajectory.py"
    analyzer_test.write_text(
        analyzer_test.read_text(encoding="utf-8")
        + "".join(f"offline test line {index}\n" for index in range(5)),
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "offline analyzer provenance")
    intermediate_revision = _git(root, "rev-parse", "HEAD")
    intermediate_sha256 = {
        path: _git_blob_sha256(root, intermediate_revision, path)
        for path in (
            "scripts/analyze_webots_risk_trajectory.py",
            "tests/test_analyze_webots_risk_trajectory.py",
        )
    }

    finalizer_path = root / "scripts/finalize_webots_risk_trajectory.py"
    finalizer_path.write_bytes(b"reviewed finalizer\n")
    finalizer_test = root / "tests/test_finalize_webots_risk_trajectory.py"
    finalizer_test.write_bytes(b"reviewed finalizer test\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "one-time finalizer")

    model = root / FINALIZER._MODEL
    webots = root / "tools/webots.exe"
    result = root / FINALIZER._RESULT_DIRECTORY
    for path, payload in (
        (model, b"model"),
        (webots, b"webots"),
        (result / "summary.json", b"summary"),
        (result / "episodes.csv", b"episodes"),
        (result / "episode_context.jsonl", b"contexts"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(FINALIZER, "_ORIGINAL_REVISION", original_revision)
    monkeypatch.setattr(FINALIZER, "_INTERMEDIATE_REVISION", intermediate_revision)
    monkeypatch.setattr(
        FINALIZER,
        "_INTERMEDIATE_FILE_SHA256",
        intermediate_sha256,
    )
    monkeypatch.setattr(
        FINALIZER,
        "_EXPECTED_SOURCE_SHA256",
        {
            portable: hashlib.sha256(payload).hexdigest()
            for portable, payload in source_payloads.items()
            if portable
            in {
                FINALIZER._CONFIG,
                FINALIZER._NORMAL_RUNNER,
                FINALIZER._CONTROLLER,
                FINALIZER._WORLD,
            }
        },
    )
    monkeypatch.setattr(FINALIZER, "_EXPECTED_MODEL_SHA256", _sha256(model))
    monkeypatch.setattr(FINALIZER, "_EXPECTED_WEBOTS_SHA256", _sha256(webots))
    monkeypatch.setattr(
        FINALIZER,
        "_EXPECTED_RESULT_SHA256",
        {name: _sha256(result / name) for name in FINALIZER._RESULT_FILENAMES},
    )
    normal = _FakeNormalRunner(root)
    return root, webots, normal


def test_finalizer_seals_exact_standard_report_without_starting_webots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    before_results = {
        name: _sha256(root / FINALIZER._RESULT_DIRECTORY / name)
        for name in FINALIZER._RESULT_FILENAMES
    }
    before_webots = _sha256(webots)

    result = FINALIZER.finalize_collection(
        repository_root=root,
        webots_executable=webots,
        normal_runner_loader=lambda _root: normal,
    )

    assert normal.parse_calls == 1
    assert normal.write_calls == 1
    assert _sha256(webots) == before_webots
    assert {
        name: _sha256(root / FINALIZER._RESULT_DIRECTORY / name)
        for name in FINALIZER._RESULT_FILENAMES
    } == before_results
    report_directory = root / FINALIZER._REPORT_DIRECTORY
    assert set(path.name for path in report_directory.iterdir()) == {
        "collection_report.json",
        "collection_report.yaml",
        "collection_report.sha256",
        "finalize_audit.json",
        "finalize_audit.sha256",
    }
    report = json.loads(
        (report_directory / "collection_report.json").read_text(encoding="utf-8")
    )
    assert set(report) == {
        "schema_version",
        "experiment_id",
        "status",
        "generated_at_utc",
        "split",
        "validation_split_was_run",
        "test_split_was_run",
        "performance_gate_applied",
        "source_bindings",
        "scenario",
        "collection",
        "model",
        "execution",
        "descriptive_metrics_not_used_as_a_gate",
        "artifacts",
        "access",
        "written_reports",
    }
    assert report["status"] == "collection_complete"
    assert report["source_bindings"]["git_revision"] == FINALIZER._ORIGINAL_REVISION
    assert report["execution"]["exit_code"] == 0
    assert report["scenario"]["axis_counts"] == {"horizontal": 60, "vertical": 60}
    assert report["validation_split_was_run"] is False
    assert report["test_split_was_run"] is False

    audit_path = report_directory / "finalize_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "sealed_without_webots_rerun"
    assert audit["no_webots_rerun"] is True
    assert audit["nonignored_untracked_files"] == []
    assert audit["original_guard_failure"]["reason"] == (
        "Git HEAD or tracked worktree changed during collection."
    )
    assert audit["original_guard_failure"]["runtime_change_total_added_lines"] == 20
    assert [item["path"] for item in audit["original_guard_failure"]["runtime_changes"]] == [
        "scripts/analyze_webots_risk_trajectory.py",
        "tests/test_analyze_webots_risk_trajectory.py",
    ]
    assert audit["reviewed_commit_chain"]["original"]["revision"] == (
        FINALIZER._ORIGINAL_REVISION
    )
    assert audit["reviewed_commit_chain"]["intermediate"] == {
        "revision": FINALIZER._INTERMEDIATE_REVISION,
        "parent": FINALIZER._ORIGINAL_REVISION,
        "changed_file_sha256": dict(
            sorted(FINALIZER._INTERMEDIATE_FILE_SHA256.items())
        ),
    }
    assert audit["reviewed_commit_chain"]["finalizer"]["parent"] == (
        FINALIZER._INTERMEDIATE_REVISION
    )
    assert audit["result_artifacts"] == {
        name: {
            "path": f"{FINALIZER._RESULT_DIRECTORY}/{name}",
            "sha256": before_results[name],
        }
        for name in sorted(FINALIZER._RESULT_FILENAMES)
    }
    assert result["standard_report_sha256"] == _sha256(
        report_directory / "collection_report.json"
    )
    assert (report_directory / "finalize_audit.sha256").read_text(
        encoding="utf-8"
    ) == f"{_sha256(audit_path)}  finalize_audit.json\n"


def test_finalizer_rejects_any_extra_original_to_head_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    controller = root / FINALIZER._CONTROLLER
    controller.write_text("changed controller\n", encoding="utf-8")
    _git(root, "add", FINALIZER._CONTROLLER)
    _git(root, "commit", "-m", "forbidden runtime change")

    with pytest.raises(PermissionError, match="directly follow the fixed intermediate"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert not (root / FINALIZER._REPORT_DIRECTORY).exists()


def test_finalizer_rejects_dirty_tracked_tree_and_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    analyzer = root / "scripts/analyze_webots_risk_trajectory.py"
    analyzer.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )

    _git(root, "restore", "scripts/analyze_webots_risk_trajectory.py")
    report = root / FINALIZER._REPORT_DIRECTORY
    report.mkdir(parents=True)
    marker = report / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_finalizer_rejects_result_artifact_sha_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    (root / FINALIZER._RESULT_DIRECTORY / "episodes.csv").write_bytes(b"changed")

    with pytest.raises(PermissionError, match="episodes.csv changed"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert not (root / FINALIZER._REPORT_DIRECTORY).exists()


def test_finalizer_rejects_nonignored_untracked_shadow_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    shadow = root / "scripts/shadow_normal_runner.py"
    shadow.write_text("raise RuntimeError('shadow')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-ignored untracked files"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert not (root / FINALIZER._REPORT_DIRECTORY).exists()


def test_finalizer_rejects_intermediate_blob_sha_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    changed = dict(FINALIZER._INTERMEDIATE_FILE_SHA256)
    changed["scripts/analyze_webots_risk_trajectory.py"] = "0" * 64
    monkeypatch.setattr(FINALIZER, "_INTERMEDIATE_FILE_SHA256", changed)

    with pytest.raises(PermissionError, match="Intermediate review blob SHA-256"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert not (root / FINALIZER._REPORT_DIRECTORY).exists()


def test_finalizer_rereads_and_rejects_tampered_standard_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    normal.tamper_standard_sidecar = True

    with pytest.raises(RuntimeError, match="collection_report.sha256"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert not (root / FINALIZER._REPORT_DIRECTORY).exists()


def test_finalizer_rereads_and_rejects_tampered_audit_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, webots, normal = _fixture(tmp_path, monkeypatch)
    original_writer = FINALIZER._write_new_json

    def tamper(path: Path, value: dict[str, Any]) -> None:
        original_writer(path, value)
        path.write_text('{"tampered":true}\n', encoding="utf-8")

    monkeypatch.setattr(FINALIZER, "_write_new_json", tamper)
    with pytest.raises(RuntimeError, match="finalize_audit.json differs"):
        FINALIZER.finalize_collection(
            repository_root=root,
            webots_executable=webots,
            normal_runner_loader=lambda _root: normal,
        )
    assert not (root / FINALIZER._REPORT_DIRECTORY).exists()
