"""Secure, read-only access to one explicit patrol result directory."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indoor_patrol_rl.application.live_status import (
    PatrolLiveSnapshot,
    _validate_successor,
    read_patrol_live_status,
)
from indoor_patrol_rl.application.runtime_abort import (
    RUNTIME_ABORT_EVIDENCE_SCOPE,
    RUNTIME_ABORT_FILENAME,
    PatrolRuntimeAbort,
)
from indoor_patrol_rl.application.task import PatrolTaskState
from indoor_patrol_rl.dashboard.alarm_history import (
    active_rule_ids_from_history,
    snapshot_alarm_history,
    verified_alarm_history,
)
from indoor_patrol_rl.inspection import verify_inspection_bundle

_TERMINAL_STATES = {"completed", "failed"}
_TERMINAL_BUNDLE_FILES = (
    "inspection_manifest.json",
    "inspection_manifest.sha256",
    "inspection_config.yaml",
    "inspection_samples.csv",
    "alarm_events.jsonl",
    "inspection_visits.csv",
    "inspection_summary.json",
)


class DashboardEvidenceError(ValueError):
    """The selected run directory is incomplete, inconsistent, or unsafe."""


@dataclass(frozen=True)
class DashboardEvidence:
    """One coherent read of live and optional terminal patrol evidence."""

    run_directory: Path
    current: PatrolLiveSnapshot
    snapshots: tuple[PatrolLiveSnapshot, ...]
    summary: Mapping[str, Any] | None
    inspection_summary: Mapping[str, Any] | None
    alarm_history: tuple[dict[str, object], ...]
    alarm_history_source: str
    terminal_validation: str | None
    runtime_abort: PatrolRuntimeAbort | None

    @property
    def terminal(self) -> bool:
        return self.current.state.value in _TERMINAL_STATES


@dataclass(frozen=True)
class EventStreamDiagnostics:
    """Observable work counters for incremental JSONL regression tests and telemetry."""

    data_read_operations: int
    parse_attempts: int
    cache_rebuilds: int


class _IncrementalLiveEventsReader:
    """Parse only complete records appended since the previous coherent read."""

    def __init__(self) -> None:
        self._identity: tuple[int, int] | None = None
        self._size = 0
        self._offset = 0
        self._pending = b""
        self._snapshots: tuple[PatrolLiveSnapshot, ...] = ()
        self._data_read_operations = 0
        self._parse_attempts = 0
        self._cache_rebuilds = 0

    @property
    def diagnostics(self) -> EventStreamDiagnostics:
        return EventStreamDiagnostics(
            data_read_operations=self._data_read_operations,
            parse_attempts=self._parse_attempts,
            cache_rebuilds=self._cache_rebuilds,
        )

    def read(self, path: Path) -> tuple[PatrolLiveSnapshot, ...]:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise ValueError(f"Cannot read live events from {path}: {error}") from error

        try:
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                identity = (before.st_dev, before.st_ino)
                rebuild = (
                    self._identity != identity
                    or before.st_size < self._offset
                    or before.st_size < self._size
                )
                offset = 0 if rebuild else self._offset
                pending = b"" if rebuild else self._pending
                snapshots = () if rebuild else self._snapshots

                payload = b""
                if before.st_size > offset:
                    stream.seek(offset)
                    payload = stream.read()
                    self._data_read_operations += 1
                after = os.fstat(stream.fileno())
                if (after.st_dev, after.st_ino) != identity or after.st_size < offset + len(
                    payload
                ):
                    raise ValueError(f"Live events changed while reading {path}.")
        except Exception:
            # os.fdopen owns descriptor after successful construction.  If construction
            # itself failed, close the original descriptor before propagating the error.
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

        combined = pending + payload
        complete_end = combined.rfind(b"\n")
        if complete_end < 0:
            complete = b""
            new_pending = combined
        else:
            complete = combined[: complete_end + 1]
            new_pending = combined[complete_end + 1 :]

        parsed = list(snapshots)
        if complete:
            records = complete.split(b"\n")[:-1]
            first_line_number = len(snapshots) + 1
            for index, record in enumerate(records):
                self._parse_attempts += 1
                line_number = first_line_number + index
                if not record.strip():
                    raise ValueError(f"Blank JSONL records are not allowed in {path}.")
                try:
                    raw = json.loads(record.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {error}"
                    ) from error
                snapshot = PatrolLiveSnapshot.from_dict(raw)
                previous = parsed[-1] if parsed else None
                _validate_successor(previous, snapshot)
                parsed.append(snapshot)

        # Commit only after every newly completed record validates.  A bad append is
        # therefore retried and can never be skipped because the byte offset advanced.
        self._identity = identity
        self._size = after.st_size
        self._offset = offset + len(payload)
        self._pending = new_pending
        self._snapshots = tuple(parsed)
        if rebuild:
            self._cache_rebuilds += 1
        return self._snapshots


class PatrolDashboardRepository:
    """Read only fixed evidence filenames beneath an explicit run directory.

    Symlinked evidence is rejected so a dashboard request can never turn into an
    arbitrary filesystem read.  There is deliberately no API for selecting a
    different run after this object is created.
    """

    def __init__(self, run_directory: str | Path) -> None:
        requested = Path(run_directory).expanduser()
        try:
            resolved = requested.resolve(strict=True)
        except OSError as error:
            raise DashboardEvidenceError(f"巡检结果目录不存在或不可访问：{requested}") from error
        if not resolved.is_dir():
            raise DashboardEvidenceError(f"巡检结果路径不是目录：{resolved}")
        self.run_directory = resolved
        self._events_reader = _IncrementalLiveEventsReader()

    @property
    def event_stream_diagnostics(self) -> EventStreamDiagnostics:
        """Return counters without exposing or modifying cached evidence."""

        return self._events_reader.diagnostics

    def read(self) -> DashboardEvidence:
        """Read a coherent snapshot, retrying once across an atomic update."""

        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                return self._read_once()
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        assert last_error is not None
        if isinstance(last_error, DashboardEvidenceError):
            raise last_error
        raise DashboardEvidenceError(f"巡检证据读取失败：{last_error}") from last_error

    def _read_once(self) -> DashboardEvidence:
        status_path = self._required_file("status.json")
        events_path = self._required_file("events.jsonl")
        current_from_status = read_patrol_live_status(status_path)
        snapshots = self._events_reader.read(events_path)
        if not snapshots:
            raise DashboardEvidenceError("events.jsonl 尚无可显示的巡检快照。")
        event_current = snapshots[-1]
        if event_current.run_id != current_from_status.run_id:
            raise DashboardEvidenceError("status.json 与 events.jsonl 的 run_id 不一致。")
        if current_from_status.sequence > event_current.sequence:
            raise DashboardEvidenceError("status.json 超前于 events.jsonl，证据流不完整。")
        if (
            current_from_status.sequence == event_current.sequence
            and current_from_status != event_current
        ):
            raise DashboardEvidenceError("status.json 与同序号事件内容不一致。")
        current = event_current
        run_id = current.run_id

        terminal = current.state.value in _TERMINAL_STATES
        runtime_abort_candidate = self._optional_runtime_abort(expected_run_id=run_id)
        runtime_abort: PatrolRuntimeAbort | None = None
        if terminal and runtime_abort_candidate is not None:
            self._validate_runtime_abort(current, runtime_abort_candidate)
            runtime_abort = runtime_abort_candidate
            summary = None
            inspection_summary = None
            history = snapshot_alarm_history(snapshots)
            history_source = "runtime_abort_snapshot_diff"
            terminal_validation = RUNTIME_ABORT_EVIDENCE_SCOPE
        elif terminal:
            summary = self._optional_json("summary.json", expected_run_id=run_id)
            inspection_summary = self._optional_json(
                "inspection_summary.json", expected_run_id=run_id
            )
            if summary is None:
                raise DashboardEvidenceError("终态巡检缺少 summary.json。")
            if inspection_summary is None:
                raise DashboardEvidenceError("终态巡检缺少 inspection_summary.json。")
            for filename in _TERMINAL_BUNDLE_FILES:
                self._required_file(filename)
            manifest_preview = _read_json_object(
                self._required_file("inspection_manifest.json"),
                "inspection_manifest.json",
            )
            patrol_binding = manifest_preview.get("patrol_summary")
            if (
                not isinstance(patrol_binding, Mapping)
                or patrol_binding.get("path") != "summary.json"
            ):
                raise DashboardEvidenceError(
                    "inspection manifest 必须绑定当前目录内的 summary.json。"
                )
            try:
                manifest = verify_inspection_bundle(self.run_directory)
            except (FileNotFoundError, OSError, TypeError, ValueError) as error:
                raise DashboardEvidenceError(
                    f"终态巡检的 inspection bundle 校验失败：{error}"
                ) from error
            if manifest.get("run_id") != run_id:
                raise DashboardEvidenceError(
                    "inspection manifest 与 status.json 的 run_id 不一致。"
                )
            records = self._required_json_lines("alarm_events.jsonl")
            history = verified_alarm_history(records, expected_run_id=run_id)
            active_from_events = set(active_rule_ids_from_history(history))
            active_from_status = {alert.rule_id for alert in current.active_alerts}
            if active_from_events != active_from_status:
                raise DashboardEvidenceError(
                    "终态 alarm_events.jsonl 与 status.json 的活动告警不一致。"
                )
            self._validate_terminal_consistency(current, summary, inspection_summary, history)
            history_source = "verified_inspection_bundle"
            terminal_validation = "inspection_bundle_and_runtime_consistency"
        else:
            summary = self._optional_json("summary.json", expected_run_id=run_id)
            inspection_summary = self._optional_json(
                "inspection_summary.json", expected_run_id=run_id
            )
            history = snapshot_alarm_history(snapshots)
            history_source = "live_snapshot_diff"
            terminal_validation = None

        return DashboardEvidence(
            run_directory=self.run_directory,
            current=current,
            snapshots=snapshots,
            summary=summary,
            inspection_summary=inspection_summary,
            alarm_history=history,
            alarm_history_source=history_source,
            terminal_validation=terminal_validation,
            runtime_abort=runtime_abort,
        )

    def _path(self, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise DashboardEvidenceError("看板只允许读取固定的直接子文件。")
        candidate = self.run_directory / filename
        if candidate.is_symlink():
            raise DashboardEvidenceError(f"拒绝读取符号链接证据：{filename}")
        if candidate.exists():
            resolved = candidate.resolve(strict=True)
            if resolved.parent != self.run_directory or not resolved.is_file():
                raise DashboardEvidenceError(f"证据路径越出巡检目录：{filename}")
        return candidate

    def _required_file(self, filename: str) -> Path:
        path = self._path(filename)
        if not path.is_file():
            raise DashboardEvidenceError(f"缺少巡检证据文件：{filename}")
        return path

    def _optional_json(
        self,
        filename: str,
        *,
        expected_run_id: str,
    ) -> Mapping[str, Any] | None:
        path = self._path(filename)
        if not path.exists():
            return None
        raw = _read_json_object(path, filename)
        if raw.get("run_id") != expected_run_id:
            raise DashboardEvidenceError(f"{filename} 的 run_id 与 status.json 不一致。")
        return raw

    def _optional_runtime_abort(
        self,
        *,
        expected_run_id: str,
    ) -> PatrolRuntimeAbort | None:
        path = self._path(RUNTIME_ABORT_FILENAME)
        if not path.exists():
            return None
        try:
            evidence = PatrolRuntimeAbort.from_dict(_read_json_object(path, RUNTIME_ABORT_FILENAME))
        except DashboardEvidenceError:
            raise
        except (TypeError, ValueError) as error:
            raise DashboardEvidenceError(
                f"{RUNTIME_ABORT_FILENAME} 巡检证据读取失败：{error}"
            ) from error
        if evidence.run_id != expected_run_id:
            raise DashboardEvidenceError(
                f"{RUNTIME_ABORT_FILENAME} 的 run_id 与 status.json 不一致。"
            )
        return evidence

    def _required_json_lines(self, filename: str) -> tuple[Mapping[str, Any], ...]:
        path = self._required_file(filename)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise DashboardEvidenceError(f"无法读取 {filename}：{error}") from error
        if any(not line.strip() for line in lines):
            raise DashboardEvidenceError(f"{filename} 不允许空记录。")
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            try:
                raw = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise DashboardEvidenceError(
                    f"{filename} 第 {line_number} 行不是有效 JSON：{error}"
                ) from error
            if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
                raise DashboardEvidenceError(f"{filename} 第 {line_number} 行必须是对象。")
            records.append(raw)
        return tuple(records)

    @staticmethod
    def _validate_runtime_abort(
        current: PatrolLiveSnapshot,
        evidence: PatrolRuntimeAbort,
    ) -> None:
        if current.state is not PatrolTaskState.FAILED or current.outcome.value != "error":
            raise DashboardEvidenceError(
                "runtime_abort.json 只允许绑定 outcome=error 的 failed 终态。"
            )
        expected = {
            "run_id": current.run_id,
            "sequence": current.sequence,
            "seed": current.seed,
            "control_step": current.control_step,
            "checkpoints_visited": current.completed_checkpoint_count,
            "checkpoints_total": current.checkpoint_count,
        }
        for name, value in expected.items():
            if getattr(evidence, name) != value:
                raise DashboardEvidenceError(
                    f"runtime_abort.json 的 {name} 与终态 status.json 不一致。"
                )
        if not math.isclose(
            evidence.simulation_time_s,
            current.simulation_time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise DashboardEvidenceError(
                "runtime_abort.json 的 simulation_time_s 与终态 status.json 不一致。"
            )

    @staticmethod
    def _validate_terminal_consistency(
        current: PatrolLiveSnapshot,
        summary: Mapping[str, Any],
        inspection_summary: Mapping[str, Any],
        history: tuple[dict[str, object], ...],
    ) -> None:
        expected_summary_outcome = {
            "success": "completed",
            "collision": "collision",
            "safety_stop": "safety_stop",
            "timeout": "timeout",
            "inspection_failed": "error",
            "error": "error",
        }[current.outcome.value]
        if summary.get("outcome") != expected_summary_outcome:
            raise DashboardEvidenceError("summary.json 与终态 status.json 的结果不一致。")
        if summary.get("scenario_seed") != current.seed:
            raise DashboardEvidenceError("summary.json 与 status.json 的 seed 不一致。")
        if summary.get("checkpoints_total") != current.checkpoint_count:
            raise DashboardEvidenceError("summary.json 的巡检点总数不一致。")
        if summary.get("checkpoints_visited") != current.completed_checkpoint_count:
            raise DashboardEvidenceError("summary.json 的已巡检点数量不一致。")
        if inspection_summary.get("scenario_seed") != current.seed:
            raise DashboardEvidenceError("inspection_summary.json 的 seed 不一致。")
        if inspection_summary.get("event_count") != len(history):
            raise DashboardEvidenceError("inspection_summary.json 的告警事件数不一致。")
        if inspection_summary.get("active_alarm_count") != len(current.active_alerts):
            raise DashboardEvidenceError("inspection_summary.json 的活动告警数不一致。")
        expected_identity = {
            "model_path": current.runtime_identity.model_id,
            "model_sha256": current.runtime_identity.model_sha256,
            "safety_shield_enabled": current.runtime_identity.safety_shield_enabled,
            "safety_shield_id": current.runtime_identity.safety_shield_id,
            "safety_shield_sha256": current.runtime_identity.safety_shield_sha256,
        }
        for key, expected in expected_identity.items():
            if summary.get(key) != expected:
                raise DashboardEvidenceError(
                    f"summary.json 的 {key} 与终态 status.json 运行身份不一致。"
                )
        if summary.get("control_step_count") != current.control_step:
            raise DashboardEvidenceError(
                "summary.json 的 control_step_count 与终态 status.json 不一致。"
            )
        elapsed = summary.get("elapsed_simulation_time_s")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or not math.isclose(
                float(elapsed), current.simulation_time_s, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise DashboardEvidenceError(
                "summary.json 的 elapsed_simulation_time_s 与终态 status.json 不一致。"
            )


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DashboardEvidenceError(f"{label} 不是有效 JSON 对象：{error}") from error
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise DashboardEvidenceError(f"{label} 必须是 JSON 对象。")
    return raw


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")
