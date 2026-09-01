"""Secure, read-only access to one explicit patrol result directory."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from indoor_patrol_rl.application.live_status import (
    PatrolLiveSnapshot,
    read_patrol_live_events,
    read_patrol_live_status,
)
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

    @property
    def terminal(self) -> bool:
        return self.current.state.value in _TERMINAL_STATES


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
            raise DashboardEvidenceError(
                f"巡检结果目录不存在或不可访问：{requested}"
            ) from error
        if not resolved.is_dir():
            raise DashboardEvidenceError(f"巡检结果路径不是目录：{resolved}")
        self.run_directory = resolved

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
        snapshots = read_patrol_live_events(events_path)
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

        summary = self._optional_json("summary.json", expected_run_id=run_id)
        inspection_summary = self._optional_json(
            "inspection_summary.json", expected_run_id=run_id
        )
        terminal = current.state.value in _TERMINAL_STATES
        if terminal:
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
            if not isinstance(patrol_binding, Mapping) or patrol_binding.get(
                "path"
            ) != "summary.json":
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
            if not isinstance(raw, Mapping) or not all(
                isinstance(key, str) for key in raw
            ):
                raise DashboardEvidenceError(f"{filename} 第 {line_number} 行必须是对象。")
            records.append(raw)
        return tuple(records)

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
