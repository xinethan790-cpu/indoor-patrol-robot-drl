#!/usr/bin/env python3
"""Offline, post-hoc audit of the Webots LiDAR ray-origin sensitivity.

This audit is deliberately subordinate to ``relative_motion_cpa_v1``.  It
recomputes only the terminal LiDAR ray/circle residual with the recorded robot
root and GPS positions.  It never launches Webots, selects a checkpoint, or
reads validation/held-out test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import analyze_webots_risk_trajectory as v1
except ModuleNotFoundError:  # pragma: no cover - supports module-style test loading
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_webots_risk_trajectory as v1


AUDIT_ID = "webots_recovery_v1_lidar_origin_sensitivity_v1"
COLLECTION_REPORT_SHA256 = "bb3e08bb20cb7d67b099323b37e3e15413d075be1b2b59fa9c95afb0317fd2cb"
V1_RISK_ANALYSIS_REPORT_SHA256 = "ff560bb2c4d3ef8df3e180c6d117e6ce324b306493fb900af7fdd65ac68c2e50"
V1_METRICS_JSON_SHA256 = "7671e26fed44813f9594c6445e9ff0fb3a3e6c489b8b2a7db151f65c16e3fb73"
V1_METRICS_CSV_SHA256 = "0487ebc927d63b1de3f53035f4e94b3a19cd26cb4ab9f51d3237af92dcdc056b"
OFFICIAL_TURTLEBOT3_BURGER_PROTO_URL = (
    "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/"
    "robots/robotis/turtlebot/protos/TurtleBot3Burger.proto"
)
OFFICIAL_ROBOTIS_LDS01_PROTO_URL = (
    "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/"
    "devices/robotis/protos/RobotisLds01.proto"
)
EXTENSION_SLOT_TRANSLATION_ROBOT_FRAME_M = (-0.03, 0.0)
EXPECTED_LIDAR_ONLY = 47
EXPECTED_CONTACT_CONFIRMED = 3
_SHA_LENGTH = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"{label} schema mismatch; missing={missing}, unknown={unknown}.")


def _sha_literal(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _path_from_report(value: Any, *, report_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path.")
    path = Path(value)
    if path.is_absolute():
        declared = path.resolve()
        if declared.is_file():
            return declared
        portable_fallback = (report_path.parent / path.name).resolve()
        if portable_fallback.is_file():
            return portable_fallback
        return declared
    return (report_path.parent / path).resolve()


def _portable_output_path(path: Path, *, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _verify_binding(
    value: Any,
    *,
    report_path: Path,
    label: str,
    expected_sha256: str,
) -> Path:
    binding = _mapping(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    declared = _sha_literal(binding["sha256"], f"{label}.sha256")
    if declared != expected_sha256:
        raise PermissionError(f"{label} is not the locked v1 artifact.")
    path = _path_from_report(binding["path"], report_path=report_path, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != declared:
        raise ValueError(f"{label} SHA-256 mismatch: expected {declared}, actual {actual}.")
    return path


def _load_v1_report(
    path: Path,
    *,
    locked_collection: v1.LockedCollection,
) -> tuple[dict[str, Any], Path, Path, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v1 risk analysis report does not exist: {path}")
    report_sha = _sha256_file(path)
    if report_sha != V1_RISK_ANALYSIS_REPORT_SHA256:
        raise PermissionError("v1 risk analysis report is not the locked artifact.")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "v1_report")
    _exact_keys(
        report,
        {
            "schema_version",
            "analysis_protocol",
            "inputs",
            "episode_count",
            "scenario_seeds",
            "aggregates",
            "mechanism_decision",
            "artifacts",
            "scope",
        },
        "v1_report",
    )
    if report["schema_version"] != 1 or report["episode_count"] != v1.DIAGNOSTIC_EPISODES:
        raise PermissionError("v1 report schema or episode count is out of scope.")
    if report["scenario_seeds"] != list(v1.DIAGNOSTIC_SEEDS):
        raise PermissionError("v1 report must contain the exact 120 diagnostic seeds.")
    scope = _mapping(report["scope"], "v1_report.scope")
    expected_scope = {
        "split": "diagnostic",
        "formal_mechanism_inference": True,
        "collection_experiment_id": v1.COLLECTION_EXPERIMENT_ID,
        "model_sha256": v1.RECOVERY_MODEL_SHA256,
        "exact_scenario_seed_start": v1.DIAGNOSTIC_SEED_START,
        "exact_scenario_seed_count": v1.DIAGNOSTIC_EPISODES,
        "exact_route_axis_counts": v1.EXPECTED_AXIS_COUNTS,
        "checkpoint_selection": "forbidden",
        "validation_or_test_access": "forbidden",
    }
    if scope != expected_scope:
        raise PermissionError("v1 report scope is not the locked diagnostic scope.")
    inputs = _mapping(report["inputs"], "v1_report.inputs")
    collection_input = _mapping(
        inputs.get("collection_report"), "v1_report.inputs.collection_report"
    )
    if (
        collection_input.get("sha256") != locked_collection.report_sha256
        or collection_input.get("experiment_id") != v1.COLLECTION_EXPERIMENT_ID
        or collection_input.get("model_sha256") != v1.RECOVERY_MODEL_SHA256
    ):
        raise ValueError("v1 report does not bind the supplied collection report/model.")
    context_input = _mapping(inputs.get("episode_context"), "v1_report.inputs.episode_context")
    episodes_input = _mapping(inputs.get("episodes_csv"), "v1_report.inputs.episodes_csv")
    if context_input.get("sha256") != locked_collection.context_sha256:
        raise ValueError("v1 report episode_context SHA disagrees with the collection.")
    if episodes_input.get("sha256") != locked_collection.episodes_sha256:
        raise ValueError("v1 report episodes_csv SHA disagrees with the collection.")
    artifacts = _mapping(report["artifacts"], "v1_report.artifacts")
    _exact_keys(
        artifacts,
        {"risk_episode_metrics_json", "risk_episode_metrics_csv"},
        "v1_report.artifacts",
    )
    metrics_json = _verify_binding(
        artifacts["risk_episode_metrics_json"],
        report_path=path,
        label="v1_report.artifacts.risk_episode_metrics_json",
        expected_sha256=V1_METRICS_JSON_SHA256,
    )
    metrics_csv = _verify_binding(
        artifacts["risk_episode_metrics_csv"],
        report_path=path,
        label="v1_report.artifacts.risk_episode_metrics_csv",
        expected_sha256=V1_METRICS_CSV_SHA256,
    )
    return report, metrics_json, metrics_csv, report_sha


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != v1.DIAGNOSTIC_EPISODES:
        raise ValueError("v1 metrics must contain exactly 120 records.")
    expected_seeds = list(v1.DIAGNOSTIC_SEEDS)
    seeds = [value.get("scenario_seed") for value in values if isinstance(value, dict)]
    if seeds != expected_seeds:
        raise PermissionError("v1 metrics must use the exact ordered diagnostic seeds.")
    required = {
        "episode_id",
        "scenario_seed",
        "condition_id",
        "planner",
        "route_axis",
        "outcome",
        "terminal_collision_source",
        "mechanism_classification",
        "deceleration_reaction_evaluable",
        "deceleration_reaction_indeterminate_reason",
        "timely_deceleration_response",
    }
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError(f"v1 metrics record {index} schema mismatch.")
    return values


def _intersection_residuals(sample: Mapping[str, Any], origin_field: str) -> list[dict[str, Any]]:
    origin = v1._vector(sample[origin_field], origin_field)
    yaw = float(sample["post_robot_yaw"])
    lidar = _mapping(sample["post_lidar_argmin"], "post_lidar_argmin")
    bearing = float(lidar["bearing_rad"])
    measured = float(lidar["range_m"])
    direction = np.asarray([math.cos(yaw + bearing), math.sin(yaw + bearing)])
    results: list[dict[str, Any]] = []
    for index, centre_value in enumerate(sample["post_obstacle_xy"]):
        expected = v1._ray_circle_intersection_range(
            origin=origin,
            direction=direction,
            centre=v1._vector(centre_value, "post_obstacle_xy"),
            radius=v1.OBSTACLE_RADIUS_M,
        )
        residual = None if expected is None else abs(expected - measured)
        results.append(
            {
                "obstacle_index": index,
                "expected_range_m": expected,
                "absolute_residual_m": residual,
                "within_tolerance": residual is not None and residual <= v1.GEOMETRY_TOLERANCE_M,
            }
        )
    return results


def _unique_match(values: Sequence[Mapping[str, Any]]) -> int | None:
    matched = [int(value["obstacle_index"]) for value in values if value["within_tolerance"]]
    return matched[0] if len(matched) == 1 else None


def classify_origin_response(root_match: int | None, gps_match: int | None) -> str:
    """Classify only the attribution response to changing the recorded origin."""

    if root_match is None and gps_match is not None:
        return "resolved_by_gps_lidar_origin_proxy"
    if root_match is None and gps_match is None:
        return "unresolved_under_both_origins"
    if root_match == gps_match:
        return "attribution_retained"
    if gps_match is None:
        return "attribution_lost_under_gps_proxy"
    return "origin_sensitive_attribution_conflict"


def _episode_row(
    record: Mapping[str, Any],
    metric: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminal_step = int(record["termination"]["terminal_step"])
    terminal = next((sample for sample in samples if int(sample["step"]) == terminal_step), None)
    if terminal is None:
        raise ValueError(f"Missing terminal trace sample for seed {metric['scenario_seed']}.")
    root = _intersection_residuals(terminal, "post_robot_root_xy")
    gps = _intersection_residuals(terminal, "post_robot_gps_xy")
    root_match = _unique_match(root)
    gps_match = _unique_match(gps)
    terminal_curve = next(
        (
            value
            for value in metric.get("risk_curve", [])
            if int(value.get("step", -1)) == terminal_step
        ),
        None,
    )
    if not isinstance(terminal_curve, dict):
        raise ValueError(
            f"Missing v1 terminal risk-curve sample for seed {metric['scenario_seed']}."
        )
    if gps_match is None:
        sensitivity_mechanism = "ambiguous"
        sensitivity_reason = "lidar_ray_not_uniquely_attributed_under_gps_proxy"
    else:
        obstacles = terminal_curve.get("obstacles")
        if not isinstance(obstacles, list) or gps_match >= len(obstacles):
            raise ValueError(
                f"Invalid v1 terminal obstacle metrics for seed {metric['scenario_seed']}."
            )
        obstacle = _mapping(obstacles[gps_match], "terminal obstacle metric")
        gap = obstacle.get("gap_m")
        proxy_cpa = obstacle.get("proxy_cpa_clearance_m")
        if not isinstance(gap, (int, float)) or not isinstance(proxy_cpa, (int, float)):
            raise ValueError("v1 terminal geometry values must be numeric.")
        if float(gap) <= v1.GEOMETRY_TOLERANCE_M or float(proxy_cpa) <= 0.0:
            sensitivity_mechanism = "physical_imminent_dynamic"
            sensitivity_reason = None
        elif float(gap) > v1.GEOMETRY_TOLERANCE_M and float(proxy_cpa) > v1.GEOMETRY_TOLERANCE_M:
            sensitivity_mechanism = "lidar_proxy_compatible"
            sensitivity_reason = None
        else:
            sensitivity_mechanism = "ambiguous"
            sensitivity_reason = "geometry_falls_inside_proxy_tolerance_band"
    root_candidates = sum(value["expected_range_m"] is not None for value in root)
    gps_candidates = sum(value["expected_range_m"] is not None for value in gps)
    return {
        "episode_id": metric["episode_id"],
        "scenario_seed": metric["scenario_seed"],
        "route_axis": metric["route_axis"],
        "terminal_step": terminal_step,
        "measured_lidar_range_m": float(terminal["post_lidar_argmin"]["range_m"]),
        "lidar_bearing_rad": float(terminal["post_lidar_argmin"]["bearing_rad"]),
        "root_xy": terminal["post_robot_root_xy"],
        "gps_lidar_origin_proxy_xy": terminal["post_robot_gps_xy"],
        "observed_origin_offset_m": float(
            np.linalg.norm(
                v1._vector(terminal["post_robot_gps_xy"], "post_robot_gps_xy")
                - v1._vector(terminal["post_robot_root_xy"], "post_robot_root_xy")
            )
        ),
        "root_circle_intersection_candidate_count": root_candidates,
        "gps_circle_intersection_candidate_count": gps_candidates,
        "root_unique_attributed_obstacle_index": root_match,
        "gps_unique_attributed_obstacle_index": gps_match,
        "root_best_absolute_residual_m": min(
            (
                value["absolute_residual_m"]
                for value in root
                if value["absolute_residual_m"] is not None
            ),
            default=None,
        ),
        "gps_best_absolute_residual_m": min(
            (
                value["absolute_residual_m"]
                for value in gps
                if value["absolute_residual_m"] is not None
            ),
            default=None,
        ),
        "post_hoc_sensitivity_classification": classify_origin_response(root_match, gps_match),
        "post_hoc_sensitivity_mechanism_classification": sensitivity_mechanism,
        "post_hoc_sensitivity_mechanism_reason": sensitivity_reason,
        "v1_mechanism_classification_unchanged": metric["mechanism_classification"],
        "included_in_corrected_physical_candidate_set": (
            sensitivity_mechanism == "physical_imminent_dynamic"
        ),
        "v1_deceleration_reaction_evaluable": metric["deceleration_reaction_evaluable"],
        "v1_deceleration_reaction_indeterminate_reason": metric[
            "deceleration_reaction_indeterminate_reason"
        ],
        "v1_timely_deceleration_response": metric["timely_deceleration_response"],
        "root_intersections": root,
        "gps_proxy_intersections": gps,
    }


def _csv_row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not isinstance(item, (dict, list))}


def audit_lidar_origin_sensitivity(
    *,
    collection_report_path: Path,
    v1_risk_analysis_report_path: Path,
    output_directory: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Run the locked offline sensitivity audit and write a new output directory."""

    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite sensitivity audit: {output_directory}")
    if _sha256_file(collection_report_path) != COLLECTION_REPORT_SHA256:
        raise PermissionError("collection_report.json is not the locked artifact.")
    locked = v1.load_locked_collection(
        collection_report_path,
        repository_root=repository_root,
    )
    v1_report, metrics_path, metrics_csv_path, v1_report_sha = _load_v1_report(
        v1_risk_analysis_report_path,
        locked_collection=locked,
    )
    episodes = v1.read_episode_rows(locked.episodes_path)
    contexts = v1.read_context_records(locked.context_path, episodes=episodes)
    v1._validate_locked_episode_scope(episodes, contexts)
    metrics = _load_metrics(metrics_path)
    metric_by_seed = {int(value["scenario_seed"]): value for value in metrics}
    context_by_seed = {
        int(episode.scenario_seed): (record, samples)
        for record, episode, samples, _memberships in contexts
    }
    if set(metric_by_seed) != set(context_by_seed) or set(metric_by_seed) != set(
        v1.DIAGNOSTIC_SEEDS
    ):
        raise PermissionError("Context and v1 metrics do not share the exact diagnostic seeds.")
    for seed, metric in metric_by_seed.items():
        episode = episodes[metric["episode_id"]]
        record, _samples = context_by_seed[seed]
        if (
            episode.scenario_seed != seed
            or episode.split != "diagnostic"
            or metric["condition_id"] != episode.condition_id
            or metric["planner"] != episode.planner
            or metric["route_axis"] != record["route_axis"]
        ):
            raise ValueError(f"v1/context identity mismatch for diagnostic seed {seed}.")
    lidar_metrics = [
        metric for metric in metrics if metric["terminal_collision_source"] == "lidar_only"
    ]
    contact_metrics = [
        metric
        for metric in metrics
        if metric["terminal_collision_source"] in {"contact_only", "both"}
    ]
    if (
        len(lidar_metrics) != EXPECTED_LIDAR_ONLY
        or len(contact_metrics) != EXPECTED_CONTACT_CONFIRMED
    ):
        raise PermissionError(
            "v1 terminal collision-source counts are not the locked 47/3 audit scope."
        )
    rows = [
        _episode_row(
            context_by_seed[int(metric["scenario_seed"])][0],
            metric,
            context_by_seed[int(metric["scenario_seed"])][1],
        )
        for metric in lidar_metrics
    ]
    rows.sort(key=lambda value: int(value["scenario_seed"]))
    response_counts = Counter(str(value["post_hoc_sensitivity_classification"]) for value in rows)
    gps_attributed = [
        value for value in rows if value["gps_unique_attributed_obstacle_index"] is not None
    ]
    corrected_lidar = [
        value for value in rows if value["included_in_corrected_physical_candidate_set"]
    ]
    corrected_seeds = {int(value["scenario_seed"]) for value in corrected_lidar} | {
        int(value["scenario_seed"]) for value in contact_metrics
    }
    corrected_metrics = [metric_by_seed[seed] for seed in sorted(corrected_seeds)]
    reaction_evaluable = [
        value for value in corrected_metrics if value["deceleration_reaction_evaluable"]
    ]
    reaction_indeterminate = [
        value for value in corrected_metrics if not value["deceleration_reaction_evaluable"]
    ]
    indeterminate_reasons = Counter(
        str(value["deceleration_reaction_indeterminate_reason"]) for value in reaction_indeterminate
    )
    sensitivity_rule_rows = [
        {
            "terminal_collision_source": "lidar_only",
            "mechanism_classification": value["post_hoc_sensitivity_mechanism_classification"],
            "deceleration_reaction_evaluable": value["v1_deceleration_reaction_evaluable"],
            "deceleration_reaction_indeterminate_reason": value[
                "v1_deceleration_reaction_indeterminate_reason"
            ],
            "timely_deceleration_response": value["v1_timely_deceleration_response"],
        }
        for value in rows
    ] + [
        {
            "terminal_collision_source": value["terminal_collision_source"],
            "mechanism_classification": "contact_confirmed",
            "deceleration_reaction_evaluable": value["deceleration_reaction_evaluable"],
            "deceleration_reaction_indeterminate_reason": value[
                "deceleration_reaction_indeterminate_reason"
            ],
            "timely_deceleration_response": value["timely_deceleration_response"],
        }
        for value in contact_metrics
    ]
    sensitivity_only_threshold_result = v1.mechanism_decision(sensitivity_rule_rows)
    report = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "audit_type": "post_hoc_sensitivity",
        "inputs": {
            "collection_report": {
                "path": _portable_output_path(
                    collection_report_path, repository_root=repository_root
                ),
                "sha256": locked.report_sha256,
            },
            "v1_risk_analysis_report": {
                "path": _portable_output_path(
                    v1_risk_analysis_report_path, repository_root=repository_root
                ),
                "sha256": v1_report_sha,
                "formal_decision_unchanged": v1_report["mechanism_decision"]["decision"],
            },
            "episode_context": {
                "path": _portable_output_path(
                    locked.context_path, repository_root=repository_root
                ),
                "sha256": locked.context_sha256,
                "schema_version": v1.EPISODE_CONTEXT_SCHEMA_VERSION,
            },
            "v1_risk_episode_metrics_json": {
                "path": _portable_output_path(
                    metrics_path, repository_root=repository_root
                ),
                "sha256": V1_METRICS_JSON_SHA256,
            },
            "v1_risk_episode_metrics_csv": {
                "path": _portable_output_path(
                    metrics_csv_path, repository_root=repository_root
                ),
                "sha256": V1_METRICS_CSV_SHA256,
            },
        },
        "protocol": {
            "official_turtlebot3_burger_proto_url": OFFICIAL_TURTLEBOT3_BURGER_PROTO_URL,
            "official_robotis_lds01_proto_url": OFFICIAL_ROBOTIS_LDS01_PROTO_URL,
            "webots_proto_revision": "R2025a",
            "network_access": "forbidden_and_not_used",
            "extension_slot_translation_robot_frame_m": list(
                EXTENSION_SLOT_TRANSLATION_ROBOT_FRAME_M
            ),
            "lds_local_xy_in_extension_slot_m": [0.0, 0.0],
            "gps_default_local_xy_in_extension_slot_m": [0.0, 0.0],
            "fixed_transform_basis": (
                "TurtleBot3Burger R2025a extensionSlot translation x=-0.03 m; "
                "RobotisLds01 and default GPS add no local XY translation"
            ),
            "gps_and_lidar_world_configuration": "same_extensionSlot",
            "root_origin_field": "post_robot_root_xy",
            "gps_lidar_origin_proxy_field": "post_robot_gps_xy",
            "obstacle_radius_m": v1.OBSTACLE_RADIUS_M,
            "absolute_residual_tolerance_m": v1.GEOMETRY_TOLERANCE_M,
            "method": "terminal_argmin_ray_to_recorded_dynamic_obstacle_circle_intersection",
            "limitation": (
                "GPS is a recorded R2025a same-extensionSlot XY proxy, not a "
                "newly logged LiDAR node pose."
            ),
        },
        "attribution_response_counts": {
            "lidar_only_terminal_episodes": len(rows),
            "root_unique_attribution": sum(
                value["root_unique_attributed_obstacle_index"] is not None for value in rows
            ),
            "gps_proxy_unique_attribution": len(gps_attributed),
            "root_unresolved_with_circle_intersection_candidate": sum(
                value["root_unique_attributed_obstacle_index"] is None
                and value["root_circle_intersection_candidate_count"] > 0
                for value in rows
            ),
            **dict(sorted(response_counts.items())),
        },
        "corrected_physical_candidate_set": {
            "definition": (
                "GPS-proxy-attributed lidar-only terminations that satisfy the "
                "unchanged v1 physical geometry rule, plus v1 contact-confirmed "
                "terminations"
            ),
            "gps_proxy_attributed_lidar_only": len(gps_attributed),
            "physical_lidar_only": len(corrected_lidar),
            "ambiguous_lidar_only": sum(
                value["post_hoc_sensitivity_mechanism_classification"] == "ambiguous"
                for value in rows
            ),
            "lidar_proxy_compatible_lidar_only": sum(
                value["post_hoc_sensitivity_mechanism_classification"] == "lidar_proxy_compatible"
                for value in rows
            ),
            "contact_confirmed": len(contact_metrics),
            "total": len(corrected_metrics),
            "reaction_fields_reused_from_v1": True,
            "reaction_evaluable": len(reaction_evaluable),
            "reaction_indeterminate": len(reaction_indeterminate),
            "reaction_indeterminate_reason_counts": dict(sorted(indeterminate_reasons.items())),
            "timely_deceleration_response": sum(
                bool(value["timely_deceleration_response"]) for value in reaction_evaluable
            ),
            "short_horizon_response_insufficient": sum(
                not bool(value["timely_deceleration_response"]) for value in reaction_evaluable
            ),
        },
        "post_hoc_sensitivity_decision": {
            "outcome": (
                f"{sensitivity_only_threshold_result['decision']}_under_sensitivity_only_rule"
            ),
            "unchanged_v1_threshold_application": sensitivity_only_threshold_result,
            "not_a_formal_mechanism_decision": True,
        },
        "scope": {
            "split": "diagnostic",
            "episode_count": v1.DIAGNOSTIC_EPISODES,
            "scenario_seeds": list(v1.DIAGNOSTIC_SEEDS),
            "model_sha256": v1.RECOVERY_MODEL_SHA256,
            "post_hoc_sensitivity": True,
            "formal_mechanism_inference": False,
            "does_not_replace_v1": True,
            "checkpoint_selection": "forbidden",
            "validation_or_test_access": "forbidden",
            "webots_execution": "forbidden_and_not_used",
        },
    }
    if (
        _sha256_file(collection_report_path) != locked.report_sha256
        or _sha256_file(v1_risk_analysis_report_path) != v1_report_sha
        or _sha256_file(locked.context_path) != locked.context_sha256
        or _sha256_file(metrics_path) != V1_METRICS_JSON_SHA256
        or _sha256_file(metrics_csv_path) != V1_METRICS_CSV_SHA256
    ):
        raise RuntimeError("Locked audit inputs changed while they were being parsed.")
    output_directory.mkdir(parents=True, exist_ok=False)
    per_episode_json = output_directory / "lidar_origin_sensitivity_per_episode.json"
    per_episode_csv = output_directory / "lidar_origin_sensitivity_per_episode.csv"
    report_path = output_directory / "lidar_origin_sensitivity_report.json"
    per_episode_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = [_csv_row(value) for value in rows]
    with per_episode_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report["artifacts"] = {
        "per_episode_json": {
            "path": _portable_output_path(per_episode_json, repository_root=repository_root),
            "sha256": _sha256_file(per_episode_json),
        },
        "per_episode_csv": {
            "path": _portable_output_path(per_episode_csv, repository_root=repository_root),
            "sha256": _sha256_file(per_episode_csv),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.with_suffix(".sha256").write_text(
        f"{_sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--v1-risk-analysis-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit_lidar_origin_sensitivity(
        collection_report_path=args.collection_report,
        v1_risk_analysis_report_path=args.v1_risk_analysis_report,
        output_directory=args.output_dir,
        repository_root=root,
    )
    gps_attributed = report["corrected_physical_candidate_set"]["gps_proxy_attributed_lidar_only"]
    print(
        f"audit={report['audit_id']} "
        f"gps_attributed={gps_attributed} "
        f"formal_mechanism_inference=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
