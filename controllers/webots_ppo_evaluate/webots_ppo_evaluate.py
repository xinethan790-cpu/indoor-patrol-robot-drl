"""Evaluate PPO, reactive, or DWA on deterministic Webots scenarios."""

from __future__ import annotations

import hashlib
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from controller import Supervisor

from indoor_patrol_rl.core import (
    SAFETY_SHIELD_CONFIG_ENV_PREFIX,
    LidarSafetyShield,
    SafetyShieldConfig,
    safety_shield_config_from_env,
)
from indoor_patrol_rl.envs import WebotsPatrolEnv
from indoor_patrol_rl.evaluation import EpisodeAccumulator, write_benchmark_outputs
from indoor_patrol_rl.evaluation.webots_formal_config import (
    DYNAMIC_OBSTACLE_DEFS,
    DYNAMIC_OBSTACLE_RADII,
    FORMAL_ROBOT_BOUNDING_RADIUS,
    HIDDEN_NODE_TRANSLATION,
    STATIC_NODE_GEOMETRY,
    WebotsFormalConfig,
    formal_reference_path_length,
    inactive_dynamic_obstacle_defs,
    load_webots_formal_config,
    obstacle_cross_lanes_for_seed,
    resolve_formal_test_request,
    static_node_translations,
)
from indoor_patrol_rl.policies import DWAConfig, DWAPlanner, ReactiveAvoidancePolicy
from indoor_patrol_rl.training.webots_curriculum import validate_evaluation_run_name

MODEL_VARIANT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
PLANNERS = {"reactive", "dwa", "ppo"}


@dataclass(frozen=True)
class EvaluationPlan:
    """Fully resolved settings for one single-condition evaluation run."""

    split: str
    condition_id: str
    planner: str
    planner_label: str
    training_seed: int | None
    scenario_seeds: tuple[int, ...]
    obstacle_count: int
    obstacle_cross_lanes: tuple[float, ...] | None
    obstacle_speed_range: tuple[float, float]
    horizontal_route_half_length: float
    vertical_route_half_length: float
    action_repeat: int
    max_episode_steps: int
    safety_shield_enabled: bool
    safety_shield_config: SafetyShieldConfig
    static_layout: str
    formal_config: WebotsFormalConfig | None = None


def sha256_file(path: Path) -> str:
    """Hash one exact evaluation input or output file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_flag(name: str, *, default: bool = False) -> bool:
    """Parse an explicit environment boolean and reject ambiguous values."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off.")


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _environment_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _required_node(supervisor: Supervisor, def_name: str) -> Any:
    node = supervisor.getFromDef(def_name)
    if node is None:
        raise RuntimeError(f"Required Webots DEF node {def_name!r} was not found.")
    return node


def _set_translation(
    supervisor: Supervisor,
    def_name: str,
    translation: tuple[float, float, float],
) -> None:
    node = _required_node(supervisor, def_name)
    field = node.getField("translation")
    if field is None:
        raise RuntimeError(f"Webots DEF node {def_name!r} has no translation field.")
    field.setSFVec3f(list(translation))
    node.resetPhysics()


def _configure_world_nodes(
    supervisor: Supervisor,
    *,
    obstacle_count: int,
    static_layout: str,
) -> None:
    """Activate exactly one scenario layout and hide every unused node below the floor."""

    for def_name, expected_radius in zip(
        DYNAMIC_OBSTACLE_DEFS, DYNAMIC_OBSTACLE_RADII, strict=True
    ):
        node = _required_node(supervisor, def_name)
        bounding_field = node.getField("boundingObject")
        bounding_node = bounding_field.getSFNode() if bounding_field is not None else None
        radius_field = bounding_node.getField("radius") if bounding_node is not None else None
        height_field = bounding_node.getField("height") if bounding_node is not None else None
        if (
            radius_field is None
            or height_field is None
            or not np.isclose(
                radius_field.getSFFloat(), expected_radius, rtol=0.0, atol=1e-9
            )
            or not np.isclose(height_field.getSFFloat(), 0.72, rtol=0.0, atol=1e-9)
        ):
            raise RuntimeError(
                f"Dynamic Webots DEF node {def_name!r} Cylinder has protocol drift."
            )
    for def_name in inactive_dynamic_obstacle_defs(obstacle_count):
        _set_translation(supervisor, def_name, HIDDEN_NODE_TRANSLATION)
    for def_name, translation in static_node_translations(static_layout).items():
        node = _required_node(supervisor, def_name)
        translation_field = node.getField("translation")
        rotation_field = node.getField("rotation")
        if translation_field is None or rotation_field is None:
            raise RuntimeError(
                f"Static Webots DEF node {def_name!r} needs translation and rotation fields."
            )
        translation_field.setSFVec3f(list(translation))
        expected_rotation, expected_size = STATIC_NODE_GEOMETRY[def_name]
        rotation_field.setSFRotation(list(expected_rotation))
        bounding_field = node.getField("boundingObject")
        bounding_node = bounding_field.getSFNode() if bounding_field is not None else None
        size_field = bounding_node.getField("size") if bounding_node is not None else None
        if size_field is None or not np.allclose(
            size_field.getSFVec3f(), expected_size, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(
                f"Static Webots DEF node {def_name!r} bounding Box size has protocol drift."
            )
        if not np.allclose(
            translation_field.getSFVec3f(), translation, rtol=0.0, atol=1e-9
        ) or not np.allclose(
            rotation_field.getSFRotation(), expected_rotation, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(f"Static Webots DEF node {def_name!r} failed readback.")
        node.resetPhysics()


def _repository_path(repository_root: Path, value: str, label: str) -> Path:
    configured = Path(value)
    path = configured if configured.is_absolute() else repository_root / configured
    resolved = path.resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository.") from error
    return resolved


def _formal_config_path(repository_root: Path) -> Path:
    primary = os.environ.get("WEBOTS_FORMAL_CONFIG")
    compatibility = os.environ.get("WEBOTS_EVAL_CONFIG")
    if primary and compatibility and Path(primary) != Path(compatibility):
        raise ValueError("WEBOTS_FORMAL_CONFIG and WEBOTS_EVAL_CONFIG disagree.")
    value = primary or compatibility or "configs/eval/webots_formal.yaml"
    return _repository_path(repository_root, value, "Formal evaluation config")


def _assert_optional_int(name: str, expected: int) -> None:
    if name in os.environ and int(os.environ[name]) != expected:
        raise ValueError(f"{name} must match the locked formal value {expected}.")


def _assert_optional_float(name: str, expected: float) -> None:
    if name in os.environ and not np.isclose(
        float(os.environ[name]), expected, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{name} must match the locked formal value {expected}.")


def _assert_optional_text(name: str, expected: str) -> None:
    if name in os.environ and os.environ[name] != expected:
        raise ValueError(f"{name} must match the locked formal value {expected!r}.")


def _resolve_formal_plan(repository_root: Path) -> EvaluationPlan:
    required_runner_fields = (
        "WEBOTS_FORMAL_CONFIG",
        "WEBOTS_EVAL_CONFIG_SHA256",
        "WEBOTS_FORMAL_CAMPAIGN_ID",
        "WEBOTS_FINAL_LOCK_SHA256",
        "WEBOTS_EVAL_RUN_NAME",
    )
    missing_runner_fields = [
        name for name in required_runner_fields if not os.environ.get(name)
    ]
    if missing_runner_fields:
        raise PermissionError(
            "Formal test must be launched by the locked matrix runner; missing "
            f"{missing_runner_fields}."
        )
    config = load_webots_formal_config(_formal_config_path(repository_root))
    condition_value = os.environ.get("WEBOTS_EVAL_CONDITION")
    planner_value = os.environ.get("WEBOTS_EVAL_PLANNER")
    if condition_value is None:
        raise ValueError("Formal test requires explicit WEBOTS_EVAL_CONDITION=G0|G1|G2|G3.")
    if planner_value is None:
        raise ValueError("Formal test requires explicit WEBOTS_EVAL_PLANNER=reactive|dwa|ppo.")
    planner = planner_value.strip().lower()
    raw_training_seed = os.environ.get("WEBOTS_MODEL_SEED")
    if planner == "ppo":
        if raw_training_seed is None:
            raise ValueError("Formal PPO test requires explicit WEBOTS_MODEL_SEED=11|22|33.")
        training_seed: int | None = int(raw_training_seed)
    else:
        if raw_training_seed is not None:
            raise ValueError("Formal baselines must not set WEBOTS_MODEL_SEED.")
        training_seed = None
    seed_start = _environment_int("WEBOTS_EVAL_SEED_START", config.scenario_seed_start)
    episode_count = _environment_int("WEBOTS_EVAL_EPISODES", config.scenario_seed_count)
    request = resolve_formal_test_request(
        config,
        condition_id=condition_value,
        planner=planner,
        training_seed=training_seed,
        seed_start=seed_start,
        episode_count=episode_count,
        confirmed=os.environ.get("WEBOTS_CONFIRM_TEST") == "1",
    )
    if environment_flag("WEBOTS_SAFETY_SHIELD"):
        raise ValueError("Formal G0-G3 evaluation fixes WEBOTS_SAFETY_SHIELD=0.")
    configured_shield_values = sorted(
        name
        for name in os.environ
        if name.startswith(SAFETY_SHIELD_CONFIG_ENV_PREFIX)
    )
    if configured_shield_values:
        raise ValueError(
            "Formal G0-G3 evaluation fixes the default SafetyShieldConfig while "
            f"WEBOTS_SAFETY_SHIELD=0; remove {configured_shield_values}."
        )
    configured_label = os.environ.get("WEBOTS_EVAL_PLANNER_LABEL")
    if configured_label is not None and configured_label != request.planner_label:
        raise ValueError(
            "WEBOTS_EVAL_PLANNER_LABEL must match the locked formal label "
            f"{request.planner_label!r}."
        )
    condition = request.condition
    _assert_optional_int("WEBOTS_OBSTACLE_COUNT", condition.obstacle_count)
    _assert_optional_float("WEBOTS_OBSTACLE_SPEED_MIN", condition.obstacle_speed_range[0])
    _assert_optional_float("WEBOTS_OBSTACLE_SPEED_MAX", condition.obstacle_speed_range[1])
    _assert_optional_float(
        "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH", condition.horizontal_route_half_length
    )
    _assert_optional_float(
        "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH", condition.vertical_route_half_length
    )
    _assert_optional_text("WEBOTS_STATIC_LAYOUT", condition.static_layout)
    _assert_optional_int("WEBOTS_ACTION_REPEAT", config.control.action_repeat)
    _assert_optional_int("WEBOTS_MAX_EPISODE_STEPS", config.control.max_episode_steps)
    _assert_optional_float("WEBOTS_MAX_LINEAR_SPEED", config.control.max_linear_speed)
    _assert_optional_float("WEBOTS_MAX_ANGULAR_SPEED", config.control.max_angular_speed)
    _assert_optional_float(
        "WEBOTS_MAX_LINEAR_ACCELERATION", config.control.max_linear_acceleration
    )
    _assert_optional_float(
        "WEBOTS_MAX_ANGULAR_ACCELERATION", config.control.max_angular_acceleration
    )
    _assert_optional_text("WEBOTS_EVAL_CONFIG_SHA256", config.source_sha256)
    campaign_id = os.environ["WEBOTS_FORMAL_CAMPAIGN_ID"]
    final_lock_sha256 = os.environ["WEBOTS_FINAL_LOCK_SHA256"]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id) is None:
        raise ValueError("WEBOTS_FORMAL_CAMPAIGN_ID is not a path-safe campaign name.")
    if re.fullmatch(r"[0-9a-f]{64}", final_lock_sha256) is None:
        raise ValueError("WEBOTS_FINAL_LOCK_SHA256 must be a lowercase SHA-256 digest.")
    return EvaluationPlan(
        split=config.split,
        condition_id=condition.condition_id,
        planner=request.planner,
        planner_label=request.planner_label,
        training_seed=request.training_seed,
        scenario_seeds=request.scenario_seeds,
        obstacle_count=condition.obstacle_count,
        obstacle_cross_lanes=condition.obstacle_cross_lanes,
        obstacle_speed_range=condition.obstacle_speed_range,
        horizontal_route_half_length=condition.horizontal_route_half_length,
        vertical_route_half_length=condition.vertical_route_half_length,
        action_repeat=config.control.action_repeat,
        max_episode_steps=config.control.max_episode_steps,
        safety_shield_enabled=False,
        safety_shield_config=SafetyShieldConfig(),
        static_layout=condition.static_layout,
        formal_config=config,
    )


def _resolve_validation_plan() -> EvaluationPlan:
    planner = os.environ.get("WEBOTS_EVAL_PLANNER", "ppo").strip().lower()
    if planner not in PLANNERS:
        raise ValueError("WEBOTS_EVAL_PLANNER must be reactive, dwa, or ppo.")
    raw_training_seed = os.environ.get("WEBOTS_MODEL_SEED")
    if planner == "ppo":
        training_seed: int | None = int(raw_training_seed or "11")
        if training_seed < 0:
            raise ValueError("WEBOTS_MODEL_SEED must be non-negative.")
    else:
        if raw_training_seed is not None:
            raise ValueError("Baseline evaluation must not set WEBOTS_MODEL_SEED.")
        training_seed = None
    seed_start = _environment_int("WEBOTS_EVAL_SEED_START", 5000)
    episode_count = _environment_int("WEBOTS_EVAL_EPISODES", 20)
    obstacle_count = _environment_int("WEBOTS_OBSTACLE_COUNT", 1)
    if seed_start < 0:
        raise ValueError("WEBOTS_EVAL_SEED_START must be non-negative.")
    if episode_count <= 0:
        raise ValueError("WEBOTS_EVAL_EPISODES must be positive.")
    if not 0 <= obstacle_count <= len(DYNAMIC_OBSTACLE_DEFS):
        raise ValueError(
            f"WEBOTS_OBSTACLE_COUNT must be between 0 and {len(DYNAMIC_OBSTACLE_DEFS)}."
        )
    obstacle_speed_range = (
        _environment_float("WEBOTS_OBSTACLE_SPEED_MIN", 0.20),
        _environment_float("WEBOTS_OBSTACLE_SPEED_MAX", 0.42),
    )
    if not 0.0 <= obstacle_speed_range[0] <= obstacle_speed_range[1]:
        raise ValueError("Webots obstacle speeds must be non-negative and ordered.")
    shield = environment_flag("WEBOTS_SAFETY_SHIELD")
    shield_config = safety_shield_config_from_env(os.environ)
    default_label = (
        f"ppo_seed{training_seed}{'_shield' if shield else ''}"
        if planner == "ppo"
        else f"{planner}{'_shield' if shield else ''}"
    )
    planner_label = os.environ.get("WEBOTS_EVAL_PLANNER_LABEL", default_label)
    condition_id = os.environ.get("WEBOTS_EVAL_CONDITION", f"W{obstacle_count}")
    if re.fullmatch(r"[A-Za-z0-9]+", condition_id) is None:
        raise ValueError("WEBOTS_EVAL_CONDITION must contain only letters and digits.")
    if not planner_label or any(character in planner_label for character in "\r\n"):
        raise ValueError("WEBOTS_EVAL_PLANNER_LABEL must be a non-empty single-line value.")
    return EvaluationPlan(
        split="validation",
        condition_id=condition_id,
        planner=planner,
        planner_label=planner_label,
        training_seed=training_seed,
        scenario_seeds=tuple(range(seed_start, seed_start + episode_count)),
        obstacle_count=obstacle_count,
        obstacle_cross_lanes=None,
        obstacle_speed_range=obstacle_speed_range,
        horizontal_route_half_length=_environment_float(
            "WEBOTS_HORIZONTAL_ROUTE_HALF_LENGTH", 1.55
        ),
        vertical_route_half_length=_environment_float(
            "WEBOTS_VERTICAL_ROUTE_HALF_LENGTH", 1.10
        ),
        action_repeat=4,
        max_episode_steps=300,
        safety_shield_enabled=shield,
        safety_shield_config=shield_config,
        static_layout="train_a",
    )


def _resolve_plan(repository_root: Path) -> EvaluationPlan:
    split = os.environ.get("WEBOTS_EVAL_SPLIT", "validation").strip().lower()
    if split == "test":
        return _resolve_formal_plan(repository_root)
    if split != "validation":
        raise ValueError("WEBOTS_EVAL_SPLIT must be validation or test.")
    return _resolve_validation_plan()


def _model_and_variant(
    repository_root: Path,
    plan: EvaluationPlan,
) -> tuple[Any, Path | None, str | None]:
    """Build one policy and bind PPO to an exact repository model artifact."""

    if plan.planner == "reactive":
        if plan.formal_config is not None and (
            "WEBOTS_EVAL_MODEL" in os.environ
            or "WEBOTS_EVAL_MODEL_SHA256" in os.environ
            or "WEBOTS_MODEL_VARIANT" in os.environ
        ):
            raise ValueError("Formal reactive evaluation must not set PPO model variables.")
        return ReactiveAvoidancePolicy(lidar_beams=36), None, None
    if plan.planner == "dwa":
        if plan.formal_config is not None and (
            "WEBOTS_EVAL_MODEL" in os.environ
            or "WEBOTS_EVAL_MODEL_SHA256" in os.environ
            or "WEBOTS_MODEL_VARIANT" in os.environ
        ):
            raise ValueError("Formal DWA evaluation must not set PPO model variables.")
        return None, None, None

    assert plan.training_seed is not None
    default_model = (
        f"artifacts/models/webots/seed_{plan.training_seed}/ppo_webots_latest.zip"
    )
    model_value = os.environ.get("WEBOTS_EVAL_MODEL")
    if plan.formal_config is not None and model_value is None:
        raise ValueError("Formal PPO evaluation requires an explicit WEBOTS_EVAL_MODEL.")
    model_path = _repository_path(
        repository_root,
        model_value or default_model,
        "WEBOTS_EVAL_MODEL",
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"WEBOTS_EVAL_MODEL does not exist: {model_path}")
    actual_model_sha256 = sha256_file(model_path)
    expected_model_sha256 = os.environ.get("WEBOTS_EVAL_MODEL_SHA256")
    if plan.formal_config is not None and expected_model_sha256 is None:
        raise ValueError("Formal PPO evaluation requires WEBOTS_EVAL_MODEL_SHA256.")
    if expected_model_sha256 is not None:
        if re.fullmatch(r"[0-9a-fA-F]{64}", expected_model_sha256) is None:
            raise ValueError("WEBOTS_EVAL_MODEL_SHA256 must be a SHA-256 digest.")
        if expected_model_sha256.casefold() != actual_model_sha256:
            raise ValueError(
                "WEBOTS_EVAL_MODEL failed the locked WEBOTS_EVAL_MODEL_SHA256 check."
            )
    if plan.formal_config is not None:
        seed_root = (
            repository_root / f"artifacts/models/webots/seed_{plan.training_seed}"
        ).resolve()
        try:
            model_path.relative_to(seed_root)
        except ValueError as error:
            raise ValueError(
                "Formal PPO model must be below the matching training-seed directory "
                f"{seed_root}."
            ) from error
        if "latest" in model_path.stem.casefold():
            raise ValueError("Formal PPO evaluation refuses a mutable 'latest' checkpoint.")
    raw_variant = os.environ.get("WEBOTS_MODEL_VARIANT")
    model_variant = raw_variant.strip() if raw_variant else None
    if model_variant is not None and (
        model_variant in {".", ".."}
        or MODEL_VARIANT_PATTERN.fullmatch(model_variant) is None
    ):
        raise ValueError("WEBOTS_MODEL_VARIANT must be a path-safe single name.")
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    return model, model_path, model_variant


def _build_dwa(env: WebotsPatrolEnv, plan: EvaluationPlan) -> DWAPlanner:
    control = plan.formal_config.control if plan.formal_config is not None else None
    return DWAPlanner(
        DWAConfig(
            lidar_beams=36,
            lidar_max_range=float(env.lidar.getMaxRange()),
            goal_distance_scale=4.0,
            max_linear_speed=control.max_linear_speed if control else 0.22,
            max_angular_speed=control.max_angular_speed if control else 1.50,
            max_linear_acceleration=control.max_linear_acceleration if control else 0.50,
            max_angular_acceleration=control.max_angular_acceleration if control else 3.00,
            wheel_radius=0.033,
            axle_length=0.16,
            max_wheel_speed=6.0,
            control_dt=env.basic_time_step * plan.action_repeat / 1000.0,
            robot_radius=FORMAL_ROBOT_BOUNDING_RADIUS,
            safety_margin=0.20,
        )
    )


def _policy_action(policy: Any, planner: str, observation: np.ndarray) -> np.ndarray:
    if planner == "ppo":
        action, _state = policy.predict(observation, deterministic=True)
    else:
        action = policy.predict(observation)
    result = np.asarray(action, dtype=np.float32)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{planner} policy returned an invalid action.")
    return result


def _portable_path(path: Path | None, repository_root: Path) -> str | None:
    if path is None:
        return None
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _provenance(
    repository_root: Path,
    plan: EvaluationPlan,
    *,
    model_path: Path | None,
    model_variant: str | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model_path": _portable_path(model_path, repository_root),
        "model_sha256": sha256_file(model_path) if model_path is not None else None,
        "model_variant": model_variant,
        "training_seed": plan.training_seed,
        "split": plan.split,
        "condition_id": plan.condition_id,
        "planner": plan.planner,
        "planner_label": plan.planner_label,
        "scenario_seed_start": plan.scenario_seeds[0],
        "scenario_seed_count": len(plan.scenario_seeds),
        "scenario_seeds": list(plan.scenario_seeds),
        "obstacle_count": plan.obstacle_count,
        "obstacle_cross_lanes": (
            list(plan.obstacle_cross_lanes)
            if plan.obstacle_cross_lanes is not None
            else None
        ),
        "obstacle_spawn_protocol": (
            "fixed_cross_lanes_v1"
            if plan.obstacle_cross_lanes is not None
            else "training_random_cross_axis"
        ),
        "obstacle_lane_seed_transform": (
            "parity_mirror_v1" if plan.obstacle_cross_lanes is not None else None
        ),
        "obstacle_speed_range": list(plan.obstacle_speed_range),
        "horizontal_route_half_length": plan.horizontal_route_half_length,
        "vertical_route_half_length": plan.vertical_route_half_length,
        "static_layout": plan.static_layout,
        "safety_shield_enabled": plan.safety_shield_enabled,
        "safety_shield_config": asdict(plan.safety_shield_config),
        "reference_path_definition": (
            "inflated_static_map_astar"
            if plan.formal_config is not None
            else "euclidean_start_to_goal"
        ),
        "robot_bounding_radius": FORMAL_ROBOT_BOUNDING_RADIUS,
    }
    if plan.formal_config is not None:
        config = plan.formal_config
        world_path = _repository_path(repository_root, config.world, "Formal Webots world")
        if not world_path.is_file():
            raise FileNotFoundError(f"Formal Webots world does not exist: {world_path}")
        provenance.update(
            {
                "experiment_id": config.experiment_id,
                "webots_version": config.webots_version,
                "formal_config_path": _portable_path(config.source_path, repository_root),
                "formal_config_sha256": config.source_sha256,
                "evaluation_config_path": _portable_path(config.source_path, repository_root),
                "evaluation_config_sha256": config.source_sha256,
                "world_path": _portable_path(world_path, repository_root),
                "world_sha256": sha256_file(world_path),
                "condition_semantic_sha256": config.condition(
                    plan.condition_id
                ).semantic_sha256,
            }
        )
        campaign_id = os.environ.get("WEBOTS_FORMAL_CAMPAIGN_ID")
        final_lock_sha256 = os.environ.get("WEBOTS_FINAL_LOCK_SHA256")
        if campaign_id is not None:
            provenance["formal_campaign_id"] = campaign_id
        if final_lock_sha256 is not None:
            if re.fullmatch(r"[0-9a-f]{64}", final_lock_sha256) is None:
                raise ValueError("WEBOTS_FINAL_LOCK_SHA256 must be a lowercase SHA-256 digest.")
            provenance["final_lock_sha256"] = final_lock_sha256
    return provenance


def main() -> None:
    repository_root = Path(__file__).absolute().parents[2].resolve()
    supervisor = Supervisor()
    exit_code = 1
    env = None
    plan: EvaluationPlan | None = None
    provenance: dict[str, Any] | None = None
    output_directory: Path | None = None
    episode_index: int | None = None
    scenario_seed: int | None = None
    try:
        plan = _resolve_plan(repository_root)
        plan = replace(
            plan,
            safety_shield_config=replace(
                plan.safety_shield_config,
                lidar_max_range=float(supervisor.getDevice("LDS-01").getMaxRange()),
            ),
        )
        _configure_world_nodes(
            supervisor,
            obstacle_count=plan.obstacle_count,
            static_layout=plan.static_layout,
        )
        env = WebotsPatrolEnv(
            supervisor,
            obstacle_defs=DYNAMIC_OBSTACLE_DEFS[: plan.obstacle_count],
            action_repeat=plan.action_repeat,
            max_episode_steps=plan.max_episode_steps,
            enable_safety_shield=plan.safety_shield_enabled,
            safety_shield=LidarSafetyShield(plan.safety_shield_config),
            obstacle_speed_range=plan.obstacle_speed_range,
            horizontal_route_half_length=plan.horizontal_route_half_length,
            vertical_route_half_length=plan.vertical_route_half_length,
        )
        if plan.formal_config is not None:
            control = plan.formal_config.control
            actual_control = (
                env.mapper.max_linear_speed,
                env.mapper.max_angular_speed,
                env.rate_limiter.max_linear_acceleration,
                env.rate_limiter.max_angular_acceleration,
            )
            expected_control = (
                control.max_linear_speed,
                control.max_angular_speed,
                control.max_linear_acceleration,
                control.max_angular_acceleration,
            )
            if not np.allclose(actual_control, expected_control, rtol=0.0, atol=1e-12):
                raise RuntimeError("Webots environment control limits have protocol drift.")
        policy, model_path, model_variant = _model_and_variant(repository_root, plan)
        if plan.planner == "dwa":
            policy = _build_dwa(env, plan)
        if plan.planner == "ppo":
            if policy.observation_space.shape != env.observation_space.shape:
                raise ValueError(
                    "PPO observation shape does not match Webots: "
                    f"{policy.observation_space.shape} != {env.observation_space.shape}."
                )
            if policy.action_space.shape != env.action_space.shape:
                raise ValueError(
                    "PPO action shape does not match Webots: "
                    f"{policy.action_space.shape} != {env.action_space.shape}."
                )
        provenance = _provenance(
            repository_root,
            plan,
            model_path=model_path,
            model_variant=model_variant,
        )
        records = []
        default_run_name = (
            f"webots_{plan.split}_{plan.condition_id.lower()}_{plan.planner_label}"
            if plan.formal_config is not None
            else (
                f"webots_{plan.split}_seed{plan.training_seed}_obstacles{plan.obstacle_count}"
                f"{'_shield' if plan.safety_shield_enabled else ''}"
            )
        )
        run_name = validate_evaluation_run_name(
            os.environ.get("WEBOTS_EVAL_RUN_NAME", default_run_name)
        )
        results_root = (repository_root / "artifacts/results").resolve()
        output_directory = (results_root / run_name).resolve()
        try:
            output_directory.relative_to(results_root)
        except ValueError as error:
            raise ValueError("WEBOTS_EVAL_RUN_NAME escapes artifacts/results.") from error

        for episode_index, scenario_seed in enumerate(plan.scenario_seeds):
            episode_cross_lanes = (
                obstacle_cross_lanes_for_seed(
                    plan.formal_config.condition(plan.condition_id),
                    scenario_seed,
                )
                if plan.formal_config is not None
                else None
            )
            reset_options = (
                {
                    "obstacle_cross_lanes": episode_cross_lanes,
                    "obstacle_lane_min_separation": 0.35,
                }
                if plan.obstacle_cross_lanes is not None
                else None
            )
            observation, initial_info = env.reset(
                seed=scenario_seed,
                options=reset_options,
            )
            if plan.formal_config is not None:
                start_values = env.robot_translation.getSFVec3f()
                start = (round(float(start_values[0]), 6), round(float(start_values[1]), 6))
                goal = (
                    round(float(env.goal_position[0]), 6),
                    round(float(env.goal_position[1]), 6),
                )
                reference_path_length = formal_reference_path_length(
                    plan.static_layout,
                    start,
                    goal,
                )
            else:
                reference_path_length = float(initial_info["distance_to_goal"])
            accumulator = EpisodeAccumulator(
                episode_id=(
                    f"{plan.split}-{plan.condition_id}-{scenario_seed}-{plan.planner_label}"
                ),
                split=plan.split,
                condition_id=plan.condition_id,
                scenario_seed=scenario_seed,
                planner=plan.planner_label,
                reference_path_length=reference_path_length,
            )
            terminated = truncated = False
            while not (terminated or truncated):
                started = time.perf_counter()
                action = _policy_action(policy, plan.planner, observation)
                latency = time.perf_counter() - started
                observation, reward, terminated, truncated, info = env.step(action)
                accumulator.update(
                    reward=reward,
                    info=info,
                    inference_latency_s=latency,
                )
            record = accumulator.finalize()
            records.append(record)
            provenance["reset_health"] = env.reset_health_diagnostics()
            write_benchmark_outputs(
                output_directory,
                records,
                provenance=provenance,
            )
            print(
                f"WEBOTS_EVAL_EPISODE index={episode_index} seed={scenario_seed} "
                f"planner={plan.planner_label} outcome={record.outcome} "
                f"steps={record.step_count}",
                flush=True,
            )

        reset_health = env.reset_health_diagnostics()
        expected_resets = len(plan.scenario_seeds)
        if (
            reset_health["attempts"] != expected_resets
            or reset_health["successful"] != expected_resets
            or reset_health["finite_settle_checks"] != 2 * expected_resets
            or reset_health["readback_checks"] != expected_resets
        ):
            raise RuntimeError(
                "Webots evaluation reset-health counters are inconsistent: "
                f"expected_resets={expected_resets}, actual={reset_health}."
            )
        provenance["reset_health"] = reset_health
        summary = write_benchmark_outputs(
            output_directory,
            records,
            provenance=provenance,
        )
        episodes_path = output_directory / "episodes.csv"
        episodes_sha256 = sha256_file(episodes_path)
        provenance["episodes_csv_sha256"] = episodes_sha256
        summary = write_benchmark_outputs(
            output_directory,
            records,
            provenance=provenance,
        )
        if sha256_file(episodes_path) != episodes_sha256:
            raise RuntimeError("episodes.csv changed while binding its formal SHA-256.")
        planner_summary = summary["by_planner"][0]
        print(
            f"WEBOTS_EVALUATION_OK episodes={len(plan.scenario_seeds)} "
            f"planner={plan.planner_label} "
            f"shield={int(plan.safety_shield_enabled)} "
            f"success_rate={planner_summary['success_rate']:.3f} "
            f"collision_rate={planner_summary['collision_rate']:.3f} "
            f"results={output_directory}",
            flush=True,
        )
        exit_code = 0
    except Exception as error:
        failure_traceback = traceback.format_exc()
        print(failure_traceback, end="", flush=True)
        try:
            failure_directory = output_directory or (
                repository_root / "artifacts/results/webots_evaluation_failures"
            )
            failure_directory.mkdir(parents=True, exist_ok=True)
            runtime_diagnostics: object = None
            if env is not None:
                try:
                    runtime_diagnostics = env.runtime_diagnostics()
                except Exception as diagnostic_error:  # pragma: no cover - live API failure
                    runtime_diagnostics = (
                        f"unavailable: {type(diagnostic_error).__name__}: {diagnostic_error}"
                    )
            plan_evidence = None
            if plan is not None:
                plan_evidence = {
                    "split": plan.split,
                    "condition_id": plan.condition_id,
                    "planner": plan.planner,
                    "planner_label": plan.planner_label,
                    "training_seed": plan.training_seed,
                    "scenario_seed_start": (
                        plan.scenario_seeds[0] if plan.scenario_seeds else None
                    ),
                    "scenario_count": len(plan.scenario_seeds),
                }
            failure_metadata = {
                "status": "failed",
                "plan": plan_evidence,
                "episode_index": episode_index,
                "scenario_seed": scenario_seed,
                "exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": failure_traceback,
                },
                "runtime_diagnostics": runtime_diagnostics,
                "provenance": provenance,
                "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            failure_path = failure_directory / f"failure_{time.time_ns()}.yaml"
            temporary_path = failure_path.with_suffix(".yaml.tmp")
            temporary_path.write_text(
                yaml.safe_dump(failure_metadata, sort_keys=False), encoding="utf-8"
            )
            temporary_path.replace(failure_path)
            print(f"WEBOTS_EVALUATION_FAILURE_METADATA path={failure_path}", flush=True)
        except Exception:
            traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        supervisor.step(int(supervisor.getBasicTimeStep()))
        supervisor.simulationQuit(exit_code)


if __name__ == "__main__":
    main()
