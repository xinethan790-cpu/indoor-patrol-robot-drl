"""Shared benchmark runner and episode metrics for fair planner comparisons."""

from indoor_patrol_rl.evaluation.benchmark import (
    BenchmarkCondition,
    BenchmarkConfig,
    SeedSet,
    load_benchmark_config,
    require_test_confirmation,
    run_kinematic_benchmark,
    summarize_records,
    write_benchmark_outputs,
)
from indoor_patrol_rl.evaluation.checkpoint_selection import (
    CheckpointSelectionGate,
    build_checkpoint_selection,
    candidate_sort_key,
    load_strict_candidate_summary,
    make_candidate_record,
    parse_result_identity,
    resolve_candidate_model,
    write_checkpoint_selection,
)
from indoor_patrol_rl.evaluation.metrics import (
    EpisodeAccumulator,
    EpisodeRecord,
    write_episode_records,
)
from indoor_patrol_rl.evaluation.patrol import (
    PatrolCheckpointRecord,
    PatrolCollisionRecord,
    PatrolModelSelection,
    PatrolRunSummary,
    PatrolTrajectorySample,
    portable_model_path,
    select_patrol_model,
    write_patrol_outputs,
)

__all__ = [
    "BenchmarkCondition",
    "BenchmarkConfig",
    "CheckpointSelectionGate",
    "EpisodeAccumulator",
    "EpisodeRecord",
    "PatrolCheckpointRecord",
    "PatrolCollisionRecord",
    "PatrolModelSelection",
    "PatrolRunSummary",
    "PatrolTrajectorySample",
    "SeedSet",
    "build_checkpoint_selection",
    "candidate_sort_key",
    "load_benchmark_config",
    "load_strict_candidate_summary",
    "make_candidate_record",
    "parse_result_identity",
    "require_test_confirmation",
    "resolve_candidate_model",
    "run_kinematic_benchmark",
    "portable_model_path",
    "select_patrol_model",
    "summarize_records",
    "write_benchmark_outputs",
    "write_checkpoint_selection",
    "write_episode_records",
    "write_patrol_outputs",
]
