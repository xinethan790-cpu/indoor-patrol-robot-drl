# R2a verified rebaseline: endpoint report

Status: **passed the pre-registered development gate**.

Date: 2026-09-02. This report records the first complete result produced under
the verified R2a/R2b rebaseline protocol. It is a single-training-seed Webots
development result for the combined PPO-plus-safety-shield system, not a formal
three-seed thesis result or a policy-only comparison.

## Locked run identity

| Item | Value |
|---|---|
| Source Git revision | `3335fa30063890a634d55c3e16f7f247f1300a1b` |
| Git worktree at launch | clean |
| Training seed | `11` |
| Variant / stage | `r2a_rebaseline_v1` / `RB2A` |
| Frozen policy source | `artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip` |
| Frozen source SHA-256 | `e01bfcfa5dba22db4a8adf8b0ae8b325a6f7ef7eb50c2b1af1840e3d0c079c67` |
| Transfer semantics | policy weights copied into a fresh PPO optimizer |
| Obstacle distribution | one obstacle, `0.10-0.20 m/s`, training layout `train_a` |
| Training budget | exactly `40,960` Webots steps |
| System under evaluation | PPO plus the pinned front-brake safety shield |
| Validation grid | `6000-6099`, exactly 100 episodes |
| Protected test grid | `10000-10019`, not run |

The file-level curriculum config SHA-256 is
`20bbea28a8f94719dfc29abd0b645334c350f5ba35c8bf4cc678614362779d48`.
The runner's normalized semantic-config SHA-256 is
`2e35a87012caf2648bd2f27b0037fc1d98aa3cc068fa94802b23b1132fddc2ea`,
and the warm-start requirement SHA-256 is
`bae50260f11e8fa0a576e13c91e62d909992c0cc3125246eb3819eb78d7f34bc`.
The frozen input is byte- and validation-bound, but its own earlier training lacks
a complete clean-revision manifest. It is therefore an explicit warm-start input,
not a claim that the pre-Webots training chain has become reproducible.

## Training endpoint

Training completed without a retry. All `370/370` environment reset attempts
passed the finite-state and supervisor readback checks. The versioned endpoint is:

- Model: `artifacts/models/webots/seed_11/r2a_rebaseline_v1/ppo_webots_s11_step000040960.zip`.
- SHA-256: `f52a4691dbe7d787f6a9dfdb02383d206f597999e15f991063bc812a66706c2c`.
- Stable-Baselines3 internal `num_timesteps`: `40,960`.
- Metadata: `run_metadata_step000040960.yaml`, SHA-256
  `944c21a52590b42a1af814ea6c10d0ed5d4b826ead3bec760ea34d4a3f28bb1e`.

The manifest-referenced `run_metadata.yaml` alias is archived as well and is
byte-identical to the versioned metadata above.

The four periodic checkpoints at 10,240-step intervals are retained as training
evidence. They were not evaluated or searched to choose a better result: the
pre-registered endpoint model above is the only model used for the gate.

| Internal step | Checkpoint SHA-256 |
|---:|---|
| 10,240 | `60f5dbb5a67ff23372926db71bef2ef563fa4c8eb632dc54c88eee3905177765` |
| 20,480 | `f552d9d37e48a5835a1b9eea28c0486ea3c911ce6332d513f96ad8e02bb9eec5` |
| 30,720 | `bab65513b75240ecfc6eec1ac16dd41979b68f1ba3de13acbac9acc3d158284a` |
| 40,960 | `97978f9419f583d75659eab014a89ea34f03c6053ed3060b284018133a9d74c9` |

The final endpoint archive and the periodic 40,960-step checkpoint are distinct
models, not duplicate ZIP packaging. Stable-Baselines3's checkpoint callback saved
the latter after the final rollout had reached step 40,960 but before that rollout's
last PPO gradient update; the endpoint was saved after `learn()` returned. Both
load with `num_timesteps=40960`, but their learned tensors differ. The post-update
endpoint archive is the pre-registered and hash-bound evaluation model.

## Validation result

The strict loader independently re-read the raw CSV and confirmed one row for
every integer seed from 6000 through 6099, with one planner, one condition and no
error rows. Validation resets passed `100/100` checks.

| Episodes | Success | Collision | Timeout | Error | Success rate | Collision rate | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | **76** | **24** | 0 | 0 | **0.7600** | **0.2400** | **Pass** |

The locked gate was success rate `>=0.75` and collision rate `<=0.25`. Both
criteria pass by only `0.01`, equivalent to one episode. This is a valid pass but
a narrow margin; it is not evidence of a robust generalization reserve.

Secondary descriptive metrics from the same raw rows are:

| Metric | Mean / percentile |
|---|---:|
| Successful completion time | `14.6105 s` mean |
| Successful path-length ratio | `0.9902` mean |
| Minimum clearance | `0.2503 m` mean |
| Near-collision step ratio | `0.06963` mean |
| Squared action delta | `0.04997` mean |
| Inference latency | `0.5421 ms` p95 |

Raw evidence:

- `episodes.csv`: SHA-256
  `0ff2c47e92e8d707a2f73fc7c7e77615cc0904e3013b58dfce3fa8439829d859`.
- `summary.json`: SHA-256
  `1f3fe8a8f0fc14f310dbfa2b65e86e790bd1df6cd204fa74d1afb50acaa09200`.
- `run_manifest.yaml`: SHA-256
  `529c58c8fb74d7ae2d4a898d1b8af266bf6a2f5c2684cce62d622ce745c01503`.
- `run_manifest.json`: SHA-256
  `5195e9a2d6d1198ae06acb69999f28cde24cc8690c5548ab6d57969483a71aca`.

The summary binds the CSV SHA-256 and endpoint model SHA-256. The schema-v2 run
manifest records `status: completed`, validation `passed: true`, the exact source
Git revision, and `test_split_was_run: false`; the versioned metadata additionally
records `git.dirty: false`. The complete portable checksum list is
`r2a_rebaseline_v1_artifacts.sha256`.

## Result boundary and next transition

This result establishes a cleanly traceable development baseline on the specified
single-obstacle speed distribution. It does not establish multi-seed statistical
reliability, safety-shield-free policy quality, unseen-layout generalization,
multi-point patrol performance, real-robot safety, or a formal thesis conclusion.

Because RB2A passed, RB2B may now be configured. Its source must be an immutable
copy of the exact endpoint bytes above, bound by the full
`f52a4691...06c2c` SHA-256 before training starts. RB2B will retain the optimizer
and learned action distribution, change only the obstacle-speed distribution to
`0.18-0.25 m/s`, add exactly 40,960 steps, and reuse the locked validation grid.
No RB2B outcome is claimed by this report.
