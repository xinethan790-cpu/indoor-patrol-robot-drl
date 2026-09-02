# R2b verified rebaseline: endpoint report

Status: **passed the pre-registered development gate**.

Date: 2026-09-02. This report records the conditional RB2B continuation that was
authorized only after the verified RB2A endpoint passed. It is a single-training-
seed Webots development result for the combined PPO-plus-safety-shield system,
not a formal three-seed thesis result or a policy-only comparison.

## Locked run identity

| Item | Value |
|---|---|
| Source Git revision | `5c9102b67e2abfd64741eb5660970141151ad88e` |
| Git worktree at launch | clean |
| Training seed | `11` |
| Variant / stage | `r2b_rebaseline_v1` / `RB2B` |
| Immutable RB2A source | `.../sources/ppo_webots_s11_r2b_rebaseline_v1_source_r2a_rebaseline_v1_40960_steps.zip` |
| Source SHA-256 | `f52a4691dbe7d787f6a9dfdb02383d206f597999e15f991063bc812a66706c2c` |
| Source internal steps | `40,960` |
| Transfer semantics | full resume, retaining optimizer and learned action distribution |
| Obstacle distribution | one obstacle, `0.18-0.25 m/s`, training layout `train_a` |
| Additional training budget | exactly `40,960` Webots steps |
| Endpoint cumulative steps | exactly `81,920` |
| System under evaluation | PPO plus the unchanged pinned front-brake safety shield |
| Validation grid | `6000-6099`, exactly 100 episodes |
| Protected test grid | `10000-10019`, not run |

The file-level curriculum config SHA-256 is
`f0878a3610f3c4df1396c12b71431b712729df98d503a3152105fcc4dcae96bd`.
The runner's normalized semantic-config SHA-256 is
`86e5ee467b121ccc329db14c43d9050b37022ce48edaf0dfcba251e3722a55ed`.
The pre-launch checksum list passed before Webots started. The run manifest and
metadata both bind the same source path and full source SHA-256; metadata records
`mode: webots_resume` and no initial-action-standard-deviation override.
The config, source and runtime inputs were first committed at `206631a`; the only
subsequent change before the `5c9102b` launch revision was a test implementation
that removed an optional CI import. All ten pre-launch checksum entries remained
unchanged and passed again immediately before launch.

## Training endpoint

Training completed without a retry. All `396/396` environment reset attempts
passed finite-state and supervisor readback checks. The versioned endpoint is:

- Model: `artifacts/models/webots/seed_11/r2b_rebaseline_v1/ppo_webots_s11_step000081920.zip`.
- SHA-256: `498d970cfcf539d55943ace88dcbaad83445713ad852291098b391fd38a86cfa`.
- Stable-Baselines3 internal `num_timesteps`: `81,920`.
- Metadata: `run_metadata_step000081920.yaml`, SHA-256
  `1e16d3e16166fb588f482fd58c24cff1797f355d2858249b07fef3f8e4b0966d`.

The manifest-referenced `run_metadata.yaml` alias is archived and byte-identical
to the versioned metadata. Four periodic checkpoints are retained as training
evidence, but no checkpoint search was performed: the pre-registered post-update
endpoint is the only model used for the validation gate.

| Internal step | Checkpoint SHA-256 |
|---:|---|
| 51,200 | `64994ebb3d3a60504201aa8563bed9e9fe378fade09e41f1ffd19fededdad76e` |
| 61,440 | `b7f58631a44df201657481b373cb7ecca2a9f7e7e59d0f60e214d4923e6aa6ba` |
| 71,680 | `e2488f504f365fd99ca825aa656d8d227618c5408b9dc61752812f3082801cc4` |
| 81,920 | `7da8a74ca2a73cd7833bca8567532f5a36400978e8e9c7f2af9060ad59784f0d` |

The 81,920-step periodic checkpoint was saved after the final rollout reached the
step count but before that rollout's last PPO gradient update. The endpoint was
saved after `learn()` returned. Both report `num_timesteps=81920`, but all 13
policy-state tensors differ; validation correctly binds the post-update endpoint.

## Validation result

The strict loader independently re-read the raw CSV and confirmed one row for
every integer seed from 6000 through 6099, with one planner, one condition and no
error rows. Validation resets passed `100/100` checks.

| Episodes | Success | Collision | Timeout | Error | Success rate | Collision rate | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | **85** | **15** | 0 | 0 | **0.8500** | **0.1500** | **Pass** |

The locked gate was success rate `>=0.75` and collision rate `<=0.25`. Both
criteria pass with `0.10` absolute margin, equivalent to ten episodes.

Secondary descriptive metrics from the same raw rows are:

| Metric | Mean / percentile |
|---|---:|
| Successful completion time | `14.4023 s` mean |
| Successful path-length ratio | `0.9862` mean |
| Minimum clearance | `0.2217 m` mean |
| Near-collision step ratio | `0.06563` mean |
| Squared action delta | `0.05114` mean |
| Inference latency | `0.6054 ms` p95 |

Raw evidence:

- `episodes.csv`: SHA-256
  `21e4f9b5e0f1147a00c80b29e17ab72b725f0b51666c3060422854cb019ea273`.
- `summary.json`: SHA-256
  `19a6419aefca1a5ad4b3d59b12a1b62a4ad9d165a76d398465715c5a5d40e1df`.
- `run_manifest.yaml`: SHA-256
  `ad3eda948a00c1f9bea7016bae56c6d9f623c6ef124bee43222e0ef3ed414590`.
- `run_manifest.json`: SHA-256
  `c6bdeeabe372041eaba5783f53ed87e8cfdda35477832594f8121206c2d92cd4`.

The summary binds the CSV and endpoint model hashes. YAML and JSON manifests are
semantically identical and record `status: completed`, validation `passed: true`,
the exact launch revision, source-model binding and `test_split_was_run: false`.
The metadata records `git.dirty: false`. The complete portable checksum list is
`r2b_rebaseline_v1_artifacts.sha256`.

## Descriptive RB2A/RB2B context

| Stage | Speed range | Success / collision | Success rate | Collision rate |
|---|---:|---:|---:|---:|
| RB2A endpoint | `0.10-0.20 m/s` | `76 / 24` | `0.76` | `0.24` |
| RB2B endpoint | `0.18-0.25 m/s` | `85 / 15` | `0.85` | `0.15` |

Across the shared seed identifiers, 65 cases succeeded in both stages, 20 changed
from RB2A collision to RB2B success, 11 changed from RB2A success to RB2B collision,
and four collided in both. This is useful failure-analysis context, not a paired
causal improvement claim: each seed samples a different physical speed because
the stage ranges differ, and the RB2B model received additional training.

## Result boundary

This result shows that the exact RB2A model-plus-shield system can be continued
under the registered faster-obstacle distribution with a traceable endpoint and
a stronger gate margin. It does not establish multi-seed statistical reliability,
safety-shield-free policy quality, unseen-layout or multi-obstacle generalization,
multi-point patrol performance, real-robot safety, or a formal thesis conclusion.
The CSV proves that the pinned shield configuration was enabled, but it does not
contain a per-episode shield-intervention aggregate and therefore cannot quantify
the shield's separate contribution. The validation grid is a development gate
reused by design; the protected test seeds remain untouched.
