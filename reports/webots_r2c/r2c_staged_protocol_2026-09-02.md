# R2C staged density-and-speed adaptation protocol

Status: **R2C1 preregistered; no R2C outcome has been observed yet**.

Date: 2026-09-02. This is a development-only continuation from the verified
RB2B endpoint. It does not authorize access to the protected formal-test split,
and no future stage or result is claimed as complete in advance.

## Why R2C is needed

The exact RB2B policy-plus-shield endpoint passed its locked 100-episode
single-obstacle validation at `85 success / 15 collision`. A later
pre-registered stress matrix on validation seeds `8000-8019` observed:

| Condition | Dynamic obstacles | Speed range | Success / collision |
|---|---:|---:|---:|
| BASE1 | 1 | `0.18-0.25 m/s` | `14 / 6` |
| DENS3 | 3 | `0.18-0.25 m/s` | `8 / 12` |
| FAST3 | 3 | `0.26-0.42 m/s` | `0 / 20` |

All three stress gates failed. The next engineering step is therefore to train
against the observed density and speed distribution gap, not to reinterpret the
stress result as a pass or tune the safety shield after seeing those outcomes.

## Locked staged path

Training seed `11`, the reward function, PPO update settings, `1.10 m x 1.10 m`
route half-lengths, and the complete 15-field front-brake shield remain fixed.
Each stage receives exactly `40,960` additional steps: 20 PPO rollouts and four
`10,240`-step checkpoint intervals.

| Stage | Obstacles | Speed range | Fresh validation seeds | Expected cumulative endpoint |
|---|---:|---:|---:|---:|
| R2C1 | 2 | `0.18-0.25 m/s` | `8200-8299` | `122,880` |
| R2C2 | 3 | `0.18-0.25 m/s` | `8300-8399` | `163,840` |
| R2C3 | 3 | `0.22-0.32 m/s` | `8400-8499` | `204,800` |
| R2C4 | 3 | `0.26-0.42 m/s` | `8500-8599` | `245,760` |

Only R2C1 is materialized now. R2C2-R2C4 are conditional designs. Every later
stage must be created in a new model variant only after the preceding stage has
passed, and its configuration must bind an immutable copy of the accepted
preceding endpoint by full SHA-256 before launch. This prevents an unavailable
future model or an outcome-dependent source choice from being hidden inside one
mutable multi-stage manifest.

Every stage uses exactly 100 validation episodes and must satisfy both locked
gates: success rate `>=0.75` and collision rate `<=0.25`, with zero evaluation
errors and complete reset-health evidence. A failed stage is archived as a valid
negative result and stops the chain. Its budget, seed window and thresholds are
not extended or relaxed after observation.

## Locked R2C1 realization

- Source endpoint:
  `artifacts/models/webots/seed_11/r2b_rebaseline_v1/ppo_webots_s11_step000081920.zip`.
- Source SHA-256:
  `498d970cfcf539d55943ace88dcbaad83445713ad852291098b391fd38a86cfa`.
- Variant-local immutable source:
  `artifacts/models/webots/seed_11/r2c_density2_v1/sources/ppo_webots_s11_r2c_density2_v1_source_r2b_rebaseline_v1_81920_steps.zip`.
- Transfer mode: full PPO resume with optimizer and cumulative timestep state;
  policy-only transfer is forbidden.
- No action-standard-deviation or action-head reset is allowed.
- Dynamic obstacles: exactly 2, each sampled at `0.18-0.25 m/s`.
- Additional budget: exactly `40,960` steps; expected endpoint `122,880`.
- Validation: previously unused development seeds `8200-8299`, exactly 100
  episodes, under the same two-obstacle and speed distribution.
- Protected formal-test seeds `10000-10019` remain untouched.

The only authorized R2C1 runner is:

```bash
.venv/bin/python scripts/run_webots_curriculum.py \
  --config configs/train/webots_r2c_density2_v1.yaml \
  --seed 11
```

The runner must start from a clean committed revision, verify the source digest
before and immediately prior to launch, preserve the optimizer, and bind the
result metadata back to that exact source. The accepted model, if any, is the
post-update `ppo_webots_s11_step000122880.zip` endpoint rather than the same-step
periodic checkpoint saved before the final PPO update.

## Evidence and claim boundary

For each observed stage, archive the four periodic checkpoints, final endpoint,
versioned metadata, training monitor, YAML/JSON curriculum manifest, raw
validation episodes, summary, runtime inventory, human-readable report and a
portable SHA-256 list. The existing BASE1/DENS3/FAST3 matrix remains a fixed
regression anchor and is not overwritten.

R2C remains a single-training-seed development curriculum on a familiar Webots
layout. A stage pass supports only a bounded policy-plus-shield engineering
claim under its locked distribution. It does not establish policy-only benefit,
unseen-layout generalization, multi-seed reliability, production readiness,
real-robot safety or a formal thesis conclusion.
