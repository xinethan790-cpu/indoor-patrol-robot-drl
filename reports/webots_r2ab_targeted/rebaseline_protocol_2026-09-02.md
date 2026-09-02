# R2a/R2b verified rebaseline protocol

Status: **RB2A completed and passed; RB2B is authorized but not yet claimed**.

Outcome update (2026-09-02): the pre-registered RB2A endpoint completed from the
clean source revision and produced `76 success / 24 collision / 0 timeout / 0
error` on the exact 100-seed validation grid. Both locked gates pass by `0.01`.
See [`r2a_rebaseline_v1_report.md`](r2a_rebaseline_v1_report.md) for the bound
model, raw evidence, hashes and limitations. The protocol text below remains the
pre-observation design; this outcome note does not redefine its thresholds.

Date: 2026-09-02. This protocol supersedes the legacy R2a/R2b chain as the active
training path. The legacy files are retained as an audit trail, but their claimed
end-to-end provenance is withdrawn because the exact upstream warm-start model and
its declared validation CSV/summary cannot be recovered.

## Why a new baseline is required

The integrity review found two incompatible kinematic source records:

| Item | Legacy claim | Frozen input archived for the new run |
|---|---|---|
| stage-2 model SHA-256 | `152060175d9f...7372f8c` | `e01bfcfa5dba...c079c67` |
| PPO validation outcome | `61 success / 19 collision` | `55 success / 25 collision` |
| summary/CSV directory | `kinematic_validation_retrained_20260830` | `kinematic_validation_seed11` |

The legacy R2a/R2b endpoint files remain useful historical and demonstration
artifacts, but downstream hashes cannot repair a missing upstream identity. They
must not be quoted as the current thesis result or used to claim a reproducible
improvement.

There is also a direct lineage gap: the old R2a-v2 metadata names Webots checkpoint
`70e84ff35292...89055` as its policy source, but that source and its curriculum-v2
training chain were local ignored artifacts rather than a complete clean-clone
evidence bundle. The new run therefore does not resume any old Webots, R2a, or R2b
weight. It starts a fresh Webots optimizer directly from the separately archived
kinematic frozen input below.

## Locked R2a rebaseline

- Training seed: `11` (development evidence, not the three-seed formal study).
- Source model: `artifacts/models/ppo_curriculum_seed11_stage2_nominal_dynamic.zip`.
- Source SHA-256: `e01bfcfa5dba22db4a8adf8b0ae8b325a6f7ef7eb50c2b1af1840e3d0c079c67`.
- Source validation: `G0-G3 × seeds 5000-5019`, PPO `55/25/0/0` over 80 episodes.
- Webots variant: `r2a_rebaseline_v1`; old variant directories are never reused.
- Transfer mode: policy weights into a fresh PPO optimizer.
- Training: one dynamic obstacle at `0.10-0.20 m/s`, `40,960` steps.
- System under evaluation: PPO plus the fully pinned front-brake safety shield,
  enabled during both training and validation.
- Validation: previously unused seeds `6000-6099`, exactly 100 episodes.
- Gate: success rate `>= 0.75`, collision rate `<= 0.25`.
- Held-out seeds `10000-10019` remain untouched.

The only authorized R2a runner is:

```bash
.venv/bin/python scripts/run_webots_curriculum.py \
  --config configs/train/webots_r2a_rebaseline_v1.yaml \
  --seed 11
```

Before Webots starts, the runner must pass the warm-start preflight, start from a
clean committed revision, and write a new schema-v2 curriculum manifest. The
model, step metadata, manifest, summary and episode CSV are archived together and
hashed after completion. A failed gate remains a valid negative result and blocks
R2b; thresholds are not relaxed after observing the output.

## Conditional R2b transition

R2b is configured only if R2a passes. Its configuration will bind an immutable
copy of the accepted R2a model by full SHA-256, retain the optimizer and learned
action distribution, add `40,960` steps at obstacle speed `0.18-0.25 m/s`, keep
the same pinned shield, and use the same fixed `6000-6099` validation grid. R2b
uses a new variant and output directory. R2a and R2b results describe different
speed distributions and are not a paired improvement comparison.

## Evidence boundary

This rebaseline is a single-training-seed, single-obstacle Webots development
experiment. Passing it supports a bounded policy-plus-shield engineering claim;
it does not establish a policy-only advantage, multi-point patrol success,
unseen-layout generalization, real-robot safety, or the formal three-seed thesis
conclusion.
