# R2a targeted fine-tuning: endpoint result

> Integrity status (2026-09-02): **historical validation record; not the active
> baseline**. Its downstream outcome and artifact identity remain verifiable, but
> its complete upstream training provenance is not clean-clone reproducible. It
> must not be presented as the current result or an end-to-end reproducible gain.
> See `rebaseline_protocol_2026-09-02.md`.

This report records the completed R2a training endpoint before checkpoint selection.
It is an exploratory single-obstacle result and is not evidence that multi-point patrol
or multi-obstacle avoidance has been solved.

## Locked protocol

- Training seed: `11`
- Source checkpoint SHA-256:
  `70e84ff352925f09c1dbbae74750f6ff2a7096d4edc303e5aa4dc06cc7289055`
- Transfer mode: policy weights only, fresh PPO optimizer
- Variant: `r2a_targeted_v1`
- Additional steps: `40,960`
- Obstacle count: `1`
- Obstacle speed: `0.10-0.20 m/s`
- Initial action standard deviation: `0.45`
- Validation seeds: `5000-5099`
- Promotion gate: success rate `>= 0.75`, collision rate `<= 0.25`

## Endpoint result

The immutable final model at step `40,960` produced:

- Success: `61/100` (`0.61`)
- Collision: `39/100` (`0.39`)
- Timeout: `0/100`
- Evaluation error: `0/100`
- Gate result: **failed**
- Final model SHA-256:
  `2a87eb5e209a5dc80cbc619b892d4424e6f3928198a0aaf48d1d941d003e1e10`

All `421` training resets and all `100` evaluation resets passed the finite-state,
settle, and readback checks.  The training metadata records Git revision
`358b34cab6cb1f6d0cebcb488519af56fa9e15ae` with a clean worktree.

## Immutable checkpoint sweep

All four checkpoints were subsequently evaluated on the same 100 episodes:

| Checkpoint | Success | Collision | Timeout | Error | Gate |
|---:|---:|---:|---:|---:|:---:|
| 10,240 | 61/100 | 39/100 | 0 | 0 | Fail |
| 20,480 | 61/100 | 39/100 | 0 | 0 | Fail |
| 30,720 | 62/100 | 38/100 | 0 | 0 | Fail |
| 40,960 | 60/100 | 40/100 | 0 | 0 | Fail |

The validation-only selector ranked step `30,720` first, but selected no formal
candidate because every checkpoint missed both locked thresholds.  R2b therefore
remains blocked.

The first targeted hypothesis did not improve the legacy R1 reference result of
`63/100` success on the broader R2 evaluation.  Because every R2a-v1 episode ended
in either success or collision and no timeout was observed, the next preregistered
single-variable ablation increases only the proximity-warning distance from
`0.60 m` to `0.70 m`.  Source model, optimizer reset, action standard deviation,
reward weights, PPO settings, training length, and validation seeds remain fixed.
