# R2a targeted fine-tuning: endpoint result

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

R2b is therefore blocked at this point.  The four immutable R2a checkpoints at
steps `10,240`, `20,480`, `30,720`, and `40,960` must be evaluated with the same
100 validation episodes before deciding whether R2a has a promotable candidate.
