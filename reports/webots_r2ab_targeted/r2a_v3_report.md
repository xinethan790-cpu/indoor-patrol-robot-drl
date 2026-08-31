# R2a-v3 collision-penalty ablation

R2a-v3 changed only the terminal collision penalty from `-30` to `-40`.
All other source, PPO, reward, scenario, and gate values remained identical to
R2a-v2.

## Training provenance

- Source SHA-256:
  `70e84ff352925f09c1dbbae74750f6ff2a7096d4edc303e5aa4dc06cc7289055`
- Variant: `r2a_targeted_v3`
- Git revision: `330d09476fff89a885acb6855007e0e6469833e1`
- Training resets: `409/409` successful
- Training length: `40,960` steps
- Validation seeds per candidate: `5000-5099`
- Gate: success `>= 0.75`, collision `<= 0.25`

## Results

| Candidate | Success | Collision | Timeout | Error | Gate |
|---:|---:|---:|---:|---:|:---:|
| checkpoint 10,240 | 61/100 | 39/100 | 0 | 0 | Fail |
| checkpoint 20,480 | 63/100 | 37/100 | 0 | 0 | Fail |
| checkpoint 30,720 | 61/100 | 39/100 | 0 | 0 | Fail |
| checkpoint 40,960 | 59/100 | 41/100 | 0 | 0 | Fail |
| final 40,960 | 60/100 | 40/100 | 0 | 0 | Fail |

The strict selector ranked checkpoint `20,480` first but selected no promotable
model. Compared with R2a-v2's best checkpoint (`64/100`), the larger terminal
penalty lost one success and did not reduce collisions. The endpoint result also
fell from `64/100` to `60/100`.

This ablation closes the pure terminal-penalty tuning path: all four checkpoints
and the final model failed the preregistered gate, with no timeout or evaluator
errors that could explain the result. R2b therefore stays blocked. The next step
is an explicit runtime safety-constraint diagnostic on the strongest R2a-v2
checkpoint, evaluated on the same fixed validation seeds before any further
training or formal model promotion.
