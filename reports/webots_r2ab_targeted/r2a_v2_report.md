# R2a-v2 early-warning distance ablation

> Integrity status (2026-09-02): **historical validation record; not the active
> baseline**. Its downstream outcome and artifact identity remain verifiable, but
> its complete upstream training provenance is not clean-clone reproducible. It
> must not be presented as the current result or an end-to-end reproducible gain.
> See `rebaseline_protocol_2026-09-02.md`.

R2a-v2 changed only the proximity-speed warning distance from `0.60 m` to
`0.70 m`.  All other source, PPO, reward, scenario, and gate values remained
identical to R2a-v1.

## Training provenance

- Source SHA-256:
  `70e84ff352925f09c1dbbae74750f6ff2a7096d4edc303e5aa4dc06cc7289055`
- Variant: `r2a_targeted_v2`
- Git revision: `921d54b5f21c3d91b6585db90ff27f5a22420b53`
- Training resets: `422/422` successful
- Training length: `40,960` steps
- Validation seeds per candidate: `5000-5099`
- Gate: success `>= 0.75`, collision `<= 0.25`

## Results

| Candidate | Success | Collision | Timeout | Error | Gate |
|---:|---:|---:|---:|---:|:---:|
| checkpoint 10,240 | 64/100 | 36/100 | 0 | 0 | Fail |
| checkpoint 20,480 | 63/100 | 37/100 | 0 | 0 | Fail |
| checkpoint 30,720 | 62/100 | 38/100 | 0 | 0 | Fail |
| checkpoint 40,960 | 62/100 | 38/100 | 0 | 0 | Fail |
| final 40,960 | 64/100 | 36/100 | 0 | 0 | Fail |

The strict selector ranked checkpoint `10,240` first but selected no promotable
model.  Compared with R2a-v1's best checkpoint (`62/100`), earlier warning gained
two successes, but it remains eleven successes below the promotion threshold.

R2b stays blocked.  The next single-variable ablation retains the R2a-v2 setup and
changes only the terminal collision penalty from `-30` to `-40`.  If that does not
produce a material improvement, further terminal-penalty tuning will stop in favor
of an explicit runtime safety constraint.
