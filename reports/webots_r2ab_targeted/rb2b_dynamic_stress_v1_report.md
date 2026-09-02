# RB2B dynamic-stress characterization report

Status: **all three development gates failed; no model promotion is permitted**.

Date: 2026-09-02. This report describes a validation-only, post-training stress
matrix for the exact RB2B endpoint. It characterizes the combined PPO policy and
pinned front-brake safety shield on one familiar Webots layout; it is not a formal
test result, a policy-only ablation, or evidence of real-robot safety.

## Protocol and evidence integrity

The matrix was pre-registered and executed from clean Git revision
`75af6a8e1096a366f0043572f53b3f40b634024a`. The exact endpoint model was fixed
before observation at cumulative step `81,920`, SHA-256
`498d970cfcf539d55943ace88dcbaad83445713ad852291098b391fd38a86cfa`.
All three conditions used training seed `11`, validation seeds `8000-8019`, the
`train_a` layout, `1.10 m x 1.10 m` route half-lengths, deterministic inference,
action repeat `4`, at most `300` episode steps, and the same 15-field front-brake
shield configuration, including `lidar_max_range=3.5 m`.

| Condition | Dynamic obstacles | Obstacle speed | Purpose |
|---|---:|---:|---|
| BASE1 | 1 | `0.18-0.25 m/s` | RB2B distribution anchor |
| DENS3 | 3 | `0.18-0.25 m/s` | Density-only increase from BASE1 |
| FAST3 | 3 | `0.26-0.42 m/s` | Speed-only increase from DENS3 |

The 60 Webots episodes ran once. FAST3 produced zero successes, for which
successful-completion time and successful-path ratio are mathematically
undefined and were correctly serialized as `null`. The original strict parser
mistakenly rejected those nullable metrics and marked the FAST3 sweep manifest
failed even though Webots exited with code `0` and all 20 episode rows were
complete. This was a software-validation defect, not an evaluator or reset error.

Revision `b1dbaa9464e93acd4abe9ef9204353c1fc8ab7cf` corrected that parser contract and
performed read-only finalization. It launched **zero** Webots processes, rewrote
none of the three original sweep manifests, and verified that all nine original
manifest/summary/CSV artifacts retained their pre-finalization SHA-256 values.
Therefore this is a software erratum with post-hoc revalidation, **not a rerun,
adaptive extension, failed-episode exclusion, or outcome-dependent retry**.

The canonical aggregate evidence is
`artifacts/results/webots_rb2b_posthoc_dynamic_stress_v1/dynamic_stress_report.json`,
SHA-256
`45dd37acb9cdb1378cf70247a1999fc9c928a05031fbae98f0d7919cabaf3e18`.
It records `test_split_was_run: false`, `no_rerun: true`, and the separate
execution and finalization revisions.

## Primary outcomes and locked gates

Each condition independently required success rate `>=0.75`, collision rate
`<=0.25`, exactly 20 episodes, zero evaluation errors, and `20/20` successful
reset checks. The intervals below are two-sided 95% Wilson score intervals for a
binomial proportion (`n=20`); they are descriptive and are not multiplicity-
adjusted hypothesis tests.

| Condition | Success | Success rate (Wilson 95% CI) | Collision | Collision rate (Wilson 95% CI) | Timeout | Error | Reset | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| BASE1 | 14/20 | `0.7000` (`0.4810-0.8545`) | 6/20 | `0.3000` (`0.1455-0.5190`) | 0 | 0 | 20/20 | **Fail** |
| DENS3 | 8/20 | `0.4000` (`0.2188-0.6134`) | 12/20 | `0.6000` (`0.3866-0.7812`) | 0 | 0 | 20/20 | **Fail** |
| FAST3 | 0/20 | `0.0000` (`0.0000-0.1611`) | 20/20 | `1.0000` (`0.8389-1.0000`) | 0 | 0 | 20/20 | **Fail** |

BASE1 missed both locked rate thresholds by one episode. DENS3 and FAST3 failed
by substantially larger margins. The zero timeout/error counts and complete reset
checks show that these negative outcomes are observed collision behavior, not
missing data or an evaluator-infrastructure failure.

## Paired seed transitions

The same 20 scenario-seed identifiers were used in every condition. Because all
episodes ended in either success or collision, the complete paired transitions
are:

| BASE1 outcome -> DENS3 outcome | Count |
|---|---:|
| Success -> success | 7 |
| Success -> collision | 7 |
| Collision -> success | 1 |
| Collision -> collision | 5 |

Increasing only the obstacle count from one to three reduced success by 6/20
episodes (`0.70` to `0.40`): seven seeds regressed from success to collision while
one moved in the opposite direction.

| DENS3 outcome -> FAST3 outcome | Count |
|---|---:|
| Success -> success | 0 |
| Success -> collision | 8 |
| Collision -> success | 0 |
| Collision -> collision | 12 |

Increasing only the speed range with three obstacles reduced success by another
8/20 episodes (`0.40` to `0.00`): every DENS3 success became a FAST3 collision,
and every DENS3 collision remained a collision. These paired counts localize the
observed robustness boundary within this simulator protocol. BASE1 -> DENS3 and
DENS3 -> FAST3 are controlled single-variable contrasts under the locked reset
generator, but 20 development seeds on one familiar layout do not establish a
population-level causal or generalization claim.

## Secondary descriptive metrics

| Metric | BASE1 | DENS3 | FAST3 |
|---|---:|---:|---:|
| Successful completion time, mean | `13.1017 s` | `17.9040 s` | undefined (0 successes) |
| Successful path-length ratio, mean | `0.9698` | `1.0076` | undefined (0 successes) |
| Minimum clearance, all-episode mean | `0.1936 m` | `0.0468 m` | `0.0000 m` |
| Near-collision step ratio, all-episode mean | `0.08432` | `0.16639` | `0.17072` |
| Squared action delta, all-episode mean | `0.05345` | `0.06181` | `0.07585` |
| Per-episode inference-latency p95, aggregate p95 | `0.4994 ms` | `0.4108 ms` | `0.4091 ms` |

Completion time and path-length ratio are averaged only over successful episodes;
FAST3's values are undefined rather than zero. The remaining means use all 20
episodes. The minimum-clearance and near-collision trends are consistent with the
primary collision counts, while the action-delta increase is descriptive only and
does not isolate a mechanism. Latency is reported for runtime completeness, not as
evidence of avoidance quality.

## Supported and unsupported conclusions

This matrix supports the following bounded conclusions:

- The exact RB2B endpoint plus pinned front-brake shield does not meet the locked
  development gate in any of the three stress conditions.
- Under the fixed `train_a` protocol, the observed success rate falls from `0.70`
  to `0.40` when obstacle count increases from one to three at the same speed
  range, then to `0.00` when the three-obstacle speed range is increased.
- The complete resets, absence of evaluator errors, fixed model/profile hashes,
  and unchanged raw artifacts make this a usable negative engineering result: it
  identifies multi-obstacle density and speed as the next robustness bottlenecks.

It does **not** support any of the following claims:

- that the stress gate passed, the current RB2B model is promotable, or indoor
  dynamic avoidance is solved;
- formal-test performance, held-out-layout generalization, multi-training-seed
  statistical reliability, production readiness, or real-robot safety;
- a policy-only benefit or a separate safety-shield contribution, because the
  evaluated system always combined both components;
- a single-variable BASE1-versus-FAST3 attribution, because those endpoints differ
  in both obstacle count and speed;
- a formal significance or population claim from 20 development seeds and Wilson
  intervals alone.

The protected formal-test split remains untouched, and formal model promotion is
forbidden by the aggregate decision.

## Recommended R2C development path

The appropriate next step is a new, pre-registered **R2C multi-obstacle staged
training curriculum**, not a reinterpretation of the current matrix as a pass.
A defensible progression is:

1. introduce two obstacles at `0.18-0.25 m/s`;
2. progress to three obstacles at the same speed range;
3. only after density competence is stable, progress three obstacles toward the
   `0.26-0.42 m/s` FAST3 range.

Before training, R2C should lock the source-model SHA-256, stage budgets and
transition rules, obstacle distributions, safety-shield treatment, new unused
development seeds, endpoint-selection rule, and per-stage gates. The existing
BASE1, DENS3 and FAST3 conditions should remain fixed regression anchors; shield
changes, if investigated, should be a separately registered ablation. Failed
stages must remain reportable negative results without threshold relaxation or
outcome-dependent retries. Multiple independent training seeds and protected
formal testing should be considered only after a traceable R2C endpoint passes
its development gates.

As of this report, **R2C is a recommendation only**. No R2C model has been trained
or evaluated, and no future pass is claimed.
