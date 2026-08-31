# webots_r2a_front_brake_paired_diagnostic_v1 diagnostic report

Decision: **PASS** — `retain_front_brake_profile_for_fixed_r2a_validation`.

> Validation-only exploratory safety diagnostic. Formal model promotion is forbidden; no model alias was created.

## Arm outcomes

| Arm | Success | Collision | Timeout | Error | Success rate | Collision rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 79 | 41 | 0 | 0 | 0.658333 | 0.341667 |
| candidate | 108 | 12 | 0 | 0 | 0.900000 | 0.100000 |

## Paired analysis

- Rescue / regression: 30 / 1
- Common success / common collision: 78 / 11
- Relative collision reduction: 0.707317073171
- Exact one-sided McNemar p: 1.49011611938e-08
- Common-success completion-time ratio median / p90: 1 / 1.10666666667

## Preregistered gate

| Check | Observed | Requirement | Result |
| --- | ---: | ---: | :---: |
| `minimum_success_rate` | 0.9 | >= 0.75 | PASS |
| `maximum_collision_rate` | 0.1 | <= 0.25 | PASS |
| `minimum_relative_collision_reduction` | 0.707317073171 | >= 0.25 | PASS |
| `exact_one_sided_mcnemar_significance_level` | 1.49011611938e-08 | <= 0.05 | PASS |
| `maximum_new_timeout_count` | 0 | <= 2 | PASS |
| `maximum_median_completion_time_ratio` | 1 | <= 1.15 | PASS |
| `minimum_common_success_count` | 78 | >= 24 | PASS |
| `maximum_error_count_per_arm` | 0 | <= 0 | PASS |

## Audit bindings

- Input config SHA-256: `7be127f48cccc3de6e3c9118074f3fc9e146cf17c75570397d7fea5fbe4126bd`
- Pre-registration revision: `db634faf918ace1e6dc97f922771b44f323c6f45`
- Pre-registration Git blob: `cf67268577576397a705fcf5bec4cdb6cd9c6317`
- Pre-registration committed at: `2026-08-31T11:33:50+08:00`
- Model SHA-256: `051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0`
- World SHA-256: `e0ae7ac8aa2a7981f94673d656d76940166cb12ceab8dd2b17a17329985d5425`
- Controller SHA-256: `91f8ab0b91a5fdb56aa616822544bceaa85ac747be2b10b79d2e2487cb378fdb`
- Git revision: `a3676e9b8183d482581d42f68c531fb53ed4087d`

## Provenance boundary

The sweep binds the evaluated model, profile, world, controller, summaries, and episode CSV files. It did not record a cryptographic snapshot of the evaluation-time worktree, the imported Webots environment adapter, or the Webots executable.
