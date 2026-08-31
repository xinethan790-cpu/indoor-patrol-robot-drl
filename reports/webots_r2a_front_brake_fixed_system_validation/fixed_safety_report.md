# webots_r2a_front_brake_fixed_system_validation_v1 fixed-system safety report

Decision: **PASS** — `admit_r2a_front_brake_system_for_r2b_preregistration`.

> This validates one model-plus-shield system. It never creates a policy-only alias
> and does not promote a formal-test model.

## Recomputed outcomes

| Episodes | Success | Collision | Timeout | Error | Success rate | Collision rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 88 | 12 | 0 | 0 | 0.880000 | 0.120000 |

## Locked gate

| Check | Observed | Requirement | Result |
| --- | ---: | ---: | :---: |
| `minimum_success_rate` | 0.88 | >= 0.75 | PASS |
| `maximum_collision_rate` | 0.12 | <= 0.25 | PASS |
| `maximum_timeout_count` | 0 | <= 2 | PASS |
| `maximum_error_count` | 0 | <= 0 | PASS |

## Audit bindings

- Model SHA-256: `051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0`
- Profile source SHA-256: `6acd0b2bcafa7ffe48ac800ab94cf13636bc4a7c2dff2fb4e50fd7381262acf3`
- Profile semantic SHA-256: `354ce9b831f01fb9e9c72a9ad1650574f18c98b1b5fafea26671f81ee000ec08`
- Paired diagnostic report SHA-256: `728b23bf0729ecf2a75d2c473ccecf5e1eda17cab17e12e05123e708d3018a97`
- Sweep manifest SHA-256: `b1a2f0727ef5836ee06d4855282dfe2d4087215f15de93c675d02251bbaa7f49`
- Summary SHA-256: `45744291e93db2e53f1b201a3d834ee196b37c5e37bb017221e4b9c07fad2776`
- Episodes CSV SHA-256: `e7546e5f91cec654a9ccba9c82c2045915f8095a8a32c386ffec12e5e38819a9`
- Git revision: `df4a9f27193fcc524fc770190779c45759b9ec59`
