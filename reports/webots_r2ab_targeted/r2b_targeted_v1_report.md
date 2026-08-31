# R2b medium-high-speed targeted fine-tuning

Date: 2026-08-31. This report records a development-stage Webots validation
result. It is not a held-out test result and must not be presented as the final
three-seed thesis conclusion.

## Objective and fixed inputs

R2b continued the accepted R2a-v2 model-plus-shield system on a harder
single-dynamic-obstacle distribution. The obstacle speed range increased to
`0.18–0.25 m/s`; the route half-lengths remained `1.10 m × 1.10 m`. The complete
R2a front-brake safety profile stayed enabled during both training and validation.

- Training code revision: `421df4b452533f9fde9b65e4ccde3120fc67ebeb`
- Git state recorded by the trainer: clean
- Training configuration SHA-256:
  `44151f43618e84e7458ddb674ea674a0fa3971aeef9767e19f84b2a992b17cb4`
- Semantic curriculum SHA-256:
  `a53f6f1490e2a065e8d670ef4dc60f7541eee41be84ea9ef60b9d91879ca854f`
- Immutable resume-model SHA-256:
  `051d1e2f07bb147d247c91ded821e58af5b87987bedc2cee4e287fdbdfcdcbe0`
- Resume step: `10,240`
- Additional training: `40,960` steps
- Final cumulative step: `51,200`
- PPO rollout/batch: `2,048 / 128`
- Initial action-standard-deviation override: none; the learned resume state and
  optimizer were retained.
- Training reset-health checks: `368/368` successful.

## Fixed validation result

The automatic gate used exactly 100 validation seeds, `5000–5099`, and required
success `>= 0.75` and collision `<= 0.25`. The evaluator re-bound the exact model,
training seed, variant, scenario seeds, condition, and all 15 safety-shield
configuration fields before accepting the summary.

| Outcome | Count | Rate |
|---|---:|---:|
| Success | 83 | **0.83** |
| Collision | 17 | **0.17** |
| Timeout | 0 | 0.00 |
| Error | 0 | 0.00 |

The stage passed both gates: success margin `+0.08` and collision margin `0.08`
below the maximum. Additional aggregate diagnostics were:

- mean successful completion time: `14.435 s`;
- mean successful path-length ratio: `1.0267`;
- mean minimum clearance: `0.2157 m`;
- mean near-collision ratio: `0.0649`;
- mean squared action delta: `0.03827`;
- inference latency p95: `1.004 ms`.

## Artifact bindings

- Final model SHA-256:
  `e782034baf09d43b28ac00c08a6bd52ed1e1847a6b76b52858271dc8ab548861`
- Training metadata SHA-256:
  `970851226a00c2eead9af5af0f1208d75b1a5b9a55851cf6d15d20f9442a70ec`
- Curriculum manifest YAML SHA-256:
  `98ab6b3e66d796bd6208909950e1caa1c79bc08061f7a0c5932cb7aaaafb9921`
- Curriculum manifest JSON SHA-256:
  `c9a126cb1625402751d32bdf7969d1b763bb0a3dab1dbf869012441166a43560`
- Episode CSV SHA-256:
  `9e57bf97b98c0fe6d6d47aab27e7e0efd8600ae772318700cd59c3efb293256e`
- Validation summary SHA-256:
  `d276405e9d2655493586a9892cc6b4ed2a9a559cd097f190c451442eab722dd0`

The immutable final model, step-specific metadata, both curriculum manifests,
per-episode CSV, and summary are archived with this report.

## Interpretation and boundary

R2b demonstrates that the accepted R2a policy-plus-safety system can be continued
to a higher obstacle-speed range and still pass the pre-registered development
gate. Its `83/17` result must not be compared as a paired improvement or regression
against R2a's `88/12`, because the two validations use different obstacle-speed
distributions. This is also one training seed, one dynamic obstacle, the familiar
training layout, and a validation split. Multi-obstacle patrol routes, unseen
layouts, multiple independent training seeds, and the protected held-out test are
still required before a final effectiveness claim.
