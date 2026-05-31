# Cap Matching Expansion Summary

Source: `resonance/outputs/cap_matching_expansion_2026_04_30_205216`. This used the older mixed cap generator, so treat results as a regime/signal probe rather than clean evidence.

## Winners By Setting

| Setting | Best condition | Acc | Std | Runner-up | Acc | Notes |
|---|---|---:|---:|---|---:|---|
| depth4_same | `standard_deberta_lite` | 0.7546 | 0.0120 | `harmonic_relation_value_normalized` | 0.7272 | known relative-position baseline wins |
| depth4_to6 | `standard_deberta_lite` | 0.7419 | 0.0143 | `standard_alibi` | 0.7389 | depth extrapolation, known relative-position baseline wins |
| depth5_same | `phase_dynamic_qk_film_alibi_normalized` | 0.7438 | 0.0216 | `standard_alibi` | 0.7432 | structural/phase variant wins |
| depth5_to7 | `phase_dynamic_qk_film_alibi_normalized` | 0.7441 | 0.0276 | `complex_directional_normalized` | 0.7354 | depth extrapolation, structural/phase variant wins |
| depth6_same | `standard_deberta_lite` | 0.7725 | 0.0032 | `complex_directional_normalized` | 0.7441 | known relative-position baseline wins |
| depth7_same | `harmonic_relation_value_normalized` | 0.7630 | 0.0322 | `complex_directional_normalized` | 0.7604 | structural/phase variant wins |

## Per-Condition Mean Across Settings

| Condition | Mean acc | Mean acc-majority | Mean label gap | Wins |
|---|---:|---:|---:|---:|
| `complex_directional_normalized` | 0.7314 | 0.1873 | 0.0810 | 0 |
| `harmonic_relation_value_normalized` | 0.7312 | 0.1872 | 0.0826 | 1 |
| `phase_dynamic_qk_film_alibi_normalized` | 0.7346 | 0.1905 | 0.0629 | 2 |
| `relation_value_qk_film_alibi_normalized` | 0.7181 | 0.1740 | 0.0571 | 0 |
| `resonance_full_normalized` | 0.7235 | 0.1794 | 0.0657 | 0 |
| `standard_alibi` | 0.7261 | 0.1821 | 0.0626 | 0 |
| `standard_deberta_lite` | 0.7477 | 0.2037 | 0.1251 | 3 |

## Interpretation

- The task is consistently learnable and non-saturated: all six settings beat majority by roughly 15-22 points.
- The strongest clean baseline is `standard_deberta_lite`, which wins depth4, depth4->6, and depth6. That means any cap claim must compare against relation/relative-position baselines, not only vanilla/ALiBi.
- Phase/structural variants are not dead: `phase_dynamic_qk_film_alibi_normalized` wins depth5 and depth5->7, while harmonic/directional variants are strongest at depth7.
- Because the generator used here had easy negatives, this result should motivate hardened cap tasks, not headline claims.
- The next scientific spine should be exact sequence-verifier tasks: lambda one-step, lambda trace, VM one-step, VM trace, behavioral equivalence, and RLVR-style posttraining.