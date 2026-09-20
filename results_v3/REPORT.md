# Structural Plasticity Bench v0.3.0

- Freeze verification: **PASS**
- Frozen-floor gate: **PASS**
- Chosen LR: **2.0**
- Final verdict: **INCONCLUSIVE**

## Frozen-floor precheck

| Task | Frozen mean | Chance |
|---|---:|---:|
| binary_classification | 0.497 | 0.500 |
| frequency_discrimination | 0.331 | 0.333 |
| temporal_xor | 0.501 | 0.500 |
| temporal_sequence | 0.124 | 0.125 |
| associative_memory | 0.169 | 0.167 |

## LR calibration

| LR | Initial norm | Final norm | Signed movement | Max abs weight | Stable |
|---:|---:|---:|---:|---:|:---:|
| 0.001 | 0.472647 | 0.472635 | -0.002% | 0.0905 | yes |
| 0.01 | 0.472647 | 0.472533 | -0.024% | 0.0904 | yes |
| 0.05 | 0.472647 | 0.472087 | -0.118% | 0.0901 | yes |
| 0.1 | 0.472647 | 0.471557 | -0.231% | 0.0901 | yes |
| 0.5 | 0.472647 | 0.467758 | -1.034% | 0.0901 | yes |
| 1 | 0.472647 | 0.463730 | -1.887% | 0.0901 | yes |
| 2 | 0.472647 | 0.458184 | -3.060% | 0.0901 | yes |

## Final arm means

| Task | Weight only | Structural only | Both | Frozen |
|---|---:|---:|---:|---:|
| binary_classification | 0.562 | 0.583 | 0.604 | 0.562 |
| frequency_discrimination | 0.375 | 0.333 | 0.375 | 0.375 |
| temporal_xor | 0.542 | 0.542 | 0.542 | 0.542 |
| temporal_sequence | 0.042 | 0.042 | 0.021 | 0.042 |
| associative_memory | 0.125 | 0.188 | 0.125 | 0.125 |

## Weight movement

- **weight_only:** mean absolute norm change 0.005720; mean percentage change 1.422% — **PASS**
- **both:** mean absolute norm change 0.585745; mean percentage change 32.676% — **PASS**

## Verdict

**INCONCLUSIVE** — whether any plasticity arm beats its frozen control by at least 10 percentage points across the three-seed mean.
