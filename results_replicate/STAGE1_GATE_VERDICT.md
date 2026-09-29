# Stage 1 Gate Verdict

Commit reviewed: `a396f8a32fa6c9fe796b23795f7828bdc1e2b170`

## Verdict

**FAIL. Do not start Stage 2.**

The run completed with 10 distinct seeds per configuration, so the seed-count requirement is met. The learning-evidence and sign-of-life requirements are not met.

## Paired confidence bounds

Effects are final plasticity-on accuracy minus the matched frozen-control accuracy for each seed. Bounds are two-sided 95% Student t intervals across 10 paired seeds, with 9 degrees of freedom and t = 2.262.

| Configuration | Mean effect | 95% CI | Signal-exists seeds | Output-spiking seeds | Gate interpretation |
|---|---:|---:|---:|---:|---|
| binary_classification_easy | +0.0000 | [-0.0211, +0.0211] | 3/10 | 10/10 | No demonstrated benefit |
| temporal_xor_easy | +0.0375 | [+0.0062, +0.0688] | 3/10 | 5/10 | Positive paired effect, but absolute accuracy remains below chance and half the seeds show no output spikes |
| binary_classification_hard_ws2 | +0.0000 | [+0.0000, +0.0000] | 1/10 | 10/10 | No demonstrated benefit |
| temporal_xor_hard_ws2 | +0.0188 | [-0.0114, +0.0489] | 5/10 | 9/10 | Inconclusive |
| temporal_sequence_hard_ws2 | -0.0313 | [-0.0629, +0.0004] | 4/10 | 10/10 | Inconclusive and directionally negative |

## Sign of life

Eligibility traces and weight updates exist, but they do not establish useful learning. The mean reward-to-weight-change correlations are near zero across all five configurations: +0.0173, +0.0004, +0.0021, +0.0118, and +0.0170. The script's own `signal_exists` criterion passes in only 10% to 50% of seeds, depending on configuration.

## Review

Ethan and Nora independently approved the mechanism change in `b370b02`: reward is multiplied by each sample's eligibility before the batch mean, and all five experiments use the intended 10 unique seeds.

Nora independently recomputed the paired bounds and returned **FAIL Stage 1**. Her reason matches the data: the code fix is a valid mechanism correction, not evidence that the bench learns.

## Required next proof

Repair the anchor so it produces above-chance learning, then rerun matched plasticity-on and frozen controls across adequate seeds. Stage 1 can pass only when the positive effect has a confidence interval excluding zero and the sign-of-life evidence is credible across seeds.
