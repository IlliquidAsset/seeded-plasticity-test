# Stage 2 r3 result: continual hidden-structure learning (two-output readout)

Spec: `docs/STAGE2_SPEC.md` frozen r3 at `def4b36220762a906f360ea44b9d82102c9cb4d2`; SHA-256 at run `695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd` (unchanged: True)
Code commit: `6b23716c3f3601a41dbe027098ae9df2a751a20a` (scientific files clean: True; equals pushed upstream ref: True)
Command: `/Users/kendrick/.hermes/hermes-agent/venv/bin/python3.11 run_stage2_r3.py --workers 5 --tests 'pytest -q tests -> 216 passed at 14522b5 (2026-10-01, Mac Mini)'`
Host: Kendricks-Mac-mini.local (i386, 6 logical CPUs); Python 3.11.15, NumPy 1.26.4, torch 2.2.2
Maintained tests: pytest -q tests -> 216 passed at 14522b5 (2026-10-01, Mac Mini)

## Resources (sec. 12), reported separately

- Fixture qualification (F0 only, seeds 1000..1019): wall 639 s (0.18 h); 5 workers; peak RSS 1.08 GiB; interruptions 0; retries 0; rows ok 20/20; decision PASS
- Experimental run (5 arms x seeds 0..19): wall 3372 s (0.94 h); 5 workers; peak RSS 1.15 GiB; interruptions 0; retries 0; rows ok 100/100

## Verdict: INVALID_OR_MECHANISM_FAIL: 6.3_freeze_at_shift_B_differential_point_ge_0.10, 6.4_scrambled_B_post_differential_point_ge_0.10, 6.5_sfa_off_sign_of_life_B, 8_primary_sign_of_life_B

All applicable section 10 labels: `INVALID_OR_MECHANISM_FAIL: 6.3_freeze_at_shift_B_differential_point_ge_0.10, 6.4_scrambled_B_post_differential_point_ge_0.10, 6.5_sfa_off_sign_of_life_B, 8_primary_sign_of_life_B`; `NO_A_COMPETENCE`

Finding taxonomy (spec sec. 1, verbatim): "If B learning passes but A retention fails, the finding is 'online adaptation with forgetting' — not continual learning."

## Gates (section 10)

| Gate | Item | Result |
|---|---|---|
| gate0 | integrity:all_100_rows_present | PASS |
| gate0 | integrity:hash_assertions_passed_all_rows | PASS |
| gate0 | integrity:shared_hashes_equal_across_arms | PASS |
| gate0 | integrity:section_8_1_schema_complete_all_rows | PASS |
| gate0 | integrity:no_nonfinite | PASS |
| gate0 | integrity:weights_within_bounds | PASS |
| gate0 | integrity:exactly_four_evaluations_every_row | PASS |
| gate0 | integrity:evaluation_isolation_byte_identical | PASS |
| gate0 | integrity:evaluation_weights_bitwise_constant | PASS |
| gate0 | integrity:evaluation_copy_start_state_ok | PASS |
| gate0 | integrity:evaluation_scored_10000 | PASS |
| gate0 | integrity:tie_coin_read_only_at_scored_ties | PASS |
| gate0 | integrity:tie_coin_never_read_by_training | PASS |
| gate0 | integrity:fixture_qualification_passed_before_run | PASS |
| gate0 | 6.2_frozen_from_start_zero_dW | PASS |
| gate0 | 6.2_frozen_from_start_chance_means_in_[0.45,0.55] | PASS |
| gate0 | 6.2_frozen_from_start_ci95_contains_0.50 | PASS |
| gate0 | 6.3_freeze_at_shift_A_dW_gt0 | PASS |
| gate0 | 6.3_freeze_at_shift_zero_dW_B | PASS |
| gate0 | 6.3_freeze_at_shift_B_differential_point_ge_0.10 | FAIL |
| gate0 | 6.3_freeze_at_shift_B_differential_lower_gt_0 | PASS |
| gate0 | 6.4_scrambled_histogram_and_spike_count_equal | PASS |
| gate0 | 6.4_scrambled_order2_lookup_le_0.55 | PASS |
| gate0 | 6.4_scrambled_B_post_differential_point_ge_0.10 | FAIL |
| gate0 | 6.4_scrambled_B_post_differential_lower_gt_0 | PASS |
| gate0 | 6.5_sfa_off_threshold_exactly_0 | PASS |
| gate0 | 6.5_sfa_off_dW_gt0_A_and_B | PASS |
| gate0 | 6.5_sfa_off_sign_of_life_A | PASS |
| gate0 | 6.5_sfa_off_sign_of_life_B | FAIL |
| gate0 | 8_primary_sign_of_life_A | PASS |
| gate0 | 8_primary_sign_of_life_B | FAIL |
| **gate0** | **overall** | **FAIL** |
| gate1 | 1_mean_A_pre_ge_0.70 | FAIL |
| gate1 | 2_lower_mean_A_pre_gt_0.50 | FAIL |
| gate1 | 3_mean_P_minus_F0_A_pre_ge_0.15 | FAIL |
| gate1 | 4_lower_P_minus_F0_A_pre_gt_0 | FAIL |
| **gate1** | **overall** | **FAIL** |
| gate2 | 1_B_improvement_ge_0.15 | FAIL |
| gate2 | 2_lower_B_improvement_gt_0 | FAIL |
| gate2 | 3_mean_B_post_ge_0.70 | FAIL |
| gate2 | 4_lower_mean_B_post_gt_0.50 | FAIL |
| gate2 | 5_freeze_and_scramble_differentials | FAIL |
| **gate2** | **overall** | **FAIL** |
| gate3 | 0_A_learned_gt_0 | PASS |
| gate3 | 1_A_retained_ge_0.80 | PASS |
| gate3 | 2_A_loss_le_0.05 | PASS |
| gate3 | 3_lower_mean_A_post_gt_0.50 | PASS |
| **gate3** | **overall** | **PASS** |
| gate4 | 1_exactly_80_weights_all_arms | PASS |
| gate4 | 2_primary_one_init_zero_resets | PASS |
| gate4 | 3_primary_200k_A_then_200k_B_plastic_every_step | PASS |
| gate4 | 4_train_hashes_ne_heldout_and_zero_heldout_consumed | PASS |
| gate4 | 5_no_replay_no_A_represented_in_B | PASS |
| gate4 | 6_provenance_serialized_before_execution | PASS |
| gate4 | 7_no_deviation_from_spec | PASS |
| **gate4** | **overall** | **PASS** |

## Mean held-out accuracy (20 seeds; chance 0.50; theoretical optimum 0.90)

| Arm | A_pre | B_pre | B_post | A_post |
|---|---:|---:|---:|---:|
| P: primary: SFA on, R-STDP live | 0.5006 [0.4986, 0.5025] | 0.4992 [0.4970, 0.5015] | 0.5003 [0.4987, 0.5020] | 0.5019 [0.4997, 0.5042] |
| F0: frozen from the start | 0.4996 [0.4959, 0.5033] | 0.5011 [0.4986, 0.5036] | 0.4999 [0.4978, 0.5019] | 0.5009 [0.4980, 0.5038] |
| FS: freeze at the A->B shift | 0.5006 [0.4986, 0.5025] | 0.4992 [0.4970, 0.5015] | 0.4992 [0.4979, 0.5004] | 0.5017 [0.4997, 0.5039] |
| SC: temporally scrambled B twin | 0.5006 [0.4986, 0.5025] | 0.4992 [0.4970, 0.5015] | 0.4985 [0.4973, 0.4996] | 0.5008 [0.4990, 0.5027] |
| NS: SFA off, R-STDP active | 0.5004 [0.4985, 0.5021] | 0.4984 [0.4961, 0.5006] | 0.5006 [0.4992, 0.5020] | 0.5023 [0.5001, 0.5046] |

## Gate quantities (bootstrap, 100,000 seed resamples, inverted-CDF percentiles)

| ID | Metric | Point | One-sided lower 95 | One-sided upper 95 | Two-sided 95 |
|---:|---|---:|---:|---:|---|
| 0 | `m(acc[F0, A_pre])` | 0.4996 | undefined | undefined | [0.4959, 0.5033] |
| 1 | `m(acc[F0, B_pre])` | 0.5011 | undefined | undefined | [0.4986, 0.5037] |
| 2 | `m(acc[F0, B_post])` | 0.4999 | undefined | undefined | [0.4978, 0.5019] |
| 3 | `m(acc[F0, A_post])` | 0.5009 | undefined | undefined | [0.4980, 0.5038] |
| 4 | `m(acc[P, A_pre])` | 0.5006 | 0.4989 | undefined | [0.4986, 0.5025] |
| 5 | `m(acc[P, A_pre] - acc[F0, A_pre])` | 0.0010 | -0.0014 | undefined |  |
| 6 | `m(acc[P, B_post] - acc[P, B_pre])` | 0.0011 | -0.0012 | undefined | [-0.0017, 0.0038] |
| 7 | `m(acc[P, B_post])` | 0.5003 | 0.4989 | undefined | [0.4987, 0.5020] |
| 8 | `m((acc[P, B_post] - acc[P, B_pre]) - (acc[FS, B_post] - acc[FS, B_pre]))` | 0.0012 | 0.0004 | undefined |  |
| 9 | `m(acc[P, B_post] - acc[SC, B_post])` | 0.0019 | 0.0009 | undefined |  |
| 10 | `m(acc[P, A_post])` | 0.5019 | 0.5001 | undefined | [0.4997, 0.5042] |
| 11 | `A_loss = m(acc[P, A_pre]) - m(acc[P, A_post])` | -0.0013 | undefined | 0.0012 | [-0.0045, 0.0017] |
| 12 | `A_retained = (m(acc[P, A_post]) - 0.50) / (m(acc[P, A_pre]) - 0.50)` | 3.0484 | -inf | undefined | [-inf, 25.5000] |
| 13 | `m(acc[P, A_post] - acc[NS, A_post])` | -0.0004 | -0.0012 | undefined |  |
| 14 | `A_learned = m(acc[P, A_pre]) - 0.50` | 0.0006 | undefined | undefined | [-0.0014, 0.0025] |

Metric 12 undefined-denominator replicates: 26688 of 100000.
A_learned = 0.0006; A_loss = -0.0013; A_retained = 3.0484.

## Controls (section 6)

- 6.2 frozen from start: PASS — {"max_abs_dW_eq_0_20of20": true, "means_in_[0.45,0.55]": {"A_pre": true, "B_pre": true, "B_post": true, "A_post": true}, "ci95_contains_0.50": {"A_pre": true, "B_pre": true, "B_post": true, "A_post": true}}
- 6.3 freeze at shift: FAIL — {"A_abs_dw_gt0_20of20": true, "max_abs_dW_B_eq_0_20of20": true, "differential_point_ge_0.10": false, "differential_lower_gt_0": true, "differential_point": 0.0011500000000000121, "differential_lower": 0.0004100000000000131}
- 6.4 scrambled B twin: FAIL — {"histogram_and_spike_count_equal_20of20": true, "order2_lookup_le_0.55_20of20": true, "order2_lookup_max": 0.507160071600716, "differential_point_ge_0.10": false, "differential_lower_gt_0": true, "differential_point": 0.0018600000000000005, "differential_lower": 0.0009149999999999991}
- 6.5 SFA off: FAIL — threshold contribution exactly 0: True; dW>0 A&B: True; sign-of-life A/B: True/False
- SFA benefit claim (metric 13 >= 0.05 and lower > 0): False (point -0.0004, lower -0.0012)

## Primary sign-of-life (section 8)

- A: PASS — dW>0 seeds 20/20; min +/- reward events 1170/1185; median r 0.743; seeds r>0.10 20/20; undefined r 0
- B: FAIL — dW>0 seeds 19/20; min +/- reward events 0/0; median r 0.814; seeds r>0.10 18/20; undefined r 2

## Claim boundary

Only a `STAGE2_PASS` licenses the sec. 2 sentence. Any other verdict is reported as named above; thresholds were not moved and no seed, arm, or run was re-run or selected.
Not established in any case: broad continual learning, biological equivalence, optimal SFA, transfer across modalities, performance in Abe.
No oracle: the target enters only as the next observation and the sign of the reward for already-emitted output spikes. The tie-coin scores evaluations only.
r3 is a second attempt under the same roadmap gates; the r2 verdict (`results_stage2/`) stands and is not re-scored.
