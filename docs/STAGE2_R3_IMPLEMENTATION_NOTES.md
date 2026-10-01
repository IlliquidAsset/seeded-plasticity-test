# Stage 2 r3 implementation notes (committed and pushed before any qualification or experimental run)

Spec: `docs/STAGE2_SPEC.md`, frozen r3 at `def4b36220762a906f360ea44b9d82102c9cb4d2`,
SHA-256 `695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd` (approved by Nora on `t_de6dcd58`).
Build card: `t_095b372d`. This file does not change the spec. It maps r3 to code and states the
operational readings, all fixed before any data exists. The r2 code map (`docs/STAGE2_IMPLEMENTATION_NOTES.md`)
and r2 code (`snn/stage2.py`, `snn/stage2_stats.py`, `run_stage2.py`) are unchanged, so the r2 package stays reproducible.

## Code map

| Spec | Code |
|---|---|
| §7 / §7.1 objects. W1, W2 row `O1`, every stream, and the scramble are imported from the r2 functions (bitwise identical). W2 row `O0` comes from `[20261001, seed, 21]`. The tie-coin is `[20261001, seed, 22, c]` | `snn/stage2_r3.py`: `initial_weights`, `tie_coin`, `seed_objects`, `object_hashes` (adds four coin hashes, which every arm shares) |
| §3.1 / §4.1 / §4.2 two-output `[2, 20, 2]` network and the global scalar reward `(2x-1)(z1-z0)` | `snn/stage2_r3.py:build_network`, `make_reward_fn` on unmodified `snn/core.py` (`PureSNN.online_step`, `RSTDPPlasticity`) |
| §3.1 / §5.2 prediction rule and tie-coin | `snn/stage2_r3.py:predict`, `CoinReader`, `evaluate` |
| §5.3 report-only readout fractions | `evaluate`: `O1_only/O0_only/both_silent/both_fire_fraction`, `nontie_accuracy`, `nontie_count` |
| §8 instrumentation | `snn/stage2_r3.py:PhaseStats`, `train_phase` (per-output spikes and rates, state fractions, +/- reward events, per-row abs dW, `reward_signed_dw_out_corr` over all 40 W2 entries, plus per-row correlations reported only) |
| §8.1 `a_end_*` / `b_end_*` and the schema-completeness assertion | `snn/stage2_r3.py:phase_end_fields`, `validate_row` (called in `run_arm_seed`), and `run_stage2_r3.py:write_rows` (validates every row first and writes nothing if any check fails) |
| §9 / §9.1 bootstrap | unchanged. `snn/stage2_r3_stats.py` imports the r2 `bootstrap_metric` and `bootstrap_all` objects themselves |
| §6 / §8 / §10 scoring | `snn/stage2_r3_stats.py:score`: the r2 scorer plus the r3 deltas listed below |
| §11 item 15 fixture qualification | `snn/stage2_r3_stats.py:qualify`, `run_stage2_r3.py --qualify` |
| §11 shakedown | `tests/test_stage2_r3.py` (non-experimental seeds only) and `run_stage2_r3.py --shakedown` |
| §12 / §13 | `run_stage2_r3.py`: RSS abort above 8 GiB (r2 `RSSWatch`), provenance before execution, SHA-256 manifest, report |

## Operational readings (fixed before the run)

1. **Training step.** At step `t` the live outputs `(z1_t, z0_t)` are the prediction state. The step calls
   `online_step(onehot(x_t), reward_fn)` with `reward_fn(f) = (2*x_t - 1) * (f[0,0] - f[0,1])`. Non-plastic phases
   pass `reward_fn=None`. This is r2 reading 1 with the r3 formula.
2. **Reward event** (§8): a transition with `r_t != 0`, which happens exactly when `z1_t != z0_t` and the arm is plastic.
   Per event, the signed W2 change summed over all 40 entries is recorded, and per row as well.
   `reward_signed_dw_out_corr` keeps the Stage 1 definition over the 40-entry sum (§0 R2). Per-row correlations are reported, not gated.
3. **Training state fractions** (§8, report only) classify `(z1_t, z0_t)` at every training step, in plastic and
   non-plastic phases alike.
4. **Phase-end capture** (§8.1). `a_end_*` is computed from the live network immediately after the A `train_phase`
   returns (after step 199,999) and before the `A_pre`/`B_pre` evaluation copies are made. `b_end_*` is computed
   immediately after the B `train_phase` returns. Every arm computes both sets from the live weights, and no field is copied from another.
5. **Tie-coin access.** `CoinReader` counts every read. The access log records `eval:coin_<ckpt>` reads.
   `coin_reads_by_training` counts `train:*coin*` keys and must be 0. A coin is read only when `z1 == z0`, including
   inside the unscored warm-up. That has no effect on scoring, because warm-up predictions are never scored.
6. **Gate 0 integrity additions (r3).** These are `section_8_1_schema_complete_all_rows`, `tie_coin_read_only_at_scored_ties`,
   `tie_coin_never_read_by_training`, `fixture_qualification_passed_before_run`, and the bounds check applied to both `a_end`
   and `b_end` extrema. Each one is an existing §6–§8 / §11 integrity rule stated for the r3 objects. None adds a threshold.
7. **Gate 4 item 1** uses the constant 80, per §10 as approved (R1).
8. **Fixture qualification** (§11 item 15) runs only the F0 arm on seeds 1000..1019 at full length. It computes §9.1 metrics 0–3
   with the unchanged bootstrap (`[20261001, 7, k]`, 100,000 resamples). In the `acc` tensor only the F0 slice is populated;
   metrics 0–3 read nothing else. The decision is PASS only if all of the following hold: 20/20 rows,
   `max_abs_dW_total == 0.0` in 20/20, every checkpoint mean in `[0.45, 0.55]`, and every two-sided 95% interval containing `0.50`.
   Otherwise the decision is STOP. The runner refuses the experimental mode unless `results_stage2_r3/qualification/qualification.json`
   says PASS. A qualification job error is a STOP.
9. **Push-before-run guard.** Outside shakedown, the preflight requires all of the following: unchanged spec hash, params
   equal to the independent spec transcription, clean scientific files, and `HEAD == upstream tracking ref` (code pushed).
10. All r2 readings 3, 4, 5, 6, 7, 8, 10, 10a, 11, and 12 (`docs/STAGE2_IMPLEMENTATION_NOTES.md`) carry over unchanged. This includes
    undefined correlation counting as 0.0, 1,000-step instrumentation sampling, verdict precedence, Gate 0 granularity,
    SFA-off point checks, and outcome-free progress lines.

## Shakedown scope (§11)

`tests/test_stage2_r3.py` covers items 1–14 and 16 (item 1 ALIF and item 8 legacy stay in `tests/test_stage2.py` and
the maintained suite). It uses seeds `1000+` and at most a few thousand steps, so no gate metric is estimated.
`run_stage2_r3.py --shakedown` exercises aggregation, bootstrap, scoring, and the report on seeds 1000–1019 with
1,000-step phases, writing to a scratch directory outside the repository.
