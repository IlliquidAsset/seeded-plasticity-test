# Stage 2 D1-D4 diagnostic implementation notes

Implementation input: `docs/STAGE2_DIAGNOSTIC_SPEC.md` at commit
`ebb90e74d088057f1d37e08ac80ed7bab4e8851e`, SHA-256
`7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf`.
The frozen Stage 2 specification remains SHA-256
`695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd`.

This implementation authorizes no diagnostic run. The full runner is locked
behind a separate Nora approval record naming the exact pushed implementation
commit. The implementation card may run only the maintained tests, the short
non-diagnostic shakedown, and the section 3.2 identity probe.

## Code map

| Specification | Implementation |
|---|---|
| Sections 2 and 3.1, namespaced weights, streams, coins, lag-1 stream, and drive | `snn/stage2_diagnostic.py`: `initial_weights`, `a_stream`, `tie_coin`, `lag1_stream`, `background_source_vectors` |
| D1 causal features and fixed decoder | `FeatureAccumulator`, `capture_d1_features`, `fit_d1_decoder`, `circular_block_lower_bound`, `run_d1_seed`, `summarize_d1` |
| D2 checkpoints, strict schema, and onset | `CheckpointAccumulator`, `validate_checkpoint_row(s)`, `train_condition`, `onset_record`, `classify_onsets`; aggregate in `snn/stage2_diagnostic_stats.py:summarize_d2` |
| D3 balanced lag-1 contrast | `lag1_stream`, `run_condition_seed`; bootstrap/readout in `stage2_diagnostic_stats.py` |
| D4 non-plastic background drive | `Stage2DiagnosticSNN.online_step_with_background`, `background_source_vectors`, `exchange_output_drive`, `run_condition_seed` |
| Section 9 bootstrap | `snn/stage2_diagnostic_stats.py`: fixed metric table 0..8, one generator per metric, whole paired seed rows |
| Section 3.2 collision probe | `tools/stage2_diagnostic_ss_probe.py`, retained output `results_stage2_diagnostic/ss_probe.json` and `SHA256SUMS` |
| Section 11 shakedown | `tests/test_stage2_diagnostic.py`, one named test for each item 1..15; `run_stage2_diagnostic.py --shakedown` exercises a short seed-4242 path |
| Later approved execution | `run_stage2_diagnostic.py --run-diagnostic --approval-file <record>`; fail-closed preflight requires exact specs, clean scientific files, pushed HEAD, five workers maximum, and Nora approval tied to HEAD |

## Frozen timing and implementation readings

1. D1 forms `phi_t` from the already-present hidden spikes before integrating
   `x_t`. Each independent stream starts with zero network and feature state.
   Rows 0..1,999 are captured but excluded. The test mutation path can call
   only the stored transform and model; it cannot refit either.
2. D2 counts the output state present at step `t`, then advances the unchanged
   r3 transition and counts the resulting hidden spikes, matching r3's existing
   `PhaseStats` timing. A checkpoint closes after each 1,000 transitions and
   records live post-transition weights.
3. The background path is a subclass seam. When background is absent it calls
   the existing r3 `online_step` directly. When present it duplicates that
   frozen order and adds one 22-vector only to ordinary postsynaptic input
   before thresholding. It creates no `Synapse` or `RSTDPPlasticity` object and
   never changes `pres`, while resulting neuron spikes enter the normal traces
   on the following transition.
4. The 212,000-step source array is generated once from component 40. Training
   consumes indices 0..199,999. Evaluation resets network state as r3 does but
   consumes source indices 200,000..211,999, preserving source-clock continuity.
5. No-drive D2 rows are named `P_A-no` and `F0_A-no` and are reused for D4.
   D3 rows are `P_lag1-no` and `F0_lag1-no`; D4 adds `P_A+drive`,
   `F0_A+drive`, and report-only `P_lag1+drive`.
6. The runner completes and validates every seed row in a stage before reading
   outcomes. D1 stops downstream work unless it passes. D3 invalidity stops D4.
   Progress lines expose status and resource use only, never accuracy.
7. The source code permits shorter lengths only in construction functions used
   by tests. The approved full runner fixes seeds 2000..2019 and all lengths to
   module constants.

## Rollback

Revert the implementation commit. No existing Stage 2 file, result package,
threshold, gate, control, validity rule, or Stage 1 rule is modified.
