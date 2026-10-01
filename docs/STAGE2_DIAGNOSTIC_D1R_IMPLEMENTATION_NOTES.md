# Stage 2 diagnostic D1R implementation notes

Implementation input: `docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md` at commit
`2d9efd8e4269057620607725a3bd615db22796f2`, SHA-256
`c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743`
(Nora APPROVE, t_1337897f). Card: t_64ba9cb4. The ebb90e74 diagnostic spec
(`7e6ada61...`) and the Stage 2 spec (`695cfaa4...`) are unchanged.

This implementation authorizes no D1R run. `--run-d1r` is locked behind a Nora
approval record naming the exact pushed implementation commit, the D1R spec
hash, `run_authorized: true`, and a separate run-authorization card id. The
card may run only the maintained tests, the seed-4242 shakedown, and the
section 3.3 collision probe.

## Code map

| D1R spec | Implementation |
|---|---|
| 3.1 namespace-31 identities, r3 draw order | `snn/stage2_diagnostic_d1r.py`: `entropy_table`, `rng`, `initial_weights`, `a_stream` (approved `_roll_a`) |
| 5.1 timing, `h_t`, `u_t` | `capture_stream` (row captured before `x_t` is integrated, no reward function), `input_layer_vector` |
| 5.2 / 5.3 single map, K = 19, width-parameterized operator | `LagFeatureAccumulator(n)`: columns `0..3n-1` from the approved `diag.FeatureAccumulator` instance, then lag 0..19 blocks; n = 20 gives `phiR_t` (460), n = 2 gives `psi_t` (46) |
| 6 three decoders, labels | `run_d1r_seed` (approved `diag.fit_d1_decoder` unchanged), `worker_labels`; coordinator labels in `object_hashes` by direct slicing (`x[1999:-1]`, `x[2000:]`) |
| 2 / 4.5 per-seed interval, component `[20261001, seed, 31, 60, q]` | `block_lower_bound` (arithmetic bit-identical to approved `circular_block_lower_bound`, tested) |
| 6.1-6.3 readouts and status, 7 invariants | `validate_decoder`, `validate_row`, `validate_package`, `section46`, `pipeline_readout`, `summarize` |
| 9 branch table | `BRANCH_TABLE`, `STATUS_TO_BRANCH`, `branch` |
| 3.3 collision probe | `tools/stage2_diagnostic_d1r_ss_probe.py`, retained `results_stage2_diagnostic_d1r/ss_probe.json` + `SHA256SUMS` |
| 8 step 1 shakedown, 11 item 15 | `run_stage2_diagnostic_d1r.py --shakedown` (seed 4242, full lengths, pipeline decoder only) |
| 8 steps 2-3, 10, 12 | `run_stage2_diagnostic_d1r.py --run-d1r` (locked): `preflight`, `run_pool`, `run_d1r`, `finalize_package` |
| 11 items 1-15 | `tests/test_stage2_diagnostic_d1r.py::test_d1r_11_01..15` (item 10 also `test_d1r_11_10b`), package tests `test_d1r_pkg_01..04` |

## Implementation readings (no scientific choice)

1. `u_t` is formed from the previous symbol, `ONEHOT[x_(t-1)]`, and zero at
   `t = 0`, read from the same `ONEHOT` tensors the network integrates.
2. Both maps share one class; `psi_t` and `phiR_t` differ only in the width
   argument and the spike vector fed in. `count20 == sum(lag blocks)` is
   asserted at every captured row of both maps and recorded as `nesting_ok`.
3. Row timing is the approved D1 order: capture, then
   `online_step_with_background(ONEHOT[x_t], None, reward_fn=None)`.
4. Per-seed lower bounds exist only for `network_positive` (q = 2) and `A`
   (q = 1). The pipeline decoder has no interval and no bootstrap identity.
5. The D1R row carries all three label hashes per split; the validator requires
   each to equal the coordinator identity, and the pipeline and
   network-positive labels to be the identical `x_(t-1)` vector.
6. Under `INVALID_PIPELINE` (section 9 row 2, "no science reading of the
   network decoders") the network readouts are not computed, and network
   accuracy, bound, and coefficient fields are redacted from the written rows
   and the package summary.
7. `run_d1r_seed` refuses every seed in 2100..2119 unless the runner's job
   wrapper passes `authorized_run=True`. The `decoders` restriction exists for
   the seed-4242 shakedown only; `validate_package` always requires all three.
   Short lengths and an injected `capture` exist for construction tests only;
   the package validator rejects non-spec lengths.
8. Resource cap per spec section 10: at most 3 workers (preflight refuses more),
   abort above 12 GiB total RSS (`D1RRSSWatch`), stop and report past 6 h.
   The 6 h stop is a real deadline (round-1 review fix): `run_pool` waits on
   futures with a timeout bounded by the remaining wall budget, so the check
   fires even when no job completes. On the deadline, RSS abort, or any
   exception, `terminate_executor` cancels every queued future, kills the
   worker processes, and shuts the pool down with `wait=False`; the
   coordinator never waits for running work. `WallLimitExceeded` then reaches
   `run_d1r`, which freezes the outcome-free `STOP_D1R_INVALID` package
   through the single finalizer. Tests:
   `test_d1r_wall_limit_bounded_return_and_no_queued_job_starts_after_deadline`,
   `test_d1r_wall_limit_terminates_running_worker_processes`,
   `test_d1r_wall_limit_freezes_outcome_free_invalid_package`.

## Fail-closed package behaviour carried from t_53bf2d59

Fresh exclusive run directory (`claim_fresh_run_dir`, `prepare_package_dir`);
`D1R.progress.jsonl` holds only `PROGRESS_FIELDS`; rows held in coordinator
memory until all 20 are present; strict row/package validators with
coordinator-derived identities and label hashes before any readout; an
incomplete-marker `STOP_D1R_INVALID` written before the stage; one
`finalize_package` for every branch and any exception; manifest over declared
artifacts only, with any undeclared file forcing `STOP_D1R_INVALID`.

## Unchanged

`snn/stage2_diagnostic.py`, `snn/stage2_diagnostic_stats.py`,
`run_stage2_diagnostic.py`, `run_stage2.py`, `snn/stage2_r3.py`,
`snn/stage2.py`, `snn/core.py`, `tools/stage2_diagnostic_ss_probe.py`, the three
approved ebb90e74 test files, `results_stage2_diagnostic/`, and all three spec
documents. `test_d1r_11_13` pins their SHA-256 values.

Rollback: `git revert <implementation commit>`.
