# Stage 2 diagnostic D1R2 implementation notes

Implementation input: `docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md` at commit
`f3d52ee91b7ec87f4e665f6ce80208954c30c7a2`, SHA-256
`e5a6a5f4eb7a7adafd9255ac6504593e45be51eefbfa92ce24c63b70fafb82a2`
(Nora APPROVE, t_aafbc69b). Card: t_8863b155. The carried D1R spec
(`c1f03efe...`), the ebb90e74 diagnostic spec (`7e6ada61...`) and the Stage 2
spec (`695cfaa4...`) are unchanged, as are the approved D1 and D1R code, tests,
and retained probe outputs (pinned by `test_d1r2_11_13`).

This implementation authorizes no D1R2 run. `--run-d1r2` is locked behind a
Nora approval record naming the exact pushed implementation commit, the D1R2
spec hash, `run_authorized: true`, and a separate run-authorization card id.
This card ran only the maintained tests, the section 3.3 probe (shakedown
execution), and the section 8 step 1 shakedown on fixture seed 4242.

## Code map

| D1R2 spec | Implementation |
|---|---|
| 3.1 namespace-32 identities, r3 draw order | `snn/stage2_diagnostic_d1r2.py`: `entropy_table`, `rng`, `initial_weights`, `a_stream` |
| 5 timing, `phiR_t` (460), `psi_t` (46), K = 19 | `LagFeatureAccumulator`, `capture_stream`, `input_layer_vector` (carried D1R construction) |
| 6, 6.1 three decoders, `max_iter = 20000`, no `maxfun` | `DECODER_CONFIG` (approved D1 config with only `max_iter` changed), `fit_decoder_slot` |
| 6.2 convergence rule | `fit_decoder_slot`, `_warning_stop`, `validate_slot`, `validate_decoder` |
| 6.3 total slot schema, stop_reason first-match order, attribution | `fit_decoder_slot`, `empty_slot`, `validate_slot` (re-derives the first-match reason), `validate_row_schema` |
| 6.3 warning durability | `_warning_call` (`catch_warnings` + `simplefilter("always")` + `showwarning` hook), `append_ledger_event` (one `O_APPEND` `os.write` + `os.fsync` per event), `scope_open` / `warning` / `scope_close` / `upstream_failure` events |
| 6.3 torn tail / malformed interior line | `read_ledger_report`, `read_ledger` (`MalformedLedger`), runner `invalid_summary` |
| 6.3 rows the worker cannot write | `reconstruct_rows_from_ledger`; runner `complete_rows` (a missing or schema-invalid worker row becomes a `coordinator_from_ledger` `worker_lost` row and forces INVALID) |
| 6.3 preflight finalizer | `preflight_failure_rows`; runner `run_d1r2` probe gate |
| 6.3 outcome redaction | `redact_slot_scoring_text` (ledger), `redact_invalid_row` (package) |
| 7 item 6 | `validate_row`, `validate_package`, `summarize` |
| 8 step 1 shakedown probe (no package) | runner `shakedown_collision_check`, `--ss-probe`; probe `main` writes nothing on failure |
| 8 step 2 run-preflight probe (preflight package) | runner `run_preflight_probe`, called by `preflight` in `--run-d1r2` mode; failure routes to the 20-row preflight finalizer |
| 9 branch table with stop-loss | `BRANCH_TABLE`, `branch` |
| 10 caps | `MAX_WORKERS`, `RSS_LIMIT_BYTES`, `WALL_LIMIT_SECONDS`; runner `run_pool`, `D1R2RSSWatch`, `terminate_executor` |
| 3.3 collision probe | `tools/stage2_diagnostic_d1r2_ss_probe.py`; retained `results_stage2_diagnostic_d1r2/ss_probe.json` + `SHA256SUMS` |
| 11 item 16 | `run_fixture_network_convergence`, `fit_convergence_only` (no predict) |

## Section 11 test map (`tests/test_stage2_diagnostic_d1r2.py`)

| Item | Test |
|---|---|
| 1-15 (carried) | `test_d1r2_11_01` .. `test_d1r2_11_15` (item 10 also `test_d1r2_11_10b`) |
| 11a | `test_d1r2_11a_every_decoder_constructed_with_max_iter_20000_and_frozen_arguments` |
| 11b | `test_d1r2_11b_slot_schema_stop_reasons_attribution_reconstruction_and_redaction`, `test_d1r2_11b_convergence_warning_subclass_with_other_name_is_non_converged_worker_and_ledger` (Nora round 1), `test_d1r2_11b_durable_classification_is_consumed_coherently_by_reconstruction_and_schema_validation` (Nora round 2) |
| 11c | `test_d1r2_11c_branch_table_reaches_every_row_and_rows_1_2_carry_stop_loss` (6 cases) |
| 11d | `test_d1r2_11d_run_preflight_collision_finalizer_writes_20_preflight_rows` |
| 11e | `test_d1r2_11e_warning_then_termination_reconstruction_by_sigkill_wall_and_rss` |
| 11f | `test_d1r2_11f_scoring_exception_slot_and_shakedown_probe_failure_writes_nothing` |
| 16 | `test_d1r2_11_16_convergence_only_network_fits_on_fixture_seed_never_predict`, `test_d1r2_11_16_committed_shakedown_record_is_fixture_only_and_convergence_only` |
| package contract | `test_d1r2_pkg_01..04`, `test_d1r2_wall_limit_*` |

## Implementation readings (no scientific choice)

1. Synthetic fixtures for 11b, 11e, and 11f use an injected capture (random
   features, no network) and a scripted model factory (no LogisticRegression),
   under row IDs 9001..9020 that lie outside every declared seed range. The
   real lbfgs iteration-limit case in 11b uses a fixture-only `max_iter=1`
   override through the same factory hook; production always uses
   `DECODER_CONFIG`.
2. An upstream exception is written to the ledger as an `upstream_failure`
   event so a coordinator reconstruction keeps `seed_error` and the
   `not_attempted_upstream_failure` reason.
3. On the INVALID branch the on-disk rows drop the `decoders` outcome key
   (approved D1R finalizer behaviour); every section 6.3 field is retained.
4. Item 16 fits the two 460-dim network decoders on the fixture-seed training
   stream only; no held-out stream is generated, `predict` is never called,
   and only `n_iter`, `converged`, and `warnings` are recorded.
5. The run preflight re-executes the committed probe in-process at the run's
   code hash; the shakedown uses the retained, manifest-verified output.
6. ConvergenceWarning classification (section 6.3, "or a subclass"; Nora
   round 1 correction). The `showwarning` hook computes
   `issubclass(category, sklearn.exceptions.ConvergenceWarning)` on the real
   class and writes it on every ledger `warning` event as the boolean
   `convergence_warning`, before the hook returns, so it is as durable as the
   warning itself. Worker-written slots use the in-memory result; coordinator
   reconstruction (open and closed slots) uses the durable ledger boolean, so
   both reach the same result after worker termination. The slot's
   `warnings` entries keep exactly the frozen `{category, phase,
   message_head}` shape. The ledger cross-check rejects a warning event
   without its boolean and a slot whose `convergence_warning` differs from
   the durable classification.
7. One classification for reconstruction and validation (Nora round 2
   correction). The schema validator (`validate_slot`, `validate_row_schema`,
   `validate_row`, `validate_package`, `summarize`) consumes the same durable
   ledger classification as `reconstruct_rows_from_ledger`, passed as
   `ledger_classifications(events)` keyed by (seed, decoder) in ledger order.
   The runner passes it in `complete_rows` and both `summarize` calls whenever
   the run ledger exists. `validated_entry_flags` uses the durable boolean when
   present; otherwise it resolves the fully qualified name and applies
   `issubclass`. It fails closed, as a schema finding (section 7 item 6,
   INVALID row 1) and never as a silent guess, when a classification is
   neither durable nor resolvable, when the durable list does not align with
   the slot's warnings, or when a durable boolean contradicts a resolvable
   class. Reconstruction of an open slot with a missing boolean still sets
   `convergence_warning = true`. A warning outside the ConvergenceWarning
   hierarchy, durable false, therefore no longer changes validity even when
   its class cannot be resolved in the coordinator. Test:
   `test_d1r2_11b_durable_classification_is_consumed_coherently_by_reconstruction_and_schema_validation`.

## Fail-closed package behaviour carried

Fresh exclusive run directory (`claim_fresh_run_dir`, `prepare_package_dir`);
`D1R2.progress.jsonl` holds only `PROGRESS_FIELDS`; rows held in coordinator
memory until all 20 are present; strict row and package validators with
coordinator-derived identities and label hashes before any readout; an
incomplete-marker `STOP_D1R2_INVALID` written before the stage; one
`finalize_package` for every branch and any exception; manifest over declared
artifacts only, with any undeclared file forcing `STOP_D1R2_INVALID`; real
wall-stop deadline that kills workers rather than waiting for them.

Rollback: `git revert <implementation commit>`.
