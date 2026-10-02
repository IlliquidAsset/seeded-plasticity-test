# Stage 2 diagnostic D1R2: final D1 harness attempt (solver budget repair) specification

**Status:** FROZEN CANDIDATE, document only. Two pre-freeze findings (effective solver ceiling; worst-case wall projection) were ruled by Amanda, `t_bbc93d9c`: both carried as disclosed in sections 6.1, 6.2, and 10, with no setting changed. Revised after Nora review round 2 (`t_aafbc69b`): the section 6.3 per-decoder schema is total (all three decoder slots on every row and every branch, every warning attributed), with consistent updates to sections 2, 6.2, 7, 11, and 12; no setting, threshold, seed, or rule changed. No D1R2 construction run, seed, decoder fit, or datum may exist until Nora independently approves the frozen bytes, a separate implementation card is approved, and a separate run card authorizes execution.  
**Task/directive:** Project AIB Kanban `t_aafbc69b`; stored task-body SHA-256 (raw DB bytes) `77b60e90aa1e61f175336f9afe4c9d7cf0bb21ace57f2e60a1202a3a4d597bdd`.  
**Governing proposal:** Amanda, `t_90a2f636`, `STAGE2_D1R2_NEXT_STEP_PROPOSAL.md`, SHA-256 `fd2988e4dbf16398f56b7b371981f08ee974af84866c8b15238c0baed7e32f80`.  
**Triggering verdict:** Nora, `t_75929454`, APPROVE of D1R `INVALID`, branch `STOP_D1R_INVALID`, table row 1, package `results_stage2_diagnostic_d1r/runs/d1r-20261002T003314006433Z-a953291091d5-pid18595` at `34aae46`.  
**Control record read:** AIB `.inbox/CONTROL.json`, epoch 46 (internal mirror). This card is doc-only and authorizes no run.  
**Base:** branch `v0.3.0-calibrated`, commit `34aae464b12bd834bd45ef46b6dbc2a122577ffe` (verified by `git ls-remote`).  
**Carried document:** `docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md` at `2d9efd8e4269057620607725a3bd615db22796f2`, SHA-256 `c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743`, referred to below as "2d9efd8". Its bytes are not edited.  
**Other frozen documents, bytes not edited:** `docs/STAGE2_DIAGNOSTIC_SPEC.md` ("ebb90e74"), SHA-256 `7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf`; `docs/STAGE2_SPEC.md`, SHA-256 `695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd`.  
**Date:** 2026-10-02.

## 0. Authority, scope, and stop boundary

2d9efd8 section 9 row 1 reads: "STOP. Return to implementation/spec review. No science reading. No D2-D4, no r4." 2d9efd8 hard stop 2 bars further D1 harness repair only "under section 9 row 3"; D1R ended at row 1. This document is the new spec that hard stop 2 requires for any change.

D1R2 = 2d9efd8, carried byte-for-byte by reference, with exactly the differences in section 2. Every 2d9efd8 section not amended in section 2 is in force for D1R2 with "D1R" read as "D1R2", seeds `2100..2119` read as `2200..2219`, and namespace `31` read as `32`. That includes: section 1 (readings (a) and (b)), section 4 (claim boundary), section 5 (timing, `h_t`, `u_t`, the single 460-dim map `phiR_t` with `K = 19`, its justification, and the 46-dim pipeline map `psi_t` with its reachability bound), section 6 (three decoders, labels, section 4.6 predicate, pipeline control, status), section 9 (branch table, extended here only by the stop-loss column), and section 12 (package, no-oracle statement, claim boundary).

Also carried unchanged: ebb90e74 in full (D2-D4 text, constants, thresholds, seeds `2000..2019` under namespace 30, section 8 run order, section 10 rows 3 onward); `docs/STAGE2_SPEC.md` including Stage 2's four exit gates, four controls, section 6 validity rules, and section 10 thresholds; the Stage 1 repository-core MSTDPET rule.

Not reinterpreted, re-scored, refit, completed, or pooled: D1 (seeds `2000..2019`, package at `5348fcc`, verdict `INVALID_HARNESS`) and D1R (seeds `2100..2119`, package at `34aae46`, verdict `INVALID`). D1R's 18 outcome-redacted rows stay redacted. D1's A-decoder median `0.648` stays non-decision-bearing. No D1R accuracy exists; all D1R readouts are null. The only D1R fact carried forward is the convergence count: seeds 2100 and 2106 each raised "lbfgs failed to converge after 2000 iteration(s) (status=1): STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"; which decoder raised it is not recorded. D1R's resource fields (wall time, RSS) are used in section 10 as engineering anchors only.

Owned artifact: `docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md` only. No runtime, result, test, frozen-spec, or source file is edited on this card. Rollback is `git revert <this-spec-commit>`.

Hard stops (2d9efd8 hard stops 1-7 carried, restated for D1R2):

1. No D1R2 execution, performance smoke on a D1R2 seed, seed screening, or partial outcome inspection before independent approval of this spec, then of the implementation, then a separate run authorization.
2. No threshold, feature, `K`, split, decoder setting (including `max_iter`), seed, or branch rule may change after any D1R2 datum is inspected.
3. Exactly one network feature map and one pipeline map, as in 2d9efd8 section 5.
4. D1R2 is one-shot on seeds `2200..2219`. No rerun, resume, second seed range, second K, second decoder, or second solver budget.
5. **Stop-loss (new).** D1R2 is the final D1 harness attempt. Any D1R2 `INVALID` (section 9 row 1) or `INVALID_PIPELINE` (row 2) closes the D1 harness-repair path. D1 then returns to Kendrick as a NEEDS KENDRICK DECISION card. There is no D1R3. Only an explicit new Kendrick authorization can reopen harness repair.
6. No D2, D3, D4, r4, Stage 3, Stage 4, Track 4, rhythmic-drive arm, retuning, or mechanism change is authorized by this document.
7. No D1R2 decoder may enter a training or evaluation path of D2-D4 or any Stage 2 arm.

## 1. Why D1R2 exists

D1R produced no reading. Two of twenty seeds hit the fixed 2,000-iteration lbfgs cap, so by 2d9efd8 sections 2, 6.3, 7, and 9 the run was mechanically `INVALID`. That is a solver budget failure, not a measurement.

L2 logistic regression with `C = 1.0` is strictly convex, so each fit has exactly one optimum. Raising the iteration budget does not change the model, the feature map, the labels, the threshold, or what counts as success. It only lets the solver reach the optimum the spec already defines. D1R2 therefore changes the solver budget and diagnostics only, on fresh seeds, and adds a stop-loss so that this is the last attempt.

## 2. What D1R2 changes relative to 2d9efd8, and nothing else

| Item | D1R (2d9efd8) | D1R2 |
|---|---|---|
| Seeds | `2100..2119` | `2200..2219` |
| SeedSequence namespace component | `31` | `32`, for every D1R2 object (section 3.1) |
| `max_iter` (all three decoders) | `2000` | `20000`, fixed before data (section 6.1) |
| Convergence rule | `n_iter < 2000`; ConvergenceWarning raised as error | `n_iter < 20000` **and** no `ConvergenceWarning`, else `INVALID` (section 6.2). No fallback solver, warm start, retry, or `tol` change |
| Per-decoder diagnostics | error text only; failing decoder not identified | every one of the 20 rows, on every branch including `INVALID`, carries all three decoder slots (`pipeline`, `network_positive`, `A`), never omitted: attempted and completed flags, `stop_reason`, nullable `n_iter`, converged flag, a `ConvergenceWarning` flag, every captured warning attributed to its decoder (category, phase, first two message lines), and any fit error (section 6.3) |
| Stop-loss | ban on repair only after row 3 | any D1R2 `INVALID` or `INVALID_PIPELINE` closes harness repair; D1 returns to Kendrick; no D1R3 (section 9) |
| Resource section | 2d9efd8 section 10 | re-estimated for the new budget, including the every-fit-capped worst case (section 10) |
| Shakedown | 2d9efd8 section 11 items 1-15 | carried, plus proposed item 16, a convergence-only check on fixture seed `4242` (section 11) |

Unchanged, by reference to 2d9efd8: frozen r3 network and F0 semantics; streams and split lengths (198,000 train rows, 10,000 scored test rows, 2,000 warm-up); timing (`h_t` and `u_t` observed before `x_t` is integrated); `K = 19`; the single 460-dim spike-only map `phiR_t`; the 46-dim pipeline map `psi_t`; scikit-learn `LogisticRegression`, L2, `C=1.0`, `solver="lbfgs"`, `tol=1e-8`, intercept enabled, `class_weight=None`, float64; training-only standardization (training mean, population standard deviation `ddof=0`, zero-variance column divided by `1.0`); probability `>= 0.5` predicts 1; 2d9efd8 section 2 block bootstrap (ebb90e74 section 4.5: circular moving blocks of 100, exactly 10,000 resamples, `numpy.percentile(replicates, 5, method="inverted_cdf")`); the section 4.6 predicate; the pipeline bar; the 6-row branch table conditions and codes; the outcome-free `INVALID` finalizer; the claim boundary and no-oracle text.

## 3. Deterministic identities and the collision probe

### 3.1 D1R2 namespace

Root `20261001`; D1R2 namespace component `32`. For `seed` in `2200..2219`:

| Object | SeedSequence entropy list |
|---|---|
| W1 then W2 row O1 (r3 draw order) | `[20261001, seed, 32, 1]` |
| W2 row O0 | `[20261001, seed, 32, 21]` |
| order-2 A decoder-train stream | `[20261001, seed, 32, 2]` |
| order-2 A decoder-test stream | `[20261001, seed, 32, 5]` |
| per-seed block bootstrap, A decoder | `[20261001, seed, 32, 60, 1]` |
| per-seed block bootstrap, network positive control | `[20261001, seed, 32, 60, 2]` |

Object list, generator rule (one generator per object, none shared), and the absence of tie-coin, background drive, lag-1 stream, across-seed bootstrap, and pipeline bootstrap are exactly as 2d9efd8 section 3.1.

### 3.2 Seed disjointness

D1R2 seeds `2200..2219` are disjoint from experimental seeds `0..19`, fixture/qualification seeds `1000..1019`, ebb90e74/D1 seeds `2000..2019`, D1R seeds `2100..2119`, and the non-diagnostic shakedown seed `4242`. Neither D1 nor D1R seeds are reused, because D1R2 is motivated by their results.

### 3.3 Collision probe (rerun including all namespace-30 and namespace-31 identities as legacy)

PASS requires all of: every new D1R2 entropy tuple unique among new tuples; every new `generate_state(8)` state unique among new states; no new tuple or state equal to any legacy tuple or state; seed ranges in section 3.2 disjoint. Legacy = every identity the approved in-repo probe `tools/stage2_diagnostic_d1r_ss_probe.py` enumerates as legacy (r2/r3 objects, tie-coins, bootstrap identities, every ebb90e74 namespace-30 identity) plus every D1R namespace-31 identity it enumerates as new (legacy since D1R ran). Legacy-only duplicates are reported separately and are not repaired or hidden.

Pre-data result (identity enumeration only; no network, stream, decoder, or D1R2 datum generated). Probe: workspace script `d1r2_ss_probe.py`, SHA-256 `cd5f1ad50d72b96f58fd60e0b8f12cb7a1b9350dec9b190483fdf5585765e4f4`, importing the approved, unchanged `tools/stage2_diagnostic_d1r_ss_probe.py` (SHA-256 `67185d8cc33100157b93f4061ccec92bfe5f80e192c9b221d573c323f5e97b8f`) and through it `tools/stage2_diagnostic_ss_probe.py` (SHA-256 `ab3dbac8a08fc7108b52a8963202145d4b8acb97ac6eba3720df5c4c21734f58`). Interpreter `/Users/aib-agent/miniforge3/envs/aib310/bin/python` 3.10.20, NumPy 1.26.4. Two runs produced byte-identical output.

| Quantity | Value |
|---|---:|
| legacy records | 824 (704 pre-D1R legacy + 120 D1R namespace 31) |
| legacy unique entropy tuples / states | 818 / 818 |
| legacy-only duplicate groups (entropy / state) | 6 / 6 (the disclosed r3 seed-7 object vs bootstrap identity, components 1-6; unchanged since ebb90e74) |
| new D1R2 records | 120 (20 seeds x 6 objects) |
| new unique entropy tuples / states | 120 / 120 |
| new-within-namespace collisions (entropy / state) | 0 / 0 |
| new-vs-legacy collisions (entropy / state) | 0 / 0 |
| seed ranges disjoint | true |
| identity-state SHA-256 | `4d8dd8b477aa2615bc14f66ed11223d0995c10df0985919577bacc9648b6848e` |
| probe output SHA-256 | `76483e93ee594bb4843a882185ab3e20eac3073c9807830ba28ec634c3f80e77` |
| decision | PASS |

The implementation card must commit an equivalent in-repo probe, reproduce these counts and the identity-state SHA-256 exactly, and retain code hash, NumPy version, collision lists, and output hash. Any collision involving a D1R2 identity is `INVALID` and stops before a D1R2 run.

## 4. Claim boundary

2d9efd8 section 4, unchanged, with "D1R" read as "D1R2". Raising the solver budget adds no claim: a converged fit at any `n_iter < 20000` is the same unique optimum the D1R design defined.

## 5. Feature maps and timing

2d9efd8 section 5, unchanged: `K = 19`; `phiR_t` (460 dims: the 60 ebb90e74 columns, then lag blocks `h_t..h_(t-19)`); `psi_t` (46 dims, same operator on `u_t = ONEHOT[x_(t-1)]`); column order, zero-fill, permitted inputs, K justification, and the pipeline reachability bound (at most 108 of 198,000 training errors) all as written there. Not re-decided.

## 6. Decoders, solver budget, convergence rule, diagnostics, and readouts

The three decoders, features, labels, per-seed lower-bound identities (now namespace 32), and roles are 2d9efd8 section 6 table exactly. The section 4.6 predicate (2d9efd8 section 6.1), pipeline control (6.2), and status rule (6.3) are unchanged:

- network positive control and A decoder, each over 20 per-seed held-out accuracies and lower bounds: `PASS` = median `>= 0.70` and at least 15/20 lower bounds strictly `> 0.50`; `FAIL` = median `< 0.55`; `INCONCLUSIVE` = neither;
- pipeline control: `PASS` iff median held-out accuracy `>= 0.99`, else `INVALID_PIPELINE`;
- status order: `INVALID`, then `INVALID_PIPELINE`, then `REPRESENTATION_LIMITED` (network positive control `FAIL` or `INCONCLUSIVE`), then the A decoder's status.

### 6.1 Solver budget

`max_iter = 20000` for all three decoders (`pipeline`, `network_positive`, `A`), fixed before any D1R2 datum. All other `LogisticRegression` arguments as 2d9efd8 section 2.

Basis, stated as required: the value was chosen from the convergence count only. No D1R accuracy exists (all D1R readouts are null and 18 rows are outcome-redacted), so no outcome informed it. It is a budget, not a tuned value.

Nominal and effective ceiling. The nominal `max_iter = 20000` is `10x` the failed 2,000 cap and about `14.6x` (`20000 / 1371`) D1's worst observed `n_iter` of 1,371 on the 60-dim map; these two ratios are nominal only. The effective ceiling is lower. scikit-learn 1.7.2 `LogisticRegression(solver="lbfgs")` calls `scipy.optimize.minimize(method="L-BFGS-B", options={"maxiter": max_iter, "maxls": 50, "gtol": tol, "ftol": 64*eps})` and does not pass `maxfun`; `LogisticRegression` has no `maxfun` parameter; scipy 1.15.3 L-BFGS-B therefore applies its default `maxfun = 15000` function-and-gradient evaluations (verified in both packages' source and by `inspect`). Each lbfgs iteration uses at least one evaluation, so no fit can exceed about 15,000 iterations. A synthetic ill-conditioned toy (workspace `maxfun_check2.py`; no AIB module, identity, or datum) fitted with `max_iter=20000` stopped at `n_iter = 13501` with "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT". At 2,000 this limit never bound (D1R's two failures report "ITERATIONS REACHED LIMIT"). The effective budget is therefore at most 7.5x the failed 2,000 cap and at most about 10.9x D1's worst observed n_iter 1,371, because sklearn 1.7.2 does not pass maxfun and scipy 1.15.3 L-BFGS-B defaults to maxfun = 15000.

No `maxfun` argument is added; the decoder call is 2d9efd8 section 2 with only `max_iter` changed (Amanda ruling, `t_bbc93d9c`, F1 option (a)). The section 6.2 rule fails closed at either limit.

### 6.2 Convergence rule

A decoder fit is converged iff `n_iter < 20000` **and** the fit raised no `ConvergenceWarning`. Otherwise that decoder is non-converged, the seed is `INVALID`, and D1R2 is `INVALID` under the first bullet of the status rule (2d9efd8 section 6.3, restated in section 6 above) and section 9 row 1.

Both clauses are required. In the installed stack a fit can stop on scipy's function-evaluation limit with `n_iter < 20000`; that stop sets lbfgs status 1 and raises `ConvergenceWarning` ("STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"), so only the warning clause catches it (section 6.1, effective ceiling). A fit that does not complete (exception, termination, or never attempted) has no `n_iter` and is non-converged.

No fallback solver, no warm start, no retry, no `tol` change, no change of `C`, no rescaling beyond the frozen standardization, no seed substitution. A non-converged fit is never re-run.

### 6.3 Per-decoder convergence diagnostics (every row, every branch, all three slots)

**Totality.** The package always contains exactly 20 rows, one per seed `2200..2219`, on every branch including `INVALID` and `INVALID_PIPELINE`. Every row always contains exactly three decoder slots, keyed `pipeline`, `network_positive`, `A`. No slot is ever omitted, whatever happened to the seed: convergence failure, fit exception, upstream exception, invariant failure, worker crash, six-hour wall stop, RSS abort, or a seed never started. A missing row, a missing slot, or a slot that fails this schema is itself a section 7 item 6 failure (`INVALID`, row 1).

**Fit order and continuation.** Each worker fits the decoders in the 2d9efd8 order (`pipeline`, `network_positive`, `A`). A `ConvergenceWarning` is captured and recorded, never raised. An exception inside one decoder's standardization or fit is caught and recorded in that decoder's slot, and the remaining decoders are still attempted whenever their features and labels were built. Invariant checks never skip a decoder fit. A decoder is not attempted only when its inputs could not be built (an upstream exception in stream generation, feature capture, or label construction) or the worker stopped before reaching it.

**Slot schema.** Each of the three slots has exactly these fields:

| Field | Type | Rule |
|---|---|---|
| `attempted` | boolean | true iff standardization for this decoder was entered |
| `completed` | boolean | true iff `fit` returned; `completed` implies `attempted` |
| `stop_reason` | enum | exactly one of, first applicable wins: `converged`; `iteration_limit` (`n_iter >= 20000`, or a `ConvergenceWarning` naming "ITERATIONS REACHED LIMIT"); `evaluation_limit` (a `ConvergenceWarning` naming "F,G EVALUATIONS EXCEEDS LIMIT"); `other_convergence_warning` (any other `ConvergenceWarning`); `fit_exception` (exception in this decoder's standardization or fit); `not_attempted_upstream_failure`; `terminated_wall_stop`; `terminated_rss_abort`; `worker_lost` (worker died without writing its row); `seed_not_started` (wall stop or RSS abort before the seed began) |
| `n_iter` | integer or null | `int(n_iter_[0])` when `completed`; null otherwise |
| `converged` | boolean | section 6.2: true iff `completed`, `n_iter < 20000`, and no `ConvergenceWarning` in this slot; false in every other case, including every slot that did not complete |
| `convergence_warning` | boolean | true iff `warnings` contains an entry whose category is `sklearn.exceptions.ConvergenceWarning` or a subclass |
| `warnings` | list, possibly empty | every warning captured while this decoder's scope was active, in emission order, each as `{category, phase, message_head}`: `category` is the fully qualified class name; `phase` is one of `standardize`, `fit`, `predict_score`; `message_head` is the first two lines of the message, each truncated to 300 characters. Capture uses `warnings.catch_warnings(record=True)` with `simplefilter("always")`, so repeats are not suppressed |
| `fit_error` | object or null | for `fit_exception`: `{exception_type, phase, message_head}` with the same truncation; for `terminated_*` and `worker_lost`: `{exception_type: null, phase: null, message_head: <coordinator stop text>}`; null otherwise |

**Attribution.** A decoder's scope opens immediately before its standardization and closes after its scoring, so every warning raised by that decoder's standardization, fit, or predict-and-score is attributed to that decoder and only that decoder. A warning raised outside all three scopes (stream generation, feature capture, label construction) is recorded in the row-level list `non_decoder_warnings`, with the same entry shape and `phase` one of `streams`, `capture`, `labels`. Every warning is therefore attributed either to exactly one decoder or explicitly to no decoder.

**Validity.** The validity rule is section 6.2 only: a slot is valid iff `converged` is true. A warning of any category other than `ConvergenceWarning` is recorded and attributed but does not by itself change validity; the other section 7 invariants (for example item 5, finiteness) still apply independently. Any slot with `converged = false`, for any reason, makes the seed and D1R2 `INVALID`.

**Rows the worker cannot write.** The worker appends outcome-free ledger events when each decoder scope opens and closes, carrying that slot's schema fields (never an accuracy, bound, prediction, coefficient, or intercept). If a worker dies, or the coordinator ends the run at the six-hour stop or the 12 GiB abort, the coordinator writes the missing rows from the ledger: slots closed in the ledger carry their ledger fields; the slot open at termination has `attempted = true`, `completed = false`, `n_iter = null`, `converged = false`, and `stop_reason` `terminated_wall_stop`, `terminated_rss_abort`, or `worker_lost`; slots never opened have `attempted = false`, `completed = false`, `n_iter = null`, `converged = false`, and `stop_reason` `not_attempted_upstream_failure`, the matching `terminated_*` or `worker_lost` reason, or `seed_not_started`. Each row records `row_source` (`worker` or `coordinator_from_ledger`) and a row-level `seed_error` (`{exception_type, phase, message_head}` for an upstream exception, else null).

**Outcome redaction.** All fields above are outcome-free and are retained on every branch, including `INVALID` and `INVALID_PIPELINE`. On the `INVALID` branch they are the only per-decoder fields written: accuracy, scored predictions, correctness vector, lower bound, coefficients, intercept, coefficient hash, and any readout stay redacted exactly as by the 2d9efd8 outcome-free finalizer. In addition, on the `INVALID` branch `message_head` is set to null for every `warnings` or `fit_error` entry with `phase = predict_score` (category, phase, and exception type are kept), so no text produced while scoring held-out rows reaches disk. The progress ledger stays outcome-free and carries only these schema fields.

## 7. Integrity invariants (any failure is INVALID)

2d9efd8 section 7 items 1-8, with seeds `2200..2219`, and item 6 replaced by:

6. Exactly 20 rows are present, each with all three section 6.3 decoder slots (`pipeline`, `network_positive`, `A`) present and schema-valid, never omitted; and all 60 slots have `converged = true` under section 6.2 (`completed`, `n_iter < 20000`, and no `ConvergenceWarning`). A missing row, a missing slot, a schema-invalid slot, or any slot with `converged = false` is `INVALID`.

## 8. Run order and stop conditions

2d9efd8 section 8, with D1R2 seeds, and step 1 extended by the section 11 item 16 check if Nora accepts it. The section 10 six-hour stop and 12 GiB abort apply. D1R2 never runs any D2, D3, or D4 row.

## 9. Predeclared D1R2 branch table (with stop-loss)

Exhaustive and ordered; earlier rows take precedence. Conditions and codes are 2d9efd8 section 9 with `D1R` read as `D1R2`; the stop-loss is added to rows 1 and 2. Rows 4-6 are ebb90e74 section 10 rows 3 onward with the D1R2 A-decoder status in place of D1's.

| Row | Condition | Code | Action |
|---:|---|---|---|
| 1 | Collision-probe failure, any section 7 invariant failure (including any non-converged decoder under section 6.2), any `INVALID` decoder or seed, six-hour wall stop, RSS abort, or incomplete package | `STOP_D1R2_INVALID` | STOP. No science reading. **Stop-loss: the D1 harness-repair path is closed. D1 returns to Kendrick as a NEEDS KENDRICK DECISION card. No D1R3.** No D2-D4, no r4. |
| 2 | Row 1 not met and pipeline median `< 0.99` | `STOP_D1R2_INVALID_PIPELINE` | STOP. Code defect, not science. No science reading of the network decoders. **Stop-loss: the D1 harness-repair path is closed. D1 returns to Kendrick as a NEEDS KENDRICK DECISION card. No D1R3.** No D2-D4, no r4. |
| 3 | Pipeline `PASS`, `network_positive` `FAIL` or `INCONCLUSIVE` | `CLOSE_D1_REPRESENTATION_LIMITED` | D1 closes as `REPRESENTATION_LIMITED` (reading (b)). No further D1 harness repair. Return to Kendrick with a roadmap decision card on task presentation and encoding. No D2-D4, no r4. |
| 4 | Pipeline `PASS`, `network_positive` `PASS`, `A` `FAIL` | `D1R2_A_FAIL_EBB90E74_ROW3` | ebb90e74 section 10 row 3. Return to Kendrick. No r4 readout or reward change. No D2-D4. |
| 5 | Pipeline `PASS`, `network_positive` `PASS`, `A` `INCONCLUSIVE` | `D1R2_A_INCONCLUSIVE_EBB90E74_ROW4` | ebb90e74 section 10 row 4. Return to Kendrick/spec review. No downstream diagnostic, no r4. |
| 6 | Pipeline `PASS`, `network_positive` `PASS`, `A` `PASS` | `D1R2_PASS_D2_D4_ELIGIBLE` | D1 is `PASS` (by D1R2). D2-D4 only per ebb90e74 section 8 steps 3-6 under a separate run authorization; outcomes governed by ebb90e74 section 10 row 1 and rows 5 onward. No r4 from D1R2 alone. |

Rows are mutually exclusive and exhaustive by the 2d9efd8 section 9 argument. After rows 1-3, no D1 harness repair of any kind remains authorized; rows 4-6 already leave the harness question.

## 10. Resource estimate and budget

Reference host: Mac mini (Macmini8,1), Intel Core i5-8500B 3.0 GHz, 6 cores, 32 GiB DDR4, aib310 interpreter, `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`. Budget unchanged from 2d9efd8 section 10: at most 3 worker processes; abort above 12 GiB total RSS (12,884,901,888 bytes); stop and report past 6 h (21,600 s) wall. Over 6 h is `INVALID` (row 1) and never shortens the map, `K`, or the decoder.

Measured anchors (engineering fields only):

- D1R at `34aae46`: 20 seeds, 3 workers, wall 4,389.5 s, peak total RSS 7,126,728,704 B, peak single-process RSS 2,738,458,624 B; 658 worker-seconds per seed on average, all work included.
- Synthetic benchmark (workspace `lbfgs_cost_bench3.py`; Gaussian matrices of the D1R2 training shapes from a toy generator; no AIB module, identity, or datum): one lbfgs function-and-gradient evaluation of sklearn's own `LinearModelLoss.loss_gradient` costs 0.071 s at 198,000 x 460 alone and 0.150 s with 3 concurrent processes; 0.017 s and 0.024 s at 198,000 x 46.

Memory: lbfgs keeps 10 correction pairs of length 461 regardless of `max_iter`, so per-process and total RSS are unchanged from D1R (peak 7.13 GB of the 12 GiB cap).

Wall time (calculated, workspace `wall_projection.py`):

| Case | Projection |
|---|---|
| D1R-like convergence | about 1.2 h |
| `L` 460-dim fits run to the effective ceiling (section 6.1: 15,000 evaluations, about 2,264 s each at 3 workers) | `L = 2`: 1.6-3.1 h; `L = 8`: 2.9-4.3 h; the 6 h stop is guaranteed not to bind for `L <= 15` of the 40 network fits |
| Worst case: every fit runs to the ceiling (all 40 network fits and 20 pipeline fits) | 10.3-10.8 h, above the 6 h stop |

In the worst case every fit is non-converged, so the run is already `INVALID` under section 6.2; the 6 h stop ends it earlier with the same row 1 code and the same stop-loss. The 6 h stop changes the branch only if more than 15 network fits converge, each only near the ceiling, which would need a convergence profile far from D1R's (18 of 20 seeds converged all three decoders under 2,000 iterations). Accepted as carried by Amanda ruling, t_bbc93d9c.

Result package under 25 MiB as 2d9efd8 section 10. Actual worker count, wall time, peak RSS, interruptions, and retries must be reported.

## 11. Implementation shakedown required before any D1R2 run

2d9efd8 section 11 items 1-15 carried with namespace 32 and seeds `2200..2219`. Item 13 adds the D1R implementation and its tests to "remain unchanged", plus the 2d9efd8 hash pin. Additional required construction tests:

- 11a. Every decoder is constructed with `max_iter=20000` and every other argument as 2d9efd8 section 2.
- 11b. On synthetic fixtures (no D1R2 seed): a fit that stops at the iteration limit, and a fit that stops at the function-evaluation limit with `n_iter < 20000`, are each recorded non-converged with the matching `stop_reason` and make the seed `INVALID`; all three decoders are still fitted and all three section 6.3 slots recorded; a fit exception in one decoder yields `fit_exception` in that slot while the other two are still attempted; an upstream exception yields three `not_attempted_upstream_failure` slots; a simulated worker loss, wall stop, and RSS abort each yield 20 rows with three schema-valid slots each via `coordinator_from_ledger`; a non-`ConvergenceWarning` warning (for example a `RuntimeWarning`) emitted inside one decoder's scope is attributed to that decoder only, and one emitted during capture lands in `non_decoder_warnings`; a row missing any slot fails section 7 item 6; no outcome field, and no `predict_score` message text, reaches disk on the `INVALID` branch.
- 11c. The branch table reaches every row on synthetic row fixtures, and rows 1 and 2 carry the stop-loss action text.

**Proposed item 16, for Nora to accept or reject (not in force unless accepted):** a convergence-only check on the non-diagnostic fixture seed `4242`, at full 2d9efd8 section 2 lengths, fitting the two 460-dim network decoders with the D1R2 settings and recording only `n_iter`, converged flag, and warnings. Network accuracies are not computed (no `predict` on test rows, no correctness vector, no bound) and not recorded. Pass = both converge under section 6.2. If it fails, block to Amanda; the cap is never raised again. Rationale: it is the only pre-data check that the 460-dim fits converge in practice at this budget. Cost: about two 460-dim fits on one seed. Residual risk for Nora to weigh: seed 4242 uses the same frozen r3 architecture, so a failure would be informative about convergence but carries no accuracy.

## 12. Result package, claim boundary, no-oracle statement, and next gate

2d9efd8 section 12, with exactly 20 rows on every branch, each carrying all three section 6.3 decoder slots (the full slot schema, never omitted), `row_source`, `seed_error`, and `non_decoder_warnings`, effective parameters including `max_iter = 20000` (nominal), the effective scipy L-BFGS-B `maxfun = 15000` (default, not passed by scikit-learn), scikit-learn `1.7.2`, scipy `1.15.3` (the versions this spec's effective-ceiling statement was verified against; the run records the installed versions), and the branch code with stop-loss action text. No-oracle statement and claim boundary unchanged.

Next gate: Nora independently reviews the frozen document for numeric closure, preservation of every carried 2d9efd8 element, the convergence rule and diagnostics, the stop-loss, the collision probe, the resource projection, and proposed item 16. This document authorizes no run.
