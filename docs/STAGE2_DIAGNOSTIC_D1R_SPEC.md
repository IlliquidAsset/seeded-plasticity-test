# Stage 2 diagnostic D1R: one-shot D1 harness repair specification

**Status:** FROZEN CANDIDATE, document only. No D1R construction run, seed, decoder fit, or datum may exist until Nora independently approves these bytes, a separate implementation card is approved, and a separate run card authorizes execution.  
**Task/directive:** Project AIB Kanban `t_1337897f`; stored task-body SHA-256 `ee51c44196e6ebb33ad2f1b96a1c0bd61996bd33a83a80005d0616b9034be0c9`.  
**Governing proposal:** Amanda, `t_69a661de`, `STAGE2_D1_NEXT_STEP_PROPOSAL.md`, SHA-256 `22b54934817d0b514a448425506630e7eb3f255ebc7c2a880c3a175c1c8be9fa`.  
**Triggering verdict:** Nora, `t_631a163b`, APPROVE of D1 `INVALID_HARNESS`, branch `STOP_D1_INVALID_HARNESS`, table row 2, package `results_stage2_diagnostic/runs/diagnostic-20261001T192134659758Z-c8cb8c688934-pid6342` at `5348fcc`.  
**Control record read:** AIB `.inbox/CONTROL.json`, epoch 46. This card is doc-only and authorizes no run.  
**Base:** branch `v0.3.0-calibrated`, commit `ca7c3f665e24208041000cf0b4164e8282b40b3a` (verified by `git ls-remote` before writing).  
**Amended document:** `docs/STAGE2_DIAGNOSTIC_SPEC.md` at `ebb90e74d088057f1d37e08ac80ed7bab4e8851e`, SHA-256 `7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf`, referred to below as "ebb90e74". Its bytes are not edited.  
**Frozen Stage 2 r3 specification:** `docs/STAGE2_SPEC.md`, SHA-256 `695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd`. Its bytes are not edited.  
**Date:** 2026-10-01.

## 0. Authority, scope, and stop boundary

ebb90e74 section 10 row 2 reads: "D1 positive control not PASS -> D1 is `INVALID_HARNESS`. Repair only under a new reviewed spec; no downstream diagnostic." This document is that new spec. It amends D1 only, and only in the ways listed in section 2.

Carried by reference, unchanged and byte-for-byte in force:

- ebb90e74 sections 5, 6, 7, and 9 (D2, D3, D4, across-seed bootstrap), every constant and threshold in them, and their seeds `2000..2019` under namespace `30`;
- ebb90e74 section 8 run order for D2-D4 and section 10 rows 3 onward, plus section 10 row 1 as it applies to any later D2-D4 `INVALID`;
- ebb90e74 section 13 package rules and claim boundary for any later D2-D4 package;
- `docs/STAGE2_SPEC.md`, including Stage 2's four exit gates, four controls, section 6 validity rules, and section 10 thresholds;
- the Stage 1 repository-core MSTDPET rule and the Rung 1 result that reward for silence alone produces exactly zero update;
- the r2/r3 result packages and verdicts, and the D1 package at `5348fcc` with its verdict `INVALID_HARNESS`.

The D1 results on seeds `2000..2019` are not reinterpreted, re-scored, pooled with D1R, or used as D1R inputs. In particular the D1 A-decoder median `0.648` remains non-decision-bearing, as Nora ruled, and is not evidence for or against anything in this document.

Owned artifact for this freeze: `docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md` only. No runtime, result, test, frozen-spec, or source file is edited on this card. Rollback is `git revert <this-spec-commit>`.

Hard stops:

1. No D1R execution, performance smoke on a D1R seed, seed screening, feature-map comparison, or partial outcome inspection before independent approval of the spec, then of the implementation, then a separate run authorization.
2. No threshold, feature, lag count `K`, split, decoder setting, seed, or branch rule may change after any D1R datum is inspected. A desired change requires a new spec and fresh review; under section 9 row 3 no further D1 harness repair is permitted at all.
3. Exactly one network feature map (section 5.2) and exactly one pipeline-control map (section 5.3). No alternative map, menu, ablation, or post-data choice among maps.
4. D1R is one-shot. It runs once on seeds `2100..2119`. No rerun on other seeds, no second K, no second decoder.
5. Any `INVALID` stops the sequence and returns to implementation/spec review. It is not a scientific `FAIL`.
6. No D2, D3, D4, r4, Stage 3, Stage 4, Track 4, rhythmic-drive arm, retuning, or mechanism change is authorized by this document.
7. No D1R decoder may enter a training or evaluation path of D2-D4 or any Stage 2 arm.

## 1. Why D1R exists and what it must separate

Observed, from `d1_rows.jsonl` at `5348fcc` (Nora-verified): the D1 positive control (decode `x_(t-1)` from the 60-dimensional smoothed hidden feature map) had per-seed held-out accuracy `0.667` to `0.735`, 9/20 seeds at or above `0.70`, 20/20 per-seed lower bounds above `0.50`, median `0.68835`, all decoders converged (`n_iter` 702 to 1,371 against a cap of 2,000), frozen weights constant. Under ebb90e74 section 4.6 that is `INCONCLUSIVE`, so D1 is `INVALID_HARNESS`.

Two readings fit that result. D1R must separate them rather than assume one:

- **(a) Harness weakness.** The D1 map is only `count20`, `trace20`, `trace25` of hidden spikes. Symbols arrive one per 1 ms integration step, so each 20-25 ms window sums about 20-25 consecutive symbol responses into one number per neuron. The hidden response to `x_(t-1)` is one spike-or-not per neuron at one exact step, and the map blurs it with up to 19 older steps. That blur alone could cap `x_(t-1)` near `0.69`.
- **(b) Representation limit.** In r3 one input spike adds at most `10 mV` against a `16 mV` gap from rest to threshold with `tau_m = 20 ms`. The frozen hidden layer may simply carry the last input weakly in its spikes, so no honest spike-only linear readout reaches `0.70`.

D1R holds the network, streams, split lengths, timing rule, decoder, standardization, bootstrap, and every section 4.6 number fixed, and changes only the temporal resolution of the hidden-spike features. It adds a pipeline-validity control so that a code defect cannot masquerade as reading (b).

## 2. What D1R changes relative to ebb90e74 section 4, and nothing else

| Item | ebb90e74 D1 | D1R |
|---|---|---|
| Seeds | `2000..2019` | `2100..2119` |
| SeedSequence namespace component | `30` | `31` |
| Network feature map | 60 dims: `count20`, `trace20`, `trace25` | 460 dims: the same 60 plus exact-lag hidden spike vectors `h_t, h_(t-1), ..., h_(t-19)` (`K = 19`) |
| Pipeline validity control | none | new: decode `x_(t-1)` from the input-layer spike encoding, same pipeline and timing, PASS iff median `>= 0.99` |
| Positive control | `x_(t-1)` from the 60-dim map | network positive control: `x_(t-1)` from the 460-dim map, section 4.6 predicate |
| A decoder | `x_t` from the 60-dim map | `x_t` from the 460-dim map, section 4.6 predicate |
| Branch table | ebb90e74 section 10 rows 1-4 for D1 | section 9 below; rows 3 onward of ebb90e74 govern after a network-positive-control PASS |
| Resource cap | 5 workers, abort above 8 GiB | at most 3 workers, abort above 12 GiB (section 10) |

Everything else in ebb90e74 section 4 is carried exactly:

- frozen r3 network (ebb90e74 section 2 table): `[2, 20, 2]`, `dt = 1.0 ms`, hidden ALIF with `tau_m = 20.0 ms`, rest/reset `-70.0 mV`, threshold `-54.0 mV`, `tau_a = 200.0 ms`, `beta_a = 1.12 mV`; W1 `U(-10,10)` mV, both W2 rows `U(0,10)` mV, r3 draw order; float64, batch size 1;
- F0 semantics: W1 and W2 frozen from initialization, SFA active, no reward computed or applied, weights must be bitwise constant;
- streams: decoder-train order-2 A stream of 200,000 observations; independent decoder-test order-2 A stream of 12,000 observations; train rows are indices `2,000..199,999` (exactly 198,000 rows); test indices `0..1,999` warm state and features only, `2,000..11,999` scored (exactly 10,000 rows); the test stream starts from the exact r3 start state and zero feature state;
- timing: the feature row at index `t` is formed from state already present when the network would predict `y_t`, before `x_t` is integrated; the current input and target never enter a feature;
- decoder: scikit-learn `LogisticRegression`, L2, `C=1.0`, intercept enabled, `solver="lbfgs"`, `tol=1e-8`, `max_iter=2000`, `class_weight=None`, float64; per-column centering and scaling by training mean and population standard deviation (`ddof=0`), zero-variance column divided by `1.0`; the stored training transform applied unchanged to test; probability `>= 0.5` predicts 1; failure to converge within 2,000 iterations makes that seed invalid; no tuning, cross-validation, alternative C, or substitution;
- ebb90e74 section 4.5 per-seed one-sided 95% lower bound: circular moving-block bootstrap, block length 100 scored steps, exactly 10,000 resamples, 100 circular blocks concatenated and truncated to 10,000 entries, `numpy.percentile(replicates, 5, method="inverted_cdf")`, with the namespace-31 component in section 3.1;
- ebb90e74 section 4.6 numbers exactly (restated in section 6.1).

## 3. Deterministic identities and the collision probe

### 3.1 D1R namespace

A fresh generator means `numpy.random.default_rng(numpy.random.SeedSequence(entropy))`. Root `20261001`; new D1R namespace component `31`. Every D1R object uses namespace 31:

| Object | SeedSequence entropy list |
|---|---|
| W1 then W2 row O1 (r3 draw order) | `[20261001, seed, 31, 1]` |
| W2 row O0 | `[20261001, seed, 31, 21]` |
| order-2 A decoder-train stream | `[20261001, seed, 31, 2]` |
| order-2 A decoder-test stream | `[20261001, seed, 31, 5]` |
| per-seed block bootstrap, A decoder | `[20261001, seed, 31, 60, 1]` |
| per-seed block bootstrap, network positive control | `[20261001, seed, 31, 60, 2]` |

for `seed` in `2100..2119`. The namespace occupies the third entropy position, so `[20261001, seed, 31, k]` is a different tuple from ebb90e74's lag-1 object `[20261001, seed, 30, 31]`; the probe in section 3.3 checks this rather than assuming it. A-stream construction is the r3 section 7.1 procedure (as implemented for ebb90e74 by `_roll_a`) with the namespaced component. Each object has one generator and no generator is shared by two objects. D1R has no tie-coin (no output is read), no background drive, no lag-1 stream, and no across-seed bootstrap (its across-seed predicates are point predicates, as in ebb90e74 section 9 last paragraph). The pipeline validity control has no interval and therefore no bootstrap identity; it reuses the network's input streams, which is required so that the pipeline control and the network decoders read the identical labels.

### 3.2 Seed disjointness

D1R seeds `2100..2119` are disjoint from experimental seeds `0..19`, fixture/qualification seeds `1000..1019`, ebb90e74 diagnostic seeds `2000..2019`, and the non-diagnostic shakedown fixture seed `4242`. Seeds `2000..2019` are not reused for D1R because the repair is motivated by their result. If D2-D4 are later reached, they stay on `2000..2019` under namespace 30 exactly as ebb90e74 freezes them; no D2-D4 datum exists on those seeds.

### 3.3 Collision probe (rerun of ebb90e74 section 3.2 including the D1R identities)

PASS requires all of: every new D1R entropy tuple is unique among new tuples; every new `generate_state(8)` state is unique among new states; no new tuple or state equals any legacy tuple or state; the seed ranges in section 3.2 are disjoint. Legacy = every r2/r3 object, tie-coin, and bootstrap identity enumerated by the approved `tools/stage2_diagnostic_ss_probe.py`, plus every ebb90e74 namespace-30 identity (which became legacy once D1 ran). Legacy-only duplicates are reported separately and are not repaired or hidden.

Pre-data result for this freeze (identity enumeration only; no network, stream, decoder, or D1R datum was generated). Probe: workspace script `d1r_ss_probe.py`, SHA-256 `f042ad245db87f84a155b7d63f796fd6593e563e623b4007462ddf613650f2e4`, which imports the approved base probe `tools/stage2_diagnostic_ss_probe.py` (SHA-256 `ab3dbac8a08fc7108b52a8963202145d4b8acb97ac6eba3720df5c4c21734f58`) for the legacy enumeration and collision functions. Interpreter `/Users/aib-agent/miniforge3/envs/aib310/bin/python` 3.10.20, NumPy 1.26.4.

| Quantity | Value |
|---|---:|
| legacy records | 704 (475 r2/r3 and bootstrap + 229 ebb90e74 namespace 30) |
| legacy unique entropy tuples / states | 698 / 698 |
| legacy-only duplicate groups (entropy / state) | 6 / 6 (the disclosed r3 seed-7 object vs bootstrap identity; unchanged) |
| new D1R records | 120 (20 seeds x 6 objects) |
| new unique entropy tuples / states | 120 / 120 |
| new-within-namespace collisions (entropy / state) | 0 / 0 |
| new-vs-legacy collisions (entropy / state) | 0 / 0 |
| seed ranges disjoint | true |
| identity-state SHA-256 | `78fb3a3a604e5e5617cb5bb224a95e9cbf8ea0b45c77484b476fe05ae14c8706` |
| probe output SHA-256 | `22df208f13fa1ae911375dbf28298fc3399f55b0525e462620fdfc792a2d580f` |
| decision | PASS |

The implementation card must commit an equivalent in-repo probe, reproduce these counts and the identity-state SHA-256 exactly, and retain its code hash, NumPy version, collision lists, and output hash. Any collision involving a D1R identity is `INVALID` and stops before a D1R run.

## 4. Claim boundary

D1R can support only these statements:

> Pipeline: under the fixed split, the pipeline-control map, preprocessing, and L2 logistic decoder below, `x_(t-1)` was or was not recovered from the input-layer spike encoding at the D1R timing (validity of the code path only).

> Network positive control and A decoder: under the fixed split, the single D1R exact-lag spike feature map, preprocessing, and L2 logistic decoder below, `x_(t-1)` (respectively the noisy A target `x_t`) was or was not linearly decodable from the hidden spikes of the frozen random r3 network.

It cannot support "the network can or cannot represent A at all", nonlinear-decoder, voltage-state, trained-weight, architecture-wide, or learning-rule claims. `REPRESENTATION_LIMITED` (section 9) is a bounded operational label: the strongest spike-only linear readout this spec permits did not clear the unchanged bar. It is a reason for a roadmap decision, not a proof that no readout could.

## 5. Feature maps and timing

### 5.1 Notation and timing

For a stream `x_0, x_1, ...` the network is stepped with `online_step` exactly as in ebb90e74 D1: at index `t` the feature row is captured first, then `x_t` is integrated. Let:

- `h_t` in `{0,1}^20`: the hidden spike vector already present at index `t` (in the existing code, `net._spikes[0]` before the step at index `t`). `h_t` is the hidden layer's first response to `x_(t-1)`, because the transition at index `t-1` integrates `x_(t-1)` into the hidden layer and emits `h_t`. `h_0 = 0`.
- `u_t` in `{0,1}^2`: the input-layer spike vector most recently integrated before index `t`, that is the one-hot encoding `ONEHOT[x_(t-1)]` for `t >= 1`, and `u_0 = 0`. This is the input-layer analogue of `h_t` at the same timing: the last input-layer state already present before `x_t` is integrated. `u_t` never contains `x_t`.
- For any lag `k`, `h_(t-k)` and `u_(t-k)` with `t-k < 0` are the zero vector. All such rows lie inside the 2,000-step warm-up and are never fitted or scored.

### 5.2 The single D1R network feature map

Declared before any data: **`K = 19`**.

```text
count20_t = sum(h_j for j=t-19..t)
trace20_t = exp(-1/20) * trace20_(t-1) + h_t
trace25_t = exp(-1/25) * trace25_(t-1) + h_t
phi_t     = concatenate(count20_t, trace20_t, trace25_t)            # 60 dims, ebb90e74 section 4.3 exactly
lag_t     = concatenate(h_t, h_(t-1), h_(t-2), ..., h_(t-19))         # 20 x 20 = 400 dims
phiR_t    = concatenate(phi_t, lag_t)                                 # 460 dims
```

Column order is fixed: the 60 ebb90e74 columns in their existing order, then lag 0 neurons `0..19`, lag 1 neurons `0..19`, ..., lag 19 neurons `0..19`. Feature state starts at zero with the network and advances continuously within each stream.

Permitted inputs: hidden spikes only. Not a feature: membrane voltage, adaptation state `a`, adaptive threshold, eligibility, STDP traces of the plastic synapses, weights, output spikes, current input `x_t`, any input spike, target, future observation, tie-coin, or arm label.

**Written justification for K = 19 (symbol rate versus window).**

1. Symbol rate: one input symbol per `dt = 1 ms` step. The D1 map's shortest window is `count20`, a 20-step window, so it sums the responses to 20 consecutive symbols into one number per neuron. `trace20` and `trace25` are exponentially weighted sums over the same 20-25 step scale. Reading (a) says this many-symbols-per-feature blur, not the network, caps `x_(t-1)`.
2. The repair should change resolution, not horizon. `K = 19` gives exactly 20 exact lags, `h_t` through `h_(t-19)`, which is the `count20` window resolved one symbol per lag. No time scale is added beyond the 20 ms window that ebb90e74 already declared, which is also the hidden membrane time constant `tau_m = 20 ms` and the STDP trace constant. A larger K would add history beyond the D1 window and confound "sharper" with "longer". A smaller K would leave part of the declared window still summed and so not fully test the blur reading.
3. It covers the targets' direct carriers. `x_(t-1)` first reaches the hidden spikes at `h_t` (lag 0); the order-2 A target's inputs `x_(t-1)` and `x_(t-2)` first reach `h_t` and `h_(t-1)`. The remaining lags 2..19 let the linear decoder condition on the recent spike and reset history that determines whether a given input pushes a neuron over threshold, within one membrane time constant, where a single input's residual is `exp(-k/20)` of its size at lag `k` (`0.39` at `k = 19`).
4. It nests the D1 map exactly. `count20_t` equals the sum of the 20 lag blocks, so the D1R map's linear span contains the D1 map's. D1R cannot be weaker than D1 for any linear readout that D1 could express. That gives an exact algebraic construction check (section 11 item 3).
5. Fit size stays well posed: 460 features against 198,000 training rows (about 430 rows per feature) under L2 with `C=1.0`.

This is the only network map. K is not swept, tuned, or revisited after data.

### 5.3 The pipeline-control map

The identical feature operator, applied to the 2-dimensional input-layer encoding `u_t` instead of `h_t`:

```text
psi_t = concatenate(count20(u)_t, trace20(u)_t, trace25(u)_t, u_t, u_(t-1), ..., u_(t-19))   # 6 + 40 = 46 dims
```

with the same recurrences, the same `K = 19`, the same column-order rule, and the same accumulator code path parameterized only by width (`n = 2` instead of `n = 20`). It is computed on the same decoder-train and decoder-test streams, the same row slices, and the same label vectors as the network positive control. Since `u_t = ONEHOT[x_(t-1)]`, a correct pipeline recovers `x_(t-1)` essentially perfectly; a label shift, feature/label misalignment, wrong lag indexing, standardization error, or scoring defect drops it. It tests the code path only and carries no scientific content.

Reachability (calculated, no data). The lag-0 block of `psi_t` is the one-hot of the label, so the training rows are linearly separable. sklearn minimizes `0.5*||w||^2 + C * sum_i logloss_i` with `C = 1.0`. Take the comparison vector that puts weight `beta = 10` on the standardized lag-0 label column and zero elsewhere, intercept 0. For a label fraction `p` in `[0.45, 0.55]` (the A stream's pair chain is doubly stochastic, so its stationary label mean is exactly 0.5), every training row then has margin at least `beta * min(p,1-p)/sqrt(p(1-p)) >= 9.0`, so the objective at that vector is at most `198,000 * log(1+exp(-9.0)) + 50 < 75`. The optimum's objective is no larger, and every misclassified training row contributes at least `log 2`, so the fitted pipeline decoder misclassifies at most 108 of 198,000 training rows (training accuracy `>= 0.9994`). The range for `p` is for concreteness only: for any `p` in `(0,1)` the margin is `beta * sqrt(min(p,1-p)/max(p,1-p))`, and choosing `beta` to restore a margin of 9 gives the same conclusion with a slightly larger regularization term (for example `p = 0.30` gives at most 180 training errors). Held-out rows come from the same generator with the same exact one-hot relation, so a correct implementation is expected at or near `1.0`; the `0.99` bar leaves margin for that. This bound is not data. Section 11 item 15 adds a construction check on the non-diagnostic fixture seed `4242`.

## 6. Decoders and readouts

Fit three separate binary decoders per seed with the section 2 decoder and standardization. Network decoders share one training feature matrix:

| Decoder | Features | Label | Per-seed lower bound | Role |
|---|---|---|---|---|
| `pipeline` | `psi_t` (46) | `x_(t-1)` | not computed | validity only |
| `network_positive` | `phiR_t` (460) | `x_(t-1)` | section 4.5, component `[20261001, seed, 31, 60, 2]` | decision-bearing control |
| `A` | `phiR_t` (460) | `x_t` (realized noisy A target) | section 4.5, component `[20261001, seed, 31, 60, 1]` | decision-bearing readout |

Labels on scored train rows `t = 2,000..199,999` and scored test rows `t = 2,000..11,999` are `x_(t-1)` (equal to `x[1999:-1]` of the stream) for `pipeline` and `network_positive`, and `x_t` (equal to `x[2000:]`) for `A`, derived by the coordinator by direct slicing, as in the approved D1 implementation.

### 6.1 Section 4.6 predicate, restated exactly

Applied separately to `network_positive` and to `A`, over the 20 per-seed held-out accuracies and lower bounds:

- `PASS`: median held-out accuracy `>= 0.70` and at least 15/20 per-seed one-sided lower bounds strictly `> 0.50`;
- `FAIL`: median held-out accuracy `< 0.55`;
- `INCONCLUSIVE`: neither PASS nor FAIL;
- `INVALID`: any deterministic identity, split, feature timing, solver, convergence, row count, finite-value, or schema invariant fails.

### 6.2 Pipeline validity control

Over the 20 per-seed held-out accuracies of `pipeline`:

- `PASS`: median held-out accuracy `>= 0.99`;
- otherwise (median `< 0.99`): `INVALID_PIPELINE`.

A pipeline decoder that fails to converge, or any pipeline integrity failure, is `INVALID` under section 6.1's last bullet and is routed by section 9 row 1.

### 6.3 D1R status

- `INVALID` if any section 6.1 `INVALID` condition holds for any decoder or seed, or the collision probe or package is incomplete;
- else `INVALID_PIPELINE` if the pipeline control is not `PASS`;
- else `REPRESENTATION_LIMITED` if `network_positive` is `FAIL` or `INCONCLUSIVE`;
- else D1R takes the `A` decoder's status: `PASS`, `FAIL`, or `INCONCLUSIVE`.

Every status is computed mechanically by code from the rows; no narrative status is permitted.

## 7. Integrity invariants (any failure is INVALID)

1. Exactly 20 rows, seeds exactly `2100..2119`, each once; rows sorted by seed before aggregation.
2. Per seed: 198,000 train rows, 10,000 scored test rows, 2,000 warm-up rows; network feature width 460; pipeline feature width 46.
3. Coordinator-derived identities match the worker row: initial-weight hash, train and test input hashes, and the three train and three test label hashes.
4. F0: W1 and W2 bitwise equal to initialization after both streams, in 20/20 seeds.
5. All features, means, scales, coefficients, intercepts, accuracies, and lower bounds finite.
6. All three decoders converged (`n_iter < 2000`) in 20/20 seeds.
7. The `count20` block of `phiR_t` equals the sum of its 20 lag blocks exactly at every captured row (float64 sums of 0/1 integers are exact); the same holds for `psi_t`. Checked by hash-free assertion during capture and recorded as a boolean per seed.
8. No D1R row is read, and no D1R outcome is computed, until all 20 seed rows are present and schema-valid. The progress ledger is outcome-free (identity, status, and resource fields only).

## 8. Run order and stop conditions

After this document, a separate implementation, and a separate run authorization are each independently approved:

1. Run the deterministic shakedown on the non-diagnostic fixture seed `4242` only, and the section 3.3 collision probe. No D1R seed is touched.
2. Run D1R on seeds `2100..2119`: all three decoders for every seed, then compute the D1R status (section 6.3) and the branch (section 9).
3. Freeze the complete package (section 12) and send it to Nora. No section 9 branch action occurs before review.

D1R never runs any D2, D3, or D4 row. A D1R `PASS` makes D2-D4 eligible only under ebb90e74 section 8 steps 3-6, on seeds `2000..2019`, under their own separate implementation review and run authorization.

## 9. Predeclared D1R branch table

Exhaustive and ordered. Earlier rows take precedence. Rows 4-6 are ebb90e74 section 10 rows 3 onward, applied with D1R's A-decoder status in the place of D1's status.

| Row | Condition | Code | Action |
|---:|---|---|---|
| 1 | Collision-probe failure, any section 7 invariant failure, any `INVALID` decoder or seed, or incomplete package | `STOP_D1R_INVALID` | STOP. Return to implementation/spec review. No science reading. No D2-D4, no r4. |
| 2 | Row 1 not met and pipeline median `< 0.99` | `STOP_D1R_INVALID_PIPELINE` | STOP. Implementation review. A code defect, not science. No science reading of the network decoders. No D2-D4, no r4. |
| 3 | Pipeline `PASS`, `network_positive` `FAIL` or `INCONCLUSIVE` | `CLOSE_D1_REPRESENTATION_LIMITED` | D1 closes as `REPRESENTATION_LIMITED` (reading (b)). No further D1 harness repair is permitted. Return to Kendrick with a roadmap decision card on task presentation and encoding. No D2-D4, no r4. |
| 4 | Pipeline `PASS`, `network_positive` `PASS`, `A` `FAIL` | `D1R_A_FAIL_EBB90E74_ROW3` | ebb90e74 section 10 row 3: the specified linear decoder cannot recover A under the fixed split. Task/architecture pairing needs roadmap revision. Return to Kendrick. No r4 readout or reward change. No D2-D4. |
| 5 | Pipeline `PASS`, `network_positive` `PASS`, `A` `INCONCLUSIVE` | `D1R_A_INCONCLUSIVE_EBB90E74_ROW4` | ebb90e74 section 10 row 4: evidence is not decision-complete. Return to Kendrick/spec review. No downstream diagnostic and no r4. |
| 6 | Pipeline `PASS`, `network_positive` `PASS`, `A` `PASS` | `D1R_PASS_D2_D4_ELIGIBLE` | D1 is `PASS` (by D1R). D2-D4 may proceed only per ebb90e74 section 8 steps 3-6 under a separate run authorization; their outcomes are governed by ebb90e74 section 10 row 1 (any D2-D4 INVALID) and rows 5 onward, unchanged. No r4 from D1R alone. |

The rows are mutually exclusive and cover every combination: row 1 catches every invalid state; given validity, the pipeline is PASS or not (row 2); given pipeline PASS, the network positive control is PASS or not (row 3); given both, A is exactly one of FAIL, INCONCLUSIVE, PASS (rows 4-6).

## 10. Resource estimate and budget

Reference host: Mac mini, Intel Core i5-8500B 3.0 GHz, 6 logical CPUs, 32 GiB RAM, interpreter `/Users/aib-agent/miniforge3/envs/aib310/bin/python`, `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `PYTHONHASHSEED=0`.

Measured anchor (D1 at `5348fcc`, 60-dim map, two decoders): 20 seeds in 705.2 s wall at 5 workers, about 170-190 s per seed per worker; peak total RSS 4,175,732,736 bytes; peak single-process RSS 794,091,520 bytes; `n_iter` 702-1,371.

Calculated D1R load per seed:

- network simulation of both streams: unchanged from D1;
- training feature matrix `200,000 x 460 x 8 B = 736 MB`, plus a standardized copy `198,000 x 460 x 8 B = 729 MB`; test matrix `12,000 x 460 x 8 B = 44 MB`; pipeline matrices about `80 MB`;
- two 460-dim logistic fits at about 7.7x the per-iteration cost of the 60-dim fits, plus one small 46-dim fit.

Estimated per-process peak about 2.5-3.0 GiB. Budget: **at most 3 worker processes**, estimated total RSS at most about 9 GiB; **abort rather than swap if total RSS exceeds 12 GiB**. Estimated wall time `0.5-3.0 h` for the 20 seeds at 3 workers (the wide upper end allows for the 460-dim fits needing more lbfgs iterations than the 60-dim fits), plus `0.1-0.3 h` for aggregation, hashing, and reports. If wall time exceeds 6 h the run is stopped and reported, not shortened by any change to K, the map, or the decoder. Result package under 25 MiB (coefficients `461 x 2` per seed for network decoders, no raw feature matrices serialized; feature hashes only).

Worker count is an engineering parameter, not a scientific one. If the section 8 step 1 shakedown on seed `4242` measures a per-process peak that projects above 12 GiB at 3 workers, the run uses fewer workers (minimum 1). It never uses more than 3, and the feature map, K, and decoder never change for resource reasons. Actual worker count, wall time, peak RSS, interruptions, and retries must be reported.

## 11. Implementation shakedown required before any D1R run

A later implementation card must pass deterministic construction tests that prove, at minimum:

1. Namespace-31 identities in section 3.1 are used for every D1R object, with r3 weight distributions and draw order; D1R seeds are exactly `2100..2119`.
2. `phiR_t` columns 0-59 are bit-identical to the approved ebb90e74 `FeatureAccumulator` output on a fixed fixture.
3. On hand-computed spike fixtures, the lag block at row `t` equals `h_t, ..., h_(t-19)` in the declared order, zero-filled before index 0; and `count20` equals the sum of the 20 lag blocks.
4. Neither `phiR_t` nor `psi_t` can contain or depend on current `x_t`: forcing a different `x_t` leaves row `t` unchanged and changes row `t+1` only.
5. `u_t` equals `ONEHOT[x_(t-1)]` for `t >= 1` and zero at `t = 0`; `psi_t` uses the identical accumulator code path with width 2.
6. Label hashes: `pipeline` and `network_positive` labels equal `x[1999:-1]`; `A` labels equal `x[2000:]`; derived by the coordinator independently of the worker.
7. Training-only standardization: forced test-feature changes cannot alter stored train means/scales or coefficients.
8. F0 weights bitwise constant; no reward computed; evaluation non-mutating.
9. Per-seed block bootstrap uses component `[20261001, seed, 31, 60, q]`, `q = 1` A and `q = 2` network positive control, and no other generator.
10. Section 6 readouts and section 9 branch codes are computed mechanically from rows; every one of rows 1-6 is reached on synthetic row fixtures (no D1R datum).
11. The progress ledger contains no accuracy, bound, coefficient, or decoder-outcome field.
12. The in-repo D1R collision probe reproduces section 3.3 exactly.
13. The approved ebb90e74 D1 path and its tests remain unchanged; both frozen-spec hash pins remain green.
14. Maintained repository tests pass (`/Users/aib-agent/miniforge3/envs/aib310/bin/python -m pytest -q`).
15. Pipeline reachability on the non-diagnostic fixture seed `4242` only: the pipeline decoder alone, at full section 2 lengths, reaches held-out accuracy `>= 0.99`. On seed `4242` the network decoders are not fitted or scored. This checks the code path; it is not a D1R datum and sets no threshold.

These are construction tests on fixtures and seed `4242`. They are not permission to execute a D1R seed, inspect an accuracy, or estimate a threshold.

## 12. Result package, claim boundary, no-oracle statement, and next gate

A later approved D1R run package (new run-unique directory, for example under `results_stage2_diagnostic_d1r/runs/`) must contain:

- exact source commit, this spec's commit and SHA-256, ebb90e74 and Stage 2 spec hashes, host, Python/NumPy/PyTorch/scikit-learn/psutil versions, argv, environment, worker count, clean/dirty state, and effective parameters including `K = 19`, feature widths 460 and 46, and all section 6 thresholds;
- the collision-probe record;
- one complete row per seed: identity hashes, feature hashes, the section 7 item 7 nesting boolean, and for each of the three decoders: accuracy, scored count, label hashes, training-only normalization hash, coefficients and intercepts and their hash, `n_iter`, convergence, and (network decoders only) the per-seed lower bound;
- the pipeline, network-positive, and A readouts; the D1R status; the machine-readable section 9 branch code and row;
- a plain-language summary naming what is supported and what is not;
- a SHA-256 manifest for every result artifact.

No-oracle statement: the target is used only as an offline decoder label after feature capture. It is never a network input, hidden-state label, plasticity feature, state reset, initialization signal, or internal policy or world-model component. No reward is computed and no weight changes. No LLM logic enters the network, the feature path, the decoder, or the simulator.

Claim boundary: passing D1R does not pass Stage 2. D1R is only linear decodability under one frozen spike-only feature map, one split, and one decoder. `REPRESENTATION_LIMITED` is bounded as in section 4. None of D1R's outcomes establishes general learning, continual learning, developmental progress, biological equivalence, or an r4 result, and none changes a Stage 2 gate, control, validity rule, or the Stage 1 rule.

Next gate: Nora independently reviews this exact document for numeric closure, leakage and timing, the K justification, pipeline-control reachability, deterministic identity and the collision probe, branch completeness, and preservation of every frozen artifact. Only after APPROVE may a separate implementation card be created. This document itself authorizes no run.
