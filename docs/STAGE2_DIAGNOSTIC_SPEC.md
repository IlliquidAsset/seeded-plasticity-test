# Stage 2 D1-D4 diagnostic specification

**Status:** FROZEN CANDIDATE, document only. No diagnostic may run until Nora independently approves these bytes.  
**Task/directive:** Project AIB Kanban `t_665f08bf`; exact stored task-body SHA-256 `7064f9bd7037205c5f344da29f98c7b8287a9a0f03823e893465b4ee98327d88`.  
**Decision:** Kendrick, 2026-10-01, via `t_61f7cf65`: diagnose first and include the fixed background-noise contrast.  
**Control record read:** AIB `.inbox/CONTROL.json`, epoch 46. It prohibits science execution without a new qualifying directive; this card is doc-only and authorizes no run.  
**Frozen bench baseline:** branch `v0.3.0-calibrated`, commit `061ef95b7fda6417d5bc2892cb22b073c6b490c5`.  
**Frozen Stage 2 r3 specification:** `docs/STAGE2_SPEC.md` at `def4b36220762a906f360ea44b9d82102c9cb4d2`, SHA-256 `695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd`.  
**Date:** 2026-10-01.

## 0. Authority, scope, and stop boundary

This document specifies four diagnostics outside Stage 2. They are not Stage 2 arms. They do not re-score r2 or r3, spend experimental seeds `0..19`, advance a phase, or license a continual-learning, biological, behavioral, or developmental claim.

The following stay byte-for-byte and conceptually unchanged:

- `docs/STAGE2_SPEC.md`, including its four exit gates, four controls, section 6 validity rules, and section 10 thresholds;
- the Stage 1 repository-core MSTDPET rule and the Rung 1 result that reward for silence alone produces exactly zero update;
- the 2-20-2 r3 architecture, ALIF/LIF parameters, MSTDPET parameters, one-global-scalar reward, and fair tie-coin, except where a diagnostic explicitly changes only the task target or adds the separately declared fixed drive;
- r2 and r3 result packages and verdicts.

Owned artifact for this freeze: `docs/STAGE2_DIAGNOSTIC_SPEC.md` only. No runtime, result, test, frozen-spec, or source file may be edited on this card. Rollback is `git revert <this-spec-commit>`.

Hard stops:

1. No diagnostic execution, performance smoke test, seed screening, constant sweep, or partial outcome inspection before independent approval.
2. No parameter, threshold, feature, split, source count, drive rate, drive weight, or branch rule may be changed after any diagnostic datum is inspected. A desired change requires a new spec and fresh review.
3. Any `INVALID` result stops the sequence and returns to spec review. It is not a scientific `FAIL`.
4. NaN/Inf, an out-of-bounds plastic weight, mismatched paired-object hash, duplicate/missing row, unapproved file change, or seed-identity failure is `INVALID`.
5. No auxiliary decoder from D1 may enter a training or evaluation path in D2-D4 or any future Stage 2 arm.
6. No Stage 3, Stage 4, Track 4, rhythmic-drive arm, parameter tuning, or extra post-data contrast is authorized.
7. Positive diagnostic findings remain simulated bench findings. They carry the no-oracle statement in section 13 and require Nora review before use.

## 1. Prior evidence and bounded question

Stage 2 r3 at `061ef95` was an honest `INVALID_OR_MECHANISM_FAIL + NO_A_COMPETENCE` result. The report-only diagnostics show median hidden rate falling from about 44.04 Hz in frozen A to 12.44 Hz in primary A and 5.19 Hz in primary B, while the both-silent fraction rose from about 0.7147 to 0.9211 to 0.9736. Primary held-out A was 0.50062 despite a median A reward/signed-output-weight-change correlation of 0.74324. Sources: `results_stage2_r3/DIAGNOSTICS.json`, `summary.json`, and `KENDRICK_SUMMARY.md` at `061ef95`.

Correction carried forward: 9/20 primary `A_pre` seeds had zero single-output/non-tie evaluation events. This is not the same as no output firing. Eight of those nine had `both_fire_fraction > 0` in `results_stage2_r3/runs.jsonl`.

The four bounded questions are:

- D1: Can one specified fixed linear decoder recover the noisy A target from frozen random hidden activity under one fixed train/test split? This does not ask whether the architecture can represent A under any decoder or parameterization.
- D2: At which 1,000-step checkpoint does primary activity first fall below half of its paired frozen baseline, and does the hidden-layer onset lead the output-layer onset, the reverse, or neither?
- D3: Can the unchanged rule and harness learn a balanced one-step-memory target under the same budget?
- D4: Does one frozen, task-independent background drive prevent silence, and if so does it also improve held-out A competence rather than merely produce activity?

## 2. Shared harness

Unless a diagnostic says otherwise, reuse r3 exactly:

| Quantity | Frozen value |
|---|---|
| Architecture | `[2, 20, 2]`: 20 hidden ALIF neurons; non-adaptive `O1` and `O0` LIF outputs |
| Integration step | `dt = 1.0 ms` |
| Membrane | `tau_m = 20.0 ms`; rest/reset `-70.0 mV`; threshold `-54.0 mV`; no synaptic filter |
| SFA | hidden `tau_a = 200.0 ms`, `beta_a = 1.12 mV` |
| Weights | W1 `U(-10,10)` mV, bounds `[-10,10]`; both W2 rows `U(0,10)` mV, bounds `[0,10]` |
| Plasticity | MSTDPET, `gamma = 0.25 mV`, `tau_plus = tau_minus = 20.0 ms`, `tau_elig = 25.0 ms`, `a_plus = a_minus = 1/25` |
| Reward | one global scalar `r_t = (2*y_t-1)(z1_t-z0_t)`, applied on the following transition through each synapse's own eligibility |
| Readout | `O1` alone -> 1; `O0` alone -> 0; a tie -> paired fair tie-coin; the coin never enters the network or reward |
| A task | r3 noisy order-2 XOR, with target `y_t = x_t`, 10% independent innovation |
| Training length | one A phase, 200,000 integration steps, developmental age 0 through 199,999 ms; no B phase in these diagnostics |
| Held-out evaluation | 12,000 integration steps from the r3 start state; indices `0..1,999` warm-up, `2,000..11,999` scored; 10,000 scored observations |
| Precision | float64, batch size 1 |
| Diagnostic seeds | exactly `2000..2019`, one independent statistical unit per seed |

The output at index `t` is read and graded before input `x_t` is integrated, exactly as in r3. The current target or current input is never exposed to a feature or network state used to predict that target.

All paired conditions for one seed use byte-identical initial weights, task streams, held-out streams, tie-coins, and, where applicable, background-drive realization. Parallel scheduling may alter wall time only. Rows are sorted by diagnostic, condition, and seed before aggregation.

## 3. Deterministic identities and the collision probe

### 3.1 Diagnostic namespace

Every diagnostic object is drawn from a new namespace. A fresh generator means:

```text
def rng(entropy):
    return numpy.random.default_rng(numpy.random.SeedSequence(entropy))
```

The root remains `20261001`; the new diagnostic namespace component is `30`.

| Object | SeedSequence entropy list |
|---|---|
| W1 then W2 row O1 | `[20261001, seed, 30, 1]` |
| W2 row O0 | `[20261001, seed, 30, 21]` |
| order-2 A training stream | `[20261001, seed, 30, 2]` |
| order-2 A held-out stream | `[20261001, seed, 30, 5]` |
| A-end evaluation tie-coin | `[20261001, seed, 30, 22, 1]` |
| lag-1 training stream | `[20261001, seed, 30, 31]` |
| lag-1 held-out stream | `[20261001, seed, 30, 32]` |
| lag-1 evaluation tie-coin | `[20261001, seed, 30, 22, 2]` |
| D4 background drive | `[20261001, seed, 30, 40]` |
| D1 per-seed block bootstrap, target `q` | `[20261001, seed, 30, 60, q]`, `q=1` A and `q=2` positive control |
| Across-seed bootstrap metric `k` | `[20261001, 30, 7, k]` |

The weight distributions and draw order are the r3 procedures, but their entropy is namespaced as above. A-stream construction is the r3 section 7.1 procedure with the namespaced component. Each object has one generator and no generator is shared by two objects.

### 3.2 Required pre-execution probe

Before implementation is approved to execute a diagnostic, an `ss_probe` must enumerate and call `SeedSequence.generate_state(8)` for:

- all r2/r3 object identities for seeds `0..19` and `1000..1019`, components `1..6`, `21`, `22` and tie-coin checkpoints;
- all r2/r3 bootstrap identities `[20261001, 7, k]`, `k=0..14` and `100..119`;
- every identity in section 3.1 for seeds `2000..2019` and every metric ID in section 9.

The frozen r3 spec already discloses the legacy entropy-list identity between seed-7 object components `[20261001,7,k]`, `k=1..6`, and bootstrap components with the same entropy lists. The diagnostic does not repair or hide that historical collision. PASS requires: every new diagnostic entropy tuple is unique among new tuples; every new generated eight-word state is unique among new states; and no new tuple or state equals any r2/r3 object or bootstrap tuple/state. The probe must separately report the known legacy-only duplicate count. Any collision involving a new diagnostic identity is `INVALID` and stops before a diagnostic run. The probe's code, NumPy version, object counts, collision lists, and SHA-256 output must be retained. Distinct seeds alone are not accepted as a substitute for this check.

## 4. D1: fixed linear decodability

### 4.1 Claim boundary

D1 can support only this statement:

> Under the fixed split, feature map, preprocessing, and L2 logistic decoder below, the noisy A target was or was not linearly decodable from the hidden activity of the frozen random r3 network.

It cannot support “the network can/cannot represent A at all,” nonlinear-decoder, trained-weight, architecture-wide, or learning-rule claims.

### 4.2 Network and streams

For every seed, construct the shared r3 network and freeze both W1 and W2 from initialization. SFA remains active. No reward is computed or applied. Run two disjoint order-2 A streams:

- decoder-train stream: 200,000 observations from component 2;
- decoder-test stream: 12,000 observations from component 5, generated independently, with the first 2,000 used only to warm dynamic state and features and the final 10,000 scored.

The decoder-training rows are indices `2,000..199,999`, exactly 198,000 rows. Indices `0..1,999` warm the network and feature recurrences and are excluded. Test fitting, threshold selection, and test-based normalization are prohibited.

### 4.3 Feature map and timing

Let `h_t` be the 20-dimensional hidden spike vector already present when the network predicts target `y_t`, before `x_t` is integrated. Maintain, per hidden neuron:

```text
count20_t = sum(h_j for j=t-19..t), with unavailable j treated as 0
trace20_t = exp(-1/20) * trace20_(t-1) + h_t
trace25_t = exp(-1/25) * trace25_(t-1) + h_t
```

Time constants are milliseconds because `dt=1 ms`. `20 ms` is the existing membrane and STDP pre/post-trace time constant. `25 ms` is the existing eligibility time constant. The 60-dimensional feature is:

```text
phi_t = concatenate(count20_t, trace20_t, trace25_t)
```

No output spike, membrane voltage, adaptation state, eligibility, weight, current `x_t`, target, future observation, tie-coin, or arm label is a feature. Feature state starts at zero with the network and advances continuously within each stream. The independent test stream starts from the exact r3 evaluation state and zero feature state.

### 4.4 Decoder

Fit two separate binary decoders per seed on the identical training feature matrix:

1. A decoder: label `x_t`, the realized noisy A target.
2. Mandatory positive control: label `x_(t-1)`.

For the first two generated observations, no labels are scored or fitted; the 2,000-step warm-up already excludes them.

The decoder is scikit-learn `LogisticRegression` with L2 penalty, `C=1.0`, intercept enabled, `solver="lbfgs"`, `tol=1e-8`, `max_iter=2000`, and class weights `None`. It uses float64. Each feature column is centered and scaled by training-set mean and population standard deviation (`ddof=0`); a zero-variance column is divided by 1.0. The stored training transform is applied unchanged to test. Probability `>=0.5` predicts 1. No tuning, cross-validation, alternative C, alternate feature window, or convergence-result substitution is permitted. Failure to converge within 2,000 iterations makes that seed invalid.

### 4.5 Per-seed confidence interval

For each decoder and seed, form the length-10,000 correctness vector. Its one-sided 95% lower interval is a circular moving-block bootstrap with block length 100 scored steps, exactly 10,000 resamples, and the target-specific component in section 3.1. Each replicate concatenates 100 independently sampled circular blocks, truncates to 10,000 entries, and computes mean correctness. The lower bound is `numpy.percentile(replicates, 5, method="inverted_cdf")`.

### 4.6 Readout

Apply the same predicate separately to the A decoder and positive control:

- `PASS`: median of the 20 held-out accuracies is `>= 0.70`, and at least 15/20 per-seed one-sided lower bounds are strictly `> 0.50`.
- `FAIL`: median held-out accuracy is `< 0.55`.
- `INCONCLUSIVE`: neither PASS nor FAIL.
- `INVALID`: any deterministic identity, split, feature timing, solver, row count, finite-value, or schema invariant fails.

D1 is `INVALID_HARNESS` unless the `x_(t-1)` positive control is `PASS`. If the positive control passes, D1 takes the A decoder's PASS/FAIL/INCONCLUSIVE status.

## 5. D2: checkpointed drift localization

### 5.1 Conditions

Run the primary plastic arm `P` through the 200,000-step order-2 A phase, paired with frozen-from-start `F0`, for seeds `2000..2019`. Use the shared objects from section 3. D2 and D4 no-drive conditions are the same rows and must not be rerun.

### 5.2 Checkpoints and fields

A checkpoint closes every non-overlapping 1,000-step window: steps `0..999`, `1,000..1,999`, ..., `199,000..199,999`, exactly 200 checkpoints per condition and seed. At every checkpoint record:

- `hidden_rate_hz`: median of the 20 per-neuron rates in that 1,000-step window;
- `output_rate_hz_O1` and `output_rate_hz_O0` in that window;
- `output_layer_rate_hz`: median of the two per-output rates;
- `both_silent_fraction` in that window;
- for W1 and W2 separately: mean, min, max, lower-bound hit count, upper-bound hit count;
- `w1_lower_fraction = w1_lower_hits / 40` and `w2_lower_fraction = w2_lower_hits / 40`;
- step index, seed, condition, stream/weight hashes, and finite/bounds flags.

Bounds and exact-hit semantics are r3 section 8.1: W1 lower/upper `-10/+10 mV`, W2 lower/upper `0/+10 mV`, float64 equality. A schema assertion equivalent to r3 section 8.1 refuses a row missing any field, a non-finite numeric value, an out-of-range count, a wrong checkpoint index, or a hash mismatch.

### 5.3 Onset rule

For layer `L` in `{hidden, output}` and checkpoint `c`, define:

```text
ratio(seed,L,c) = rate_P(seed,L,c) / rate_F0(seed,L,c)
```

A checkpoint whose F0 rate is exactly zero is not onset-eligible for that layer, because the ratio is undefined. The onset is the first eligible checkpoint with `rate_P < 0.50 * rate_F0`. If no checkpoint qualifies, onset is `+infinity` and is serialized as JSON `null` plus `onset_detected=false`, never as non-finite JSON.

For one seed:

- `HIDDEN_FIRST` if hidden onset is finite and at least 5,000 steps earlier than output onset, treating finite as earlier than `+infinity`;
- `OUTPUT_FIRST` by the symmetric rule;
- `CO_ONSET` otherwise, including both undetected, an absolute separation under 5,000 steps, or contradictory/undefined eligible windows.

Across seeds:

- `LOCALIZED_HIDDEN` if at least 14/20 seeds are `HIDDEN_FIRST`;
- `LOCALIZED_OUTPUT` if at least 14/20 seeds are `OUTPUT_FIRST`;
- `CO_ONSET` otherwise.

Report the W1 lower-bound fraction and W2 lower-bound fraction at each finite layer onset, plus the fractions at the earlier onset used for the seed label. An undetected onset reports `null` for its at-onset fractions.

### 5.4 Readout

D2 has a process verdict and a numeric localization result; localization is not a competence gate.

- process `PASS`: exactly 2 conditions x 20 seeds x 200 checkpoints are present, every section 5.2 field passes schema, and exactly one localization label is computed by section 5.3;
- process `INVALID`: any expected row/field is absent or duplicated, any invariant fails, or the label is not mechanically derivable.

For the card-level PASS/FAIL/INVALID vocabulary, map the numeric result without changing the process verdict:

- `D2_PASS_LOCALIZED_HIDDEN`: process PASS and at least 14/20 seeds are `HIDDEN_FIRST`;
- `D2_PASS_LOCALIZED_OUTPUT`: process PASS and at least 14/20 seeds are `OUTPUT_FIRST`;
- `D2_FAIL_CO_ONSET`: process PASS and neither layer reaches 14/20 first-onset seeds, so the predeclared result is `CO_ONSET`;
- `D2_INVALID`: process INVALID.

`D2_FAIL_CO_ONSET` means the requested 5,000-step/14-seed localization contrast failed. It is still a successfully executed, schema-complete diagnostic and is not a mechanism or competence failure. No narrative-only localization is permitted.

## 6. D3: balanced lag-1 solvable contrast

### 6.1 Generator and timing

For training, component 31 draws exactly 200,001 independent fair bits as one NumPy call, `u = g.integers(0,2,size=200_001).astype(int8)`. Define context `x_-1=u[0]` and inputs `x_t=u[t+1]` for `t=0..199,999`. The target graded before integrating `x_t` is:

```text
y_t = x_(t-1)
```

Thus `y_0=x_-1`; for `t>=1`, `y_t` is the prior input. The held-out generator is identical using component 32 and 12,001 draws. It has its own `x_-1`, 2,000 warm-up steps, and 10,000 scored steps. Since inputs are i.i.d. Bernoulli(0.5), the target is non-degenerate, balanced at nominal chance 0.50, and independent of the current input.

Use the same network, state initialization, training length, plasticity parameters, global reward formula with this `y_t`, evaluation length, and tie rule as section 2.

### 6.2 Conditions

Run:

- `P_lag1`: SFA on and MSTDPET live for all 200,000 steps;
- `F0_lag1`: frozen from start, same weights, input/target stream, held-out stream, and tie-coin.

The frozen companion is a mandatory harness validity control. Its weights must be bitwise constant in 20/20 seeds.

### 6.3 Readout

The across-seed interval is the section 9 paired seed bootstrap of the median `P_lag1` held-out accuracy, retaining the complete P/F0 seed row per draw.

- `INVALID`: the two-sided 95% seed-bootstrap interval for the mean `F0_lag1` held-out accuracy does not contain `0.50`, any frozen weight changes, or any shared identity/integrity invariant fails.
- `PASS`: median `P_lag1` held-out accuracy is `>= 0.70` and the one-sided 95% lower bootstrap bound for the median is strictly `> 0.55`.
- `FAIL`: the two-sided 95% bootstrap interval for the median `P_lag1` accuracy contains `0.50`.
- `INCONCLUSIVE`: valid and neither PASS nor FAIL. This includes any unusual interval wholly below chance, which must be reported rather than silently reclassified.

## 7. D4: fixed background-drive contrast

### 7.1 Why this diagnostic exists

Kendrick's Abe lesson is narrower than “noise causes learning.” The read-only Abe record says noise can unstick movement while failing to create brain-driven learning, and too much random babble trades diversity against predictivity:

- Obsidian `04 Findings - Interpretations - Retractions/2026-05-27 — Body-gate 3-of-4 confirmed at scale; forward-model test structurally infeasible; contingency pivot.md`, section “Why claim_c fails”: high babble improved diversity but destroyed predictivity; low babble preserved predictivity but collapsed diversity. The corresponding repository chronology includes `3de1b9c8` (“Pareto confirmed n=4”) and `f18a8386` at the end of that diagnostic sequence.
- Obsidian `2026-05-28 — Contingency detection passes 4-of-4; claim_b vs sustained-contingency tension through weight magnitude.md`, addendum: stillness-gated random babble achieved 0/14 sustained contingency and could either fail to prevent settling or swamp command signal. Relevant commits are `aef83ce8` (stillness-gated motor babble) and `154c418b` (random perturbation exhausted; CPG fork).
- Obsidian `2026-06-05 — Contingency-v0 NO-GO; SNN-body decoupling is a temporal fixed-point lock.md`: decode temperature 8 plus babble standard deviation 5 restored roughly 2,000x movement, but movement was injected rather than brain-driven and the action layer remained near fixed point, CV about 0.0013. The note names related commits `174d3f8`, `9e639b3`, and `d0c974f`; the direct sweep commits include `b594d029` and `35ec19f1`.

Therefore D4 scores both activity and held-out competence. It does not add a rhythmic arm. Abe used downstream motor babble in a body loop; D4 uses upstream fixed synaptic drive in this feedforward bench. They are analogous anti-silence perturbations, not the same executed mechanism.

### 7.2 Frozen drive constants

For each of the 22 target neurons, allocate exactly 8 independent, non-plastic Poisson sources, 176 sources total. Every source has:

- rate: exactly `25.0 Hz`;
- fixed excitatory synaptic weight: exactly `+2.0 mV`;
- no synaptic filtering beyond the target neuron's existing membrane integration;
- no plastic state, bounds, eligibility, reward, adaptation, or learned parameter.

At each 1 ms step, each source emits one spike iff `u < 1-exp(-25/1000)`, using one vectorized `random((212_000,22,8))` call from component 40. Sum the eight source spikes for each target and add `2.0 mV` per spike to that target's ordinary synaptic input before its membrane threshold test.

Targets `0..19` are hidden neurons; target 20 is O1 and 21 is O0. O1 and O0 have identical source count, rate, weight, and mapping rule. Their random columns are independent but exchangeable. The drive contains no task, target, phase, reward, prediction, or arm input.

Written pre-data justification: the resting threshold gap is 16 mV. Eight 25 Hz sources produce 200 source events/s per neuron and a subthreshold first-moment membrane offset of approximately `8 * 25/s * 2 mV * 0.020 s = 8 mV` before reset effects. The selected value is large enough to test escape from silence but does not deterministically cross threshold in one step. This is one hypothesis test, not an estimate of an optimum. D4 may not change it, sweep it, or add a second amplitude.

### 7.3 Common-random-number and continuity contract

The 212,000-step realization is generated once per seed and hashed. Steps `0..199,999` drive training. Steps `200,000..211,999` drive the held-out evaluation in order, including its 2,000-step warm-up. The source clock is not restarted, redrawn, or arm-dependent at the training/evaluation boundary.

Every with-drive condition for that seed reads the identical array and index. This includes `P_A+drive`, `F0_A+drive`, and, if D3 is run, `P_lag1+drive`. Different task streams do not alter the drive array. No-drive conditions receive exactly zero background input and do not consume a substitute RNG stream. Paired hashes are asserted before execution.

Drive is present identically during training and held-out evaluation. It is never plastic and never appears as a presynaptic input to either plastic W1 or W2 eligibility. It can cause hidden or output postsynaptic spikes; those spikes enter the unchanged MSTDPET post traces and therefore create noise-driven post tags. This is an intended, disclosed consequence. The Rung 1 rule and update equation are unchanged.

### 7.4 Conditions and reuse

On the order-2 A task, run the paired 2x2 contrast:

| Plasticity | no drive | fixed drive |
|---|---|---|
| P | `P_A-no` | `P_A+drive` |
| F0 | `F0_A-no` | `F0_A+drive` |

`P_A-no` and `F0_A-no` are the exact D2 rows, reused rather than rerun. Every condition records the D2 1,000-step activity fields, A-end evaluation, and integrity fields.

If D3 is run, also compare `P_lag1-no` (the D3 row) to `P_lag1+drive` using the same per-seed background realization. This lag-1 comparison is mandatory report-only context. It does not change the D4 A-task verdict.

### 7.5 Readouts

All medians below are across the 20 seed-level values.

**D4(i), drive validity**

- `PASS`: the two-sided 95% seed-bootstrap interval for mean `F0_A+drive` held-out accuracy contains `0.50`, and F0 weights are bitwise constant in 20/20 seeds.
- `INVALID`: the interval excludes `0.50`, a frozen weight changes, drive identity differs across with-drive conditions, or another integrity invariant fails.

This control detects task-signal leakage or a broken chance model. D4 has no competence verdict if D4(i) is invalid.

**D4(ii), silence prevention**

Use the final 1,000-step training window, steps `199,000..199,999`.

```text
hidden_rate_ratio = median(hidden_rate_hz[P_A+drive]) /
                    median(hidden_rate_hz[F0_A+drive])
```

If the denominator is zero, D4(ii) fails. Let `silent_end` be the median seed-level both-silent fraction of `P_A+drive` in that window.

- `PASS`: `hidden_rate_ratio >= 0.50` and `silent_end <= 0.80`.
- `FAIL`: either numeric predicate fails.
- `INVALID`: a required end-window field is absent/non-finite or its row fails integrity.

**D4(iii), A competence**

For each seed define `delta_A = accuracy(P_A+drive) - accuracy(P_A-no)`. The paired bootstrap statistic is the median of the 20 `delta_A` values.

- `PASS`: median held-out accuracy of `P_A+drive` is `>= 0.70`, median `delta_A >= 0.05`, and the one-sided 95% paired-bootstrap lower bound for median `delta_A` is strictly `> 0`.
- `FAIL`: the two-sided 95% paired-bootstrap interval for median `delta_A` contains `0`.
- `INCONCLUSIVE`: valid and neither PASS nor FAIL.
- `INVALID`: D4(i) invalid or any paired identity/integrity invariant fails.

Overall D4 is `PASS` only when (i), (ii), and (iii) pass. It is `FAIL` when (i) passes and either (ii) fails or (iii) fails. It is `INCONCLUSIVE` when (i) and (ii) pass but (iii) is inconclusive. Any invalid sub-readout makes D4 `INVALID`.

For the mandatory lag-1 report-only comparison, report both medians, median paired drive-minus-no-drive difference, and its two-sided interval. Do not use it to tune drive or alter D3/D4 status.

## 8. Run order and stop conditions

After this document and a separate implementation are independently approved:

1. Run deterministic shakedowns and the section 3.2 collision probe only.
2. Run D1 first. If the positive control is not PASS, stop `INVALID_HARNESS`. If D1 is FAIL, INCONCLUSIVE, or INVALID, stop and return to Kendrick/spec review. Do not run D2-D4.
3. If D1 passes, run the shared no-drive D2 rows and compute D2.
4. Run D3 and its frozen companion. If D3 is INVALID, stop. A D3 FAIL or INCONCLUSIVE is decision-bearing and does not authorize mechanism modification.
5. Run only the predeclared D4 drive rows and compute D4. No amplitude retry or extra arm is permitted.
6. Freeze the complete package and send it to Nora. No branch action in section 10 occurs before review.

Nothing in this order allows partial-seed inspection. A stage is inspected only after all its predeclared seed rows are present and schema-valid.

## 9. Bootstrap and arithmetic

The seed is the independent unit for aggregate intervals. Use exactly 100,000 nonparametric percentile bootstrap resamples. For metric ID `k`:

```text
g = default_rng(SeedSequence([20261001, 30, 7, k]))
idx = g.integers(0, 20, size=(100_000,20))
```

A row is one seed's complete paired record for the conditions in that metric. Each replicate selects whole rows and recomputes the full mean, median, difference, or ratio. Paired sides always use the same resampled indices. Percentiles use `numpy.percentile(..., method="inverted_cdf")`; one-sided lower 95% is q=5; two-sided 95% is q=2.5 and q=97.5. No per-tick pseudo-replication, dropped seed, normal approximation, outlier removal, alternate interval, or post-data metric is permitted.

| k | Statistic | Interval/use |
|---:|---|---|
| 0 | mean D3 `F0_lag1` accuracy | two-sided; D3 validity |
| 1 | median D3 `P_lag1` accuracy | one-sided lower and two-sided; D3 verdict |
| 2 | mean D4 `F0_A+drive` accuracy | two-sided; D4(i) |
| 3 | median D4 `P_A+drive` accuracy | two-sided report |
| 4 | median paired `P_A+drive - P_A-no` accuracy | one-sided lower and two-sided; D4(iii) |
| 5 | median paired `P_lag1+drive - P_lag1-no` accuracy | two-sided report only |
| 6 | median `P_lag1+drive` accuracy | two-sided report only |
| 7 | median `P_A-no` accuracy | two-sided report only |
| 8 | median `F0_A-no` accuracy | two-sided report only |

D1's per-seed temporal block intervals are separately fixed in section 4.5. D1's across-seed median and count are point predicates and are not given an additional aggregate interval.

## 10. Predeclared branch table

This table is exhaustive and ordered. Earlier rows take precedence.

| Condition | Action |
|---|---|
| Any D1-D4 `INVALID`, collision-probe failure, or incomplete package | STOP. Return to spec/implementation review. No r4. |
| D1 positive control not PASS | D1 is `INVALID_HARNESS`. Repair only under a new reviewed spec; no downstream diagnostic. |
| D1 FAIL | The specified linear decoder cannot recover A under the fixed split. Task/architecture pairing needs roadmap revision. Return to Kendrick. No r4 readout or reward change. |
| D1 INCONCLUSIVE | Evidence is not decision-complete. Return to Kendrick/spec review. No downstream diagnostic and no r4. |
| D1 PASS and D3 FAIL | Frozen activity is linearly decodable but the unchanged rule/harness does not establish lag-1 competence. Treat as a rule-limit or harness question; return to Kendrick. D4 observations cannot independently authorize r4. |
| D1 PASS and D3 INCONCLUSIVE | Rule capacity is unresolved. Return to Kendrick. D4 may be reported if already completed in the fixed sequence, but cannot authorize r4. |
| D1 PASS, D3 PASS, D4(ii) FAIL | The frozen drive did not prevent silence at its predeclared constants. No constant retuning inside this diagnostic. No r4 with this drive. D2 localization is reported. |
| D1 PASS, D3 PASS, D4(ii) PASS, D4(iii) FAIL | Silence was prevented but competence did not improve decisively. Silence alone is not shown to be the bottleneck. Drive alone does not justify r4. D2 localization is reported. |
| D1 PASS, D3 PASS, D4(ii) PASS, D4(iii) INCONCLUSIVE | Silence was prevented but learning evidence is unresolved. Return to Kendrick; no r4 and no retuning. |
| D1 PASS, D3 PASS, D4(ii) PASS, D4(iii) PASS | An r4 with this exact frozen drive is warranted for consideration. Create a new doc-only r4 spec card, inform Kendrick, retain all Stage 2 gates/controls/rule, and require Nora review before execution. |
| D1 PASS, D3 PASS, D4(ii) FAIL, D4(iii) PASS | Activity criterion still failed, so the anti-silence mechanism is not validated even if competence happened to pass. No r4 with this drive; return to Kendrick. |
| D1 PASS, D3 PASS, D4 overall INCONCLUSIVE for any other valid combination | Return to Kendrick. No r4 and no post-data constant change. |

D2 is always reported if run. `LOCALIZED_HIDDEN`, `LOCALIZED_OUTPUT`, and `CO_ONSET` guide the explanation of a valid branch but do not override D1, D3, or D4. A localized onset does not itself authorize a mechanism change.

## 11. Implementation shakedown required before any diagnostic

A later implementation card must pass tests that prove, at minimum:

1. r3 initial-weight distributions and draw order are reproduced under namespace 30.
2. A streams and tie-coins are deterministic and paired; train/test hashes differ.
3. Feature `phi_t` cannot contain or depend on current `x_t`; hand-computed hidden spike fixtures reproduce count20, trace20, and trace25.
4. Decoder preprocessing fits on training only; forced test-feature changes cannot alter stored train means/scales or coefficients.
5. The D1 positive-control label is exactly prior input, not current input.
6. D2 emits exactly 200 checkpoints and refuses a row with any required field removed; onset boundary cases at 4,999 and 5,000 steps follow section 5.3.
7. The lag-1 generator is bit-identical across two constructions, balanced by construction, and uses `x_(t-1)` with the declared `x_-1` context.
8. F0 weights are bitwise constant and evaluation is non-mutating.
9. Background-source vectors are deterministic, identical across all with-drive conditions, absent from no-drive conditions, and continuous from training index 199,999 to evaluation index 200,000.
10. O1/O0 receive identical drive parameters and distinct exchangeable columns; swapping their columns and labels preserves construction.
11. Background synapses have no plasticity object and never enter plastic pre traces, while a forced background-caused post spike does enter the unchanged postsynaptic trace.
12. With drive disabled, the implementation is bit-identical to the existing r3 path on a fixed short fixture.
13. Bootstrap results are invariant to metric evaluation order, and paired metrics preserve complete seed rows.
14. The section 3.2 collision probe passes.
15. Maintained repository tests pass.

These are deterministic construction tests, not permission to execute a full seed, inspect accuracy, or estimate a threshold.

## 12. Resource estimate and budget

Reference host: Mac mini, Intel Core i5-8500B 3.0 GHz, 6 logical CPUs, 32 GiB RAM. Use at most 5 worker processes and leave one logical CPU for the OS/coordinator. r3 observed roughly 166 seconds per 400,000-training-step arm/seed job plus its four evaluations; these diagnostics use 200,000 training steps and one evaluation, but D1 also fits two 198,000-row logistic decoders.

Predeclared workload if D1 passes:

- D1: 20 frozen 200,000-step streams plus 20 held-out evaluations and 40 decoder fits. Estimate `0.3-0.8 h` wall at 5 workers.
- Shared D2/D4 no-drive: P and F0 x 20 seeds = 40 jobs, each 200,000 training steps plus one 12,000-step evaluation. Estimate `0.4-0.9 h`.
- D3: P and F0 x 20 = 40 jobs. Estimate `0.4-0.9 h`.
- D4 A drive additions: P+drive and F0+drive x 20 = 40 jobs. Estimate `0.4-1.0 h`.
- D4 lag-1 drive addition if D3 runs: P+drive x 20 = 20 jobs; P-no is reused. Estimate `0.2-0.5 h`.
- Aggregation, bootstrap, hashing, and reports: `0.2-0.5 h`.

Maximum expected total: `1.9-4.6 h` wall after implementation approval, with 5 workers, under 4 GiB resident memory and a result package under 25 MiB if raw per-step features and Poisson uniforms are not serialized. Serialize source/feature/drive hashes and required aggregates, not the 212,000 x 22 x 8 uniform array. Abort rather than swap if RSS exceeds 8 GiB. Actual worker count, wall time, peak RSS, interruptions, and retries must be reported separately by diagnostic.

D1 is first specifically to avoid spending the remaining budget if the fixed decoder result is FAIL, INCONCLUSIVE, or INVALID.

## 13. Result package, claim boundary, and next gate

A later approved run package must contain:

- exact source commit, spec commit/hash, host, Python/NumPy/PyTorch/scikit-learn versions, argv, environment, worker count, clean/dirty state, and effective parameters;
- one complete row per predeclared condition/seed, deterministic object hashes, schema verdicts, and the section 3 collision-probe record;
- D1 coefficients/intercepts, training-only normalization hashes, convergence status, held-out accuracy, per-seed block interval, positive-control result, and bounded claim text;
- D2 all 200 checkpoint rows per seed/condition, exact onset records, bound fractions at onset, seed labels, and aggregate label;
- D3 generator hashes, P/F0 rows, frozen-weight proof, intervals, and verdict;
- D4 drive constants/hash, pairing proof, activity and competence readouts, A and lag-1 reports, intervals, and verdict;
- a machine-readable branch-table outcome and a plain-language summary that names what is supported and not supported;
- SHA-256 manifest for every result artifact.

No-oracle statement: the target is used only to grade already-emitted output spikes and, in D1 only, as an offline decoder label after feature capture. It is never an input, hidden-state label, plasticity feature, state reset, initialization signal, or internal policy/world-model component. D4 drive is generated without task or target input. No LLM logic enters the network, policy, memory, reward rule, decoder feature path, or simulator.

Passing any diagnostic does not pass Stage 2. D1 is only linear decodability under one frozen specification. D2 is localization. D3 is a simpler-task contrast. D4 is one anti-silence perturbation at one frozen amplitude. None establishes general learning, continual learning, developmental progress, biological equivalence, Abe transfer, or an r4 result.

Next gate: Nora independently reviews this exact document for numeric closure, leakage, deterministic identity, statistical validity, common-random-number fidelity, Abe-evidence accuracy, and branch completeness. Only after APPROVE may a separate implementation card be created. This document itself authorizes no run.