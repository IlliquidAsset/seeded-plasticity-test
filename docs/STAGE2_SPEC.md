# Stage 2 — Continual hidden-structure learning: frozen specification

**Status:** FROZEN CANDIDATE — document only. Implementation and execution require independent approval of this document.  
**Task:** `t_83684b37`  
**Branch:** `v0.3.0-calibrated`  
**Frozen against baseline:** `608c4255a700f93318da9eb66a8b26adc55ead2c`  
**Stage 1 evidence:** `docs/FLORIAN_CORE_GATE.md` and `results_florian_core/` (`CORE PASS`, independently approved by Nora).  
**Date frozen:** 2026-10-01  

No Stage 2 experiment has been run. This document freezes the experiment before implementation or data collection. After independent approval, implementation must reproduce this contract exactly; a changed scientific parameter requires a new spec and a new review, not an edit made after seeing results.

## 1. Roadmap contract (verbatim)

### Stage 2 — Continual hidden-structure learning

> One continuous run: learn regime A -> unexpected mid-stream shift to regime B -> local plasticity stays live, no reset, no retrain -> learn B -> test whether A was retained. The net receives observations only (no hidden-state labels, no oracle — the next tick of reality grades each prediction). Spike-frequency adaptation (SFA) in from the start, per the Bellec et al. 2020 e-prop/ALIF formulation (SpikingBrain's 'adaptive threshold' is input-magnitude normalization, not SFA, and is not to be used).

### Exit gates (all required; roadmap wording verbatim)

1. Competence on A before the shift.
2. Held-out improvement on B after the shift.
3. Retention of >=80% of A's learned-above-chance performance, with <=5-point absolute loss, and a confidence bound still above chance.
4. Fixed capacity, no reset, no replay contamination, no post-hoc hyperparameter selection.

### Controls (all required; roadmap wording verbatim)

- Frozen from the start.
- Freeze at the A->B shift.
- Temporally-scrambled B (twin stream, identical statistics, scrambled temporal structure).
- SFA off with R-STDP still active.

### Finding taxonomy (verbatim)

> If B learning passes but A retention fails, the finding is 'online adaptation with forgetting' — not continual learning.

It will be named honestly either way.

## 2. Question and claim boundary

The experiment asks whether one fixed-capacity spiking network can learn a noisy temporal rule A, continue local R-STDP without reset across an unannounced switch to the opposite rule B, improve on held-out B, and still predict held-out A from the same final weights.

A Stage 2 PASS licenses only:

> Under this predeclared noisy order-2 binary stream, the fixed-capacity repository network learned B online after an unannounced shift and retained the required amount of its learned A performance.

It does not establish broad continual learning, biological equivalence, optimal SFA, transfer across modalities, or performance in Abe. The SFA-off control determines whether this experiment demonstrates an SFA contribution; Stage 2 can pass without an SFA-specific claim if both SFA-on and SFA-off satisfy the behavioral gates.

## 3. Prediction stream and hidden regimes

### 3.1 Observation and prediction timing

The environment emits a binary observation `x_t in {0, 1}` every SNN integration step (`dt = 1 ms`). The input at step `t` is a one-hot spike vector:

- `x_t = 0` -> `[1, 0]`
- `x_t = 1` -> `[0, 1]`

The network's single output spike already present at step `t`, `z_t in {0, 1}`, is its prediction of `x_t`, made from earlier observations. The environment then reveals `x_t` and grades that prediction before `x_t` is integrated into the network. This ordering uses `PureSNN.online_step` exactly: the reward associated with `z_t` is applied on transition `t -> t+1`, and only afterward does the network integrate the input spike encoding `x_t`.

The Florian reward is retained exactly:

```text
r_t = (2*x_t - 1) * z_t
```

Thus an output spike is rewarded `+1` when the newly revealed observation is `1`, punished `-1` when it is `0`, and silence causes no weight update. Accuracy, however, scores both outcomes: `prediction_t = z_t`, so a correct silence for `x_t=0` counts as correct.

The target is never supplied as an input, feature, regime label, loss gradient, weight initializer, or state reset. It enters only as the next observation and the sign of the reward for an already-emitted output spike. This is the no-oracle boundary.

### 3.2 Regime A

For `t >= 2`, regime A is a noisy order-2 XOR transition:

```text
ideal_A(t) = x_(t-2) XOR x_(t-1)
x_t = ideal_A(t)             with probability 0.90
x_t = 1 - ideal_A(t)         with probability 0.10
```

The first two observations are independent Bernoulli(0.5). The 10% innovation draw is independent at every step.

### 3.3 Regime B

Regime B is the complementary noisy order-2 XNOR transition:

```text
ideal_B(t) = 1 - (x_(t-2) XOR x_(t-1))
x_t = ideal_B(t)             with probability 0.90
x_t = 1 - ideal_B(t)         with probability 0.10
```

A and B have the same binary marginal, the same one-hot input magnitude, the same innovation probability, the same theoretical optimum accuracy (90%), and opposite hidden transition rules. There is no shift marker. At the fixed boundary, B uses the last two realized A observations as its starting context.

This is deliberately a hidden-structure problem rather than a label-remapping problem: the current observation alone is insufficient, and the regime can only be inferred from temporal history. Chance accuracy is exactly `0.50` because the stationary binary marginal is balanced.

### 3.4 Training length and shift

Each run contains exactly `400,000` observed steps:

- A phase: steps `0..199,999` (`200,000 ms` developmental age; `200 s` SNN integration time).
- Unexpected A->B shift: immediately before step `200,000`.
- B phase: steps `200,000..399,999` (`200,000 ms`; cumulative `400 s`).

At the shift there is no membrane reset, SFA reset, eligibility reset, spike reset, reward-running-statistic reset, weight restore, optimizer/retrainer invocation, new neuron, new synapse, context bit, phase label, or pause in plasticity in the primary arm. The only changed variable is the environment's transition rule.

## 4. Frozen network and learning rule

The implementation must reuse the Stage 1 repository path `snn/core.py` and reward mode `REWARD_PER_SPIKE_NEXT_STEP`. No separate NumPy learner is allowed in the experimental path.

### 4.1 Fixed architecture

| Quantity | Frozen value |
|---|---:|
| Layer sizes | `[2, 20, 1]` |
| Trainable synapses | 40 input-to-hidden + 20 hidden-to-output = 60 |
| `dt` | `1.0 ms` |
| LIF `tau_m` | `20.0 ms` |
| Synaptic filter | none, `tau_syn = 0.0 ms` (direct voltage jump) |
| Rest/reset | `-70.0 mV` / `-70.0 mV` |
| Baseline threshold | `-54.0 mV` |
| Hidden neurons | ALIF/SFA on in primary arm |
| Output neuron | ordinary LIF, non-adaptive |
| Precision | float64 |
| Batch size | 1 |

Initial input-to-hidden weights are independent uniform `U(-10, 10) mV`; hidden-to-output weights are independent uniform `U(0, 10) mV`. Bounds remain `[-10, 10] mV` and `[0, 10] mV`, respectively. Initialization and all stream generation use NumPy `default_rng` as specified in section 7.

Capacity is fixed at construction. Structural plasticity, neuron birth/death, masks that add connections, auxiliary readouts, replay buffers, and external classifiers are prohibited.

### 4.2 R-STDP

Use Stage 1 temporal MSTDPET parameters without tuning:

| Quantity | Frozen value |
|---|---:|
| Credit mode | `eligibility` (MSTDPET) |
| `gamma` / learning rate | `0.25 mV` |
| STDP `tau_plus`, `tau_minus` | `20.0 ms`, `20.0 ms` |
| Eligibility `tau_elig` | `25.0 ms` |
| Pairing amplitudes | `a_plus = a_minus = 1/25` |
| Reward schedule | per-output-spike, following-step, online |
| Plasticity state | continuous across patterns and A->B |

The update ordering, signs, and bounds are those already qualified in `CoreFlorianBench`: update pairing and eligibility, apply the per-spike reward, clamp, then integrate the current input with the updated weight.

### 4.3 SFA / ALIF

Use threshold adaptation, not an adaptation current and not SpikingBrain activation normalization. For each hidden neuron:

```text
rho = exp(-dt / tau_a)
theta_t = v_thresh + beta_a * a_t
v_pre = v_rest + exp(-dt/tau_m) * (v_t - v_rest) + I_t
z_t = 1[v_pre >= theta_t]
v_(t+1) = v_reset if z_t else v_pre
a_(t+1) = rho * a_t + z_t
```

Frozen values:

- `tau_a = 200.0 ms`
- `beta_a = 1.12 mV` (`0.07 * (v_thresh - v_rest) = 0.07 * 16 mV`)
- initial `a = 0`
- no adaptation clamp
- no refractory period (preserves the existing core neuron contract)
- hard membrane reset (preserves the existing core behavior)
- adaptation is incremented only by emitted hidden spikes
- adaptation is initialized once at run start and is not reset at the A->B shift

This is the Bellec et al. ALIF state equation mapped to the bench's voltage convention. The SFA-off control sets `beta_a = 0.0` and allocates no effective adaptation contribution; every other network and R-STDP parameter remains identical.

## 5. Non-mutating held-out evaluations

Training never consumes an evaluation observation. Evaluation pauses the simulated training clock and runs on a deep copy of the checkpoint; it cannot alter the live model, RNG, membranes, SFA, eligibility, weights, or stream position.

For each seed, immutable A and B evaluation streams are generated before training from independent RNG streams. Each evaluation stream has `12,000` observations: the first `2,000` are an unscored state warm-up, and the final `10,000` are scored. Evaluation copies start with zero membrane, spike, eligibility, and adaptation state and have plasticity disabled. This common reset is confined to disposable evaluation copies; the continuous training run is never reset.

The same held-out A stream is used for that seed at all A checkpoints, and the same held-out B stream is used at all B checkpoints. No evaluation output or state returns to training.

Checkpoints:

1. Initial: optional report-only A and B accuracy before training.
2. `A_pre`: after step 199,999 and before the first B step; evaluate held-out A.
3. `B_pre`: from the same pre-shift weights; evaluate held-out B.
4. `B_post`: after step 399,999; evaluate held-out B.
5. `A_post`: from the same final weights as `B_post`; evaluate held-out A.

Primary metric per seed and checkpoint:

```text
accuracy = correct binary predictions / 10,000 scored observations
```

A spike is prediction 1 and silence prediction 0. Chance is `0.50`. No threshold is estimated from data.

## 6. Required arms and control validity

All five arms use the same 20 seeds, initial weights, A stream, structured B stream (except the declared permutation), evaluation streams, length, and checkpoint code.

### 6.1 Primary: SFA on, R-STDP live

Plasticity and SFA are active from the start through both phases. This is the only arm whose outcome can satisfy Stage 2.

### 6.2 Frozen from the start

No reward is applied to weights in either phase. Validity requires:

- bitwise equality of all final and initial weights (`max_abs_dW = 0.0`) in 20/20 seeds;
- mean held-out accuracy at each checkpoint in `[0.45, 0.55]`; and
- the two-sided 95% seed-bootstrap interval at each checkpoint contains `0.50`.

A failure of the weight equality check invalidates the run. A performance failure means the nominal 50% chance model is wrong for the realized fixture and invalidates the predeclared chance-based gates.

### 6.3 Freeze at the A->B shift

R-STDP is live during A and disabled immediately before step 200,000. SFA remains active and all dynamic state remains continuous. Validity requires:

- A-phase total absolute weight change `> 0` in 20/20 seeds;
- bitwise equality of shift and final weights (`max_abs_dW_B = 0.0`) in 20/20 seeds; and
- the primary arm's held-out B improvement exceeds this arm's B improvement by at least `0.10` in the across-seed point estimate, with the one-sided 95% paired-bootstrap lower bound `> 0`.

This separates B weight learning from inference caused only by persistent neural state.

### 6.4 Temporally scrambled B twin

The arm receives exactly the primary A stream. Its B observation vector is an exact Fisher-Yates permutation of the primary arm's 200,000 realized B observations. Therefore stream length, zero/one counts, one-hot spike counts, reward opportunities, and one-point input statistics are identical, while order-2 structure is destroyed. No permutation is retried or selected.

Validity requires:

- exact equality of the B symbol histogram and total input spike count to the primary B stream;
- empirical best order-2 lookup accuracy on the scrambled B training stream `<= 0.55`; and
- primary `B_post` accuracy exceeds scrambled-arm `B_post` accuracy by at least `0.10`, with the one-sided 95% paired-bootstrap lower bound `> 0`.

The lookup accuracy uses the four contexts `(x_(t-2), x_(t-1))`, chooses the majority next bit separately for each context on the whole scrambled stream, and reports total correct / total eligible steps. This check measures only whether the scramble accidentally retained usable temporal structure.

### 6.5 SFA off, R-STDP active

Set `beta_a = 0.0`; R-STDP remains live for both phases. Validity requires:

- effective SFA threshold contribution exactly `0.0 mV` at every sampled instrumentation point;
- total absolute weight change `> 0` in both A and B in 20/20 seeds; and
- the Stage 1 sign-of-life criterion in section 8 passes separately in A and B.

Interpretation is frozen in advance:

- If the primary passes Stage 2 and SFA-off fails any behavioral gate, report `CONTINUAL_LEARNING_WITH_SFA_DEPENDENCE_AT_THIS_PARAMETERIZATION`.
- If both arms pass all behavioral gates, report `CONTINUAL_LEARNING; SFA_NOT_SHOWN_NECESSARY`.
- An SFA benefit may be claimed only if primary minus SFA-off `A_post` accuracy is at least `0.05` and its one-sided 95% paired-bootstrap lower bound is `> 0`.
- The SFA-off arm cannot rescue a failure of the predeclared primary arm.

## 7. Seeds and deterministic identities

There are exactly 20 experimental seeds: integers `0..19`. A seed is the independent statistical unit. Arms are paired by seed.

Use NumPy `SeedSequence` components below; no Python process hash or worker order may affect any draw:

| Object | SeedSequence entropy list |
|---|---|
| Initial weights | `[20261001, seed, 1]` |
| A training stream | `[20261001, seed, 2]` |
| B innovation draws | `[20261001, seed, 3]` |
| B scramble permutation | `[20261001, seed, 4]` |
| A held-out stream | `[20261001, seed, 5]` |
| B held-out stream | `[20261001, seed, 6]` |
| Bootstrap resampling | `[20261001, 7]` |

The B generator starts from the final two realized A observations but uses its independent innovation RNG. All arms must serialize SHA-256 hashes of initial weights, A observations, structured B observations, scrambled B observations where applicable, and both held-out streams. Paired arms must assert matching hashes for every object they share before running.

Parallel workers may change wall time only. Result rows are sorted by arm then seed before aggregation.

## 8. Stage 1 sign-of-life instrumentation carried forward

Record separately for A and B and for every seed/arm:

- total output spikes;
- positive and negative reward-event counts;
- total absolute weight change in each layer;
- min/max final weight and bound hits;
- reward and signed hidden-to-output `delta_w` for each reward event, or exact streaming sufficient statistics for their Pearson correlation;
- `reward_signed_dw_out_corr` (same definition as Stage 1);
- hidden and output firing rates;
- SFA adaptation mean, maximum, and threshold contribution for SFA-on arms;
- all held-out accuracies and the prediction-one fraction;
- the maximum weight difference for each freeze interval.

The primary arm's mechanism sign-of-life gate passes separately in A and B only if:

1. total absolute weight change is `> 0` in 20/20 seeds;
2. each seed has at least `100` positive and `100` negative reward events;
3. the median `reward_signed_dw_out_corr` across seeds is `>= 0.30`; and
4. at least 18/20 seeds have `reward_signed_dw_out_corr > 0.10`.

NaN/Inf in any state or metric, any weight outside its declared bounds, a failed stream hash assertion, or a sign-of-life failure is `INVALID_OR_MECHANISM_FAIL`, not a behavioral PASS.

## 9. Confidence intervals and arithmetic

All accuracy summaries first compute one accuracy per seed. The 20 seed values—not the 200,000 individual predictions—are the independent observations.

For every reported mean, difference, absolute loss, and retained fraction:

- point estimate: arithmetic on the 20 seed-level values;
- interval: nonparametric percentile bootstrap over seed indices;
- resamples: exactly `100,000`;
- RNG: `default_rng(SeedSequence([20261001, 7]))` freshly initialized for the ordered metric table, with metrics evaluated in the order listed in the result schema;
- ordinary accuracy: resample the 20 seed accuracies;
- arm or checkpoint difference: resample the 20 paired seed rows together;
- one-sided lower 95% bound: 5th percentile;
- one-sided upper 95% bound: 95th percentile;
- two-sided 95% interval: 2.5th and 97.5th percentiles.

No normal approximation, per-tick pseudo-replication, dropped seed, outlier exclusion, or alternate interval is allowed. All comparisons are strict where written `>`; equality fails a strict gate. The four behavioral exit gates are conjunctive, so no multiplicity correction is applied.

Definitions using across-seed mean accuracies:

```text
A_learned       = mean(A_pre_primary) - 0.50
A_loss          = mean(A_pre_primary) - mean(A_post_primary)
A_retained      = (mean(A_post_primary) - 0.50) / A_learned
B_improvement   = mean(B_post_primary - B_pre_primary)
```

If `A_learned <= 0`, retention is undefined and Gate 3 fails.

## 10. Exact pass/fail gates

### Gate 0 — integrity and mechanism

PASS only if every deterministic identity, bound, required control-validity check, and primary A/B sign-of-life check in sections 6–8 passes. Otherwise the experiment is invalid or a mechanism failure and no Stage 2 PASS is permitted.

### Gate 1 — competence on A before the shift

PASS only if all are true:

1. `mean(A_pre_primary) >= 0.70`;
2. the one-sided 95% lower seed-bootstrap bound for `mean(A_pre_primary)` is `> 0.50`;
3. `mean(A_pre_primary - A_pre_frozen_start) >= 0.15`; and
4. the one-sided 95% lower paired-bootstrap bound for that primary-minus-frozen difference is `> 0`.

### Gate 2 — held-out improvement on B after the shift

PASS only if all are true:

1. `B_improvement >= 0.15`;
2. the one-sided 95% lower paired-bootstrap bound for `B_post - B_pre` is `> 0`;
3. `mean(B_post_primary) >= 0.70`;
4. the one-sided 95% lower bound for `mean(B_post_primary)` is `> 0.50`;
5. the freeze-at-shift differential and temporally-scrambled differential in sections 6.3 and 6.4 both pass.

### Gate 3 — retention of A

PASS only if all are true:

1. `A_retained >= 0.80`;
2. `A_loss <= 0.05` (five accuracy points; a gain is allowed);
3. the one-sided 95% lower seed-bootstrap bound for `mean(A_post_primary)` is `> 0.50`.

For reporting, bootstrap retained-fraction and loss intervals are mandatory, but the roadmap's confidence-bound clause is adjudicated by item 3. No clipping of negative losses or retained fractions above 1 is allowed.

### Gate 4 — fixed capacity and clean continual protocol

PASS only if all are true:

1. every arm has exactly 60 trainable synaptic weights before and after the run;
2. the primary arm has exactly one initialization and zero calls to network, membrane, spike, eligibility, SFA, or weight reset after step 0;
3. the primary executes exactly 200,000 A steps followed immediately by 200,000 B steps, with R-STDP enabled on every eligible step;
4. training-stream hashes do not equal either held-out-stream hash, and an access log shows zero held-out observations consumed by training;
5. no replay buffer exists and no A observation is deliberately re-presented during B;
6. source commit, effective argv, full parameter JSON, host identity, and clean/dirty status are serialized before execution;
7. no parameter, threshold, seed, checkpoint, or metric differs from this approved spec.

Any violation fails Gate 4 even if numerical performance is strong.

### Overall verdict

`STAGE2_PASS` requires Gates 0, 1, 2, 3, and 4.

If Gate 2 passes but Gate 3 fails, the mandatory verdict is:

> online adaptation with forgetting — not continual learning

If Gate 1 fails, report `NO_A_COMPETENCE`. If Gate 1 passes but Gate 2 fails, report `NO_HELD_OUT_B_LEARNING`. If an integrity/mechanism rule fails, report `INVALID_OR_MECHANISM_FAIL` and name the exact failed invariant. Results are reported once under this frozen spec; thresholds are not moved to convert a result.

## 11. Implementation shakedown before the experimental run

After this spec is independently approved, implementation may run unit and deterministic shakedown tests only. They must not use experimental seeds `0..19` or run enough steps to estimate a gate metric.

Required tests:

1. ALIF hand check: one emitted spike increments `a` by exactly 1, next threshold rises by exactly `1.12 mV`, and `a` decays by `exp(-1/200)` per 1 ms step.
2. Ordering check: changing `x_t` cannot affect `z_t` before reward at transition `t -> t+1`.
3. No-reset check: instrumented primary shift records zero reset calls.
4. Freeze checks: frozen-from-start changes zero weights; freeze-at-shift changes zero weights after the shift.
5. Stream checks: A/XOR and B/XNOR equations hold against hand-computed fixtures; scrambled B preserves the exact symbol multiset.
6. Evaluation isolation: checkpoint evaluation leaves a byte-identical live training state and unchanged RNG states.
7. SFA-off check: with `beta_a=0`, output is bit-identical to the ordinary LIF path on a fixed short fixture.
8. Legacy regression: the maintained Stage 1 core and provenance tests stay green.

Performance smoke tests, seed screening, hyperparameter sweeps, or looking at partial seed outcomes before all 100 arm/seed jobs complete are prohibited.

## 12. Resource and wall-time estimate (Mac mini)

Target host observed while freezing this spec: Mac mini, Intel Core i5-8500B 3.0 GHz, 6 logical CPUs, 32 GiB RAM. Use 5 worker processes, leaving one logical CPU for the OS and coordinator.

Workload:

- 5 arms x 20 seeds = 100 continuous runs;
- 400,000 training steps per run = 40,000,000 training steps total;
- four 12,000-step held-out evaluations per run (A/B pre and A/B post; the pre pair shares one checkpoint and the post pair shares one checkpoint) = 4,800,000 evaluation steps total;
- total approximately 44.8 million SNN steps.

The Stage 1 temporal core jobs took roughly 241–275 seconds for 400,000 steps while also re-running the independent anchor. Removing the anchor comparison but adding ALIF state and held-out evaluation gives a planning estimate of 300–450 seconds per Stage 2 arm/seed job.

At 5 workers, expected wall time is `1.7–2.5 hours`; budget `3 hours` including coordination, hashing, aggregation, and variance. Expected peak memory is `< 2 GiB` for five small float64 networks plus processes. Store event-correlation sufficient statistics rather than per-tick tensors; expected committed result package is `< 25 MiB`. Abort rather than swap if resident memory exceeds 8 GiB.

These are planning estimates, not gates. Actual wall time, peak RSS, worker count, interruptions, and retries must be reported.

## 13. Frozen result schema and provenance

The future result package must contain:

- one JSON row for every one of the 100 arm/seed runs;
- seed-level checkpoint accuracies, sign-of-life metrics, freeze deltas, firing/SFA summaries, hashes, timing, and status;
- aggregate point estimates and all required bootstrap intervals;
- exact executable argv as both JSON array and `shlex.join` display string;
- source/gate commit and confirmation that scientific files were clean;
- maintained test command and result;
- a plain-language report with every gate and control marked PASS/FAIL;
- SHA-256 for all result artifacts.

The gate code must be committed and pushed before the first experimental seed runs. Result aggregation may not silently omit a failed, timed-out, or non-finite seed; an incomplete 20-seed arm cannot pass.

## 14. Freeze, rollback, and next gate

This card changes only `docs/STAGE2_SPEC.md`. Rollback is `git revert <spec-commit>`; no runtime or science artifact exists to remove.

Next gate: Nora independently reviews this frozen document for roadmap fidelity, numerical closure, no-oracle timing, control sufficiency, and implementation ambiguity. Only after approval may a separate build card implement Stage 2. Stage 3, Stage 4, and Track 4 remain out of scope.