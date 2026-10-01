# Stage 2 — Continual hidden-structure learning: frozen specification

**Status:** FROZEN CANDIDATE r3 — document only. Implementation and execution require independent approval of this document.  
**Task:** `t_83684b37` (r1, r2); `t_de6dcd58` (r3, from Amanda's proposal on `t_89fc68c8`)  
**Branch:** `v0.3.0-calibrated`  
**Frozen against baseline:** `608c4255a700f93318da9eb66a8b26adc55ead2c` (r1/r2); r3 edits the r2 text of `ef85a15b0382bdc86b8dba377671e4c2b2e4f79f` (SHA-256 `0c1bbdba036b13955c81f2ce0ffee0557b09842ef7d56b0dec0834faf8d8ec87`) on branch head `084aec8`  
**Stage 1 evidence:** `docs/FLORIAN_CORE_GATE.md` and `results_florian_core/` (`CORE PASS`, independently approved by Nora).  
**Date frozen:** 2026-10-01  
**Revision:** r3 (2026-10-01). See the revision history and the r3 change ledger (section 0).

### Revision history

- **r1** (`b8a0168`): first freeze.
- **r2** (`ef85a15`, 2026-10-01): closes the four implementation ambiguities in Nora's REVISE review of `b8a0168` (evaluation membrane initialization; bootstrap RNG/metric order and ratio recomputation; initial evaluation status; held-out stream starting context and RNG draw order). No scientific threshold, arm, seed, length, or hyperparameter changed. r2 was approved, implemented (`d04fbd9`), and run once (`1418436`; review amendment `084aec8`). Verdict under r2: `INVALID_OR_MECHANISM_FAIL + NO_A_COMPETENCE; G0-G3 FAIL, G4 PASS`. Nora approved that verdict on `t_26c046ac`. The result package was INCOMPLETE and the process card FAILED, because the §8 A-end weight min/max and bound hits were never serialized (9/100 rows unrecoverable). Diagnosis (`results_stage2/KENDRICK_SUMMARY.md`): the single output fell silent. Under `r_t = (2x_t-1)*z_t` silence earns no update, while accuracy credits silence as a correct "0". In 19/20 seeds the prediction-one fraction was 0 at `A_pre`, and in 16/20 seeds there were zero B output spikes. The r2 verdict stands. It is not re-scored, superseded, or rescued by r3.
- **r3** (this revision): one mechanism change, an active-"0" two-output readout (2-20-2), plus the edits that follow from it in the wording, a §8 A-end/B-end instrumentation fix with a schema-completeness assertion, and a pre-run frozen-control fixture qualification (§11). No roadmap gate, control, §10 numeric threshold, §6 validity rule, MSTDPET/ALIF parameter, regime, length, shift, experimental seed, evaluation length, or bootstrap procedure changed.

No r3 experiment has been run. r3 is a new frozen experiment and not a rerun of r2. It uses the same predeclared seeds `0..19`, and its outcome will be reported next to the r2 FAIL as a second attempt under the same roadmap gates. This document freezes the experiment before implementation or data collection. After independent approval, implementation must reproduce this contract exactly; a changed scientific parameter requires a new spec and a new review, not an edit made after seeing results.

## 0. r3 change ledger (complete list of differences from r2 `ef85a15`)

| # | Change | Sections | Kind |
|---:|---|---|---|
| 1 | **Single mechanism change.** The single LIF output is replaced by two non-adaptive LIF outputs, `O1` ("predict 1", output index 0) and `O0` ("predict 0", output index 1). Network `[2, 20, 2]`. Reward is ONE global scalar, delivered online on the following step, `r_t = (2*x_t - 1) * (z1_t - z0_t)`, applied to every synapse through that synapse's own eligibility (unchanged MSTDPET path). It is not a per-neuron or vector reward (that would be Track 4, out of scope). | §3.1, §4.1, §4.2 | mechanism |
| 2 | `O0` hidden-to-output weights: bounds `[0, 10] mV`, init `U(0, 10) mV`, drawn from a NEW SeedSequence component `[20261001, seed, 21]`. Every r2 draw (components 1–6, bootstrap) is bitwise unchanged; `O1`'s row is r2's `W2` draw. | §4.1, §7, §7.1 | follows from 1 |
| 3 | Evaluation prediction rule: `O1` alone -> 1, `O0` alone -> 0, tie (both or neither) -> a fair coin bit from a NEW SeedSequence component `[20261001, seed, 22]`. The coin is fixed per seed and per checkpoint, identical across arms, and independent of streams, weights, and network state. The coin is used only to score evaluations. It never enters the network, the reward, or training. | §3.1, §5.2, §5.3, §7.1 | follows from 1 |
| 4 | Consequential wording: layer sizes, 80 trainable synapses, auxiliary-readout clause narrowed to "anything beyond the declared O1/O0 pair", output-spike and prediction-one definitions, §8 `reward_signed_dw_out_corr` summed over all hidden-to-output weights (both rows). | §4.1, §5.3, §8 | wording follows architecture |
| 5 | **Gate 4 item 1 constant `60` -> `80`.** Gate wording and logic are unchanged; the constant is the trainable-synapse count of the declared architecture (40 + 40). **Flagged for Nora to confirm this is not a gate change.** | §10 Gate 4 | constant tracks architecture |
| 6 | §8 instrumentation fix: separate `a_end_*` (after step 199,999, before the first B step) and `b_end_*` (after step 399,999) fields per arm/seed/layer. Each set records weight min, weight max, lower-bound hits, and upper-bound hits for W1 and W2, and for W2 also per output neuron. A schema-completeness assertion makes the writer refuse any row missing a field. Per-output spike counts, +/- reward-event counts, and both-silent/both-fire fractions for A and B separately (report only). | §8, §8.1, §13 | instrumentation |
| 7 | Frozen-control fixture qualification: before any experimental seed, run the frozen-from-start arm only, on non-experimental seeds `1000..1019`, at full length. If any §6.2 validity condition fails at any checkpoint (including a 95% CI that excludes `0.50`), the build card stops and returns to spec review. §6.2 itself is unchanged. | §6.2 (note), §11 | pre-run validity screen |
| 8 | New §11 shakedown items: rule-unchanged bit-identity per output, `a_end_*`/`b_end_*` presence and shift identity, tie-coin determinism and independence. | §11 | test |
| 9 | §12 resource estimate updated for 80 synapses, 2 outputs, the r2 observed timing, and the fixture qualification. | §12 | estimate |

Unchanged, explicitly: the four roadmap exit gates and the four controls (§1); every §10 numeric threshold; every §6 validity rule; MSTDPET (`gamma = 0.25 mV`, `tau_plus = tau_minus = 20 ms`, `tau_elig = 25 ms`, `a_plus = a_minus = 1/25`); ALIF (`tau_a = 200 ms`, `beta_a = 1.12 mV`); hidden layer and W1; regimes A/B; 400,000 steps with the shift before step 200,000; experimental seeds `0..19`; held-out stream construction and lengths (12,000 = 2,000 warm-up + 10,000 scored); exactly four checkpoints; the bootstrap (§9, §9.1). No tuning, no extra arm, no Stage 3/4, no Track 4.

Items for the reviewer to adjudicate (flagged, not decided here):

- **R1.** Gate 4 item 1 constant `60 -> 80` (ledger #5).
- **R2.** §8 `reward_signed_dw_out_corr` keeps the Stage 1 definition, Pearson(reward, signed change of the summed hidden-to-output weights). With two outputs the sum runs over all 40 W2 entries. The threshold is unchanged. Per-output correlations are reported, not gated.
- **R3.** Pre-existing SeedSequence coincidence, carried unchanged from r2 (the r2 run already used it). For seed `7`, the object components `[20261001, 7, c]`, `c = 1..6`, are the same entropy lists as the bootstrap generators `[20261001, 7, k]` for `k = 1..6`. The weights/streams of seed 7 and bootstrap metrics 1–6 are therefore drawn from identically-seeded generators (different draw calls). The card freezes the bootstrap, so r3 does not change it. The new r3 components (`21`, `22`) collide with no r2 or bootstrap identity, which was checked with `SeedSequence.generate_state` on numpy 1.26.4 for seeds `0..19` and `1000..1019` and bootstrap IDs `0..14`, `100..119`.
- **R4.** The §11 fixture qualification looks at frozen-control accuracy on non-experimental seeds before the run. It is predeclared, it covers the frozen-from-start arm only, and its only permitted response is STOP. It is an explicit exception to the last paragraph of §11.
- **Residual risks (reported, not gated):** (i) if both outputs fall silent, or both fire on every step, `r_t = 0` and that state is still absorbing. r3 makes it unrewarded (scored at coin-flip accuracy) and no longer favored by the observed A-phase reward imbalance, but it does not make it impossible. (ii) A global scalar reward multiplies every output row's eligibility, including the row of the output that did not cause the reward, through its decaying `tau_elig = 25 ms` trace. This is the standard global-reward credit problem of Florian's multi-output setting, and it is exactly what the "one global scalar, not Track 4" decision accepts.

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

The network has two output neurons (r3, section 0 ledger #1): `O1` (output index 0, "predict 1") and `O0` (output index 1, "predict 0"). Their spikes already present at step `t`, `z1_t, z0_t in {0, 1}`, are the network's prediction of `x_t`, made from earlier observations. The environment then reveals `x_t` and grades that prediction before `x_t` is integrated into the network. This ordering uses `PureSNN.online_step` exactly: the reward associated with `(z1_t, z0_t)` is applied on transition `t -> t+1`, and only afterward does the network integrate the input spike encoding `x_t`.

The Florian per-spike, next-step, global scalar reward is retained. Its output term is the signed difference of the two declared outputs:

```text
r_t = (2*x_t - 1) * (z1_t - z0_t)        r_t in {-1, 0, +1}
```

| `z1_t` | `z0_t` | `x_t = 1` | `x_t = 0` |
|---:|---:|---:|---:|
| 1 | 0 | `+1` | `-1` |
| 0 | 1 | `-1` | `+1` |
| 0 | 0 | `0` | `0` |
| 1 | 1 | `0` | `0` |

`r_t` is ONE scalar per transition. It is passed unchanged as the `reward` of `online_step`, so every synapse in both layers (W1 and both rows of W2) is updated by `gamma * r_t * e_ij` with its own eligibility `e_ij`. This is exactly the Stage 1 `REWARD_PER_SPIKE_NEXT_STEP` path. There is no per-neuron, per-row, or vector reward channel; that is Track 4 and is out of scope. With one output and `z0 = 0` the formula reduces exactly to r2's `r_t = (2*x_t - 1) * z_t`.

The prediction rule (section 5.2) is:

```text
prediction_t = 1                     if z1_t = 1 and z0_t = 0
prediction_t = 0                     if z1_t = 0 and z0_t = 1
prediction_t = coin_t in {0, 1}      if z1_t = z0_t   (tie: both silent or both firing)
```

`coin_t` is a fair coin from its own SeedSequence component (section 7.1). It is independent of the streams, weights, and network state, and it never enters the network, the reward, or the plasticity. A "0" is therefore an active choice: it must be emitted as an `O0` spike, so it can be rewarded or punished. A tie earns no update, and it is scored at coin-flip accuracy rather than as a correct base-rate "0". This removes r2's asymmetry, in which silence received full credit for `x_t = 0` but could never be corrected. Prediction (and therefore the coin) is computed only to score held-out evaluations. Training accuracy is neither scored nor used.

The target is never supplied as an input, feature, regime label, loss gradient, weight initializer, or state reset. It enters only as the next observation and the sign of the reward for already-emitted output spikes. This is the no-oracle boundary.

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

A and B have the same binary marginal, the same one-hot input magnitude, the same innovation probability, the same theoretical optimum accuracy (90%), and opposite hidden transition rules. There is no shift marker. At the fixed boundary, the training B stream uses the last two realized A observations as its starting context: the first B observation `x_200000` is generated from `x_199998` and `x_199999`. (The standalone held-out B stream has its own declared starting context; see section 7.1.)

This is deliberately a hidden-structure problem rather than a label-remapping problem: the current observation alone is insufficient, and the regime can only be inferred from temporal history. Chance accuracy is exactly `0.50` because the stationary binary marginal is balanced.

### 3.4 Training length and shift

Each run contains exactly `400,000` observed steps:

- A phase: steps `0..199,999` (`200,000 ms` developmental age; `200 s` SNN integration time).
- Unexpected A->B shift: immediately before step `200,000`.
- B phase: steps `200,000..399,999` (`200,000 ms`; cumulative `400 s`).

The training run is initialized exactly once, before step 0, with the state defined in section 5.1 (identical to `PureSNN.reset_online_state` plus `a = 0`). At the shift there is no membrane reset, SFA reset, eligibility reset, spike reset, reward-running-statistic reset, weight restore, optimizer/retrainer invocation, new neuron, new synapse, context bit, phase label, or pause in plasticity in the primary arm. The only changed variable is the environment's transition rule.

## 4. Frozen network and learning rule

The implementation must reuse the Stage 1 repository path `snn/core.py` and reward mode `REWARD_PER_SPIKE_NEXT_STEP`. No separate NumPy learner is allowed in the experimental path.

### 4.1 Fixed architecture

| Quantity | Frozen value |
|---|---:|
| Layer sizes | `[2, 20, 2]` (r3; r2 was `[2, 20, 1]`) |
| Trainable synapses | 40 input-to-hidden + 40 hidden-to-output (20 to `O1` + 20 to `O0`) = 80 |
| `dt` | `1.0 ms` |
| LIF `tau_m` | `20.0 ms` |
| Synaptic filter | none, `tau_syn = 0.0 ms` (direct voltage jump) |
| Rest/reset | `-70.0 mV` / `-70.0 mV` |
| Baseline threshold | `-54.0 mV` |
| Hidden neurons | ALIF/SFA on in primary arm |
| Output neurons | two, `O1` (index 0, "predict 1") and `O0` (index 1, "predict 0"); each an ordinary LIF, non-adaptive, with the same LIF parameters as above; no lateral, recurrent, or inhibitory connection between them |
| Precision | float64 |
| Batch size | 1 |

Initial input-to-hidden weights are independent uniform `U(-10, 10) mV`. Hidden-to-output weights, for both rows (`O1` and `O0`), are independent uniform `U(0, 10) mV`. Bounds remain `[-10, 10] mV` for W1 and `[0, 10] mV` for both W2 rows. Row `O1` is r2's `W2` draw, unchanged. Row `O0` comes from the new component `[20261001, seed, 21]` (section 7.1). Initialization and all stream generation use NumPy `default_rng` as specified in section 7.

Capacity is fixed at construction. Structural plasticity, neuron birth/death, masks that add connections, replay buffers, and external classifiers are prohibited. Auxiliary readouts are prohibited: the only readout is the declared `O1`/`O0` pair, decoded by the fixed section 3.1 rule. Any further output neuron, readout, decoder, read-out of hidden activity, or learned or fitted decision rule is forbidden.

### 4.2 R-STDP

Use Stage 1 temporal MSTDPET parameters without tuning:

| Quantity | Frozen value |
|---|---:|
| Credit mode | `eligibility` (MSTDPET) |
| `gamma` / learning rate | `0.25 mV` |
| STDP `tau_plus`, `tau_minus` | `20.0 ms`, `20.0 ms` |
| Eligibility `tau_elig` | `25.0 ms` |
| Pairing amplitudes | `a_plus = a_minus = 1/25` |
| Reward schedule | per-output-spike, following-step, online; one global scalar `r_t = (2*x_t - 1) * (z1_t - z0_t)` (section 3.1) |
| Plasticity state | continuous across patterns and A->B |

The update ordering, signs, and bounds are those already qualified in `CoreFlorianBench`: update pairing and eligibility, apply the per-spike reward, clamp, then integrate the current input with the updated weight. The rule is the unmodified `RSTDPPlasticity` on both synapse layers. Each W2 row's eligibility is driven only by its own output neuron's spikes and the shared hidden spikes. Implementation calls `online_step(onehot(x_t), reward_fn)` with `reward_fn(f) = (2*x_t - 1) * (f[0, 0] - f[0, 1])`. No new learning code is permitted: no change to `snn/core.py`'s plasticity, reward application, or clamp.

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

For each seed, immutable A and B evaluation streams are generated before training from independent RNG streams by the exact procedure in section 7.1. Each evaluation stream has `12,000` observations: the first `2,000` (indices `0..1,999`) are an unscored state warm-up, and the final `10,000` (indices `2,000..11,999`) are scored. Evaluation copies have plasticity disabled (no reward is ever applied, so weights are bitwise constant during evaluation). The common start state in section 5.1 is confined to disposable evaluation copies; the continuous training run is never reset.

### 5.1 Exact dynamic-state initialization (training start and every evaluation copy)

The single training initialization before step 0 and the start of every evaluation copy use exactly this state, and no other:

| State | Value |
|---|---|
| Membrane potential, every neuron in every layer (hidden and output) | `v_rest = -70.0 mV` (as `PureSNN.reset_online_state`, `snn/core.py:277-285`) |
| Current spikes, every layer (so the first output state is `z1 = z0 = 0`, a tie: no training reward, and in evaluation it is resolved by the coin, inside the unscored warm-up) | `0` |
| R-STDP `eligibility`, `pre_trace`, `post_trace`, `last_pairing` (`RSTDPPlasticity.reset`) | `0` |
| ALIF adaptation `a`, every hidden neuron (SFA-on and SFA-off arms) | `0` |
| Batch size / dtype | `1` / float64 |

An evaluation copy is produced by `copy.deepcopy` of the live network at the checkpoint and then set to the table above; therefore the weights are the only quantity carried from training into evaluation. The live network, its dynamic state, and every RNG are not touched. No zero-membrane or any alternative initialization is permitted.

### 5.2 Evaluation step contract

At evaluation index `i` of checkpoint `c`, the copy's current output spikes `(z1_i, z0_i)` are recorded. The prediction of held-out observation `x_i` is formed by the section 3.1 rule: `1` if `O1` alone fired, `0` if `O0` alone fired, otherwise `coin[seed, c][i]`. Then `online_step(onehot(x_i), reward_fn=None)` is called. Prediction `i` is correct iff `prediction_i == x_i`. Only indices `2,000..11,999` are scored.

The tie-coin vector `coin[seed, c]` (length 12,000, `int8` in `{0, 1}`) is drawn once per seed and per checkpoint before training by the section 7.1 procedure. It is indexed by evaluation index `i` and consulted only when `z1_i == z0_i`. It is the same vector in every arm for that seed and checkpoint, so arms remain paired. It does not depend on the streams, the weights, the arm, or the network state. It is never consumed by training, is never an input to the network, and never enters a reward. The four checkpoints use four distinct coin vectors.

The same held-out A stream is used for that seed at all A checkpoints, and the same held-out B stream is used at all B checkpoints. No evaluation output or state returns to training.

### 5.3 Checkpoints (exactly four evaluations per arm/seed; no others)

1. `A_pre`: after step 199,999 and before the first B step; evaluate held-out A.
2. `B_pre`: from the same pre-shift weights; evaluate held-out B.
3. `B_post`: after step 399,999; evaluate held-out B.
4. `A_post`: from the same final weights as `B_post`; evaluate held-out A.

No initial (step-0) evaluation is executed in any arm. It would be redundant: the frozen-from-start arm's four evaluations are evaluations of the untouched initial weights under the identical start state and streams, and they are the predeclared untrained baseline. Executing any evaluation not listed above violates Gate 4 item 7.

Primary metric per seed and checkpoint:

```text
accuracy = correct binary predictions / 10,000 scored observations
```

Predictions follow the section 3.1 rule (`O1` alone -> 1, `O0` alone -> 0, tie -> the checkpoint's fair coin). Chance is `0.50`: a network whose output carries no information about `x_t` is right with probability exactly `0.50` on every step, because the coin is fair and independent and the held-out marginal is balanced. No threshold is estimated from data.

Also recorded per seed and checkpoint (report only, not gated): the prediction-one fraction (scored predictions equal to `1`, coin outcomes included), the `O1`-only, `O0`-only, both-silent, and both-fire fractions of scored indices, and the accuracy restricted to non-tie indices together with its count.

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

r3 note: these three conditions are unchanged from r2 and are not relaxed. Under r2 the frozen-from-start `B_pre` interval `[0.50004, 0.51342]` excluded `0.50`. r3 keeps nominal chance at `0.50` (section 5.3). It also applies this same validity check once before the experimental run, on non-experimental seeds, as a fixture qualification (section 11, item 15). If that check fails, the experimental run does not start.

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
| Initial weights: W1 and W2 row `O1` (unchanged from r2) | `[20261001, seed, 1]` |
| A training stream | `[20261001, seed, 2]` |
| B innovation draws | `[20261001, seed, 3]` |
| B scramble permutation | `[20261001, seed, 4]` |
| A held-out stream | `[20261001, seed, 5]` |
| B held-out stream | `[20261001, seed, 6]` |
| Initial weights: W2 row `O0` (new in r3) | `[20261001, seed, 21]` |
| Evaluation tie-coin, checkpoint `c` (new in r3; `c = 1, 2, 3, 4` = `A_pre, B_pre, B_post, A_post`) | `[20261001, seed, 22, c]` |
| Bootstrap resampling, metric `k` (section 9.1) | `[20261001, 7, k]` |

Every generator is `numpy.random.default_rng(numpy.random.SeedSequence(<entropy list>))`, created fresh for its one object and used only by the draws listed in section 7.1, in the order listed. The B generator starts from the final two realized A observations but uses its independent innovation RNG. The r3 components `21` and `22` are new. Adding them changes no draw of components `1..6` or of the bootstrap generators, so every r2 object (W1, W2 row `O1`, all streams, the scramble, the bootstrap) is bitwise identical to its r2 value for the same seed. All arms must serialize SHA-256 hashes of initial weights (the full `(20, 2)` W1 and `(2, 20)` W2), A observations, structured B observations, scrambled B observations where applicable, both held-out streams, and the four tie-coin vectors. Paired arms must assert matching hashes for every object they share before running. A §11 test asserts that W1, W2 row `O1`, and every stream hash for a non-experimental seed equal those produced by the r2 procedure.

Parallel workers may change wall time only. Result rows are sorted by arm then seed before aggregation.

### 7.1 Exact draw procedures (streams and weights uniquely defined)

Notation: `rng(e)` is a fresh `default_rng(SeedSequence(e))`; observations are `int8` values in `{0, 1}`; `flip_j = (u_j < 0.10)` with `u = rng.random(n)` (float64, NumPy's default `[0, 1)` uniform); `ideal_A = x_(t-2) XOR x_(t-1)` and `ideal_B = 1 - ideal_A`; `x_t = ideal(t) XOR flip`.

| Object | Draws, in this exact order, and nothing else | Starting context |
|---|---|---|
| Initial weights | `g = rng([20261001, seed, 1])`; `W1 = g.uniform(-10.0, 10.0, size=(20, 2))` (hidden x input); then `W2_O1 = g.uniform(0.0, 10.0, size=(1, 20))` (identical to r2's `W2` draw). Then `h = rng([20261001, seed, 21])`; `W2_O0 = h.uniform(0.0, 10.0, size=(1, 20))`. `W2 = concatenate([W2_O1, W2_O0], axis=0)`, shape `(2, 20)` (output x hidden; row 0 = `O1`, row 1 = `O0`). Copied into `synapses[0].weight` and `synapses[1].weight` as float64 | — |
| Evaluation tie-coin, checkpoint `c` in `1..4`, length 12,000 | `g = rng([20261001, seed, 22, c])`; `coin = g.integers(0, 2, size=12_000)` cast to `int8`; `coin[i]` is used only at evaluation index `i` of checkpoint `c` when `z1_i == z0_i` | n/a |
| A training stream, length 200,000 (steps `0..199,999`) | `g = rng([20261001, seed, 2])`; `c = g.integers(0, 2, size=2)` -> `x_0, x_1`; then `u = g.random(199_998)` -> flips for `t = 2..199,999` in order, rule A | `x_0, x_1` = the two drawn context bits (they are observed and trained on) |
| Structured B training stream, length 200,000 (steps `200,000..399,999`) | `g = rng([20261001, seed, 3])`; `u = g.random(200_000)` -> flips for `t = 200,000..399,999` in order, rule B. No context draw. | `x_199,998, x_199,999` of the realized A training stream |
| Scrambled B twin | `g = rng([20261001, seed, 4])`; `perm = g.permutation(200_000)`; `B_scr = B_struct[perm]` | n/a (a permutation of the realized structured B vector) |
| Held-out A stream, length 12,000 | `g = rng([20261001, seed, 5])`; `c = g.integers(0, 2, size=2)` -> `h_0, h_1`; then `u = g.random(11_998)` -> flips for `i = 2..11,999` in order, rule A | `h_0, h_1` = the two drawn context bits (indices 0-1, inside the unscored warm-up) |
| Held-out B stream, length 12,000 | `g = rng([20261001, seed, 6])`; `c = g.integers(0, 2, size=2)` -> `h_0, h_1`; then `u = g.random(11_998)` -> flips for `i = 2..11,999` in order, rule B | `h_0, h_1` = the two drawn context bits (indices 0-1, inside the unscored warm-up); **not** taken from any training stream |

`g.integers(0, 2, size=2)` uses NumPy's default `int64` dtype and is cast to `int8`. Both held-out streams are therefore functions of `(seed, entropy component)` alone: they do not depend on the training streams, the arm, the weights, or the checkpoint, and each is generated once per seed before training, hashed, and reused unchanged at its checkpoints in every arm. Because the held-out streams' contexts and innovations come from entropy components 5 and 6, which no training object uses, they are independent of training as claimed in section 5. The NumPy version used is recorded in provenance; the stream hashes, not the version, are the identity checked across arms.

## 8. Stage 1 sign-of-life instrumentation carried forward

Record separately for A and B and for every seed/arm:

- total output spikes, and per output neuron (`O1`, `O0`);
- positive and negative reward-event counts (a reward event is a transition with `r_t != 0`);
- the both-silent fraction (`z1_t = z0_t = 0`), the both-fire fraction (`z1_t = z0_t = 1`), and the `O1`-only and `O0`-only fractions of training steps (report only, not a gate);
- total absolute weight change in each layer, and for W2 per output row;
- the phase-end weight extrema and bound hits as the separate `a_end_*` and `b_end_*` fields of section 8.1 (r2 serialized only the final values; r3 makes both sets mandatory);
- reward and signed hidden-to-output `delta_w` for each reward event, or exact streaming sufficient statistics for their Pearson correlation;
- `reward_signed_dw_out_corr` (Stage 1 definition: Pearson correlation, over reward events, between the reward and the signed change of the sum of all hidden-to-output weights on that transition; with two outputs the sum runs over all 40 W2 entries, both rows). Also reported, not gated, are the per-row correlations `reward_signed_dw_O1_corr` and `reward_signed_dw_O0_corr`;
- hidden and output firing rates (each output separately);
- SFA adaptation mean, maximum, and threshold contribution for SFA-on arms;
- all held-out accuracies, the prediction-one fraction, and the section 5.3 tie/readout fractions;
- the maximum weight difference for each freeze interval.

### 8.1 Phase-end weight fields and schema completeness (r3)

For every arm/seed row, the writer serializes two separate, complete field sets. These are snapshots of the live training network's weights; they are never taken from an evaluation copy:

- `a_end_*`: captured on the live network after training step `199,999` completes and before the first B step (`200,000`) begins. That is the same instant and the same weights as the `A_pre`/`B_pre` checkpoint copies.
- `b_end_*`: captured on the live network after training step `399,999` completes. That is the same instant and the same weights as the `B_post`/`A_post` checkpoint copies.

Each set contains exactly these fields (`P` = `a_end` or `b_end`). Here `lo`/`hi` are the declared bounds: W1 `[-10, 10] mV`, W2 `[0, 10] mV`. A bound hit is an entry exactly equal (float64 `==`) to that bound.

| Field | Meaning |
|---|---|
| `P_w1_min`, `P_w1_max` | min / max over all 40 W1 entries (mV) |
| `P_w1_lower_hits`, `P_w1_upper_hits` | count of W1 entries `== -10.0` / `== 10.0` |
| `P_w2_min`, `P_w2_max` | min / max over all 40 W2 entries (mV) |
| `P_w2_lower_hits`, `P_w2_upper_hits` | count of W2 entries `== 0.0` / `== 10.0` |
| `P_w2_O1_min`, `P_w2_O1_max`, `P_w2_O1_lower_hits`, `P_w2_O1_upper_hits` | the same four quantities over the 20 entries of W2 row `O1` |
| `P_w2_O0_min`, `P_w2_O0_max`, `P_w2_O0_lower_hits`, `P_w2_O0_upper_hits` | the same four quantities over the 20 entries of W2 row `O0` |
| `P_weights_sha256` | SHA-256 of the float64 bytes of W1 then W2 (C order) at capture |

That is 17 fields per set and 34 per row. Every arm, including frozen-from-start, records both sets. In arms whose weights cannot change in a phase, the fields are still computed from the live weights and are never copied from another field.

Schema-completeness assertion: before writing any row, the result writer checks that every `a_end_*` and `b_end_*` field above is present, finite, and of the declared type (float for min/max, non-negative int for hits, 64-hex string for the hash). It also checks that `P_w2_lower_hits == P_w2_O1_lower_hits + P_w2_O0_lower_hits`, and the same for the upper hits. If any check fails, the writer raises and emits no row. A run with any missing row is incomplete (section 13) and cannot pass. This turns the r2 process failure, where 9/100 rows had A-end values that could not be recovered, into a hard failure at write time instead of a gap discovered in review.

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
- RNG: one independent generator per metric, `default_rng(SeedSequence([20261001, 7, k]))`, where `k` is the metric ID in section 9.1; generators are never shared between metrics, so the metric evaluation order cannot change any draw;
- resample draw: exactly one call per metric, `idx = g.integers(0, 20, size=(100_000, 20))`; replicate `b` uses seed rows `idx[b, :]` (with replacement);
- a seed row is that seed's complete paired record: all five arms' accuracies at all four checkpoints. Every quantity entering one metric (both sides of a difference, numerator and denominator of a ratio) is taken from the same resampled seed rows `idx[b, :]` in replicate `b`;
- each replicate recomputes the metric's full formula from section 9.1 on its 20 resampled rows (means first, then differences/ratios exactly as in the point estimate);
- percentiles: `numpy.percentile(replicates, q, method="inverted_cdf")` over all 100,000 replicates;
- one-sided lower 95% bound: `q = 5`;
- one-sided upper 95% bound: `q = 95`;
- two-sided 95% interval: `q = 2.5` and `q = 97.5`.

No normal approximation, per-tick pseudo-replication, dropped seed, outlier exclusion, or alternate interval is allowed. All comparisons are strict where written `>`; equality fails a strict gate. The four behavioral exit gates are conjunctive, so no multiplicity correction is applied.

Definitions using across-seed mean accuracies:

```text
A_learned       = mean(A_pre_primary) - 0.50
A_loss          = mean(A_pre_primary) - mean(A_post_primary)
A_retained      = (mean(A_post_primary) - 0.50) / A_learned
B_improvement   = mean(B_post_primary - B_pre_primary)
```

If `A_learned <= 0`, retention is undefined and Gate 3 fails.

### 9.1 Frozen bootstrap metric table (complete; IDs fix the RNG)

Notation: `acc[arm, ckpt]` is the length-20 vector of seed accuracies, rows aligned by seed; `m(v)` is the arithmetic mean over the (resampled) rows. Arms: `P` primary, `F0` frozen from start, `FS` freeze at shift, `SC` scrambled-B twin, `NS` SFA off. In replicate `b` every vector is indexed by the same `idx[b, :]`.

| ID `k` | Metric (point estimate and every replicate use this formula) | Interval(s) required | Used by |
|---:|---|---|---|
| 0 | `m(acc[F0, A_pre])` | two-sided 95% | §6.2 validity |
| 1 | `m(acc[F0, B_pre])` | two-sided 95% | §6.2 validity |
| 2 | `m(acc[F0, B_post])` | two-sided 95% | §6.2 validity |
| 3 | `m(acc[F0, A_post])` | two-sided 95% | §6.2 validity |
| 4 | `m(acc[P, A_pre])` | one-sided lower; two-sided 95% | Gate 1.2 |
| 5 | `m(acc[P, A_pre] - acc[F0, A_pre])` | one-sided lower | Gate 1.4 |
| 6 | `m(acc[P, B_post] - acc[P, B_pre])` (= `B_improvement`) | one-sided lower; two-sided 95% | Gate 2.2 |
| 7 | `m(acc[P, B_post])` | one-sided lower; two-sided 95% | Gate 2.4 |
| 8 | `m((acc[P, B_post] - acc[P, B_pre]) - (acc[FS, B_post] - acc[FS, B_pre]))` | one-sided lower | §6.3 / Gate 2.5 |
| 9 | `m(acc[P, B_post] - acc[SC, B_post])` | one-sided lower | §6.4 / Gate 2.5 |
| 10 | `m(acc[P, A_post])` | one-sided lower; two-sided 95% | Gate 3.3 |
| 11 | `A_loss = m(acc[P, A_pre]) - m(acc[P, A_post])` | one-sided upper; two-sided 95% | Gate 3 report |
| 12 | `A_retained = (m(acc[P, A_post]) - 0.50) / (m(acc[P, A_pre]) - 0.50)` | one-sided lower; two-sided 95% | Gate 3 report |
| 13 | `m(acc[P, A_post] - acc[NS, A_post])` | one-sided lower | §6.5 SFA-benefit claim |
| 14 | `A_learned = m(acc[P, A_pre]) - 0.50` | two-sided 95% | report |
| 100 + 4·a + c | `m(acc[arm_a, ckpt_c])`, arm order `a = 0..4` = `P, F0, FS, SC, NS`; checkpoint order `c = 0..3` = `A_pre, B_pre, B_post, A_post` | two-sided 95% | report (all 20 arm×checkpoint means) |

Metric 12 is a ratio of resampled means: in replicate `b` its numerator and denominator are both computed from rows `idx[b, :]`, never from separately resampled vectors and never as a mean of per-seed ratios. If the replicate denominator `m(acc[P, A_pre]) - 0.50 <= 0`, the replicate value is `-inf` (retention undefined counts toward failure); it is retained among the 100,000 replicates, and the count of such replicates is reported. `inverted_cdf` percentiles make this well-defined. No metric outside this table may be bootstrapped or reported as a gate quantity.

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

1. every arm has exactly 80 trainable synaptic weights before and after the run (r3: the constant follows the declared `[2, 20, 2]` architecture, 40 + 40; r2 read 60 for `[2, 20, 1]`; wording and logic unchanged; flagged in section 0 ledger #5 for review);
2. the primary arm's live training network has exactly one initialization (section 5.1, before step 0) and zero calls to network, membrane, spike, eligibility, SFA, or weight reset after step 0 (initialization of a disposable evaluation copy per section 5.1 is not a call on the live network and is logged separately);
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
2. Ordering check: changing `x_t` cannot affect `z_t` (r3: either output, `z1_t` or `z0_t`) before reward at transition `t -> t+1`.
3. No-reset check: instrumented primary shift records zero reset calls.
4. Freeze checks: frozen-from-start changes zero weights; freeze-at-shift changes zero weights after the shift.
5. Stream checks: A/XOR and B/XNOR equations hold against hand-computed fixtures; scrambled B preserves the exact symbol multiset.
6. Evaluation isolation: checkpoint evaluation leaves a byte-identical live training state and unchanged RNG states.
7. SFA-off check: with `beta_a=0`, output is bit-identical to the ordinary LIF path on a fixed short fixture.
8. Legacy regression: the maintained Stage 1 core and provenance tests stay green.
9. Start-state check: training start and every evaluation copy have all membranes at exactly `-70.0 mV` and zero spikes, eligibility, traces, and adaptation (section 5.1).
10. Stream-procedure check: on non-experimental seeds, streams and weights regenerated by the section 7.1 procedure are bit-identical across two independent constructions, and the held-out B stream's indices 0-1 equal the two context bits drawn from component 6.
11. Bootstrap determinism: on a synthetic (non-experimental) 20-row table, every section 9.1 metric's interval is identical when metrics are computed in forward vs. reverse ID order, and metric 12 replicates equal the ratio recomputed from the same resampled rows.

r3 additions (items 12–16):

12. Rule-unchanged check (two-output path vs. qualified single-output core path). On a fixed short non-experimental fixture, record a pre/post spike history: input spikes, hidden spikes, both output spike trains `z1`, `z0`, and the scalar reward sequence `r_t` of section 3.1. Replay it through the r3 two-output network's unmodified `RSTDPPlasticity` objects. Separately, for each output `k` in `{O1, O0}`, replay through a single-output `[2, 20, 1]` `RSTDPPlasticity` on the qualified Stage 1 core path (`snn/core.py`, `credit="eligibility"`, section 4.2 parameters), with the same initial weights for that row, the same hidden spike history, that output's spike train as the post spikes, and the identical scalar `r_t` sequence. Assert at every step that W2 row `k`'s eligibility, traces, and weight after the update and clamp are bit-identical (float64 `==`) to the single-output run. Also assert that W1's update is bit-identical to a single-output run with the same input/hidden history and `r_t` sequence. This shows that only the readout and the reward's output term changed. The learning rule did not.
13. Phase-end fields check (non-experimental seeds, short phases). Every emitted row contains every section 8.1 `a_end_*` and `b_end_*` field. In an instrumented run, `a_end_*`, including `a_end_weights_sha256`, equals the values recomputed from the live network's weights read immediately after the last A step and before the first B step. `b_end_*` equals the live weights after the last B step. A row with any section 8.1 field deleted is refused by the writer (negative test). In a plastic arm whose weights change during B, `a_end_*` differs from `b_end_*`.
14. Tie-coin check. (a) Determinism: the four coin vectors regenerated by the section 7.1 procedure are bit-identical across two independent constructions. (b) Independence: they are unchanged when the training streams, the held-out streams, the weights, or the arm are changed (that is, they are a function of `(seed, c)` only), and they are bit-identical across all five arms for a seed. (c) Use: the coin is read only at tie indices of evaluations. With a forced-tie fixture (both outputs held silent), scored predictions equal `coin[i]` exactly. With a forced `O1`-only or `O0`-only fixture, the coin is never read. Training never reads a coin (access log). (d) Distinctness: the coin vectors and the new `O0` weight row are drawn from components `21`/`22`, and their SeedSequence states equal no state in components `1..6` or bootstrap `[20261001, 7, k]`.
15. **Frozen-control fixture qualification (pre-run STOP condition).** After items 1–14 and 16 pass, and before any experimental seed runs, execute the frozen-from-start arm (section 6.2) only, unchanged and at full length: 400,000 training steps and the four 12,000-step held-out evaluations with the r3 readout and tie-coin, on the non-experimental seeds `1000..1019`, with the section 7.1 procedures. Compute the section 9.1 metrics 0–3 on these 20 rows with the section 9 bootstrap, using the same entropy `[20261001, 7, k]`. Check all three section 6.2 conditions: weights bitwise constant in 20/20 seeds; mean accuracy in `[0.45, 0.55]` at each checkpoint; two-sided 95% interval containing `0.50` at each checkpoint. If ANY condition fails at ANY checkpoint, the build card STOPS, runs no experimental seed, and returns to spec review with the qualification record. No retry, no different non-experimental seed set, no parameter change, and no partial experimental run is permitted. This step applies the existing validity rule early. It adds no arm and moves no threshold. It is the one predeclared exception to the prohibition below on looking at accuracy before the experimental run: it looks only at the frozen-from-start arm on non-experimental seeds, and its only permitted action is STOP or proceed unchanged. The qualification record (rows, intervals, verdict) is committed with the result package.
16. r2-object identity: for a non-experimental seed, W1, W2 row `O1`, the A, structured B, and scrambled B training streams, and both held-out streams are bit-identical to those produced by the r2 (`ef85a15`) section 7.1 procedure.

Apart from item 15, the following are prohibited: performance smoke tests, seed screening, hyperparameter sweeps, and looking at partial seed outcomes before all 100 arm/seed jobs complete.

## 12. Resource and wall-time estimate (Mac mini)

Target host observed while freezing this spec: Mac mini, Intel Core i5-8500B 3.0 GHz, 6 logical CPUs, 32 GiB RAM. Use 5 worker processes, leaving one logical CPU for the OS and coordinator.

Workload:

- 5 arms x 20 seeds = 100 continuous runs;
- 400,000 training steps per run = 40,000,000 training steps total;
- four 12,000-step held-out evaluations per run (A/B pre and A/B post; the pre pair shares one checkpoint and the post pair shares one checkpoint) = 4,800,000 evaluation steps total;
- total approximately 44.8 million SNN steps.

The Stage 1 temporal core jobs took roughly 241–275 seconds for 400,000 steps while also re-running the independent anchor. Removing the anchor comparison but adding ALIF state and held-out evaluation gave the r1/r2 planning estimate of 300–450 seconds per Stage 2 arm/seed job.

r3 update (80 synapses, 2 outputs). Observed under r2 on this host (`results_stage2/summary.json`): 100 jobs in `3,317 s` wall at 5 workers, which is about 166 s per job, with total peak RSS `1.16 GiB`. The r2 jobs were cheap partly because the output was mostly silent: few reward events, so the per-event weight snapshot and correlation statistics rarely ran. r3 adds one 20-weight output row, a `(2, 20)` eligibility instead of `(1, 20)`, one tie-coin lookup per evaluation step, and the section 8/8.1 counters. Each is O(20) work per step and small next to the per-step Python/torch overhead. Reward events can occur on every non-tie step, though, so the per-event bookkeeping can run up to roughly every step. Planning estimate: 170–450 s per arm/seed job.

- Experimental run: 100 jobs, about 44.8 million SNN steps (unchanged count); `1.0–2.5 h` at 5 workers.
- Section 11 item 15 fixture qualification: 20 frozen-from-start jobs on seeds `1000..1019`, about 8.96 million steps; `0.2–0.5 h` at 5 workers. It runs before, and separately from, the experimental run.
- Budget: `3.5 hours` in total, including both, coordination, hashing, aggregation, and variance. Expected peak memory remains `< 2 GiB`, and the committed result package `< 25 MiB` (the 34 section 8.1 fields per row add under 10 KiB in total). Abort rather than swap if resident memory exceeds 8 GiB.

These are planning estimates, not gates. Actual wall time, peak RSS, worker count, interruptions, and retries must be reported, separately for the qualification step and the experimental run.

## 13. Frozen result schema and provenance

The future result package must contain:

- one JSON row for every one of the 100 arm/seed runs;
- seed-level accuracies and prediction-one fractions at exactly the four checkpoints `A_pre`, `B_pre`, `B_post`, `A_post` (no initial-checkpoint field exists), with the section 5.3 tie/readout fractions;
- every section 8.1 `a_end_*` and `b_end_*` field in every row, enforced by the schema-completeness assertion; per-output spike counts, +/- reward-event counts, and both-silent/both-fire fractions for A and B separately;
- SHA-256 of the full initial W1/W2 and of the four tie-coin vectors per seed;
- the section 11 item 15 fixture-qualification record (20 rows, metrics 0–3 with intervals, PASS/STOP), produced before the experimental run;
- one bootstrap record per metric ID in section 9.1, sorted by ID, each containing ID, formula name, point estimate, the required interval bounds, the SeedSequence entropy `[20261001, 7, k]`, and (for ID 12) the count of undefined-denominator replicates;
- sign-of-life metrics, freeze deltas, firing/SFA summaries, hashes, timing, and status;
- aggregate point estimates and all required bootstrap intervals;
- exact executable argv as both JSON array and `shlex.join` display string;
- source/gate commit and confirmation that scientific files were clean;
- maintained test command and result;
- a plain-language report with every gate and control marked PASS/FAIL;
- SHA-256 for all result artifacts.

The gate code must be committed and pushed before the first experimental seed runs. Result aggregation may not silently omit a failed, timed-out, or non-finite seed; an incomplete 20-seed arm cannot pass.

## 14. Freeze, rollback, and next gate

r2 freeze record (historical; quoted verbatim from ef85a15, where "this card" is the r2 card, not r3):

> This card changes only `docs/STAGE2_SPEC.md`. Rollback is `git revert <spec-commit>`; no runtime or science artifact exists to remove.

r3 (`t_de6dcd58`) changes exactly three paths: `docs/STAGE2_SPEC.md` (modified; this text), `docs/STAGE2_SPEC_r2_ef85a15.md` (added) and `tests/test_stage2.py` (modified). The last two are non-science test-maintenance edits that the r3 edit forces:

- `docs/STAGE2_SPEC_r2_ef85a15.md`: a byte-identical archive of the r2 spec (`git show ef85a15:docs/STAGE2_SPEC.md`, SHA-256 `0c1bbdba…ec87`). It preserves the exact r2 bytes that the r2 result package identifies by path and hash: `results_stage2/provenance.json`, `results_stage2/summary.json` and `results_stage2/STAGE2_REPORT.md` cite `docs/STAGE2_SPEC.md` with SHA-256 `0c1bbdba…ec87`. The package does not cite this archive path. r3 adds the archive so those bytes stay available at a stable path after the live file changes.
- `tests/test_stage2.py::test_spec_file_hash_is_frozen_ef85a15`: the test pinned the live spec file to the r2 hash, so freezing r3 necessarily broke it. It now pins the r2 archive to the r2 implementation's `SPEC_SHA256`. It also asserts that the r2 runner's preflight, which still hashes the live `docs/STAGE2_SPEC.md`, now sees a different hash and therefore refuses to run the r2 code against r3. No assertion was weakened.

It does not touch r2's code (`snn/`, `run_stage2.py`, `d04fbd9`), its results (`results_stage2/`, `1418436`, `084aec8`), or `docs/STAGE2_IMPLEMENTATION_NOTES.md`, which stays the r2 code map. A later r3 build card must write its own implementation notes and code pin against the r3 spec hash. Rollback is `git revert <r3-spec-commit>`, which restores the r2 text and the original test exactly.

Next gate: Nora independently reviews this frozen document for roadmap fidelity, numerical closure, no-oracle timing, control sufficiency, and implementation ambiguity, and adjudicates the section 0 items R1–R4. Only after approval may a separate build card implement Stage 2 r3, and that card must stop at section 11 item 15 if the fixture qualification fails. Stage 3, Stage 4, and Track 4 remain out of scope.