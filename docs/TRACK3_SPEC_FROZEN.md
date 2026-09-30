# Track 3 Wave-Gated Plasticity — Frozen Specification

**Status:** FROZEN — doc-only deliverable. Implementation is a separate build card.  
**Repo:** `/Users/kendrick/Projects/pure-snn-learning`  
**Baseline:** Track 1 A–F revisions already landed (`v0.3.0-calibrated`, `cc12d61`).  
**Date frozen:** 2026-09-29  

## 1. Scope and non-scope

This document freezes the design of the Track 3 (wave-gated eligibility) experiment.  No code is to be written under this card.  The frozen spec is the input to a future implementation card.

What the spec covers:
- The theta-wave phase reference and its controls.
- The new `WaveGatedPlasticity` learning rule and its exact update equation.
- The four mandatory experiment arms and the optional fifth arm.
- Predeclared pass / null / stop criteria.
- Claim boundaries — what a positive result licenses and what it does not.
- Estimated footprint and rollback.

What the spec does **not** cover:
- Track 1 (Florian source-matched anchor) — that is a parallel track.
- Track 2 instrumentation — already landed in `v0.3.0-calibrated`.
- Any change to the existing `RSTDPPlasticity` class or the task definitions beyond parameter overrides.

## 2. Revision G — Oscillator

The wave is a **pure phase reference**.  It injects no current into any neuron.  Its only function is to supply a phase `phi(t)` that is read at reward time and used to gate eligibility in the learning rule.

Parameters:
- Frequency: `f = 8 Hz` (theta band).
- Phase evolution: `phi(t) = 2 * pi * f * t + phi_0`.
- Time is measured in the same units as the simulation: `dt = 1.0` per timestep = 1 ms per timestep.

Control arms:
- **Wave-coherent:** `phi_0 = 0` fixed across all trials.  The reward is delivered at the peak of the wave.
- **Phase-scrambled:** `phi_0 ~ U(0, 2 * pi)` per trial.  The phase is randomized each trial so that, on average, the wave carries no coherent information about reward timing.  This is the primary negative control for phase specificity.
- **Phase-uniform gain-only (optional Arm 5):** `phi_0 = pi / 2` so `cos(phi_reward) = 0` everywhere, producing a constant non-negative gain.  This is a secondary discriminator for non-specific wave effects; see §6.

Reset rule:
- The oscillator phase is reset together with the eligibility trace whenever a trial starts (`reset` called on the plasticity object).  This keeps the wave and eligibility locked to the trial boundary and prevents phase drift across trials.

## 3. Revision H — Learning rule

### 3.1 Class and file

- New file: `snn/wave_plasticity.py`.
- New class: `class WaveGatedPlasticity(RSTDPPlasticity):`.
- **No edits to the existing `RSTDPPlasticity` class.**  The wave class inherits the existing per-sample `apply_reward` and overrides only what is necessary.

### 3.2 Dual-trace design

The class maintains **two eligibility traces** per synapse:

- **Fast trace `e_fast`:** same as the standard `RSTDPPlasticity` eligibility trace.  `tau_elig_fast = 20 ms`.  This trace is retained so that the wave class does not disturb ordinary STDP trace computation.
- **Slow trace `e_slow`:** a second eligibility trace computed from the same pre/post spike pairing but with a longer time constant.  `tau_elig_slow = 500 ms`.

Both traces use the same STDP window parameters as the base class (`tau_plus = 20 ms`, `tau_minus = 20 ms`, `a_plus = 0.01`, `a_minus = 0.01`).

### 3.3 Update equation

At the end of a trial, the reward vector `R` (shape `(batch,)`) and the phase at reward time `phi_reward` are used to compute the weight update:

```
gate = max(0, cos(phi_reward))
Δw = eta * mean_over_batch( R * gate * e_slow )
```

Equivalently, using the per-sample product semantics from Revision B:

```python
r_expanded = R.reshape(-1, 1, 1)          # (batch, 1, 1)
gate = max(0.0, math.cos(phi_reward))       # scalar
scaled = r_expanded * gate * e_slow        # (batch, post, pre)
delta_w = self.lr * scaled.mean(dim=0)     # (post, pre)
```

After the update, weights are clamped to `[-2.0, 2.0]`, identical to the base class.

### 3.4 Degenerate-gate instrumentation

Because `gate` can be zero whenever `cos(phi_reward) <= 0`, the rule may produce no update for large portions of the reward landscape.  The implementation must record:

- `fraction_updates_gate_zero`: the fraction of weight updates (across trials and, if batching multiple updates, across batches) for which `gate == 0`.
- `phase_at_reward`: the phase `phi_reward` at reward time for each trial.
- `gate_at_reward`: `max(0, cos(phi_reward))` for each trial.

If `fraction_updates_gate_zero` is close to 1, the wave-coherent arm degenerates to no-learning and the experiment should be reported as a STOP regardless of task performance.

## 4. Revision I — Task scope

Phase 1 uses the same bench as the post-A–F Track 1 anchor:

1. `binary_classification` (rate-based, easy config with `weight_scale=2.0`).
2. `temporal_xor` (paper protocol: 500 ms trains, exactly 50 uniformly distributed spikes per active channel).
3. `temporal_sequence` (dead-network diagnosis config).

Phase 2 (delayed-reward stress variant) is **conditional**: it is run only if Phase 1 shows learning on at least one task.  The definition of "shows learning" is the same as the predeclared stop/null criteria in §8: at least one arm beats frozen on ≥2/3 seeds by the required margin.

## 5. Nora amendment 1 — Mandatory third arm: eligibility-timescale-only

To separate "the wave did it" from "longer eligibility alone did it", the experiment must include a third arm that uses a biologically grounded eligibility time constant with **no wave**.

- **Arm name:** `eligibility-timescale-only`.
- **Mechanism:** standard `RSTDPPlasticity` with a longer `tau_elig`.
- **Exact time constant:** `tau_elig = 1000.0 ms`.

### 5.1 Why 1000 ms

The post-A–F bench already moved the default eligibility time constant from 20 ms to 1000 ms (`snn/core.py`, `RSTDPPlasticity.__init__`), separating the STDP pairing window (~20 ms) from the eligibility memory (~1 s).  This value is grounded in Izhikevich (2007) distal-reward model (`tau_c = 1 s`) and sits at the lower end of the biologically plausible 1–2 s BTSP plateau range.

The third arm therefore tests the strongest fair confound: a properly biological eligibility trace, without any wave gating.  If the wave-gated arm outperforms this arm, the effect is not merely a longer eligibility time constant.

### 5.2 Conflict resolved

Nora's original transcript phrase was "tau_elig = wave period" (which would be 125 ms for an 8 Hz wave).  The verbatim verdict text says "tau_elig matched to the wave timescale".  A strict reading would place `tau_elig` at either the wave period (125 ms) or the slow-trace time constant (500 ms).  However, a subsequent principal directive (Kendrick via Claude, 2026-09-29) explicitly applies to the Track 3 third arm and instructs:

> spec it at the biological value so the comparison is wave-gating versus a properly biological trace, not versus a crippled one.

This directive is binding and resolves the ambiguity.  The exact tau for the eligibility-timescale-only arm is therefore `1000.0 ms`, matching the new biological default.

## 6. Nora amendment 5 — Wave-eligibility coupling and shakedown

### 6.1 What the wave couples to

The wave gates **only** the eligibility trace at the moment of reward delivery.  Specifically:

- The pre/post spike pairing that drives the STDP window is unchanged.
- The fast eligibility trace `e_fast` is unchanged.
- The slow eligibility trace `e_slow` is computed with the same pairing but a slower decay.
- The wave does **not** modulate input spikes, membrane potential, synaptic current, or reward magnitude.
- The only coupling is the multiplicative gate `max(0, cos(phi_reward))` applied to `e_slow` in `apply_reward`.

The phase-scrambled control destroys the coupling by making `phi_0` random per trial.  Because the eligibility trace is reset each trial, there is no trial-averaged phase relationship to exploit.

### 6.2 Shakedown check: wave does not disturb STDP trace computation

Before any claim is made, the implementation must pass a shakedown test that verifies the wave class does not alter the underlying STDP traces relative to the base class.

Procedure:
1. Create two identical networks synapse instances with the same seed and weights.
2. Attach `RSTDPPlasticity` to one and `WaveGatedPlasticity` to the other.
3. Feed both the same synthetic spike patterns for at least 100 timesteps, including pre-before-post and post-before-pre cases.
4. Compare the fast trace `e_fast` (or the inherited `eligibility` buffer) of the wave class against the `eligibility` buffer of the base class.

Pass criterion:
- `max_abs_diff < 1e-5`.
- The wave class's slow trace `e_slow` must be a pure low-pass of the same STDP signal with `tau = 500 ms`; i.e. it must equal the base-class eligibility trace computed with `tau_elig = 500 ms` on the same spike pattern.

If the shakedown fails, the wave implementation is touching the base STDP computation and the experiment is invalid.

## 7. Experiment arms

| Arm | Name | Wave? | tau_elig | Notes |
|-----|------|-------|----------|-------|
| 1 | wave-coherent | Yes, `phi_0 = 0` | 1000 ms (base) + 500 ms slow trace | Main hypothesis |
| 2 | phase-scrambled | Yes, `phi_0 ~ U(0, 2pi)` per trial | 1000 ms (base) + 500 ms slow trace | Primary negative control |
| 3 | eligibility-timescale-only | No | 1000 ms | Confound control |
| 4 | frozen | No | N/A | Plasticity disabled; same seeds |
| 5 (optional) | phase-uniform gain-only | Yes, `phi_0 = pi/2` | 1000 ms (base) + 500 ms slow trace | Optional secondary discriminator |

Seeds: **3 seeds per arm per task**, identical seed set across arms within a task to control seed-luck variance.

## 8. Predeclared three-way criteria

All comparisons are on the primary learning metric of each task (e.g. accuracy, rate-difference score), averaged over the seed set.  Let `A` = wave-coherent, `B` = phase-scrambled, `C` = eligibility-timescale-only, `D` = frozen.  In the mandatory four-arm configuration, the criteria are:

### 8.1 PASS — wave gating does real work

- `A > C` on ≥2/3 seeds for at least one task.
- `C > D` on ≥2/3 seeds for the same task, with margin `C - D >= 0.05` over frozen.
- Reward-weight correlation `r(R, w)` (from Track 2 instrumentation) > 0.1 for arm A on the winning task.

A PASS licenses only: **"wave-gated eligibility rescues learning under delayed reward on this benchmark"** (see §9 for claim boundaries).

### 8.2 NULL — longer eligibility is the whole story

- `A ≈ C` (difference within ±0.05 or within one standard error across seeds) on ≥2/3 seeds for at least one task.
- `C > D` on ≥2/3 seeds for the same task, with margin `C - D >= 0.05` over frozen.

A NULL is a **real, reportable result**: the biological eligibility time constant is sufficient; the wave gate adds no meaningful benefit.  It redirects the next stage to timescale policy, not wave mechanism.

### 8.3 STOP — neither mechanism rescues the benchmark

- Neither `A` nor `C` consistently beats `D` on any task (i.e. neither satisfies the PASS margin on ≥2/3 seeds).
- OR any arm produces NaN values.
- OR any synaptic weight exceeds `|w| > 5.0` at any point.

A STOP is also a **real, reportable result**: the action-to-reward delay in this benchmark cannot be rescued by eligibility timescale alone or by wave-gated plasticity within this network architecture.  The next step is deeper diagnosis (reward shaping, network capacity, BTSP-like plateau, per-cell eligibility).

### 8.4 Phase-scrambled guardrail

Regardless of PASS/NULL/STOP, the result is invalid unless `A > B` on ≥2/3 seeds for at least the task that produced the PASS/NULL.  If the phase-scrambled arm performs equal to or better than the coherent arm, the effect is not phase-specific and the verdict collapses to STOP.

## 9. Claim boundaries

A PASS in Track 3 licenses **only** the following claim:

> "Wave-gated eligibility rescues learning under delayed reward on this benchmark."

A PASS does **not** license:
- Any claim about biological theta.
- Any claim about generality beyond the three tasks tested.
- Any claim about frequency optimality (8 Hz was chosen; it was not optimized).
- Any claim about the mechanism in other network architectures or tasks.

A NULL or STOP is an equally valid outcome and must be reported as such.

## 10. Footprint and rollback

Estimated implementation footprint:

- `snn/wave_plasticity.py`: ~250 lines.
  - `WaveGatedPlasticity(RSTDPPlasticity)`.
  - `PhaseOscillator` helper.
  - Instrumentation hooks for gate-zero fraction and phase metrics.
- `run_wave_replicate.py`: ~300 lines.
  - Four-arm runner, seed management, optional fifth arm, metric collation.
- Tests: ~50 lines (shakedown check plus degenerate-gate instrumentation).
- **Total: ~550 lines of new code.**

Rollback: `git rm snn/wave_plasticity.py run_wave_replicate.py` and remove the `docs/TRACK3_SPEC_FROZEN.md` reference from any downstream build card.  No existing files are modified.

## 11. Open questions and next-card decisions

The following items are intentionally left for the implementation card:

1. Exact integration of the `PhaseOscillator` with the trial loop in `run_wave_replicate.py`.
2. Whether to expose the optional fifth arm as a CLI flag or a separate config entry.
3. Whether the delayed-reward stress variant in Phase 2 uses a fixed delay or a sampled delay distribution.
4. Exact filename and class name for the runner.

These are implementation details and do not affect the frozen specification above.
