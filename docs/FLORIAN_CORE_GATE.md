# Stage 1 gate: Florian per-spike reward on the repository core (`snn/core.py`)

Task: t_2b6665e2. This document is PREDECLARED: it is committed before any 20-seed core-substrate run. The results commit must come after this commit, and the gate below is not edited after results are seen. A deviation is allowed only as a dated addendum with its reason, and it cannot change the verdict of a run that already happened.

Base: `v0.3.0-calibrated` @ 490cb1c. The independent anchor is `ladder/florian.py` (NumPy), with committed results in `results_florian_anchor/summary.json` (20/20, 19/20, 20/20, 20/20 trained; 0/20, 0/20, 16/20, 16/20 frozen).

## What is being tested

The question is whether the repository's own substrate learns Florian's 2007 XOR tasks when it receives Florian's reward schedule. The substrate is `LIFNeuron`, `Synapse`, `RSTDPPlasticity`, and `PureSNN`, driven by a new online step. The schedule is +1 (target 1) or -1 (target 0) for each output spike, delivered on the following 1 ms step, online, batch 1, with state continuing across patterns. Network sizes, encodings, bounds, gammas, and time constants match the faithful anchor exactly (`docs/FLORIAN_DIFFERENCE_TABLE.md`).

The engine is `snn/core.py` via `snn/florian_bench.py`. Neither module imports `ladder/`. `run_florian_core.py` imports `ladder.florian` only to record the independent anchor's spike trains for the seed-for-seed comparison. The core path never calls it.

The legacy end-of-trial scalar reward (`PureSNN.forward` + `RSTDPPlasticity.apply_reward` after the trial) stays available, unchanged, and labeled `end_of_trial`.

## Mapping onto core primitives (decided before running)

| Paper quantity | Core realization |
|---|---|
| IF neuron, rest/reset -70 mV, threshold -54 mV, tau_m 20 ms | `LIFNeuron(v_rest=-70, v_reset=-70, v_thresh=-54, tau_m=20)`. Units are mV, unnormalized |
| Weight = direct voltage jump, no synaptic filter | `Synapse(tau_syn=0)`, so `beta_syn = 0` and the current is `W @ pre` for one step only |
| One discrete update per layer (eq. 4.1) | `PureSNN.online_step`: every layer integrates the previous step's presynaptic spikes |
| MSTDPET z(t+dt) = beta z + zeta/tau_z, A+ = 1, A- = -1 | `RSTDPPlasticity(a_plus=1/tau_z, a_minus=1/tau_z, tau_elig=25, credit="eligibility")` |
| MSTDP dw = gamma r zeta(t) | `RSTDPPlasticity(a_plus=1, a_minus=1, credit="pairing")` |
| gamma (mV) | `lr` = 0.1 / 0.625 (rate) and 0.01 / 0.25 (temporal) |
| Sign-specific bounds | `RSTDPPlasticity(w_min=..., w_max=...)` with per-synapse tensors |
| Initial weights, input spikes, pattern order | Same NumPy `default_rng(seed)` draw order as the anchor, so seeds pair one-to-one |
| Precision | float64 (`net.double()`), as in the anchor |

## Gate

G0, the engineering precondition (checked by tests before the run):
- (a) The full test suite passes, including the legacy-mode tests.
- (b) Over a 2-epoch parity run (4,000 steps), the core per-spike mode reproduces the anchor's output and hidden spike trains exactly for all four task/rule cells.
- (c) Final weights agree to within 1e-9 mV.

If G0 fails, no 20-seed result is interpreted. The divergence is diagnosed first.

G0 status at predeclaration: PASS (2026-09-30, Mac Mini, `pytest -q tests`, 160 passed).
- `tests/test_core_florian.py::test_core_matches_independent_anchor_exactly` covers all four cells at seed 3 over 2 epochs. Output trains are identical, and the max final-weight difference is 2.7e-15 mV.
- `test_end_of_trial_mode_bit_identical_to_pre_change_core` shows the legacy mode is bit-identical to `snn/core.py` at 490cb1c.

A pre-gate smoke run (2 seeds, 4 epochs, gate commit `SMOKE`) checked only the plumbing. It is not evidence and its output was discarded.

Per task/rule group (rate MSTDP, rate MSTDPET, temporal MSTDP, temporal MSTDPET), each with 20 seeds (0-19) and 200 epochs:

- G1, success rate. Core trained successes k/20 use the paper's final-epoch gate: rate11 < rate01 and rate11 < rate10. G1 passes only if both hold:
  - (a) the 95% Wilson interval of k/20 contains the published success rate (99.1, 98.2, 89.7, 99.5%);
  - (b) the two-sided Fisher exact test against the anchor's committed count for the same group gives p >= 0.05.
- G2, frozen controls. Every seed also runs as a frozen twin: identical RNG stream, inputs, order, and initial weights, with no reward ever applied. G2 passes only if all of these hold:
  - (a) Every frozen twin's weights are unchanged (total absolute change exactly 0).
  - (b) Rate task: frozen success is at most 2/20. A network that cannot learn should essentially never pass a gate that needs rate11 below both rate01 and rate10, so this is chance level.
  - (c) Temporal task: frozen success gives Fisher p >= 0.05 against the anchor's frozen 16/20. The temporal gate can be passed without learning (Nora, t_a0a9775c), so for this task "chance" means the no-learning baseline. Learning has to be shown by G3b instead.
- G3, sign of life.
  - (a) Mechanism. For every delivered reward event during training, take the reward r and the signed change it produced in the summed hidden-to-output weight. Compute the per-seed Pearson correlation between them. This differs from the Track 2 metric, which correlated reward with the absolute weight change. G3a passes only if the median across 20 seeds is >= 0.3 and at least 18/20 seeds have a correlation > 0.1.
  - (b) Outcome. The paired margin movement is (min(rate01, rate10) - rate11), trained minus frozen twin, on the last epoch. It must be > 0 in at least 15/20 seeds (one-sided sign test p ~= 0.021).

CORE PASS requires G0 and every group passing G1, G2, and G3 (a and b).

## Seed-for-seed comparison (reported; exact parity is G0)

For every trained and frozen seed, the anchor is re-run from the same seed and its spike trains are recorded. The report shows:
- whether the full 400 s output spike train is identical;
- the first divergent step, if any;
- agreement on success;
- the margin difference;
- the maximum final-weight difference.

The re-run anchor must reproduce the committed anchor summary seed by seed (success and margin). If it does not, that is flagged as an anchor determinism defect.

## Null and stop rules

- If a group fails G1, G2, or G3, the verdict is CORE_NOT_REPRODUCED for that group.
- On failure, find the first step where core and anchor diverge (spikes, membrane, traces, or weights) and name the mechanism. Do not tune parameters toward the target, and do not re-run under this gate with changed parameters.
- Frozen post-training retest and new-symbol generalization use the anchor's protocol and evaluation seeds. They are reported but not gated here.
- Twenty seeds cannot distinguish 98% from 100%. A PASS means "compatible with the paper and with the independent anchor on our own substrate", not equality of rates.

## No-oracle statement

The target affects learning only through the sign of the per-output-spike reward. No target, expected answer, or success value enters weights or spikes by any other path. Frozen twins never receive a weight change.
