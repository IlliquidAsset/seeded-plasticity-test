# Adaptive threshold and spike-frequency adaptation reference

Purpose: source material for Stage 2 review only. This note does not implement or approve the Stage 2 build.

## Bottom line

SpikingBrain's "adaptive-threshold spiking neuron" is not spike-frequency adaptation in the usual ALIF sense. Its threshold is recomputed from the current activation vector or activation group. It has no spike-history state, no per-spike threshold increment, and no exponential return to baseline. Therefore, its answer to "threshold rise per spike" is zero, and its answer to "adaptation decay time constant" is none.[1][2][3]

For the bench's intended intrinsic short-term memory, Bellec et al.'s ALIF formulation is the correct reference. It adds one per-neuron state variable, raises the firing threshold after each emitted spike, and lets that elevation decay exponentially over a task-relevant timescale.[5][6]

Recommended formulation for the bench:

```text
rho = exp(-dt / tau_a)
theta_t = v_thresh + beta_a * a_t
v_pre = v_rest + exp(-dt / tau_m) * (v_t - v_rest) + I_syn_t
z_t = 1[v_pre >= theta_t]              # and not refractory, if used
v_{t+1} = v_reset if z_t else v_pre    # preserve the bench's current hard reset
a_{t+1} = rho * a_t + z_t
```

This is threshold adaptation, not an adaptation current. Each spike raises the next effective threshold by `beta_a`; `tau_a` controls the exponential decay back to the baseline `v_thresh`.

## 1. What SpikingBrain actually implements

### SpikingBrain 1.0 paper

The paper defines the threshold from the current activation tensor:

```text
V_th(x) = mean(abs(x)) / k
s_INT = round(x / V_th(x))
```

Its virtual-time IF interpretation is:

```text
v_{t+1} = v_t - V_th * s_t + x_{t+1}
s_t = 1 if v_t >= V_th(x), else 0
```

The authors explicitly remove membrane decay and use soft reset. During optimization they collapse time and directly compute integer spike counts. The threshold follows activation magnitude, not prior spikes.[1]

Consequences:

- Threshold rise per emitted spike: `0`.
- Decay time constant back to baseline: none.
- Baseline threshold: none in the SFA sense. The threshold is recomputed from each activation vector.
- State carried through time: none for threshold adaptation.
- Mechanism type: per-token or per-group activation quantization with spike-count semantics, not intrinsic SFA memory.

### SpikingBrain 1.0 repository

The shipped function is `W8ASpike/quant_linear.py::dynamic_spikes`:

```python
vth = x.abs().mean([-1], keepdim=True).float() / k
vth = vth.clamp(min=1e-5, max=1e4)
spikes_int = (x / vth).round()
```

`QuantLinear` ships with `dynamic_sfr=3.0`, so the default is `k=3.0`. It dequantizes immediately with `x = spikes_int * vth` before the linear operation. The optional Int2Spike path converts integer counts into virtual-time spike sequences; it does not add SFA state.[2]

Implementation details worth retaining only as quantization references:

- Reduce over the last activation dimension.
- Compute the threshold in float32.
- Clamp threshold to `[1e-5, 1e4]` to avoid division by zero and extreme scales.
- Round to an integer spike count.
- `k` is a firing-rate and precision knob. Larger `k` lowers the threshold and yields more counts; smaller `k` increases sparsity. The paper explains this tradeoff but does not report a formal selection procedure beyond the shipped default and evaluation.[1][2]

### SpikingBrain 2.0

SpikingBrain 2.0 inherits the same activation-dependent idea. It applies 1x128 group-level INT8 activation quantization, describes each group as equivalent to the adaptive-threshold mechanism, and expands each signed INT8 value into a seven-step bitwise spike sequence. It again emphasizes that explicit LIF decay is absent.[3]

The checked public repository commit does not expose a second ALIF/SFA neuron with a per-spike increment or a `tau_a`. The public tree includes deployment and quantization code, but no spike-history threshold state matching the requested SFA mechanism.[4]

Parameters explicitly shipped or stated:

- SpikingBrain 1.0 code: `k=3.0` default, threshold clamp `[1e-5, 1e4]`.[2]
- SpikingBrain 1.0 weight quantization: default group size `128` in `QuantLinear`; spike coding is applied to projection-layer activations.[1][2]
- SpikingBrain 2.0: activation groups `1x128`, weight blocks `128x128`, signed INT8 represented as seven virtual spike steps.[3]
- Neither paper or checked repository supplies a per-spike adaptation amplitude or adaptation decay time constant, because this mechanism is not SFA.

## 2. Bellec et al. ALIF formulation

Bellec et al. use an adaptive LIF neuron whose second hidden state is a threshold-adaptation trace. In discrete time:

```text
A_j^t = v_th + beta * a_j^t
z_j^t = H((v_j^t - A_j^t) / v_th)
a_j^{t+1} = rho * a_j^t + z_j^t
rho = exp(-dt / tau_a)
```

Therefore:

- Each emitted spike increments `a` by `1`.
- Each spike raises the effective threshold by exactly `beta` in threshold-voltage units.
- The elevation decays exponentially with time constant `tau_a`.
- The adaptation is a threshold term. It is not subtracted as an adaptation current from membrane voltage.[5]

The official e-prop code implements the same structure:

```python
adaptive_thr = self.thr + b * self.beta
new_b = self.decay_b * b + old_z
self.decay_b = exp(-dt / tau_adaptation)
```

It initializes membrane, adaptation, spikes, and refractory counters to zero. Its membrane reset is subtractive (`new_v = decay*v + input - spike*thr`), while the adaptation state is not reset by a spike. Refractory gating suppresses emitted spikes, and only emitted spikes should increment adaptation.[6]

### Bellec parameter values

The publication says `tau_a` should be on the timescale of the working memory required by the task. It reports adaptation lasting hundreds of milliseconds to seconds, with `rho` typically around `0.995` to `0.9995` at `dt=1 ms`, and `beta` typically on the order of `0.07`. The working-memory store-recall experiment used `tau_a = 1.2 s`; the stored bit was usually requested about `1.2 s` later.[5]

The official repository contains several task-specific settings rather than one universal pair:

- Numerical verification: `v_th=0.62`, `beta=0.07`, `tau_a=500 ms`, `tau_m=20 ms`, `dt=1 ms`, refractory period `2` steps.
- TIMIT framewise defaults: user-facing `beta=1.8`, `tau_a=200 ms`, `tau_m=20 ms`, refractory period `2` steps. The code rescales threshold and beta for its voltage update convention before constructing the ALIF cell. With those defaults, the value passed to adaptive neurons is approximately `beta=0.1841`, not `1.8`.
- The repository's ALIF cell default is `tau_adaptation=200`, `beta=0.16`, `thr=0.615`, `dt=1`, refractory period `1` step.

These are task and normalization dependent. The stable lesson is the equation and timescale selection, not copying a raw beta from a differently scaled voltage model.[5][6]

## 3. Mapping onto the pure-SNN bench

Current bench path: `snn/core.py` at commit `8a601b07b2fd3112ad9da81612e4487eea9c3d70`.[7]

The current `LIFNeuron.forward` has only membrane state:

```text
v = v_rest + beta_m * (v - v_rest) + I_syn
fired = v >= v_thresh
v = v_reset where fired
```

The smallest ALIF lift changes the neuron call from `(I_syn, v) -> (spikes, v)` to `(I_syn, v, a) -> (spikes, v, a)`. In `PureSNN.forward`, add one adaptation-state slot per layer beside `v`, initialize it to zero at the start of each trial, and carry it through every timestep.[7]

Suggested update order for this codebase:

```text
if v is None: v = v_rest
if a is None: a = 0

rho = exp(-dt / tau_a)
theta = v_thresh + beta_a * a
v_pre = v_rest + beta_m * (v - v_rest) + I_syn
fired = (v_pre >= theta) and not_refractory
v_next = where(fired, v_reset, v_pre)
a_next = rho * a + fired
```

This preserves the bench's existing hard reset instead of importing Bellec's subtractive reset. Reset choice and threshold adaptation are separable. Do not reset `a` when the membrane resets. Reset `a` only at the same trial boundary where the bench currently initializes `v` to `None`.

Parameter recommendation for the first Stage 2 bench experiment, not a final design verdict:

- Use Bellec's form.
- Treat `beta_a` as a fraction of the bench's baseline threshold. A source-faithful starting point is `beta_a = 0.07 * v_thresh`.
- Choose `tau_a` from the bench's memory horizon. Current calibrated tasks use 50 one-millisecond steps. A practical reference sweep is `tau_a in {50, 100, 200}` steps. Those retain about 37%, 61%, and 78% of one spike's threshold effect after 50 steps, respectively.
- Start adaptation state at zero.
- Do not clamp `a` in the reference implementation. If a safety cap is later added, log cap hits because silent saturation can hide a firing-rate failure.
- Keep the output/readout population non-adaptive in the first comparison unless the Stage 2 roadmap explicitly calls for adaptive outputs. The working-memory mechanism belongs most naturally in hidden recurrent or temporal-processing populations.

## 4. Overlap with Abe's existing machinery

This is not a new mathematical primitive for the wider AIB stack.

- `/Volumes/Studio Backup 2/dev/AIB/aib/stress.py` already implements the same event-plus-exponential-decay pattern: add `spike_amount` on an event, clamp to a maximum, then multiply by `exp(-1 / decay_tau)` each tick.
- `/Volumes/Studio Backup 2/dev/AIB/aib/chemicals.py` already has reusable baseline, target, exponential decay, reset, and clamping patterns in `TonicModulator`. It also stores baseline layer thresholds and applies threshold modulation through `SerotoninChemical` and `ChemicalManager.apply_to_network`.
- `/Volumes/Studio Backup 2/dev/AIB/aib/engine_eventdriven.py` already tracks per-neuron refractory counters, suppresses integration while refractory, applies the threshold test, resets membrane voltage, and reloads the refractory counter after a spike.
- The older `/Volumes/Studio Backup 2/dev/AIB/aib/signal_stubs.py::inhibition` function is only a neutral stub. The mature reusable pieces are the current refractory counter, threshold baseline handling, and stress/chemical leaky-state patterns, not that stub itself.

The repurposing is straightforward conceptually: move the leaky event trace from a global stress or chemical scalar to a per-neuron vector, and use it to raise each neuron's own threshold. That turns an existing control pattern into intrinsic short-term memory.

## 5. Recommendation

Use Bellec ALIF for Stage 2. It matches the requested SFA mechanism exactly, adds only one state tensor per layer, and maps directly into the bench's timestep loop. It also has published evidence connecting slow threshold adaptation to working-memory performance.[5][6]

Do not lift SpikingBrain's mechanism as SFA. Keep it as a separate reference for adaptive activation quantization and spike-count coding. It is useful if the roadmap later includes sparse integer encoding, but it will not provide spike-history memory because it has no spike-history state or decay constant.[1][2][3]

## Source paths inspected

- SpikingBrain-7B paper source: arXiv `2509.05276`, `main.tex`, adaptive-threshold section around lines 312-383.
- SpikingBrain-7B code: commit `ef99987167cf7386ab3348312c8c7aa00a6696ee`, `W8ASpike/quant_linear.py`, lines 11-49; `W8ASpike/Int2Spike/neuron.py` for spike-count expansion.
- SpikingBrain2.0 paper source: arXiv `2604.22575`, `main.tex`, quantization section around lines 426-456.
- SpikingBrain2.0 repository: commit `02561132682c2a66925bb0df9f6a1d3f9e816723`.
- Bellec et al. paper: Nature Communications 11, 3625 (2020), DOI `10.1038/s41467-020-17236-y`; arXiv `1901.09049`, Methods ALIF equations around lines 796-812.
- Official e-prop code: commit `efd02e6879c01cda3fa9a7838e8e2fd08163c16e`, `Figure_2_TIMIT/alif_eligibility_propagation.py`, especially lines 60-98 and 171-220.
- Bench: commit `8a601b07b2fd3112ad9da81612e4487eea9c3d70`, `snn/core.py`, especially lines 15-48 and 219-275.

## Sources

[1] https://arxiv.org/abs/2509.05276
[2] https://github.com/BICLab/SpikingBrain-7B/blob/ef99987167cf7386ab3348312c8c7aa00a6696ee/W8ASpike/quant_linear.py
[3] https://arxiv.org/abs/2604.22575
[4] https://github.com/BICLab/SpikingBrain2.0/tree/02561132682c2a66925bb0df9f6a1d3f9e816723
[5] https://doi.org/10.1038/s41467-020-17236-y
[6] https://github.com/IGITUGraz/eligibility_propagation/blob/efd02e6879c01cda3fa9a7838e8e2fd08163c16e/Figure_2_TIMIT/alif_eligibility_propagation.py
[7] https://github.com/IlliquidAsset/seeded-plasticity-test/blob/8a601b07b2fd3112ad9da81612e4487eea9c3d70/snn/core.py
