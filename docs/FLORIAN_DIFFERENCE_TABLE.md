# Florian 2007 anchor: pre-run difference table

This table was frozen before running the new anchor. It distinguishes the old Track 1 run from the new implementation and lists remaining limits honestly.

Primary source: R. V. Florian (2007), sections 2, 3, 4.1-4.3, equations 2.7-2.8, 3.6-3.12, 4.1.

| Dimension | Florian 2007 | Old `run_track1_anchor.py` | New `run_florian_anchor.py` | Remaining difference / consequence |
|---|---|---|---|---|
| Reward timing | +1 or -1 for each output spike, delivered on following 1 ms step | One +/-1 scalar after full 500 ms trial | Per-output-spike reward, applied online after the spike and fresh eligibility update | Same discrete transition as eqs. 2.7-2.8; bookkeeping occurs in one loop iteration rather than a separate stored event |
| Batch | One pattern at a time | Batch 16 | Batch 1 | None known |
| Output | One output neuron; rate encodes 0/1 | Two-output argmax | One output neuron | None known |
| Rate network | 60-60-1 | Not run | 60-60-1 | None known |
| Temporal network | 2-20-1 | 2-20-2 | 2-20-1 | None known |
| Rate input | 30 neurons per bit; 1=40 Hz Poisson, 0=silent | N/A | Same | PRNG differs from the paper's unspecified generator |
| Temporal input | Fixed random 50-spike train for symbol 0 and another for symbol 1, generated once per experiment; both 100 Hz | 0=silent, 1=fresh random 50-spike train every sample | Two fixed 50-spike symbol trains generated once per seed and reused | PRNG differs |
| Pattern schedule | Four patterns, random order per epoch | Random batch samples | Four patterns, shuffled each epoch | PRNG differs |
| Duration | 200 epochs = 400 s | 200 trials = 100 s/sample | 200 epochs = 400 s | None known |
| State continuity | Continuous simulation | Membrane and traces reset each trial | Membrane, spike traces, and eligibility continue across patterns/epochs | None known |
| Neuron | IF: rest -70 mV, threshold -54 mV, reset -70 mV, tau 20 ms | Normalized IF with threshold 1 | Same mV parameters | Equality at threshold is treated as firing; paper says “surpassed,” an ambiguous one-point boundary |
| Synaptic action | Weight is direct voltage jump on step after presynaptic spike | Exponentially filtered current, tau 10 ms; same-step propagation | Direct voltage jump, one-step propagation | None known |
| Axonal delay | No extra delays beyond discrete update | Same-step layer traversal | One discrete update step per layer | Matches eq. 4.1 interpretation |
| STDP windows | tau+=tau-=20 ms; A+=1, A-=-1 | a+=0.02, a-=0.015 | Same tau and balanced amplitudes | None known |
| Eligibility | tau_z=25 ms; z(t+dt)=beta*z+zeta/tau_z | tau=25 but adds raw STDP delta without /tau_z | Same equation including /tau_z | Floating-point implementation differs |
| Rate gamma | MSTDP 0.1 mV; MSTDPET 0.625 mV | lr 0.005 unitless | Same paper values | None known |
| Temporal gamma | MSTDP 0.01 mV; MSTDPET 0.25 mV | lr 0.005 unitless | Same paper values | None known |
| Rate bounds | Input excitatory [0,5], input inhibitory [-5,0]; downstream [0,5] mV | All [-2,2] | Same sign-specific bounds | The paper does not explicitly say hidden neurons are inhibitory; downstream is implemented excitatory as implied by its wording |
| Temporal bounds | Input-hidden [-10,10]; hidden-output [0,10] mV | All [-2,2] | Same | None known |
| Initialization | Random within bounds | Gaussian normalized | Uniform within each bound | PRNG differs |
| Success gate | rate(1,1) lower than rate(0,1) and rate(1,0); 0,0 is zero | Read neuron 1 of argmax pair; checked only 1,1 vs 0,1 | Exact one-output gate over last epoch | Finite 20-seed estimate has wide uncertainty vs paper's 1000 runs |
| Published comparison | Rate: 99.1% MSTDP, 98.2% MSTDPET. Temporal: 89.7% MSTDP, 99.5% MSTDPET | Incorrectly called source-matched | Runs both rules, 20 seeds each per task | 20 seeds cannot tightly reproduce a 1000-run percentage; it can detect gross failure |
| Runtime implementation | Original code not supplied in paper | PyTorch repo core | Independent NumPy reference engine | This tests the paper equations, not reuse of `snn/core.py`; ladder tests separately cross-check the repository eligibility decay |

## Addendum (t_26f59ce3, after Nora review t_a0a9775c)

Added before the second run and declared before any of its results were seen:

| Item | Paper | This anchor | Consequence |
|---|---|---|---|
| Paired no-learning baseline | Not reported | Every seed also runs as a frozen twin: same RNG stream, inputs, order, and initial weights, with plasticity off | Separates "the gate passes" from "learning moved the gate". Nora's read-only probe on eb972c7 found frozen hits of 0/20 (rate) and 16/20 (temporal), so the temporal gate alone is not evidence of learning |
| Post-learning fixed-synapse retest | Sec 4.2: performance "remained constant" with reward removed and synapses fixed | Ten frozen presentations of the training code per seed, for trained and frozen twin | Tests one presentation at a time and cannot resolve seed-level variance in the claim |
| Temporal generalization | Sec 4.3: the network solves the task "for any pair of random input signals similarly generated" | Ten newly generated 50-spike symbol pairs per seed, synapses fixed, trained vs frozen twin | Previously absent; now run. The paper gives no count or protocol, so our operationalization (the gate per presentation) is our own |
| Equation / reward-order check | Eqs 2.7-2.8, 3.9-3.12 | `ladder/florian_check.py` recomputes the final weights of 2-epoch runs from recorded spikes and targets with explicit pair sums; tests show it fails under sign, late-reward, and reward-before-eligibility mutations | Verifies the implementation matches the written equations. Does not verify that our reading of the equations is the author's |
| Engine | Author's code (not published) | Independent NumPy (`ladder/florian.py`) | Claims are bounded to an independent reproduction, not `snn/core.py` |

## Predeclared interpretation

- A 20/20 result is compatible with each published rate but does not establish equality.
- A 19/20 result is also unsurprising for a 98-99.5% source rate.
- A grossly lower result, saturation/silence, or failure of the exact gate points to an implementation mismatch or a missing paper detail; it does not refute R-STDP.
- The ladder must pass rungs 0-3 before this anchor is interpreted.
