# R-STDP learning ladder: rungs 0-3

Generated: 2026-09-30T14:37:27.993469+00:00
Baseline commit: `95210de17dc6bff7f71bb3e6ecc149349754a316`
Command: `python run_ladder.py`

## Result table

| Rung | Mechanism | Outcome | Gate |
|---|---:|---:|---:|
| 0 | PASS | PASS | PASS |
| 1 | PASS | PASS | PASS |
| 2 | PASS | PASS | PASS |
| 3 | PASS | PASS | PASS |

## In plain language

- Rung 0: One recorded event fades exponentially. It never abruptly stops; a delayed reward gets `exp(-delay/tau)` of the original tag.
- Rung 1: Rewarding every output spike drives the one synapse to its cap. Subtracting expected reward makes the drive die away and prevents saturation in this test.
- Rung 2: With two synapses and no background pairings, only the synapse paired with the rewarded output changes. Background pairings can contaminate credit; the JSON maps that contamination by delay and firing rate.
- Rung 3: Reward changes every eligible link in a causal series, only the paired link in a parallel motif, and never an unconnected link with zero eligibility.

Learning here means both tests passed: (1) every weight change had the predicted sign and size, and (2) firing moved in the rewarded direction versus an identical frozen copy.

## Important boundary

A positive reward delivered only when the output is silent cannot change a pair-based R-STDP synapse, because silence creates no postsynaptic tag. To teach silence, this rung punishes an unwanted emitted spike (`reward=-1`); that shrinks the weight and reduces firing.

The red line in the plot is not a universal noise floor. It is the measured RMS trace from independent 5 Hz pre- and postsynaptic background spikes with the repository's default STDP amplitudes. Exact values and crossings are in `summary.json`.

## Track 1 correction

Track 1's NULL result applies to our protocol, not to Florian's published protocol. Our run rewarded once after a 500 ms trial, used two output neurons and an argmax, redrew its input trains, reset traces each trial, filtered synaptic currents, and used different bounds and STDP scales. Florian rewarded each output spike on the following 1 ms step, used one output, fixed the temporal code for the whole experiment, and ran continuously.

See `docs/FLORIAN_DIFFERENCE_TABLE.md` for the complete pre-run difference table.
