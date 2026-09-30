# What the single-synapse ladder showed

![Reward delay versus credited amount](rung0_delay_vs_credit.png)

| Step | Result | One-sentence answer |
|---|---:|---|
| One fading record | PASS | A recorded event does not suddenly disappear: after delay `d`, the amount left is exactly `exp(-d/tau)`; with the stated 5 Hz background model it falls below background RMS at about 178 ms (`tau=25`), 2.82 s (`tau=500`), and 5.28 s (`tau=1000`). |
| One synapse | PASS | Rewarding every output spike makes the synapse run to its maximum, while subtracting the expected reward stops that runaway; punishing unwanted spikes makes the synapse weaker and the neuron fires less. |
| Two synapses | PASS | With no background pairings, only the synapse that helped cause the rewarded spike grew; by the declared 10% median-leakage rule, credit first broke at 100 ms with 5 Hz background and at 200 ms even with 1 Hz background. |
| Tiny wiring motifs | PASS | Reward changed eligible links in a causal chain and the paired branch in a parallel circuit, but it never changed an unconnected link whose eligibility was zero. |

For every step, “learning” required both of these to pass across 10 seeded frozen-copy comparisons:

1. The weight change had the predicted sign and size (`reward x eligibility`).
2. Firing moved in the rewarded direction while an identical frozen copy did not.

A subtle but important limit: positive reward for silence alone cannot change pair-based R-STDP, because no output spike means no postsynaptic tag. Teaching silence therefore requires punishing an unwanted emitted spike; that is what the mirror test does.

## Why Track 1 did not answer the Florian question

The old Track 1 test waited until the end of a 500 ms trial to reward once, so with a 25 ms memory only about the last 5% of the trial was credited. It also used a different input code, two outputs instead of one, reset the traces every trial, filtered synaptic currents, and used different bounds and learning scales. Its NULL result means our old protocol failed; it does not mean Florian's rule failed.

## Florian 2007 anchor

After the four ladder steps passed, the anchor was rerun with reward on every output spike, one output neuron, continuous state, and the paper's network sizes, input codes, bounds, and learning rates.

| Task | Rule | This run (20 seeds) | Paper (1000 runs) |
|---|---|---:|---:|
| Rate-coded XOR | MSTDP | 20/20 (100%) | 99.1% |
| Rate-coded XOR | MSTDP with eligibility trace | 19/20 (95%) | 98.2% |
| Temporal XOR | MSTDP | 20/20 (100%) | 89.7% |
| Temporal XOR | MSTDP with eligibility trace | 20/20 (100%) | 99.5% |

This reproduces the published behavior at the resolution of a 20-seed check. It does not prove the exact percentages are identical: 20 seeds give wide confidence intervals, and the original source ran 1000 experiments. The one failure was rate-coded eligibility-trace seed 15: its `{0,1}` output remained silent, so it failed the paper's two-comparison gate.

Full machine-readable results are in `summary.json` here and `../results_florian_anchor/summary.json`. Remaining source differences were declared before execution in `../docs/FLORIAN_DIFFERENCE_TABLE.md`.
