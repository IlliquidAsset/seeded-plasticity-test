# Florian 2007 XOR anchor (paired frozen control, independent NumPy engine)

Generated: 2026-09-30T16:19:34.662501+00:00
Code commit: `b1824502f417c7187d3ba958675a3d3009552782` (working tree dirty for ladder/anchor: False)
Command: `python run_florian_anchor.py --seeds 20 --epochs 200 --workers 5`

Provenance repair (not a rerun): the run that produced these results was `~/.hermes/hermes-agent/venv/bin/python3.11 run_florian_anchor.py --seeds 20 --epochs 200 --workers 5`. The first version of this report said `--workers 6` because the script wrote a hard-coded string. The line was corrected by hand, and the script now builds the command from its parsed arguments. No results changed; the worker count only sets how many processes run in parallel.
Engine: independent NumPy reference implementation of Florian 2007 (ladder/florian.py); not snn/core.py

## Claim boundary

This is an independent NumPy reproduction of Florian's equations. It is not execution of the repository core (`snn/core.py`). A match here shows the paper's protocol works when implemented as written; it says nothing yet about the repository's own network code.

No oracle: the target enters only as the sign of the per-output-spike reward. Frozen twins receive the identical RNG stream, input, order, and initial weights, with plasticity off.

## Final-epoch gate (paper criterion) vs paired frozen twin

| Task | Rule | Trained | Frozen | Paper | Margin improved / same / worse | Median margin move (Hz) | Trained pass & frozen fail |
|---|---|---:|---:|---:|---:|---:|---:|
| rate | MSTDP | 20/20 | 0/20 | 99.1% | 20/0/0 | +160.0 | 20 |
| rate | MSTDPET | 19/20 | 0/20 | 98.2% | 20/0/0 | +138.0 | 19 |
| temporal | MSTDP | 20/20 | 16/20 | 89.7% | 17/2/1 | +10.0 | 4 |
| temporal | MSTDPET | 20/20 | 16/20 | 99.5% | 20/0/0 | +23.0 | 4 |

Margin = min(rate01, rate10) - rate11 on the last training epoch; the paper gate is margin > 0. Movement = trained margin - frozen margin for the same seed.

## Frozen post-training evaluation

| Task | Rule | Retest trained | Retest frozen | New-symbol generalization trained | New-symbol generalization frozen |
|---|---|---:|---:|---:|---:|
| rate | MSTDP | 87/200 | 4/200 | n/a | n/a |
| rate | MSTDPET | 190/200 | 4/200 | n/a | n/a |
| temporal | MSTDP | 195/200 | 167/200 | 192/200 | 179/200 |
| temporal | MSTDPET | 190/200 | 167/200 | 189/200 | 179/200 |

Retest = the training code presented again, ten times, with synapses fixed (paper section 4.2). Generalization = ten newly generated pairs of 50-spike symbol trains per seed, with synapses fixed (paper section 4.3: "for any pair of random input signals similarly generated"). Each count is presentations that pass the gate.

## Equation and reward-order check

`ladder/florian_check.py` recomputes the final weights from recorded spikes and targets using explicit pair sums. It shares no update code with the anchor. It covers the 2-epoch runs for both tasks and both rules; worst error 2.22e-15 mV; all pass: True. `tests/test_florian_equations.py` shows the check fails under three deliberate mutations: flipped sign, reward delivered one step late, and reward applied before the eligibility update.

Remaining implementation differences: `docs/FLORIAN_DIFFERENCE_TABLE.md`. Twenty seeds cannot resolve 98% vs 100%. They can only detect a gross failure.
