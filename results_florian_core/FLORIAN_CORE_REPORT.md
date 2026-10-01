# Stage 1 gate: Florian 2007 on the repository core (`snn/core.py`)

Generated: 2026-10-01T04:23:02.476696+00:00
Code commit: `ee442bc2c0e4e776dbda9c96754e3f00f9c97e4e` (core/bench files dirty: False)
Command: `/Users/kendrick/.hermes/hermes-agent/venv/bin/python3.11 run_florian_core.py --seeds 20 --epochs 200 --workers 5 --gate-commit ee442bc2c0e4e776dbda9c96754e3f00f9c97e4e --g0 'PASS: pytest 160 passed at ee442bc; core==anchor spike-for-spike (4 cells, 2 epochs, max dW 2.7e-15 mV); legacy end_of_trial bit-identical to 490cb1c'` on Mac Mini
Argv (JSON, exact): `["/Users/kendrick/.hermes/hermes-agent/venv/bin/python3.11", "run_florian_core.py", "--seeds", "20", "--epochs", "200", "--workers", "5", "--gate-commit", "ee442bc2c0e4e776dbda9c96754e3f00f9c97e4e", "--g0", "PASS: pytest 160 passed at ee442bc; core==anchor spike-for-spike (4 cells, 2 epochs, max dW 2.7e-15 mV); legacy end_of_trial bit-identical to 490cb1c"]`
Provenance repair: metadata-only repair after Nora review of 2c7cd6b; no science rerun; runs/groups unchanged. Exact argv recovered verbatim from Hermes process record proc_4affc9917d7f (exit 0).
Engine: snn/core.py PureSNN.online_step, reward_mode=per_spike_next_step (snn/florian_bench.py); comparison anchor ladder/florian.py
Predeclared gate: `docs/FLORIAN_CORE_GATE.md` (committed at `ee442bc2c0e4e776dbda9c96754e3f00f9c97e4e` before this run)

## Verdict: CORE PASS

| Task | Rule | Core trained | Wilson 95% | Paper | Anchor (committed) | Fisher p | Core frozen | Anchor frozen | G2 | Median r(reward, signed dW_out) | Seeds r>0.1 | Margin improved | G1 | G3a | G3b | Group |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| rate | MSTDP | 20/20 | 83.9-100.0% | 99.1% | 20/20 | 1.000 | 0/20 | 0/20 | PASS | 0.357 | 20/20 | 20/20 | PASS | PASS | PASS | PASS |
| rate | MSTDPET | 19/20 | 76.4-99.1% | 98.2% | 19/20 | 1.000 | 0/20 | 0/20 | PASS | 0.503 | 20/20 | 20/20 | PASS | PASS | PASS | PASS |
| temporal | MSTDP | 20/20 | 83.9-100.0% | 89.7% | 20/20 | 1.000 | 16/20 | 16/20 | PASS | 0.893 | 20/20 | 17/20 | PASS | PASS | PASS | PASS |
| temporal | MSTDPET | 20/20 | 83.9-100.0% | 99.5% | 20/20 | 1.000 | 16/20 | 16/20 | PASS | 0.740 | 20/20 | 20/20 | PASS | PASS | PASS | PASS |

G0 (engineering precondition, tests): PASS: pytest 160 passed at ee442bc; core==anchor spike-for-spike (4 cells, 2 epochs, max dW 2.7e-15 mV); legacy end_of_trial bit-identical to 490cb1c

## Seed-for-seed against the independent NumPy anchor

| Task | Rule | Trained 400 s output trains identical | Frozen identical | Success agrees w/ committed anchor | Margin agrees | Anchor re-run reproduces committed | Max final dW (mV) |
|---|---|---:|---:|---:|---:|---:|---:|
| rate | MSTDP | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0.00e+00 |
| rate | MSTDPET | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 7.11e-15 |
| temporal | MSTDP | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0.00e+00 |
| temporal | MSTDPET | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 3.29e-14 |

## Frozen post-training evaluation (reported, not gated)

| Task | Rule | Retest trained | Retest frozen | Generalization trained | Generalization frozen |
|---|---|---:|---:|---:|---:|
| rate | MSTDP | 87/200 | 4/200 | n/a | n/a |
| rate | MSTDPET | 190/200 | 4/200 | n/a | n/a |
| temporal | MSTDP | 195/200 | 167/200 | 192/200 | 179/200 |
| temporal | MSTDPET | 190/200 | 167/200 | 189/200 | 179/200 |

## Claim boundary

Supported: the repository substrate (`LIFNeuron`, `Synapse`, `RSTDPPlasticity`, `PureSNN.online_step`) learns Florian's rate-coded and temporal XOR with per-output-spike next-step reward. Results are compatible with the paper's published rates and with the independent anchor, under the predeclared gate.

Not supported: equality with the paper's 1000-run percentages (20 seeds cannot resolve 98% vs 100%); any claim about the legacy end-of-trial mode, which is preserved unchanged and was not re-run; continual learning (Stage 2); any claim about AIB developmental runs.

No oracle: the target enters only as the sign of the per-output-spike reward. Frozen twins receive no weight change.
