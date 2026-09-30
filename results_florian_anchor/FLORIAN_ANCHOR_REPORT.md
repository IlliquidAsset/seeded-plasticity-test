# Florian 2007 XOR anchor

Generated: 2026-09-30T14:32:31.719204+00:00
Baseline commit before anchor: `95210de17dc6bff7f71bb3e6ecc149349754a316`
Command: `python run_florian_anchor.py --seeds 20 --epochs 200`

The old Track 1 NULL was a result for our old protocol, not for Florian's protocol. This run uses online reward on each output spike, one output neuron, continuous state, the paper's network sizes, input codes, bounds, and learning rates.

| Task | Rule | This run | 95% Wilson interval | Paper (1000 runs) |
|---|---|---:|---:|---:|
| rate | MSTDP | 20/20 (100.0%) | 83.9-100.0% | 99.1% |
| rate | MSTDPET | 19/20 (95.0%) | 76.4-99.1% | 98.2% |
| temporal | MSTDP | 20/20 (100.0%) | 83.9-100.0% | 89.7% |
| temporal | MSTDPET | 20/20 (100.0%) | 83.9-100.0% | 99.5% |

Twenty seeds are enough to catch a gross failure, but not enough to distinguish 98% from 100%. The table must therefore be read as a finite replication estimate, not proof that the rates are identical.

Remaining implementation differences were frozen before execution in `docs/FLORIAN_DIFFERENCE_TABLE.md`.
