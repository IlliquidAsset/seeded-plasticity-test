# Stage 2 r3, plain-language summary

**Result: Stage 2 r3 FAILS. The network learned nothing usable on A or on B.**

Verdict (spec §10, as scored): `INVALID_OR_MECHANISM_FAIL` (failed invariants: `8_primary_sign_of_life_B`,
`6.5_sfa_off_sign_of_life_B`, `6.3` and `6.4` B differentials below 0.10) plus `NO_A_COMPETENCE`.
Gates: G0 FAIL, G1 FAIL, G2 FAIL, G3 PASS (numerically only, see below), G4 PASS.
Finding taxonomy (§1): the "online adaptation with forgetting" label does **not** apply, because B learning (Gate 2) failed.
No continual-learning claim of any kind is supported.

## What was run

- The plan was frozen at `def4b36` (spec hash `695cfaa4…31cd`, unchanged). Code `14522b5` was pushed before anything ran.
- Fixture qualification: frozen-from-start arm only, seeds 1000–1019, full length. **PASS.** All four means were about 0.500,
  and every 95% interval contained 0.50. That record was committed and pushed (`6b23716`) before any experimental seed ran.
- Experiment: 5 arms × seeds 0–19, run once, 100/100 runs OK. Wall time 3,372 s (0.94 h) on 5 workers, peak RSS 1.15 GiB,
  0 interruptions, 0 retries. Qualification separately: 639 s, 5 workers, 1.08 GiB, 0 interruptions, 0 retries.

## What happened (observed)

- **Held-out accuracy is chance everywhere.** Primary A before the shift was 0.5006 (gate needs ≥ 0.70). B after the shift was
  0.5003. The other arms were the same, 0.498 to 0.502.
- **The new readout did make learning signals happen during A.** Every primary seed had at least 1,170 positive and
  1,185 negative reward events. The reward/weight-change correlation had a median of 0.74. A-phase sign-of-life PASSES,
  which r2 never achieved.
- **The network still drifted into silence.** Learning pushed both outputs toward being quiet together. With frozen weights
  the outputs are both silent about 71% of the time. After A training the median rose to 98.5% at evaluation, and 9/20 seeds
  never once fired a single output. A silent network is scored by the coin flip, so it lands at 50%.
- **B phase lost even the learning signal.** In B the outputs were silent about 97% of the time, 10/20 seeds had fewer than
  100 events of a sign, and one seed had none. B sign-of-life therefore FAILS. That is the "mechanism" failure in the verdict.
- **Gate 3 "PASS" is not retention.** A was never learned (A_learned = 0.0006), so "retaining" it is arithmetic on
  noise. The spec scores Gate 3 literally and we report it literally. It carries no meaning here.

## Interpretation (inferred, for review)

The spec named this failure mode in advance as residual risk (i): a state where both outputs are silent earns
no reward, so nothing pushes the network out of it. r3 removed r2's unfair credit for silence, but did not stop the
learning rule from drifting there. Two attempts under the same frozen gates have now failed in the same direction:
the output goes quiet. Per the card, this is reported as-is. No fix-and-rerun, tuning, or extra arm was done.

## Next gate

Nora reviews this package independently. Any r4 change to the mechanism is a new spec and a new review, for Amanda
and Kendrick to decide. It is not a rerun of r3.
