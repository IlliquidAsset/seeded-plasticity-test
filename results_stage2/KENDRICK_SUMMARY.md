# Stage 2: plain-language summary (Ethan, 2026-10-01)

## Result: Stage 2 FAILED. The network never learned rule A, so nothing could be retained.

## Correction after review (Nora, round 1): the result package is INCOMPLETE and this card's process FAILED

The spec required min/max weights and bound-hit counts at the end of phase A and again at the end of phase B, for every run. The code saved only the end-of-B values. In 91 of 100 runs the weights did not change during B, so the end-of-A values are identical to the saved ones. In 9 runs (P seeds 2, 11, 14, 18; SC seed 2; NS seeds 0, 11, 14, 18) the end-of-A values are lost. Weights were checked to stay inside their bounds at the end of A in every run. Re-running experimental seeds to recover the lost values is forbidden, so this gap is recorded as a process failure.

None of the missing values feeds any gate. The science verdict below is unchanged, and Nora recomputed it independently.

Scored exactly per spec section 10 (spec `ef85a15`, unchanged; code `d04fbd9`, pushed before any experimental seed ran):

| Gate | Result | What it means |
|---|---|---|
| 0: integrity and mechanism | FAIL | Every integrity check passed (hashes, bounds, no resets, isolated evaluations). The failures are the mechanism and control items listed below. |
| 1: competence on A | FAIL | Mean held-out A accuracy was 0.4999 against the 0.70 bar (chance is 0.50; the best possible is 0.90). |
| 2: held-out B improvement | FAIL | B accuracy moved by -0.0003 against the +0.15 bar. |
| 3: retention of A | FAIL | Retention is undefined because nothing was learned (A_learned = -0.0001). |
| 4: fixed capacity, clean protocol | PASS | 60 weights throughout; one initialization and no resets; no replay; no held-out data touched training; provenance recorded first; no deviation from the spec. |

Overall verdict: `INVALID_OR_MECHANISM_FAIL` (Gate 0), alongside `NO_A_COMPETENCE` (Gate 1).

The spec defines "online adaptation with forgetting — not continual learning" as the outcome where B learning passes but A retention fails. That outcome does not apply here: B learning did not pass, and A was never learned. This is not continual learning. It is also not online adaptation.

## What happened, in one sentence

The network learned to stop predicting "1": its output went silent, so it guesses "0" every time and scores the base rate of 50%. Because reward only arrives when the output spikes, a silent network never gets feedback again and stays stuck there.

## Evidence (recorded instrumentation, no new runs)

- Before training, the frozen network fires and predicts "1" about 16% of the time (median prediction-one fraction 0.16).
- The learning rule was clearly active during A. In all 20 seeds, reward and the output-weight change were strongly linked (median correlation 0.86), and A sign-of-life passed.
- Training pushed the output down. A-phase punishments outnumbered rewards (median 340.5 vs 227). By the end of A, the trained network predicted "1" essentially never: the prediction-one fraction was exactly 0 at A_pre in 19 of 20 seeds.
- In B, 16 of 20 seeds produced zero output spikes over all 200,000 steps, so zero reward events. Reward requires a spike (Florian's rule; ladder Rung 1 found the same thing: "reward for silence yields exactly zero update"). Silence is therefore a trap the rule cannot climb out of. B sign-of-life failed (4/20 seeds had any weight change).
- With SFA off, the network went silent the same way. SFA neither caused the collapse nor prevented it. That arm also failed its own validity check: one seed had only 33 rewarded spikes in A, below the 100 minimum.
- Frozen-from-start control: its mean B accuracy was 0.5066. The 95% interval [0.50004, 0.51342] just misses 0.50 at the low end. By the spec's own rule, that is a separate validity failure, and it means the nominal 50% chance model was not exact for this fixture. The cause is that the untrained network's spikes happen to be slightly better than chance on held-out B. This does not change the outcome: the primary arm sits at chance on every checkpoint.

## What this does and does not show

- Shows: under this frozen protocol (noisy order-2 XOR then XNOR, per-spike reward, 2-20-1 network, Stage 1 temporal parameters), the network does not learn the hidden rule. Instead it learns to fall silent.
- Does not show that R-STDP cannot do continual learning in general. It also does not show a code bug: integrity, ordering, isolation, and hand-check tests all passed, and A-phase learning activity was strong. The failure is in what the reward pushes the network toward.
- No rerun, no tuning, and no seed selection were done; the spec forbids all three. Any follow-up needs a new, separately reviewed spec. For example, reward that also scores silence, or a readout whose "0" is an active choice. That decision is Kendrick's and Amanda's, not this card's.

## Run facts

- 100 of 100 jobs completed (5 arms x 20 seeds). 0 errors, 0 interruptions, 0 retries.
- Wall time 3,317 s (0.92 h) on 5 worker processes (Mac mini i5-8500B). The spec estimated 1.7–2.5 h.
- Peak memory 1.16 GiB for coordinator plus workers, against the 8 GiB abort limit.
- Tests: `pytest -q tests` reported 187 passed at `d04fbd9`.
- Independent review: Nora (requested on card t_26c046ac).
