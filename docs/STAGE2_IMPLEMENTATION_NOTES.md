# Stage 2 implementation notes (committed before any experimental seed runs)

Spec: `docs/STAGE2_SPEC.md`, frozen and approved at `ef85a15b0382bdc86b8dba377671e4c2b2e4f79f`,
SHA-256 `0c1bbdba036b13955c81f2ce0ffee0557b09842ef7d56b0dec0834faf8d8ec87`. This file does not change
the spec. It maps each spec section to code, and records how the scorer reads the few points where
the spec needs an explicit operational reading. Every reading below is fixed before data exists.

## Code map

| Spec | Code |
|---|---|
| §3, §7.1 streams, weights, SeedSequence identities | `snn/stage2.py`: `initial_weights`, `a_training_stream`, `b_training_stream`, `scrambled_b`, `heldout_stream`, `seed_objects`, `object_hashes` |
| §4.1–4.2 network, MSTDPET, `REWARD_PER_SPIKE_NEXT_STEP` | `snn/stage2.py:build_network` on `snn/core.py` `PureSNN`/`Synapse`/`RSTDPPlasticity`/`online_step` |
| §4.3 ALIF / SFA | `snn/core.py:ALIFNeuron` (hidden layer only; output stays `LIFNeuron`) |
| §5 evaluations, §5.1 start state, §5.3 exactly four checkpoints | `snn/stage2.py:evaluate`, `init_state`, `run_arm_seed` |
| §6 five arms | `run_arm_seed(arm=P/F0/FS/SC/NS)` |
| §8 instrumentation | `snn/stage2.py:PhaseStats`, `train_phase` |
| §9, §9.1 bootstrap | `snn/stage2_stats.py:bootstrap_metric`, `bootstrap_all` |
| §6 validity, §8 sign-of-life, §10 gates + verdict | `snn/stage2_stats.py:score`, `verdict_from` |
| §11 shakedown | `tests/test_stage2.py` (non-experimental seeds only) and `run_stage2.py --shakedown` (seeds 1000–1019, 1,000-step phases) |
| §12 resources, §13 package | `run_stage2.py` (RSS watchdog, provenance before execution, SHA-256 manifest) |

## Operational readings (fixed before the run)

1. **Training step.** At step `t` the live network's current output spike `z_t` is the prediction; the
   step calls `online_step(onehot(x_t), reward_fn)` with `reward_fn(z) = (2*x_t - 1) * z`. Arms with
   plasticity off call `online_step(onehot(x_t), reward_fn=None)`; traces still run, no weight changes.
   This is the Stage 1 `CoreFlorianBench.step` path.
2. **Freeze-at-shift.** "Disabled immediately before step 200,000" means every B step uses
   `reward_fn=None`. SFA, membranes, traces, and eligibility continue without reset.
3. **Undefined correlation.** If `reward_signed_dw_out_corr` is undefined (fewer than 2 reward events or zero
   variance), it counts as `0.0` in the §8 median and in the `> 0.10` count. This is the Stage 1 convention
   in `run_florian_core._group`. The number of undefined seeds is reported.
4. **Instrumentation sampling (§6.5 "every sampled instrumentation point").** The ALIF threshold contribution
   `beta_a * a` is sampled every 1,000 training steps and at the end of each phase. The per-step maximum of
   `a` is tracked as well. SFA-off validity requires both the sampled maximum `|beta_a * a|` and
   `beta_a * max(a)` to equal exactly `0.0`.
5. **Non-finite and bounds checks.** Weights, membranes, adaptation, and eligibility are checked for
   non-finite values, and weights against their bounds, at the same sampling points and at the end. Final
   weights are checked directly. Accuracies must be finite.
6. **Scrambled-B histogram (§6.4).** The SC arm's B training vector must have the same 0/1 counts as P's
   structured B vector for the same seed. The P row's `B_train` hash must equal the SC row's `B_train` hash.
   Because the input is one-hot, total input spike count equals stream length.
7. **Gate 4 item 2.** The live network's single initialization is `reset_online_state` (called once,
   followed by 2 plasticity resets and 1 ALIF reset inside it). After step 0 it must make zero further
   `reset_online_state`, `restore_frozen_weights`, plasticity-reset, or ALIF-reset calls. Evaluation copies
   are separate deepcopies with their own init. Their counts are logged per evaluation and are not added to the live counts.
8. **Gate 4 item 5.** The training access log must equal exactly `{A_train: 200,000, <B stream>: 200,000}`.
   No training phase reads from any held-out stream, and A is never re-read during B.
9. **Gate 4 items 6–7.** `provenance.json` is written before the first job. It records the spec hash, the
   `FROZEN_PARAMS == spec transcription` check, commit, clean status, argv, and host. The runner refuses to
   start the experimental run if the spec hash changed, the parameters differ, or a scientific file is dirty.
10. **Verdict precedence.** Every applicable §10 label is listed, in this order:
    `INVALID_OR_MECHANISM_FAIL: <failed invariants>` (Gate 0), `NO_A_COMPETENCE` (Gate 1 fail),
    `NO_HELD_OUT_B_LEARNING` (Gate 1 pass, Gate 2 fail), `online adaptation with forgetting — not continual learning`
    (Gate 2 pass, Gate 3 fail), `GATE4_FAIL: <items>`. The headline verdict is the first label in that list.
    `STAGE2_PASS` requires all five gates.
10a. **Gate 0 granularity.** The §6.3 and §6.4 validity lists include the B differentials (also Gate 2.5).
    Read literally, a primary B-learning failure therefore also fails Gate 0. The scorer follows the text
    literally and does not move these items out of Gate 0. It does split every Gate 0 component into its own
    named item (e.g. `6.3_freeze_at_shift_B_differential_lower_gt_0`). The `INVALID_OR_MECHANISM_FAIL` label
    names exactly which items failed, and every other applicable label (for example `NO_HELD_OUT_B_LEARNING`)
    is still listed. The reviewer can then tell a structural or mechanism invalidity (hash, bounds, zero-dW,
    sign-of-life) apart from a behavioral differential failure without anyone re-scoring.
11. **SFA-off interpretation labels (§6.5).** These apply only if the primary arm is `STAGE2_PASS`. The §9.1
    table defines no bootstrap metrics for the SFA-off arm's own behavioral gates, and §9.1 forbids adding
    any. If the primary passes, the scorer reports the SFA-off point estimates against the Gate 1–3 point
    thresholds and routes the label to the reviewer as a recorded spec gap. It does not invent new bootstrap
    metrics. If the primary does not pass, no SFA-dependence label is issued.
12. **Progress output.** While jobs run, the runner prints only arm, seed, status, wall time, and RSS. No
    accuracy or outcome is shown before all 100 jobs complete (§11 last paragraph).

## Shakedown scope (§11)

Tests use seeds `1000+` (never 0–19) and lengths of at most a few thousand steps. No gate metric is estimated.
`run_stage2.py --shakedown` exercises the full aggregation, bootstrap, scoring, and report pipeline on seeds
1000–1019 with 1,000-step phases and 400-step evaluations, writing to a scratch directory outside the repository.
