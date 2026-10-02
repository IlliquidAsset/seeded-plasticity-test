# Stage 2 diagnostic D2-D4 entry (D1 = PASS by D1R2): implementation notes

Card: `t_a8064329` (construction and shakedown only; no D2, D3, or D4 row on any real seed).
Base: `v0.3.0-calibrated` at `3077119f4dd5710245e0b49a652a067ed58d2bff` (verified by `git ls-remote`).

Governing text, `docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md` (f3d52ee, SHA-256 `e5a6a5f4...`) section 9 row 6:
"D1 is `PASS` (by D1R2). D2-D4 only per ebb90e74 section 8 steps 3-6 under a separate run authorization;
outcomes governed by ebb90e74 section 10 row 1 and rows 5 onward. No r4 from D1R2 alone."

## Why a new entry was needed

`run_stage2_diagnostic.py --run-diagnostic` (approved, pinned, unchanged) always runs the original D1 stage
on seeds 2000..2019 first (`run_diagnostic`, lines 832-850) and stops unless that D1 is PASS. The original
D1 closed `INVALID_HARNESS` (package at `5348fcc`). The approved code therefore had no path into
ebb90e74 section 8 step 3 without re-running D1, which this card forbids. Verified by reading the code;
no run was attempted.

## What was added (new files only; no existing file changed)

| File | Role |
|---|---|
| `run_stage2_diagnostic_d2d4.py` | entry: D1R2 gate, preflight, locked `--run-d2d4`, steps 3-6, finalizer, `--shakedown`, `--ss-probe` |
| `tests/test_stage2_diagnostic_d2d4.py` | 47 construction tests (23 test functions, parametrized) |
| `docs/STAGE2_DIAGNOSTIC_D2D4_IMPLEMENTATION_NOTES.md` | this note |
| `results_stage2_diagnostic_d2d4/` | fixture-seed shakedown record and probe check only (no diagnostic seed) |

## Design

1. **D1 = PASS by D1R2, fail closed.** `verify_d1r2_package` checks the committed package
   `results_stage2_diagnostic_d1r2/runs/d1r2-20261002T103331160786Z-e8e014913c00-pid75833`:
   exact file set; each of the seven files against a pinned SHA-256 (the bytes at `3077119`);
   `SHA256SUMS` consistency; recorded `branch_outcome.json` and `summary.json` both
   `PASS` / `D1R2_PASS_D2_D4_ELIGIBLE` / row 6; all three readout statuses `PASS`; no violations;
   source commit `e8e0149`; frozen D1R2 and ebb90e74 hashes recorded at run; and, through Git, that
   `3077119` is an ancestor of HEAD, every working-tree blob equals the blob at `3077119`, and the
   directory has no uncommitted or untracked change. Any failure refuses the run before any package file
   is written. The gate reads the recorded outcome only. It never recomputes a median, bound, or status.
2. **Lock.** `--run-d2d4` requires `--approval-file` with `decision: APPROVE`, `reviewer: nora`,
   `implementation_commit` = exact HEAD, both spec hashes, `d1r2_results_commit` = `3077119...`,
   `run_authorized: true`, and a non-empty `run_authorization_card`, plus HEAD = upstream, clean
   scientific files (including the D1R2 package paths), frozen specs unchanged, probe PASS, and effective
   parameters matching the hand transcription. `run_d2_d4` also refuses to reach the real worker pool
   unless its provenance comes from an approved `d2d4` preflight.
3. **Steps 3-6 carried, not rewritten.** `run_d2_d4` is the approved `run_diagnostic` body after the D1
   stage, line for line, with `statuses["D1"] = "PASS"` set from the gate. It imports and calls the
   approved objects unchanged: `base._condition_job` -> `snn.stage2_diagnostic.run_condition_seed`,
   `base.run_pool` (8 GiB RSS abort, at most 5 workers), `stats.summarize_d2/d3/d4`, the namespace-30
   bootstrap, `base.validated_effective_parameters`, `base.branch_outcome` (the ordered 12-row
   section 10 table), `base.write_stage_rows` (redaction of invalid stages), `base.manifest`. No constant,
   threshold, seed, drive value, checkpoint, bootstrap ID, stop condition, or branch row is defined in the
   new module. Test `d2d4_14` proves equality: for five plans, the new entry's D2/D3/D4 stage records,
   statuses, branch outcome, and row files are identical (byte-for-byte for row files) to the approved
   runner fed the same rows after a synthetic D1 PASS.
4. **Branch reachability.** With D1 fixed to PASS only ebb90e74 section 10 row 1 and rows 5 onward are
   reachable (exhaustive enumeration in `d2d4_16`; every reachable code exercised end to end in `d2d4_15`).
   The finalizer refuses rather than mislabels if a D1-only row (2-4) were ever selected.
5. **Hard limit 7 (no D1R2 decoder in D2-D4).** The entry imports no D1R/D1R2 module and no scikit-learn
   (AST check and a subprocess `sys.modules` check after the gate and a condition row). The gate hashes
   `d1r2_rows.jsonl` and `D1R2.events.jsonl` as bytes and never parses them (spy test `d2d4_12`). Worker
   jobs carry only `(seed, task, plastic, drive, expected hashes)` (`d2d4_13`).
6. **Package.** Same contract as ebb90e74 section 13 via the approved helpers, plus `d1_reference.json`
   (D1 by reference: results commit, source commit, file hashes, recorded branch and readout statuses,
   governing text, D1R2 network claim). No `d1_rows` file is written. Plain-language D1 line reads
   "D1 PASS by D1R2 ...".

## Implementation readings for the reviewer (no new science wording)

- ebb90e74 section 13 lists D1 coefficients, normalization hashes, and bounds as package contents. For a
  D1 = PASS by D1R2 entry those live in the frozen D1R2 package. The D2-D4 package references them by
  path, commit, and SHA-256 rather than copying or re-deriving them. The D1 claim text is the D1R2
  package's own `network_claim`, read verbatim. No new claim wording was written.
- `branch_outcome` condition strings for rows 5 onward say "D1 PASS ..." exactly as ebb90e74 section 10.
  The package's `governing_text` and `d1_reference.json` state that D1 PASS is by D1R2.
- ebb90e74 section 3.2 probe: `--ss-probe` re-executes the approved `tools/stage2_diagnostic_ss_probe.py`
  in memory, requires the same identity-state SHA-256 (`2f6b54fb...`), counts, and 6 legacy-only groups as
  the retained output, and requires the retained D1R and D1R2 probes (which enumerate every namespace-30
  identity as legacy) to show zero new-vs-legacy collisions.

## Verification on this card

- `PYTHONHASHSEED=0 OMP/MKL/OPENBLAS_NUM_THREADS=1 /Users/aib-agent/miniforge3/envs/aib310/bin/python -m pytest -q -p no:cacheprovider`
  (full suite) and the new file alone; results in the card handoff.
- `--ss-probe` and `--shakedown` on fixture seed 4242 only (short lengths, seven conditions through the
  approved worker body, construction checks only, no accuracy or rate recorded).
- No D2, D3, or D4 row on seeds 2000..2019 was produced. D1, D1R, D1R2 were not run or re-scored.

## Rollback

`git revert <construction commit> <shakedown results commit>`. Nothing existing was modified.

## Next gate

Nora: independent review against ebb90e74 sections 5-8 and 10-13 and D1R2 spec section 9 row 6, with
independent test execution. Approval covers implementation only. The run needs a separate run-authorization
card and the approval file described above.
