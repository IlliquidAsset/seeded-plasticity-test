#!/usr/bin/env python3
"""Stage 2 r3 runner: docs/STAGE2_SPEC.md (frozen r3 at def4b36).

Modes (exactly one):
  --shakedown   non-experimental seeds 1000..1019, tiny lengths, all 5 arms, scratch output dir;
                exercises aggregation/bootstrap/scoring/report only (sec. 11).
  --qualify     sec. 11 item 15 fixture qualification: frozen-from-start arm ONLY, seeds 1000..1019,
                full length (400,000 steps + four 12,000-step evaluations). Writes qualification.json
                with decision PASS or STOP. On STOP no experimental seed may run.
  (default)     the experiment: 5 arms x seeds 0..19, full length. Refuses to start unless
                <out>/qualification/qualification.json exists with decision PASS, the spec hash is
                the frozen r3 hash, params equal the spec transcription, and scientific files are clean.

Progress lines report arm/seed/status/wall/RSS only, never an outcome (sec. 11 last paragraph).
RSS watchdog aborts (not swaps) above 8 GiB (sec. 12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_stage2 import RSS_LIMIT, RSSWatch, _jsonable  # unchanged r2 utilities

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "docs" / "STAGE2_SPEC.md"
SCI_FILES = ["snn/core.py", "snn/florian_bench.py", "snn/stage2.py", "snn/stage2_stats.py",
             "snn/stage2_r3.py", "snn/stage2_r3_stats.py", "run_stage2.py", "run_stage2_r3.py",
             "tests/test_stage2.py", "tests/test_stage2_r3.py", "docs/STAGE2_SPEC.md",
             "docs/STAGE2_R3_IMPLEMENTATION_NOTES.md"]
SHAKEDOWN_SEEDS = tuple(range(1000, 1020))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results_stage2_r3")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--shakedown", action="store_true", help="non-experimental seeds 1000..1019, tiny lengths")
    mode.add_argument("--qualify", action="store_true", help="sec. 11 item 15 frozen-from-start fixture qualification")
    ap.add_argument("--tests", default="not recorded", help="maintained test command and result, recorded verbatim")
    return ap


def argv_from_args(args, parser=None):
    parser = parser or build_parser()
    out = []
    for a in parser._actions:
        if not a.option_strings or a.dest == "help":
            continue
        v = getattr(args, a.dest)
        if isinstance(v, bool):
            if v:
                out.append(a.option_strings[0])
        else:
            out += [a.option_strings[0], str(v)]
    return out


def _git(*cmd):
    return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expected_params_from_spec() -> dict:
    """Values transcribed from docs/STAGE2_SPEC.md r3 sec. 3-9 (independent of snn/stage2_r3.py)."""
    return {
        "spec_revision": "r3", "layer_sizes": [2, 20, 2], "n_trainable_synapses": 80,
        "outputs": ["O1 (index 0, predict 1)", "O0 (index 1, predict 0)"],
        "output_neuron": "LIF, non-adaptive, no lateral/recurrent/inhibitory connection",
        "dt_ms": 1.0, "tau_m_ms": 20.0, "tau_syn_ms": 0.0, "v_rest_mv": -70.0, "v_reset_mv": -70.0,
        "v_thresh_mv": -54.0, "dtype": "float64", "batch_size": 1,
        "w1_init": "U(-10,10) mV [20261001, seed, 1]", "w2_O1_init": "U(0,10) mV [20261001, seed, 1] (r2 W2 draw)",
        "w2_O0_init": "U(0,10) mV [20261001, seed, 21]", "w1_bounds_mv": [-10.0, 10.0], "w2_bounds_mv": [0.0, 10.0],
        "credit": "eligibility", "gamma_mv": 0.25, "tau_plus_ms": 20.0, "tau_minus_ms": 20.0, "tau_elig_ms": 25.0,
        "a_plus": 1.0 / 25.0, "a_minus": 1.0 / 25.0, "reward_mode": "per_spike_next_step",
        "reward": "r_t = (2*x_t - 1) * (z1_t - z0_t)",
        "prediction_rule": "O1 alone -> 1; O0 alone -> 0; tie -> coin[seed, c][i]",
        "tie_coin": "[20261001, seed, 22, c], c=1..4 (A_pre,B_pre,B_post,A_post); integers(0,2,size=12000) int8",
        "tau_a_ms": 200.0, "beta_a_mv": 1.12, "beta_a_mv_sfa_off": 0.0, "noise_p": 0.10, "phase_steps": 200_000,
        "total_steps": 400_000, "eval_len": 12_000, "eval_warmup": 2_000, "eval_scored": 10_000,
        "seeds": list(range(20)), "qualification_seeds": list(range(1000, 1020)),
        "arms": ["P", "F0", "FS", "SC", "NS"], "checkpoints": ["A_pre", "B_pre", "B_post", "A_post"],
        "bootstrap_resamples": 100_000, "root_entropy": 20261001,
    }


def _job(job):
    import torch

    torch.set_num_threads(1)
    from snn.stage2_r3 import run_arm_seed

    arm, seed, expected, lengths = job
    try:
        row = run_arm_seed(arm, seed, expected_hashes=expected, **lengths)  # validate_row() inside
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:  # recorded, never silently dropped (sec. 13); schema failure lands here too
        return {"arm": arm, "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


def preflight(args, out: Path, mode: str, seeds, arms, lengths) -> dict:
    from snn import stage2_r3

    spec_sha = file_sha256(SPEC)
    params_ok = stage2_r3.FROZEN_PARAMS == expected_params_from_spec()
    dirty = _git("status", "--porcelain", "--", *SCI_FILES)
    head = _git("rev-parse", "HEAD")
    try:
        upstream = _git("rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = None
    prov = {
        "schema_version": 1,
        "stage": "Stage 2 r3 continual hidden-structure learning",
        "mode": mode,
        "spec_path": "docs/STAGE2_SPEC.md",
        "spec_commit_frozen": stage2_r3.SPEC_COMMIT,
        "spec_sha256_frozen": stage2_r3.SPEC_SHA256,
        "spec_sha256_at_run": spec_sha,
        "spec_hash_unchanged": spec_sha == stage2_r3.SPEC_SHA256,
        "params_match_spec": bool(params_ok),
        "params": stage2_r3.FROZEN_PARAMS,
        "code_commit": head,
        "code_commit_equals_upstream_tracking_ref": head == upstream,
        "upstream_tracking_ref_sha": upstream,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "scientific_files": SCI_FILES,
        "scientific_files_clean": dirty == "",
        "scientific_files_status": dirty,
        "argv": [sys.executable, *sys.argv],
        "argv_display": shlex.join([sys.executable, *sys.argv]),
        "argv_normalized": [sys.executable, "run_stage2_r3.py", *argv_from_args(args)],
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "python": sys.version,
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "workers": args.workers,
        "seeds": list(seeds),
        "arms": list(arms),
        "lengths": lengths,
        "maintained_tests": args.tests,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    return prov


def run_jobs(jobs, workers, out: Path):
    watch = RSSWatch()
    watch.start()
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    partial = out / "runs.partial.jsonl"
    partial.write_text("")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            with partial.open("a") as fh:
                fh.write(json.dumps(row, default=_jsonable) + "\n")
            print(f"[{i}/{len(jobs)}] arm={row['arm']} seed={row['seed']} status={row['status']} "
                  f"wall={row.get('wall_seconds', 0):.0f}s elapsed={time.time() - t0:.0f}s "
                  f"rss_peak={watch.peak_total / 2**30:.2f}GiB", flush=True)
    watch.sample()
    watch.stop()
    wall = time.time() - t0
    errors = [r for r in rows if r["status"] != "ok"]
    meta = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall,
        "workers": workers,
        "peak_rss_total_bytes": watch.peak_total,
        "peak_rss_single_process_bytes": watch.peak_single,
        "max_worker_processes_observed": watch.max_children,
        "rss_limit_bytes": RSS_LIMIT,
        "rss_exceeded": watch.exceeded,
        "jobs": len(jobs),
        "rows_ok": len(rows) - len(errors),
        "rows_error": len(errors),
        "interruptions": 0,
        "retries": 0,
    }
    partial.unlink()
    return rows, errors, meta


def write_rows(path: Path, rows) -> None:
    """Result writer with the sec. 8.1 schema-completeness assertion: every ok row is validated
    BEFORE anything is written; any failure raises and no row (no file) is emitted."""
    from snn.stage2_r3 import validate_row

    for r in rows:
        if r.get("status", "ok") == "ok":
            validate_row(r)
    path.write_text("".join(json.dumps(r, sort_keys=True, default=_jsonable) + "\n" for r in rows))


def _manifest(out: Path):
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{file_sha256(p)}  {p.relative_to(out)}\n" for p in files))


def main():
    ap = build_parser()
    args = ap.parse_args()
    from snn import stage2_r3, stage2_r3_stats

    lengths_full = {"phase_steps": stage2_r3.PHASE_STEPS, "eval_len": stage2_r3.EVAL_LEN, "eval_warmup": stage2_r3.EVAL_WARMUP}
    if args.shakedown:
        mode, seeds, arms = "shakedown", SHAKEDOWN_SEEDS, stage2_r3.ARMS
        lengths = {"phase_steps": 1_000, "eval_len": 400, "eval_warmup": 100}
        n_boot = 2_000
        out = Path(args.out)
    elif args.qualify:
        mode, seeds, arms = "qualification", stage2_r3.QUALIFICATION_SEEDS, ("F0",)
        lengths, n_boot = lengths_full, stage2_r3_stats.N_BOOT
        out = ROOT / args.out / "qualification"
    else:
        mode, seeds, arms = "experiment", stage2_r3.EXPERIMENTAL_SEEDS, stage2_r3.ARMS
        lengths, n_boot = lengths_full, stage2_r3_stats.N_BOOT
        out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    qual = None
    if mode == "experiment":
        qpath = ROOT / args.out / "qualification" / "qualification.json"
        if not qpath.exists():
            sys.exit("refusing experiment: no fixture-qualification record (sec. 11 item 15)")
        qual = json.loads(qpath.read_text())
        if qual.get("decision") != "PASS":
            sys.exit(f"refusing experiment: fixture qualification decision is {qual.get('decision')!r} (sec. 11 item 15 STOP)")

    prov = preflight(args, out, mode, seeds, arms, lengths)
    if mode != "shakedown":
        if not (prov["spec_hash_unchanged"] and prov["params_match_spec"] and prov["scientific_files_clean"]
                and prov["code_commit_equals_upstream_tracking_ref"]):
            sys.exit(f"preflight failed: spec_hash_unchanged={prov['spec_hash_unchanged']} params_match_spec={prov['params_match_spec']} "
                     f"clean={prov['scientific_files_clean']} pushed={prov['code_commit_equals_upstream_tracking_ref']}")
    print(f"preflight ok; mode={mode}; provenance written {out / 'provenance.json'}; commit {prov['code_commit']}", flush=True)

    expected = {s: stage2_r3.object_hashes(stage2_r3.seed_objects(s, lengths["phase_steps"], lengths["eval_len"])) for s in seeds}
    jobs = [(a, s, expected[s], lengths) for a in arms for s in seeds]
    rows, errors, run_meta = run_jobs(jobs, args.workers, out)
    rows.sort(key=lambda r: (stage2_r3.ARMS.index(r["arm"]), r["seed"]))
    write_rows(out / "runs.jsonl", rows)
    summary = {"provenance": prov, "run": run_meta}

    if mode == "qualification":
        if errors:
            q = {"decision": "STOP", "reason": "job errors: " + ", ".join(f"{r['arm']}/{r['seed']}: {r['error']}" for r in errors)}
        else:
            q = stage2_r3_stats.qualify(rows, seeds, n_boot)
        q.update({"run": run_meta, "code_commit": prov["code_commit"], "spec_sha256_at_run": prov["spec_sha256_at_run"]})
        (out / "qualification.json").write_text(json.dumps(q, indent=2, sort_keys=True, default=_jsonable) + "\n")
        _manifest(out)
        print("QUALIFICATION", q["decision"])
        print(json.dumps({k: q.get(k) for k in ("means", "conditions")}, default=_jsonable))
        sys.exit(0 if q["decision"] == "PASS" else 4)

    if errors:
        summary["verdict"] = "INVALID_OR_MECHANISM_FAIL: incomplete (job errors) " + ", ".join(f"{r['arm']}/{r['seed']}" for r in errors)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
        print(summary["verdict"])
        sys.exit(2)
    acc = stage2_r3_stats.acc_tensor(rows, seeds)
    boot = stage2_r3_stats.bootstrap_all(acc, n_boot=n_boot)
    preflight_ok = {
        "provenance_before_execution": (out / "provenance.json").exists(),
        "params_match_spec": prov["params_match_spec"],
        "spec_hash_unchanged": prov["spec_hash_unchanged"] and file_sha256(SPEC) == stage2_r3.SPEC_SHA256,
    }
    qual_pass = (qual is not None and qual.get("decision") == "PASS") if mode == "experiment" else True
    sc = stage2_r3_stats.score(rows, boot, preflight_ok, seeds=seeds, phase_steps=lengths["phase_steps"],
                               eval_scored=lengths["eval_len"] - lengths["eval_warmup"], qualification_pass=qual_pass)
    summary.update({"bootstrap": boot, "scoring": sc, "verdict": sc["verdict"], "verdict_labels": sc["verdict_labels"],
                    "qualification_decision": qual.get("decision") if qual else None,
                    "accuracy_table": {a: {c: [r["accuracy"][c] for r in rows if r["arm"] == a] for c in stage2_r3.CHECKPOINTS}
                                       for a in stage2_r3.ARMS}})
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    write_report(out, summary, qual)
    _manifest(out)
    print("VERDICT", sc["verdict"])
    print(json.dumps({k: v["pass"] for k, v in sc["gates"].items()}))


def _f(x, nd=4):
    return "undefined" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(out: Path, s: dict, qual):
    from snn.stage2_r3 import ARM_NAMES, ARMS, CHECKPOINTS

    sc = s["scoring"]
    B = {b["id"]: b for b in s["bootstrap"]}
    prov, run = s["provenance"], s["run"]
    ok = lambda b: "PASS" if b else "FAIL"
    L = [
        "# Stage 2 r3 result: continual hidden-structure learning (two-output readout)",
        "",
        f"Spec: `docs/STAGE2_SPEC.md` frozen r3 at `{prov['spec_commit_frozen']}`; SHA-256 at run `{prov['spec_sha256_at_run']}` (unchanged: {prov['spec_hash_unchanged']})",
        f"Code commit: `{prov['code_commit']}` (scientific files clean: {prov['scientific_files_clean']}; equals pushed upstream ref: {prov['code_commit_equals_upstream_tracking_ref']})",
        f"Command: `{prov['argv_display']}`",
        f"Host: {prov['host']['hostname']} ({prov['host']['processor']}, {prov['host']['cpu_count']} logical CPUs); Python {prov['python'].split()[0]}, NumPy {prov['numpy']}, torch {prov['torch']}",
        f"Maintained tests: {prov['maintained_tests']}",
        "",
        "## Resources (sec. 12), reported separately",
        "",
    ]
    if qual is not None:
        qr = qual["run"]
        L.append(f"- Fixture qualification (F0 only, seeds 1000..1019): wall {qr['wall_seconds']:.0f} s ({qr['wall_seconds'] / 3600:.2f} h); "
                 f"{qr['workers']} workers; peak RSS {qr['peak_rss_total_bytes'] / 2**30:.2f} GiB; interruptions {qr['interruptions']}; retries {qr['retries']}; "
                 f"rows ok {qr['rows_ok']}/{qr['jobs']}; decision {qual['decision']}")
    L += [
        f"- Experimental run (5 arms x seeds 0..19): wall {run['wall_seconds']:.0f} s ({run['wall_seconds'] / 3600:.2f} h); {run['workers']} workers; "
        f"peak RSS {run['peak_rss_total_bytes'] / 2**30:.2f} GiB; interruptions {run['interruptions']}; retries {run['retries']}; rows ok {run['rows_ok']}/{run['jobs']}",
        "",
        f"## Verdict: {s['verdict']}",
        "",
        "All applicable section 10 labels: " + "; ".join(f"`{x}`" for x in s["verdict_labels"]),
        "",
        "Finding taxonomy (spec sec. 1, verbatim): \"If B learning passes but A retention fails, the finding is 'online adaptation with forgetting' — not continual learning.\"",
        "",
        "## Gates (section 10)",
        "",
        "| Gate | Item | Result |",
        "|---|---|---|",
    ]
    for g in ("gate0", "gate1", "gate2", "gate3", "gate4"):
        for k, v in sc["gates"][g]["items"].items():
            L.append(f"| {g} | {k} | {ok(v)} |")
        L.append(f"| **{g}** | **overall** | **{ok(sc['gates'][g]['pass'])}** |")
    m = sc["means"]
    L += ["", "## Mean held-out accuracy (20 seeds; chance 0.50; theoretical optimum 0.90)", "",
          "| Arm | " + " | ".join(CHECKPOINTS) + " |", "|---|" + "---:|" * len(CHECKPOINTS)]
    for ai, a in enumerate(ARMS):
        cells = []
        for ci, c in enumerate(CHECKPOINTS):
            ci95 = B[100 + 4 * ai + ci]["two_sided_95"]
            cells.append(f"{m[a][c]:.4f} [{ci95[0]:.4f}, {ci95[1]:.4f}]")
        L.append(f"| {a}: {ARM_NAMES[a]} | " + " | ".join(cells) + " |")
    g3 = sc["gates"]["gate3"]
    L += ["", "## Gate quantities (bootstrap, 100,000 seed resamples, inverted-CDF percentiles)", "",
          "| ID | Metric | Point | One-sided lower 95 | One-sided upper 95 | Two-sided 95 |", "|---:|---|---:|---:|---:|---|"]
    for k in range(15):
        b = B[k]
        two = b.get("two_sided_95")
        L.append(f"| {k} | `{b['formula']}` | {_f(b['point_estimate'])} | {_f(b.get('one_sided_lower_95'))} | "
                 f"{_f(b.get('one_sided_upper_95'))} | {('[' + _f(two[0]) + ', ' + _f(two[1]) + ']') if two else ''} |")
    L += [
        "",
        f"Metric 12 undefined-denominator replicates: {B[12]['undefined_denominator_replicates']} of {B[12]['resamples']}.",
        f"A_learned = {g3['A_learned']:.4f}; A_loss = {g3['A_loss']:.4f}; A_retained = {_f(g3['A_retained'])}.",
        "",
        "## Controls (section 6)",
        "",
        f"- 6.2 frozen from start: {ok(sc['validity_6_2']['pass'])} — {json.dumps({k: v for k, v in sc['validity_6_2'].items() if k != 'pass'})}",
        f"- 6.3 freeze at shift: {ok(sc['validity_6_3']['pass'])} — {json.dumps({k: v for k, v in sc['validity_6_3'].items() if k != 'pass'})}",
        f"- 6.4 scrambled B twin: {ok(sc['validity_6_4']['pass'])} — {json.dumps({k: v for k, v in sc['validity_6_4'].items() if k != 'pass'})}",
        f"- 6.5 SFA off: {ok(sc['validity_6_5']['pass'])} — threshold contribution exactly 0: {sc['validity_6_5']['threshold_contribution_exactly_0_all_samples']}; "
        f"dW>0 A&B: {sc['validity_6_5']['abs_dw_gt0_A_and_B_20of20']}; sign-of-life A/B: {sc['validity_6_5']['sign_of_life_A']}/{sc['validity_6_5']['sign_of_life_B']}",
        f"- SFA benefit claim (metric 13 >= 0.05 and lower > 0): {sc['sfa']['sfa_benefit_claim_allowed']} (point {sc['sfa']['metric13_point']:.4f}, lower {sc['sfa']['metric13_lower']:.4f})",
        "",
        "## Primary sign-of-life (section 8)",
        "",
    ]
    for ph in ("A", "B"):
        x = sc["sign_of_life_primary"][ph]
        L.append(f"- {ph}: {ok(x['pass'])} — dW>0 seeds {x['abs_dw_gt0_seeds']}/20; min +/- reward events {x['min_positive_reward_events']}/{x['min_negative_reward_events']}; "
                 f"median r {x['median_reward_signed_dw_out_corr']:.3f}; seeds r>0.10 {x['seeds_corr_gt_0.10']}/20; undefined r {x['undefined_corr_seeds']}")
    L += [
        "",
        "## Claim boundary",
        "",
        "Only a `STAGE2_PASS` licenses the sec. 2 sentence. Any other verdict is reported as named above; thresholds were not moved and no seed, arm, or run was re-run or selected.",
        "Not established in any case: broad continual learning, biological equivalence, optimal SFA, transfer across modalities, performance in Abe.",
        "No oracle: the target enters only as the next observation and the sign of the reward for already-emitted output spikes. The tie-coin scores evaluations only.",
        "r3 is a second attempt under the same roadmap gates; the r2 verdict (`results_stage2/`) stands and is not re-scored.",
    ]
    (out / "STAGE2_R3_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
