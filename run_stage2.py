#!/usr/bin/env python3
"""Stage 2 runner: docs/STAGE2_SPEC.md (frozen at ef85a15), 5 arms x 20 seeds.

Order of operations:
  1. preflight: spec SHA-256 == frozen hash, FROZEN_PARAMS == spec values,
     git commit + clean/dirty status of scientific files, argv, host, params
     -> provenance.json written BEFORE any job starts (Gate 4 item 6);
  2. coordinator computes every seed's object hashes; each worker asserts its
     own regenerated hashes against them before running (sec. 7);
  3. 100 jobs on N workers; progress lines report arm/seed/wall/status only,
     never an outcome (sec. 11: no looking at partial seed outcomes);
  4. RSS watchdog: abort (not swap) if coordinator+workers RSS > 8 GiB (sec. 12);
  5. after all 100 rows: bootstrap (sec. 9.1), score (sec. 6, 8, 10), report, SHA-256 manifest.

``--shakedown`` runs the same pipeline on non-experimental seeds 1000..1019 with
tiny lengths, into a separate output dir, to exercise aggregation (sec. 11).
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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "docs" / "STAGE2_SPEC.md"
SCI_FILES = ["snn/core.py", "snn/florian_bench.py", "snn/stage2.py", "snn/stage2_stats.py", "run_stage2.py",
             "tests/test_stage2.py", "docs/STAGE2_SPEC.md", "docs/STAGE2_IMPLEMENTATION_NOTES.md"]
RSS_LIMIT = 8 * 1024 ** 3
SHAKEDOWN_SEEDS = tuple(range(1000, 1020))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results_stage2")
    ap.add_argument("--shakedown", action="store_true", help="non-experimental seeds 1000..1019, tiny lengths")
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


def _job(job):
    import torch

    torch.set_num_threads(1)
    from snn.stage2 import run_arm_seed

    arm, seed, expected, lengths = job
    try:
        row = run_arm_seed(arm, seed, expected_hashes=expected, **lengths)
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:  # recorded, never silently dropped (sec. 13)
        return {"arm": arm, "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


class RSSWatch:
    def __init__(self, limit=RSS_LIMIT, period=2.0):
        import psutil

        self.psutil = psutil
        self.proc = psutil.Process()
        self.limit = limit
        self.period = period
        self.peak_total = 0
        self.peak_single = 0
        self.max_children = 0
        self.exceeded = False
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        procs = [self.proc] + self.proc.children(recursive=True)
        rss = []
        for p in procs:
            try:
                rss.append(p.memory_info().rss)
            except self.psutil.Error:
                pass
        total = sum(rss)
        self.peak_total = max(self.peak_total, total)
        self.peak_single = max(self.peak_single, max(rss) if rss else 0)
        self.max_children = max(self.max_children, len(procs) - 1)
        if total > self.limit:
            self.exceeded = True
        return total

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            if self.exceeded:
                print(f"ABORT: RSS {self.peak_total / 2**30:.2f} GiB > 8 GiB", flush=True)
                for c in self.proc.children(recursive=True):
                    try:
                        c.kill()
                    except self.psutil.Error:
                        pass
                os._exit(3)
            self._stop.wait(self.period)

    def start(self):
        self.t.start()

    def stop(self):
        self._stop.set()
        self.t.join()


def preflight(args, out: Path, seeds, lengths) -> dict:
    from snn import stage2

    spec_sha = file_sha256(SPEC)
    params_ok = stage2.FROZEN_PARAMS == expected_params_from_spec()
    dirty = _git("status", "--porcelain", "--", *SCI_FILES)
    prov = {
        "schema_version": 1,
        "stage": "Stage 2 continual hidden-structure learning",
        "spec_path": "docs/STAGE2_SPEC.md",
        "spec_commit_frozen": stage2.SPEC_COMMIT,
        "spec_sha256_frozen": stage2.SPEC_SHA256,
        "spec_sha256_at_run": spec_sha,
        "spec_hash_unchanged": spec_sha == stage2.SPEC_SHA256,
        "params_match_spec": bool(params_ok),
        "params": stage2.FROZEN_PARAMS,
        "code_commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "scientific_files": SCI_FILES,
        "scientific_files_clean": dirty == "",
        "scientific_files_status": dirty,
        "argv": [sys.executable, *sys.argv],
        "argv_display": shlex.join([sys.executable, *sys.argv]),
        "argv_normalized": [sys.executable, "run_stage2.py", *argv_from_args(args)],
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "python": sys.version,
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "workers": args.workers,
        "shakedown": bool(args.shakedown),
        "seeds": list(seeds),
        "lengths": lengths,
        "maintained_tests": args.tests,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    return prov


def expected_params_from_spec() -> dict:
    """Values transcribed from docs/STAGE2_SPEC.md sec. 3-9 (independent of snn/stage2.py)."""
    return {
        "layer_sizes": [2, 20, 1], "n_trainable_synapses": 60, "dt_ms": 1.0, "tau_m_ms": 20.0, "tau_syn_ms": 0.0,
        "v_rest_mv": -70.0, "v_reset_mv": -70.0, "v_thresh_mv": -54.0, "dtype": "float64", "batch_size": 1,
        "w1_init": "U(-10,10) mV", "w2_init": "U(0,10) mV", "w1_bounds_mv": [-10.0, 10.0], "w2_bounds_mv": [0.0, 10.0],
        "credit": "eligibility", "gamma_mv": 0.25, "tau_plus_ms": 20.0, "tau_minus_ms": 20.0, "tau_elig_ms": 25.0,
        "a_plus": 1.0 / 25.0, "a_minus": 1.0 / 25.0, "reward_mode": "per_spike_next_step",
        "reward": "r_t = (2*x_t - 1) * z_t", "tau_a_ms": 200.0, "beta_a_mv": 1.12, "beta_a_mv_sfa_off": 0.0,
        "noise_p": 0.10, "phase_steps": 200_000, "total_steps": 400_000, "eval_len": 12_000, "eval_warmup": 2_000,
        "eval_scored": 10_000, "seeds": list(range(20)), "arms": ["P", "F0", "FS", "SC", "NS"],
        "checkpoints": ["A_pre", "B_pre", "B_post", "A_post"], "bootstrap_resamples": 100_000, "root_entropy": 20261001,
    }


def main():
    ap = build_parser()
    args = ap.parse_args()
    from snn import stage2, stage2_stats

    if args.shakedown:
        seeds = SHAKEDOWN_SEEDS
        lengths = {"phase_steps": 1_000, "eval_len": 400, "eval_warmup": 100}
        n_boot = 2_000
        out = Path(args.out)
    else:
        seeds = stage2.EXPERIMENTAL_SEEDS
        lengths = {"phase_steps": stage2.PHASE_STEPS, "eval_len": stage2.EVAL_LEN, "eval_warmup": stage2.EVAL_WARMUP}
        n_boot = stage2_stats.N_BOOT
        out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    prov = preflight(args, out, seeds, lengths)
    if not args.shakedown:
        if not (prov["spec_hash_unchanged"] and prov["params_match_spec"] and prov["scientific_files_clean"]):
            sys.exit(f"preflight failed: spec_hash_unchanged={prov['spec_hash_unchanged']} "
                     f"params_match_spec={prov['params_match_spec']} clean={prov['scientific_files_clean']}")
    print(f"preflight ok; provenance written {out / 'provenance.json'}; commit {prov['code_commit']}", flush=True)

    expected = {s: stage2.object_hashes(stage2.seed_objects(s, lengths["phase_steps"], lengths["eval_len"])) for s in seeds}
    jobs = [(a, s, expected[s], lengths) for a in stage2.ARMS for s in seeds]
    watch = RSSWatch()
    watch.start()
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    partial = out / "runs.partial.jsonl"
    partial.write_text("")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            rows.append(row)
            with partial.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            # Progress only: no accuracy/outcome is printed before all jobs complete.
            print(f"[{i}/{len(jobs)}] arm={row['arm']} seed={row['seed']} status={row['status']} "
                  f"wall={row.get('wall_seconds', 0):.0f}s elapsed={time.time() - t0:.0f}s "
                  f"rss_peak={watch.peak_total / 2**30:.2f}GiB", flush=True)
    watch.sample()
    watch.stop()
    wall = time.time() - t0
    rows.sort(key=lambda r: (stage2.ARMS.index(r["arm"]), r["seed"]))
    errors = [r for r in rows if r["status"] != "ok"]
    run_meta = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall,
        "workers": args.workers,
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
    (out / "runs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    partial.unlink()
    summary = {"provenance": prov, "run": run_meta}
    if errors:
        summary["verdict"] = "INVALID_OR_MECHANISM_FAIL: incomplete (job errors) " + ", ".join(f"{r['arm']}/{r['seed']}" for r in errors)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(summary["verdict"])
        sys.exit(2)
    acc = stage2_stats.acc_tensor(rows, seeds)
    boot = stage2_stats.bootstrap_all(acc, n_boot=n_boot)
    preflight_ok = {
        "provenance_before_execution": (out / "provenance.json").exists(),
        "params_match_spec": prov["params_match_spec"],
        "spec_hash_unchanged": prov["spec_hash_unchanged"] and file_sha256(SPEC) == stage2.SPEC_SHA256,
    }
    sc = stage2_stats.score(rows, boot, preflight_ok, seeds=seeds, phase_steps=lengths["phase_steps"],
                            eval_scored=lengths["eval_len"] - lengths["eval_warmup"])
    summary.update({"bootstrap": boot, "scoring": sc, "verdict": sc["verdict"], "verdict_labels": sc["verdict_labels"],
                    "accuracy_table": {a: {c: [r["accuracy"][c] for r in rows if r["arm"] == a] for c in stage2.CHECKPOINTS}
                                       for a in stage2.ARMS}})
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    write_report(out, summary)
    manifest = {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in manifest.items()))
    print("VERDICT", sc["verdict"])
    print(json.dumps({k: v["pass"] for k, v in sc["gates"].items()}))


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _f(x, nd=4):
    return "undefined" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(out: Path, s: dict):
    from snn.stage2 import ARMS, ARM_NAMES, CHECKPOINTS

    sc = s["scoring"]
    B = {b["id"]: b for b in s["bootstrap"]}
    prov, run = s["provenance"], s["run"]
    ok = lambda b: "PASS" if b else "FAIL"
    L = [
        "# Stage 2 result: continual hidden-structure learning",
        "",
        f"Spec: `docs/STAGE2_SPEC.md` frozen at `{prov['spec_commit_frozen']}`; SHA-256 at run `{prov['spec_sha256_at_run']}` (unchanged: {prov['spec_hash_unchanged']})",
        f"Code commit: `{prov['code_commit']}` (scientific files clean: {prov['scientific_files_clean']})",
        f"Command: `{prov['argv_display']}`",
        f"Host: {prov['host']['hostname']} ({prov['host']['processor']}, {prov['host']['cpu_count']} logical CPUs); Python {prov['python'].split()[0]}, NumPy {prov['numpy']}, torch {prov['torch']}",
        f"Maintained tests: {prov['maintained_tests']}",
        f"Wall time: {run['wall_seconds']:.0f} s ({run['wall_seconds'] / 3600:.2f} h) with {run['workers']} workers; peak RSS (coordinator+workers) {run['peak_rss_total_bytes'] / 2**30:.2f} GiB; "
        f"interruptions {run['interruptions']}; retries {run['retries']}; rows ok {run['rows_ok']}/{run['jobs']}",
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
    L += [
        "",
        "## Mean held-out accuracy (20 seeds; chance 0.50; theoretical optimum 0.90)",
        "",
        "| Arm | " + " | ".join(CHECKPOINTS) + " |",
        "|---|" + "---:|" * len(CHECKPOINTS),
    ]
    for ai, a in enumerate(ARMS):
        cells = []
        for ci, c in enumerate(CHECKPOINTS):
            ci95 = B[100 + 4 * ai + ci]["two_sided_95"]
            cells.append(f"{m[a][c]:.4f} [{ci95[0]:.4f}, {ci95[1]:.4f}]")
        L.append(f"| {a}: {ARM_NAMES[a]} | " + " | ".join(cells) + " |")
    g3 = sc["gates"]["gate3"]
    L += [
        "",
        "## Gate quantities (bootstrap, 100,000 seed resamples, inverted-CDF percentiles)",
        "",
        "| ID | Metric | Point | One-sided lower 95 | One-sided upper 95 | Two-sided 95 |",
        "|---:|---|---:|---:|---:|---|",
    ]
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
        "No oracle: the target enters only as the next observation and the sign of the reward for an already-emitted output spike.",
    ]
    (out / "STAGE2_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
