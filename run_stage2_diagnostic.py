#!/usr/bin/env python3
"""Stage 2 D1-D4 diagnostic runner.

The implementation card may use only ``--shakedown`` or ``--ss-probe``.
``--run-diagnostic`` is fail-closed behind a separate Nora approval record tied
to the exact pushed implementation commit. Progress output is outcome-free.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from run_stage2 import RSS_LIMIT, RSSWatch, _jsonable

ROOT = Path(__file__).resolve().parent
DIAG_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_SPEC.md"
STAGE2_SPEC = ROOT / "docs" / "STAGE2_SPEC.md"
SCI_FILES = (
    "snn/core.py",
    "snn/stage2.py",
    "snn/stage2_r3.py",
    "snn/stage2_diagnostic.py",
    "snn/stage2_diagnostic_stats.py",
    "run_stage2_diagnostic.py",
    "tools/stage2_diagnostic_ss_probe.py",
    "tests/test_stage2_diagnostic.py",
    "docs/STAGE2_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_IMPLEMENTATION_NOTES.md",
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shakedown", action="store_true")
    mode.add_argument("--ss-probe", action="store_true")
    mode.add_argument("--run-diagnostic", action="store_true")
    p.add_argument("--approval-file", help="separate implementation-review APPROVE record (full run only)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", default="results_stage2_diagnostic")
    p.add_argument("--tests", default="not recorded")
    return p


def _approval(path: str, head: str, spec_sha: str) -> Dict[str, object]:
    if not path:
        raise SystemExit("refusing diagnostic: --approval-file is required")
    record = json.loads(Path(path).read_text())
    required = {
        "decision": "APPROVE",
        "implementation_commit": head,
        "diagnostic_spec_sha256": spec_sha,
        "reviewer": "nora",
    }
    mismatches = {k: (record.get(k), v) for k, v in required.items() if record.get(k) != v}
    if mismatches:
        raise SystemExit(f"refusing diagnostic: approval record mismatch {mismatches}")
    return record


def verified_probe() -> Dict[str, object]:
    path = ROOT / "results_stage2_diagnostic" / "ss_probe.json"
    manifest_path = path.parent / "SHA256SUMS"
    if not path.exists() or not manifest_path.exists():
        return {"pass": False, "reason": "missing retained ss_probe output or manifest"}
    record = json.loads(path.read_text())
    code_path = ROOT / "tools" / "stage2_diagnostic_ss_probe.py"
    output_sha = file_sha(path)
    code_sha = file_sha(code_path)
    manifest_text = manifest_path.read_text()
    passed = bool(
        record.get("decision") == "PASS"
        and record.get("numpy_version") == np.__version__
        and record.get("probe_code_sha256") == code_sha
        and f"{code_sha}  tools/stage2_diagnostic_ss_probe.py" in manifest_text
        and f"{output_sha}  ss_probe.json" in manifest_text
        and not record.get("new_entropy_collision_groups")
        and not record.get("new_state_collision_groups")
        and not record.get("new_vs_legacy_entropy_collisions")
        and not record.get("new_vs_legacy_state_collisions")
    )
    return {
        "pass": passed,
        "decision": record.get("decision"),
        "numpy_version": record.get("numpy_version"),
        "counts": record.get("counts"),
        "known_legacy_only_duplicate_count": record.get("known_legacy_only_duplicate_count"),
        "identity_state_sha256": record.get("identity_state_sha256"),
        "output_sha256": output_sha,
        "code_sha256": code_sha,
    }


def preflight(args: argparse.Namespace, mode: str, out: Path) -> Dict[str, object]:
    from snn import stage2_diagnostic as diag

    if not 1 <= args.workers <= 5:
        raise SystemExit("workers must be 1..5")
    head = git("rev-parse", "HEAD")
    try:
        upstream = git("rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = None
    dirty = git("status", "--porcelain", "--", *SCI_FILES)
    diag_sha = file_sha(DIAG_SPEC)
    stage2_sha = file_sha(STAGE2_SPEC)
    probe = verified_probe()
    approval = _approval(args.approval_file, head, diag_sha) if mode == "diagnostic" else None
    prov: Dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "code_commit": head,
        "upstream_tracking_ref_sha": upstream,
        "code_commit_equals_upstream": head == upstream,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "diagnostic_spec_commit": diag.SPEC_COMMIT,
        "diagnostic_spec_sha256_frozen": diag.SPEC_SHA256,
        "diagnostic_spec_sha256_at_run": diag_sha,
        "diagnostic_spec_unchanged": diag_sha == diag.SPEC_SHA256,
        "stage2_spec_sha256_frozen": diag.FROZEN_STAGE2_SHA256,
        "stage2_spec_sha256_at_run": stage2_sha,
        "stage2_spec_unchanged": stage2_sha == diag.FROZEN_STAGE2_SHA256,
        "scientific_files": list(SCI_FILES),
        "scientific_files_clean": dirty == "",
        "scientific_files_status": dirty,
        "ss_probe": probe,
        "approval": approval,
        "argv": [sys.executable, *sys.argv],
        "argv_display": shlex.join([sys.executable, *sys.argv]),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "python": sys.version,
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "sklearn": __import__("sklearn").__version__,
        "workers": args.workers,
        "seeds": list(diag.DIAGNOSTIC_SEEDS) if mode == "diagnostic" else [4242],
        "maintained_tests": args.tests,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True, default=_jsonable) + "\n")
    if mode == "diagnostic":
        checks = (
            prov["code_commit_equals_upstream"],
            prov["diagnostic_spec_unchanged"],
            prov["stage2_spec_unchanged"],
            prov["scientific_files_clean"],
            probe["pass"],
            approval is not None,
        )
        if not all(checks):
            raise SystemExit(
                "diagnostic preflight failed: pushed={0} diag_spec={1} stage2_spec={2} clean={3} ss_probe={4} approval={5}".format(*checks)
            )
    return prov


def _d1_job(seed: int) -> Dict[str, object]:
    from snn.stage2_diagnostic import run_d1_seed

    try:
        row = run_d1_seed(seed)
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:
        return {"diagnostic": "D1", "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


def _condition_job(job: Tuple[int, str, bool, bool, Mapping[str, object]]) -> Dict[str, object]:
    from snn.stage2_diagnostic import run_condition_seed

    seed, task, plastic, drive, expected = job
    condition = ("P" if plastic else "F0") + "_" + task + ("+drive" if drive else "-no")
    try:
        row = run_condition_seed(seed, task, plastic, drive, expected_hashes=expected)
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:
        return {"condition": condition, "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


def run_pool(fn, jobs: Sequence[object], workers: int, stage: str, out: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Complete a whole predeclared stage before any outcome inspection."""
    watch = RSSWatch()
    watch.start()
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rows: List[Dict[str, object]] = []
    partial = out / f"{stage}.partial.jsonl"
    partial.write_text("")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            with partial.open("a") as handle:
                handle.write(json.dumps(row, default=_jsonable) + "\n")
            print(
                f"[{stage} {i}/{len(jobs)}] seed={row.get('seed')} condition={row.get('condition', 'D1')} "
                f"status={row.get('status')} elapsed={time.time() - t0:.0f}s rss_peak={watch.peak_total / 2**30:.2f}GiB",
                flush=True,
            )
            if watch.exceeded:
                raise RuntimeError("RSS exceeded 8 GiB; aborting rather than swapping")
    watch.sample()
    watch.stop()
    partial.unlink()
    meta = {
        "stage": stage,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - t0,
        "workers": workers,
        "jobs": len(jobs),
        "rows_ok": sum(r.get("status") == "ok" for r in rows),
        "rows_error": sum(r.get("status") != "ok" for r in rows),
        "peak_rss_total_bytes": watch.peak_total,
        "peak_rss_single_process_bytes": watch.peak_single,
        "max_worker_processes_observed": watch.max_children,
        "rss_limit_bytes": RSS_LIMIT,
        "rss_exceeded": watch.exceeded,
        "interruptions": 0,
        "retries": 0,
    }
    return rows, meta


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True, default=_jsonable) + "\n" for r in rows))


def manifest(out: Path) -> None:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{file_sha(p)}  {p.relative_to(out)}\n" for p in files))


def branch_outcome(d1: str, d3: str, d4: Mapping[str, object]) -> str:
    if d1 in ("INVALID", "INVALID_HARNESS") or d3 == "INVALID" or d4["status"] == "INVALID":
        return "STOP_INVALID"
    if d1 != "PASS":
        return f"STOP_D1_{d1}"
    if d3 != "PASS":
        return f"STOP_D3_{d3}"
    if d4["silence_prevention"] != "PASS":
        return "STOP_D4_SILENCE_PREVENTION_FAIL"
    if d4["competence"] == "FAIL":
        return "STOP_D4_COMPETENCE_FAIL"
    if d4["competence"] == "INCONCLUSIVE":
        return "STOP_D4_COMPETENCE_INCONCLUSIVE"
    if d4["competence"] == "PASS":
        return "CONSIDER_DOC_ONLY_R4_SPEC"
    return "STOP_UNCLASSIFIED"


def run_shakedown(out: Path) -> None:
    """Short deterministic construction path on one non-diagnostic seed."""
    from snn import stage2_diagnostic as diag

    seed = 4242
    w1, w2 = diag.initial_weights(seed)
    net = diag.build_network(w1, w2)
    diag.init_state(net)
    x = diag.a_stream(seed, True, 2_000)
    rows = diag.train_condition(net, x, x, True, "fixture_P_A-no", seed)
    ev_x = diag.a_stream(seed, False, 300)
    ev = diag.evaluate_task(net, ev_x, ev_x, diag.tie_coin(seed, "A", 300), warmup=100, source_offset=0)
    result = {
        "mode": "shakedown",
        "seed": seed,
        "full_diagnostic_seed": False,
        "checkpoint_rows": len(rows),
        "checkpoint_schema": "PASS",
        "evaluation_scored": ev["scored"],
        "evaluation_nonmutating": ev["weights_bitwise_constant"],
    }
    (out / "shakedown.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest(out)
    print(json.dumps(result, sort_keys=True))


def run_diagnostic(args: argparse.Namespace, out: Path, provenance: Mapping[str, object]) -> None:
    from snn import stage2_diagnostic as diag
    from snn import stage2_diagnostic_stats as dstat

    seeds = diag.DIAGNOSTIC_SEEDS
    all_meta: List[Dict[str, object]] = []

    def condition_jobs(task: str, plastics: Sequence[bool], drive: bool):
        # Construct and hash every shared object in the coordinator before a
        # worker executes a network transition. Each worker reconstructs and
        # asserts the same hash set before training.
        expected = {seed: diag.condition_object_hashes(seed, task, drive) for seed in seeds}
        return [(seed, task, plastic, drive, expected[seed]) for plastic in plastics for seed in seeds]

    d1_rows, meta = run_pool(_d1_job, list(seeds), args.workers, "D1", out)
    all_meta.append(meta)
    d1_rows.sort(key=lambda r: r["seed"])
    write_jsonl(out / "d1_rows.jsonl", d1_rows)
    if any(r.get("status") != "ok" for r in d1_rows):
        raise SystemExit("INVALID: D1 job error; downstream diagnostics not run")
    d1 = diag.summarize_d1(d1_rows)
    (out / "d1_summary.json").write_text(json.dumps(d1, indent=2, sort_keys=True) + "\n")
    if d1["status"] != "PASS":
        manifest(out)
        raise SystemExit(f"D1 {d1['status']}: stop sequence")

    d2_jobs = condition_jobs("A", (True, False), False)
    d2_rows, meta = run_pool(_condition_job, d2_jobs, args.workers, "D2_A_no_drive", out)
    all_meta.append(meta)
    if any(r.get("status") != "ok" for r in d2_rows):
        raise SystemExit("INVALID: D2 job error")
    d2_rows.sort(key=lambda r: (r["condition"], r["seed"]))
    write_jsonl(out / "d2_rows.jsonl", d2_rows)
    d2_process = dstat.validate_d2_package(d2_rows)
    onset_rows = []
    for seed in seeds:
        p = next(r for r in d2_rows if r["condition"] == "P_A-no" and r["seed"] == seed)
        f = next(r for r in d2_rows if r["condition"] == "F0_A-no" and r["seed"] == seed)
        onset = diag.onset_record(p["checkpoints"], f["checkpoints"])
        onset.update({"seed": seed})
        onset_rows.append(onset)
    d2 = dstat.summarize_d2(onset_rows)
    d2["process_integrity"] = d2_process

    d3_jobs = condition_jobs("lag1", (True, False), False)
    d3_rows, meta = run_pool(_condition_job, d3_jobs, args.workers, "D3_lag1_no_drive", out)
    all_meta.append(meta)
    if any(r.get("status") != "ok" for r in d3_rows):
        raise SystemExit("INVALID: D3 job error")
    d3_rows.sort(key=lambda r: (r["condition"], r["seed"]))
    write_jsonl(out / "d3_rows.jsonl", d3_rows)
    pre_drive_rows = d2_rows + d3_rows
    pre_boot = dstat.bootstrap_all(pre_drive_rows, order=(0, 1, 7, 8))
    d3 = dstat.summarize_d3(d3_rows, pre_boot)
    if d3["status"] == "INVALID":
        manifest(out)
        raise SystemExit("D3 INVALID: stop sequence")

    drive_jobs = condition_jobs("A", (True, False), True) + condition_jobs("lag1", (True,), True)
    drive_rows, meta = run_pool(_condition_job, drive_jobs, args.workers, "D4_drive", out)
    all_meta.append(meta)
    if any(r.get("status") != "ok" for r in drive_rows):
        raise SystemExit("INVALID: D4 job error")
    drive_rows.sort(key=lambda r: (r["condition"], r["seed"]))
    write_jsonl(out / "d4_drive_rows.jsonl", drive_rows)

    all_rows = d2_rows + d3_rows + drive_rows
    boot = dstat.bootstrap_all(all_rows)
    d4 = dstat.summarize_d4(all_rows, boot)
    lag_table = dstat.complete_seed_table(all_rows, ("P_lag1+drive", "P_lag1-no"))
    lag_context = {
        "median_P_lag1_drive": float(np.median(lag_table["P_lag1+drive"])),
        "median_P_lag1_no": float(np.median(lag_table["P_lag1-no"])),
        "median_drive_minus_no": next(b["point_estimate"] for b in boot if b["id"] == 5),
        "two_sided_95": next(b["two_sided_95"] for b in boot if b["id"] == 5),
    }
    summary = {
        "provenance": provenance,
        "runs": all_meta,
        "D1": d1,
        "D2": d2,
        "D2_onsets": onset_rows,
        "D3": d3,
        "D4": d4,
        "D4_lag1_report_only": lag_context,
        "bootstrap": boot,
        "branch_outcome": branch_outcome(str(d1["status"]), str(d3["status"]), d4),
        "no_oracle": (
            "Targets grade already-emitted spikes and label D1 offline features only; they never enter network state. "
            "D4 drive is task-independent, and no LLM logic enters the network, reward, decoder feature path, or simulator."
        ),
    }
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    manifest(out)
    print(f"PACKAGE COMPLETE branch_outcome={summary['branch_outcome']}")


def main() -> None:
    args = parser().parse_args()
    out = Path(args.out)
    if args.ss_probe:
        command = [sys.executable, str(ROOT / "tools" / "stage2_diagnostic_ss_probe.py"), "--out", str(out / "ss_probe.json")]
        subprocess.run(command, cwd=ROOT, check=True)
        return
    mode = "diagnostic" if args.run_diagnostic else "shakedown"
    provenance = preflight(args, mode, out)
    if args.shakedown:
        run_shakedown(out)
    else:
        run_diagnostic(args, out, provenance)


if __name__ == "__main__":
    main()
