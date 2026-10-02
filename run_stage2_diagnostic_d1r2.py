#!/usr/bin/env python3
"""Stage 2 diagnostic D1R2 runner (docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md at f3d52ee).

The implementation card may use only ``--shakedown`` (non-diagnostic fixture
seed 4242: pipeline decoder scored for section 11 item 15, the two network
decoders fit for convergence only for item 16, never predicted or scored) or
``--ss-probe`` (section 3.3, shakedown execution: never writes a package). ``--run-d1r2`` is fail-closed behind a separate Nora approval
record tied to the exact pushed implementation commit and a separate run
authorization.

Fail-closed package contract (carried from the approved ebb90e74 runner):
fresh exclusive run directory; outcome-free progress ledger while the stage
runs; seed rows held in coordinator memory until all 20 are present; strict
row/package validators with coordinator-derived label hashes before any
readout; one package finalizer for every branch, including exceptions; a
SHA-256 manifest over declared artifacts only.

The approved ebb90e74 runner (run_stage2_diagnostic.py) is imported for its
package helpers and is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_stage2_diagnostic as base
from run_stage2 import RSSWatch, _jsonable
from run_stage2_diagnostic import (
    PROGRESS_FIELDS,
    StaleOutputError,
    claim_fresh_run_dir,
    file_sha,
    git,
    manifest,
    package_files,
    parameter_mismatches,
    progress_record,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent
D1R2_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_D1R2_SPEC.md"
DIAG_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_SPEC.md"
STAGE2_SPEC = ROOT / "docs" / "STAGE2_SPEC.md"
RESULTS = ROOT / "results_stage2_diagnostic_d1r2"
SCI_FILES = (
    "snn/core.py", "snn/stage2.py", "snn/stage2_r3.py", "snn/stage2_diagnostic.py",
    "snn/stage2_diagnostic_stats.py", "snn/stage2_diagnostic_d1r.py", "snn/stage2_diagnostic_d1r2.py",
    "run_stage2.py", "run_stage2_diagnostic.py", "run_stage2_diagnostic_d1r.py", "run_stage2_diagnostic_d1r2.py",
    "tools/stage2_diagnostic_ss_probe.py", "tools/stage2_diagnostic_d1r_ss_probe.py", "tools/stage2_diagnostic_d1r2_ss_probe.py",
    "tests/test_stage2_diagnostic_d1r.py", "tests/test_stage2_diagnostic_d1r2.py",
    "docs/STAGE2_SPEC.md", "docs/STAGE2_DIAGNOSTIC_SPEC.md", "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md", "docs/STAGE2_DIAGNOSTIC_D1R_IMPLEMENTATION_NOTES.md",
    "docs/STAGE2_DIAGNOSTIC_D1R2_IMPLEMENTATION_NOTES.md",
)
STAGE = "D1R2"

NO_ORACLE = (
    "The target is used only as an offline decoder label after feature capture. It is never a network input, "
    "hidden-state label, plasticity feature, state reset, initialization signal, or internal policy or world-model "
    "component. No reward is computed and no weight changes. No LLM logic enters the network, the feature path, "
    "the decoder, or the simulator."
)
CLAIM_BOUNDARY = (
    "Passing D1R2 does not pass Stage 2. D1R2 is only linear decodability under one frozen spike-only feature map, "
    "one split, and one decoder. REPRESENTATION_LIMITED is a bounded operational label: the strongest spike-only "
    "linear readout this spec permits did not clear the unchanged bar; it is not a proof that no readout could. "
    "None of D1R2's outcomes establishes general learning, continual learning, developmental progress, biological "
    "equivalence, or an r4 result, and none changes a Stage 2 gate, control, validity rule, or the Stage 1 rule."
)
PIPELINE_CLAIM = (
    "Pipeline: under the fixed split, the pipeline-control map, preprocessing, and L2 logistic decoder, x_(t-1) was "
    "or was not recovered from the input-layer spike encoding at the D1R2 timing (validity of the code path only)."
)
NETWORK_CLAIM = (
    "Network positive control and A decoder: under the fixed split, the single D1R2 exact-lag spike feature map, "
    "preprocessing, and L2 logistic decoder, x_(t-1) (respectively the noisy A target x_t) was or was not linearly "
    "decodable from the hidden spikes of the frozen random r3 network."
)
NOT_SUPPORTED = (
    "the network can or cannot represent A at all",
    "any nonlinear-decoder, voltage-state, trained-weight, architecture-wide, or learning-rule claim",
    "passing Stage 2, re-scoring r2/r3, or reinterpreting the D1 results on seeds 2000..2019",
    "general or continual learning, developmental progress, or biological equivalence",
    "any r4 result, D2-D4 result, retuning, or post-data change to K, the map, or the decoder",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shakedown", action="store_true", help="seed 4242 only, pipeline decoder only, full lengths")
    mode.add_argument("--ss-probe", action="store_true", help="section 3.3 collision probe")
    mode.add_argument("--run-d1r2", action="store_true", help="locked: requires a Nora approval and run authorization")
    p.add_argument("--approval-file", help="separate implementation-review APPROVE + run authorization record")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None, help="package directory; must not exist")
    p.add_argument("--tests", default="not recorded")
    return p


# ------------------------------------------------------- effective parameters
def expected_parameters_from_spec() -> Dict[str, object]:
    """Transcribed by hand from docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md sections 2, 3.1, 5, 6, 10.

    Deliberately independent of snn/stage2_diagnostic_d1r2.py constants; the
    runner refuses a D1R2 run on any difference from the constructed objects.
    """
    import math

    return {
        "layer_sizes": [2, 20, 2],
        "dt_ms": 1.0,
        "tau_m_ms": [20.0, 20.0],
        "v_rest_mv": [-70.0, -70.0],
        "v_reset_mv": [-70.0, -70.0],
        "v_thresh_mv": [-54.0, -54.0],
        "hidden_tau_a_ms": 200.0,
        "hidden_beta_a_mv": 1.12,
        "noise_p": 0.10,
        "train_stream_len": 200_000,
        "test_stream_len": 12_000,
        "warmup_rows": 2_000,
        "train_rows": 198_000,
        "test_rows": 10_000,
        "seeds": list(range(2200, 2220)),
        "namespace": 32,
        "root_entropy": 20261001,
        "K": 19,
        "network_feature_dim": 460,
        "pipeline_feature_dim": 46,
        "trace20_decay": math.exp(-1 / 20),
        "trace25_decay": math.exp(-1 / 25),
        "count_window_steps": 20,
        "decoder": {"penalty": "l2", "C": 1.0, "fit_intercept": True, "solver": "lbfgs", "tol": 1e-8, "max_iter": 20000, "class_weight": None},
        "prediction_threshold": 0.5,
        "block_length": 100,
        "block_resamples": 10_000,
        "block_percentile": 5,
        "decoders": ["pipeline", "network_positive", "A"],
        "max_workers": 3,
        "rss_limit_bytes": 12 * 1024**3,
        "wall_limit_seconds": 6 * 3600,
        "thresholds": {
            "pass_median_accuracy": 0.70,
            "pass_min_lower_bounds_gt_half": 15,
            "lower_bound_reference": 0.50,
            "fail_median_below": 0.55,
            "pipeline_pass_median_accuracy": 0.99,
        },
    }


def validated_effective_parameters() -> Dict[str, object]:
    from snn import stage2_diagnostic_d1r2 as d1r2

    effective = d1r2.effective_parameters()
    mismatches = parameter_mismatches(effective, expected_parameters_from_spec())
    return {"effective": effective, "matches_spec": not mismatches, "mismatches": mismatches}


# ------------------------------------------------------------- preflight
def _approval(path: Optional[str], head: str, spec_sha: str) -> Dict[str, object]:
    if not path:
        raise SystemExit("refusing D1R2 run: --approval-file is required")
    record = json.loads(Path(path).read_text())
    required = {
        "decision": "APPROVE",
        "reviewer": "nora",
        "implementation_commit": head,
        "d1r2_spec_sha256": spec_sha,
        "run_authorized": True,
    }
    mismatches = {k: (record.get(k), v) for k, v in required.items() if record.get(k) != v}
    if not isinstance(record.get("run_authorization_card"), str) or not record.get("run_authorization_card"):
        mismatches["run_authorization_card"] = (record.get("run_authorization_card"), "<separate run card id>")
    if mismatches:
        raise SystemExit(f"refusing D1R2 run: approval record mismatch {mismatches}")
    return record


class ShakedownProbeFailure(RuntimeError):
    """Section 11 item 12 failed in the shakedown: implementation failure, no package or row."""


def load_probe_module():
    import importlib.util

    path = ROOT / "tools" / "stage2_diagnostic_d1r2_ss_probe.py"
    spec = importlib.util.spec_from_file_location("stage2_diagnostic_d1r2_ss_probe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PROBE_COLLISION_KEYS = ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions")


def shakedown_collision_check(out_json: Path, **probe_overrides) -> Dict[str, object]:
    """Section 8 step 1 / section 11 item 12: the shakedown execution of the probe.

    On PASS the output and manifest are retained. On any failure nothing is
    written (no D1R2 package and no row) and ShakedownProbeFailure is raised;
    the preflight finalizer never applies here.
    """
    mod = load_probe_module()
    result = mod.probe(**probe_overrides)
    if result["decision"] != "PASS":
        raise ShakedownProbeFailure(json.dumps({k: result[k] for k in ("decision", "reproduces_spec_section_3_3", "identity_state_sha256", *PROBE_COLLISION_KEYS)}, default=str))
    return mod.write_retained(Path(out_json), result)


def run_preflight_probe(**probe_overrides) -> Dict[str, object]:
    """Section 8 step 2: the coordinator re-executes the committed probe before any seed.

    Returns the provenance record; ``pass`` is false on any collision involving
    a D1R2 identity or any mismatch with section 3.3, in which case run_d1r2
    writes the section 6.3 preflight-finalizer package.
    """
    import hashlib

    mod = load_probe_module()
    result = mod.probe(**probe_overrides)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    record = {
        "execution": "run_preflight",
        "pass": result["decision"] == "PASS",
        "decision": result["decision"],
        "reproduces_spec_section_3_3": result["reproduces_spec_section_3_3"],
        "counts": result["counts"],
        "legacy_only_duplicate_count": result["legacy_only_duplicate_count"],
        "legacy_only_entropy_duplicate_groups": result["legacy_only_entropy_duplicate_groups"],
        "legacy_only_state_duplicate_groups": result["legacy_only_state_duplicate_groups"],
        "seed_ranges_disjoint": result["seed_ranges_disjoint"],
        "identity_state_sha256": result["identity_state_sha256"],
        "numpy_version": result["numpy_version"],
        "code_sha256": file_sha(Path(mod.__file__)),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "synthetic_override": bool(probe_overrides),
    }
    record.update({k: result[k] for k in PROBE_COLLISION_KEYS})
    return record


def probe_failure_summary(record: Mapping[str, object]) -> str:
    counts = [len(record.get(k) or []) for k in PROBE_COLLISION_KEYS]  # type: ignore[arg-type]
    return f"run-preflight collision probe failed: decision={record.get('decision')} reproduces_3_3={record.get('reproduces_spec_section_3_3')} new_collisions={counts}"


def verified_probe() -> Dict[str, object]:
    """The retained section 3.3 probe, verified against its manifest and frozen counts."""
    path = RESULTS / "ss_probe.json"
    manifest_path = RESULTS / "SHA256SUMS"
    code_path = ROOT / "tools" / "stage2_diagnostic_d1r2_ss_probe.py"
    if not path.exists() or not manifest_path.exists():
        return {"pass": False, "reason": "missing retained D1R2 ss_probe output or manifest"}
    record = json.loads(path.read_text())
    output_sha, code_sha = file_sha(path), file_sha(code_path)
    text = manifest_path.read_text()
    passed = bool(
        record.get("decision") == "PASS"
        and record.get("reproduces_spec_section_3_3") is True
        and record.get("numpy_version") == np.__version__
        and record.get("probe_code_sha256") == code_sha
        and f"{code_sha}  tools/stage2_diagnostic_d1r2_ss_probe.py" in text
        and f"{output_sha}  ss_probe.json" in text
        and not record.get("new_entropy_collision_groups")
        and not record.get("new_state_collision_groups")
        and not record.get("new_vs_legacy_entropy_collisions")
        and not record.get("new_vs_legacy_state_collisions")
        and record.get("seed_ranges_disjoint") is True
    )
    return {
        "pass": passed,
        "decision": record.get("decision"),
        "numpy_version": record.get("numpy_version"),
        "counts": record.get("counts"),
        "legacy_only_duplicate_count": record.get("legacy_only_duplicate_count"),
        "identity_state_sha256": record.get("identity_state_sha256"),
        "output_sha256": output_sha,
        "code_sha256": code_sha,
    }


def preflight(args: argparse.Namespace, mode: str, out: Path) -> Dict[str, object]:
    from snn import stage2_diagnostic_d1r2 as d1r2

    if not 1 <= args.workers <= d1r2.MAX_WORKERS:
        raise SystemExit(f"workers must be 1..{d1r2.MAX_WORKERS} (spec section 10)")
    head = git("rev-parse", "HEAD")
    try:
        upstream: Optional[str] = git("rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = None
    dirty = git("status", "--porcelain", "--", *SCI_FILES)
    d1r2_sha, diag_sha, stage2_sha = file_sha(D1R2_SPEC), file_sha(DIAG_SPEC), file_sha(STAGE2_SPEC)
    approval = _approval(args.approval_file, head, d1r2_sha) if mode == "d1r2" else None
    # Section 8 step 2: the run preflight re-executes the committed probe at the
    # run's code hash before any seed; the shakedown uses the retained output.
    probe = run_preflight_probe() if mode == "d1r2" else verified_probe()
    params = validated_effective_parameters()
    prov: Dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "code_commit": head,
        "upstream_tracking_ref_sha": upstream,
        "code_commit_equals_upstream": head == upstream,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "d1r2_spec_commit": d1r2.SPEC_COMMIT,
        "d1r2_spec_sha256_frozen": d1r2.SPEC_SHA256,
        "d1r2_spec_sha256_at_run": d1r2_sha,
        "d1r2_spec_unchanged": d1r2_sha == d1r2.SPEC_SHA256,
        "ebb90e74_sha256_frozen": d1r2.EBB90E74_SHA256,
        "ebb90e74_sha256_at_run": diag_sha,
        "ebb90e74_unchanged": diag_sha == d1r2.EBB90E74_SHA256,
        "stage2_spec_sha256_frozen": d1r2.FROZEN_STAGE2_SHA256,
        "stage2_spec_sha256_at_run": stage2_sha,
        "stage2_spec_unchanged": stage2_sha == d1r2.FROZEN_STAGE2_SHA256,
        "scientific_files": list(SCI_FILES),
        "scientific_files_clean": dirty == "",
        "scientific_files_status": dirty,
        "collision_probe": probe,
        "approval": approval,
        "effective_parameters": params["effective"],
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "argv": [sys.executable, *sys.argv],
        "argv_display": shlex.join([sys.executable, *sys.argv]),
        "environment": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "VIRTUAL_ENV")},
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "python": sys.version,
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "sklearn": __import__("sklearn").__version__,
        "psutil": __import__("psutil").__version__,
        "workers": args.workers,
        "seeds": list(d1r2.D1R2_SEEDS) if mode == "d1r2" else [d1r2.FIXTURE_SEED],
        "maintained_tests": args.tests,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out),
    }
    claim_fresh_run_dir(out)
    (out / "provenance.json").write_text(_provenance_text(prov))
    if mode == "d1r2":
        checks = (
            prov["code_commit_equals_upstream"],
            prov["d1r2_spec_unchanged"],
            prov["ebb90e74_unchanged"],
            prov["stage2_spec_unchanged"],
            prov["scientific_files_clean"],
            approval is not None,
            params["matches_spec"],
        )
        if not all(checks):
            raise SystemExit(
                "D1R2 preflight failed before collision gate: pushed={0} d1r2_spec={1} ebb90e74={2} stage2_spec={3} clean={4} approval={5} params={6}".format(*checks)
            )
    return prov


def _provenance_text(provenance: Mapping[str, object]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True, default=_jsonable) + "\n"


# ------------------------------------------------------------- job + pool
def _d1r2_job(job: object) -> Dict[str, object]:
    import torch

    from snn.stage2_diagnostic_d1r2 import run_d1r2_seed

    seed, ledger = job if isinstance(job, tuple) else (job, None)
    torch.set_num_threads(1)
    try:
        row = run_d1r2_seed(int(seed), authorized_run=True, ledger_path=None if ledger is None else Path(str(ledger)))
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:
        return {"diagnostic": "D1R2", "seed": int(seed), "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


class D1R2RSSWatch(RSSWatch):
    """RSSWatch at the D1R2 cap: abort above 12 GiB total RSS rather than swap (spec section 10)."""

    def __init__(self, period: float = 2.0):
        from snn.stage2_diagnostic_d1r2 import RSS_LIMIT_BYTES

        super().__init__(limit=RSS_LIMIT_BYTES, period=period)

    def _run(self):
        """Set the abort flag; the coordinator owns termination and package finalization."""
        while not self._stop.is_set():
            self.sample()
            if self.exceeded:
                print(f"ABORT REQUESTED: RSS {self.peak_total / 2**30:.2f} GiB > 12 GiB", flush=True)
                return
            self._stop.wait(self.period)


POOL_POLL_SECONDS = 5.0


class WallLimitExceeded(RuntimeError):
    """Section 10 wall stop."""


class RSSLimitExceeded(RuntimeError):
    """Section 10 RSS abort."""


def terminate_executor(pool: Executor, join_timeout: float = 10.0) -> None:
    """Stop an executor without waiting for running work.

    Queued futures are cancelled so nothing new starts, worker processes of a
    ProcessPoolExecutor are killed (running jobs cannot continue), and the
    executor is shut down with wait=False so the coordinator returns at once.
    """
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pending = getattr(pool, "_pending_work_items", None)
    if isinstance(pending, dict):
        for item in list(pending.values()):
            item.future.cancel()
    for p in procs:
        try:
            if p.is_alive():
                p.kill()
        except (OSError, ValueError, AttributeError):
            pass
    pool.shutdown(wait=False, cancel_futures=True)
    end = time.monotonic() + join_timeout
    for p in procs:
        try:
            p.join(timeout=max(0.0, end - time.monotonic()))
        except (OSError, ValueError, AttributeError, AssertionError):
            pass


def run_pool(
    fn: Callable[[object], Dict[str, object]],
    jobs: Sequence[object],
    workers: int,
    out: Path,
    executor_factory: Callable[[int], Executor] = lambda n: ProcessPoolExecutor(max_workers=n),
    watch: Optional[object] = None,
    wall_limit: Optional[float] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Complete the whole predeclared stage before any outcome is persisted.

    Rows stay in coordinator memory. The only file written during the stage
    is ``D1R2.progress.jsonl`` with exactly ``PROGRESS_FIELDS`` per finished
    job (identity, status, resources). An exception leaves only that ledger.
    """
    from snn.stage2_diagnostic_d1r2 import WALL_LIMIT_SECONDS

    wall_limit = WALL_LIMIT_SECONDS if wall_limit is None else wall_limit
    watch = watch if watch is not None else D1R2RSSWatch()
    watch.start()  # type: ignore[attr-defined]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    deadline = time.monotonic() + wall_limit
    rows: List[Dict[str, object]] = []
    ledger = out / f"{STAGE}.progress.jsonl"
    ledger.open("x").close()
    pool = executor_factory(workers)
    completed_normally = False
    try:
        futures = [pool.submit(fn, job) for job in jobs]
        pending = set(futures)
        while pending:
            # The deadline is enforced independently of any future completing:
            # wait() returns at the deadline even if every job is still running.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WallLimitExceeded(f"wall time exceeded {wall_limit} s (section 10 six-hour stop); stopped and reported, nothing changed")
            done, pending = wait(pending, timeout=min(remaining, POOL_POLL_SECONDS), return_when=FIRST_COMPLETED)
            if watch.exceeded:  # type: ignore[attr-defined]
                raise RSSLimitExceeded("RSS exceeded 12 GiB; aborted rather than swapping")
            for future in done:
                row = future.result()
                rows.append(row)
                rec = progress_record(STAGE, len(rows), len(jobs), row, time.time() - t0, watch.peak_total)  # type: ignore[attr-defined]
                with ledger.open("a") as handle:
                    handle.write(json.dumps(rec, sort_keys=True) + "\n")
                print(" ".join(f"{k}={rec[k]}" for k in PROGRESS_FIELDS), flush=True)
        completed_normally = True
    finally:
        if completed_normally:
            pool.shutdown(wait=True)
        else:
            terminate_executor(pool)
        watch.sample()  # type: ignore[attr-defined]
        watch.stop()  # type: ignore[attr-defined]
    meta = {
        "stage": STAGE,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - t0,
        "workers": workers,
        "jobs": len(jobs),
        "rows_ok": sum(r.get("status") == "ok" for r in rows),
        "rows_error": sum(r.get("status") != "ok" for r in rows),
        "peak_rss_total_bytes": watch.peak_total,  # type: ignore[attr-defined]
        "peak_rss_single_process_bytes": watch.peak_single,  # type: ignore[attr-defined]
        "max_worker_processes_observed": watch.max_children,  # type: ignore[attr-defined]
        "rss_limit_bytes": watch.limit,  # type: ignore[attr-defined]
        "rss_exceeded": watch.exceeded,  # type: ignore[attr-defined]
        "interruptions": 0,
        "retries": 0,
    }
    return rows, meta


# ------------------------------------------------------------- package
def default_run_dir(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    head = git("rev-parse", "--short=12", "HEAD")
    return RESULTS / "runs" / f"{mode}-{stamp}-{head}-pid{os.getpid()}"


def prepare_package_dir(out: Path, provenance: Mapping[str, object]) -> None:
    """Accept only an absent location, or one holding exactly this run's own provenance."""
    if not out.exists() and not out.is_symlink():
        claim_fresh_run_dir(out)
        (out / "provenance.json").write_text(_provenance_text(provenance))
        return
    if out.is_symlink() or not out.is_dir():
        raise StaleOutputError(f"refusing package: {out} is not a fresh directory")
    present = package_files(out)
    if present != ["provenance.json"]:
        raise StaleOutputError(f"refusing package: {out} already holds {present}; packages require a fresh run-unique directory")
    if (out / "provenance.json").read_text() != _provenance_text(provenance):
        raise StaleOutputError(f"refusing package: {out}/provenance.json belongs to a different run")


def write_incomplete_marker(out: Path) -> None:
    """Fail-closed default before the stage starts: a hard kill leaves STOP_D1R2_INVALID on disk."""
    from snn import stage2_diagnostic_d1r2 as d1r2

    record = {**d1r2.branch("INVALID"), "status": None, "incomplete_reason": "sequence has not reached finalize_package"}
    (out / "branch_outcome.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def plain_language(summary: Mapping[str, object]) -> Dict[str, object]:
    code = summary["branch"]["code"]  # type: ignore[index]
    readouts = summary.get("readouts") or {}
    supported: List[str] = []
    if code == "STOP_D1R2_INVALID":
        supported.append("Nothing beyond the invalidity itself: the sequence stopped before any decision-bearing readout could be inspected.")
    elif code == "STOP_D1R2_INVALID_PIPELINE":
        supported.append("The pipeline control did not reach median 0.99, which indicates a code defect. No science reading of the network decoders is made.")
    else:
        supported.append(f"Pipeline PASS: {PIPELINE_CLAIM}")
        positive = readouts.get("network_positive") if isinstance(readouts, Mapping) else None
        a = readouts.get("A") if isinstance(readouts, Mapping) else None
        if positive:
            supported.append(f"Network positive control {positive['status']}: {NETWORK_CLAIM}")
        if code == "CLOSE_D1_REPRESENTATION_LIMITED":
            supported.append("D1 closes as REPRESENTATION_LIMITED: the single permitted spike-only linear readout did not clear the unchanged bar for x_(t-1).")
        elif a:
            supported.append(f"A decoder {a['status']}: {NETWORK_CLAIM}")
    return {
        "status": summary["status"],
        "branch": code,
        "table_row": summary["branch"]["table_row"],  # type: ignore[index]
        "next_action": summary["branch"]["action"],  # type: ignore[index]
        "supported": supported,
        "not_supported": list(NOT_SUPPORTED),
    }


REQUIRED_PACKAGE_FIELDS = (
    "provenance",
    "effective_parameters",
    "effective_parameters_match_spec",
    "collision_probe",
    "run_metadata",
    "status",
    "readouts",
    "branch_outcome",
    "per_seed",
    "package_summary",
    "claim_boundary",
    "no_oracle",
    "artifacts",
)


def finalize_package(
    out: Path,
    provenance: Mapping[str, object],
    d1r2_summary: Optional[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    run_meta: Sequence[Mapping[str, object]],
    artifacts: Sequence[str],
    incomplete_reason: Optional[str] = None,
) -> Dict[str, object]:
    """The single exit for every D1R2 branch (spec sections 6.3, 8, 9, 12)."""
    from snn import stage2_diagnostic_d1r2 as d1r2

    params = validated_effective_parameters()
    if d1r2_summary is None or incomplete_reason is not None:
        d1r2_summary = {
            "status": "INVALID",
            "valid": False,
            "violations": [incomplete_reason or "no D1R2 summary"],
            "readouts": {"pipeline": None, "network_positive": None, "A": None},
            "branch": d1r2.branch("INVALID"),
        }
    per_seed: Dict[str, object] = {}
    if d1r2_summary.get("valid"):
        # Under INVALID_PIPELINE (section 9 row 2) only pipeline fields are packaged.
        names = ("pipeline",) if d1r2_summary["status"] == "INVALID_PIPELINE" else d1r2.DECODERS
        for r in rows:
            decs = r["decoders"]  # type: ignore[index]
            per_seed[str(r["seed"])] = {
                "nesting_ok": r["nesting_ok"],
                "normalization_sha256": {n: decs[n]["normalization_sha256"] for n in names},  # type: ignore[index]
                "coefficients_sha256": {n: decs[n]["coefficients_sha256"] for n in names},  # type: ignore[index]
                "n_iter": {n: decs[n]["n_iter"] for n in names},  # type: ignore[index]
                "converged": {n: decs[n]["converged"] for n in names},  # type: ignore[index]
            }
    summary: Dict[str, object] = {
        "schema_version": 1,
        "provenance": provenance,
        "effective_parameters": params["effective"],
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "collision_probe": provenance.get("collision_probe"),
        "run_metadata": {
            "stages": list(run_meta),
            "workers": provenance.get("workers"),
            "total_wall_seconds": sum(float(m.get("wall_seconds", 0.0)) for m in run_meta),
            "peak_rss_total_bytes": max([int(m.get("peak_rss_total_bytes", 0)) for m in run_meta] or [0]),
            "peak_rss_single_process_bytes": max([int(m.get("peak_rss_single_process_bytes", 0)) for m in run_meta] or [0]),
            "interruptions": sum(int(m.get("interruptions", 0)) for m in run_meta),
            "retries": sum(int(m.get("retries", 0)) for m in run_meta),
            "incomplete_reason": incomplete_reason,
        },
        "status": d1r2_summary["status"],
        "violations": list(d1r2_summary.get("violations", [])),
        "readouts": d1r2_summary["readouts"],
        "branch_outcome": d1r2_summary["branch"],
        "branch": d1r2_summary["branch"],
        "per_seed": per_seed,
        "claim_boundary": CLAIM_BOUNDARY,
        "no_oracle": NO_ORACLE,
        "pipeline_claim": PIPELINE_CLAIM,
        "network_claim": NETWORK_CLAIM,
        "artifacts": sorted(set(artifacts) | {"summary.json", "branch_outcome.json", "provenance.json"}),
    }
    present = set(package_files(out))
    declared = set(summary["artifacts"]) - {"summary.json"}  # type: ignore[arg-type]
    unexpected = sorted(present - declared - {"summary.json", "SHA256SUMS"})
    missing = sorted(declared - present)
    if unexpected or missing:
        reason = f"package directory contents differ from declared artifacts: unexpected={unexpected} missing={missing}"
        summary["status"] = "INVALID"
        summary["readouts"] = {"pipeline": None, "network_positive": None, "A": None}
        summary["branch_outcome"] = summary["branch"] = d1r2.branch("INVALID")
        summary["per_seed"] = {}
        summary["run_metadata"]["incomplete_reason"] = reason  # type: ignore[index]
        summary["unexpected_artifacts"] = unexpected
        summary["missing_artifacts"] = missing
        summary["artifacts"] = sorted((declared - set(missing)) | {"summary.json"})
    summary["package_summary"] = plain_language(summary)
    absent = [k for k in REQUIRED_PACKAGE_FIELDS if k not in summary]
    if absent or summary["branch"]["code"] not in d1r2.BRANCH_INDEX:  # type: ignore[index]
        raise AssertionError(f"package summary incomplete: {absent}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    (out / "branch_outcome.json").write_text(json.dumps({"status": summary["status"], **summary["branch"]}, indent=2, sort_keys=True) + "\n")  # type: ignore[arg-type]
    manifest(out, summary["artifacts"])  # type: ignore[arg-type]
    print(f"PACKAGE FROZEN d1r2_status={summary['status']} branch={summary['branch']['code']} table_row={summary['branch']['table_row']}", flush=True)  # type: ignore[index]
    return summary


NETWORK_OUTCOME_FIELDS = ("accuracy", "lower_95", "coefficients", "intercept", "coefficients_sha256")


def redact_network_outcomes(row: Mapping[str, object]) -> Dict[str, object]:
    """Copy of a row with every network-decoder outcome field removed (pipeline kept)."""
    import copy

    out = copy.deepcopy(dict(row))
    for name in ("network_positive", "A"):
        dec = out.get("decoders", {}).get(name)  # type: ignore[union-attr]
        if isinstance(dec, dict):
            for k in NETWORK_OUTCOME_FIELDS:
                dec.pop(k, None)
    return out


Executer = Callable[[Callable[..., Dict[str, object]], Sequence[object]], Tuple[List[Dict[str, object]], Dict[str, object]]]


def run_d1r2(
    args: argparse.Namespace,
    out: Path,
    provenance: Mapping[str, object],
    execute: Optional[Executer] = None,
    seeds: Optional[Sequence[int]] = None,
    identity: Optional[Callable[[int], Mapping[str, str]]] = None,
) -> Dict[str, object]:
    """Section 8 run path, including total preflight and termination finalizers."""
    from snn import stage2_diagnostic_d1r2 as d1r2

    prepare_package_dir(out, provenance)
    seeds = tuple(d1r2.D1R2_SEEDS if seeds is None else seeds)
    identity = identity or d1r2.object_hashes
    artifacts: List[str] = []
    run_meta: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    event_ledger = out / f"{STAGE}.events.jsonl"

    progress_ledger = out / f"{STAGE}.progress.jsonl"

    def _declare_run_ledgers() -> None:
        # The two run-owned ledgers are declared artifacts whenever they exist.
        for path in (progress_ledger, event_ledger):
            if path.exists() and path.name not in artifacts:
                artifacts.append(path.name)

    def _default_execute(fn, jobs):
        return run_pool(fn, [(job, str(event_ledger)) for job in jobs], args.workers, out)

    run_stage: Executer = execute or _default_execute  # type: ignore[assignment]
    probe_pass = bool((provenance.get("collision_probe") or {}).get("pass"))  # type: ignore[union-attr]
    write_incomplete_marker(out)

    if not probe_pass:
        rows = d1r2.preflight_failure_rows(probe_failure_summary(provenance.get("collision_probe") or {}), seeds)  # type: ignore[arg-type]
        summary = d1r2.summarize(rows, None, False, seeds)
        name = "d1r2_rows.invalid_redacted.jsonl"
        write_jsonl(out / name, [d1r2.redact_invalid_row(row) for row in rows])
        artifacts.append(name)
        return finalize_package(out, provenance, summary, rows, run_meta, artifacts)

    expected: Optional[Dict[int, Mapping[str, str]]] = None
    try:
        expected = {seed: identity(seed) for seed in seeds}
        rows, meta = run_stage(_d1r2_job, list(seeds))
        run_meta.append(meta)
        events, torn, malformed = d1r2.read_ledger_report(event_ledger)
        ledger_present = event_ledger.exists()
        _declare_run_ledgers()
        rows, replaced = complete_rows(rows, events, torn, seeds, ledger_present)
        # Section 6.3: schema validation consumes the durable per-warning classification.
        summary = d1r2.summarize(rows, expected, True, seeds, d1r2.ledger_classifications(events) if ledger_present else None)
        problems = list(replaced)
        if malformed:
            # Section 6.3: a malformed interior ledger line makes the run INVALID.
            problems.append(f"progress ledger has malformed interior line(s) {malformed}")
        if execute is None and not ledger_present:
            problems.append("progress event ledger missing")
        if problems:
            summary = invalid_summary(summary, "; ".join(problems))
        if summary["rows_schema_valid"] and summary["status"] == "INVALID_PIPELINE":
            name = "d1r2_rows.network_outcomes_redacted.jsonl"
            payload = [redact_network_outcomes(row) for row in rows]
        elif summary["rows_schema_valid"]:
            name = "d1r2_rows.jsonl"
            payload = rows
        else:
            name = "d1r2_rows.invalid_redacted.jsonl"
            payload = [d1r2.redact_invalid_row(row) for row in rows]
        write_jsonl(out / name, payload)
        artifacts.append(name)
        return finalize_package(out, provenance, summary, rows, run_meta, artifacts)
    except Exception as exc:
        # Worker loss, six-hour wall stop, RSS abort, or any coordinator exception:
        # the coordinator writes all 20 rows from the durable ledger (section 6.3).
        reason = "terminated_wall_stop" if isinstance(exc, WallLimitExceeded) else "terminated_rss_abort" if isinstance(exc, RSSLimitExceeded) else "worker_lost"
        events, torn, malformed = d1r2.read_ledger_report(event_ledger)
        rows = d1r2.reconstruct_rows_from_ledger(events, reason, seeds, torn=torn, stop_text=f"{reason}: {type(exc).__name__}")
        _declare_run_ledgers()
        name = "d1r2_rows.invalid_redacted.jsonl"
        write_jsonl(out / name, [d1r2.redact_invalid_row(row) for row in rows])
        if name not in artifacts:
            artifacts.append(name)
        summary = d1r2.summarize(rows, expected, True, seeds, d1r2.ledger_classifications(events))
        if malformed:
            summary = invalid_summary(summary, f"progress ledger has malformed interior line(s) {malformed}")
        return finalize_package(out, provenance, summary, rows, run_meta, artifacts, incomplete_reason=f"exception during sequence: {type(exc).__name__} ({reason})")


def invalid_summary(summary: Mapping[str, object], reason: str) -> Dict[str, object]:
    from snn import stage2_diagnostic_d1r2 as d1r2

    return {
        **dict(summary), "status": "INVALID", "valid": False, "rows_schema_valid": False,
        "violations": [reason, *list(summary.get("violations", []))],  # type: ignore[arg-type]
        "readouts": {"pipeline": None, "network_positive": None, "A": None},
        "branch": d1r2.branch("INVALID"),
    }


def complete_rows(
    rows: Sequence[Mapping[str, object]], events: Sequence[Mapping[str, object]], torn: bool,
    seeds: Sequence[int], ledger_present: bool = True,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Exactly one total row per seed (section 6.3 totality), plus replacement notes.

    A schema-valid worker row is kept, stamped with ``ledger_torn_tail``, and
    (when the durable ledger exists) its warnings must equal the ledger's
    warning events; a mismatch makes the row schema-invalid (status ``error``,
    so the run is INVALID). A seed with no worker row, an error row, or a
    schema-invalid row is written by the coordinator from the ledger as
    ``worker_lost`` so no slot is omitted; each replacement is returned as a
    note that forces INVALID and preserves the original evidence. Rows for
    undeclared seeds and duplicate rows are noted the same way.
    """
    from snn import stage2_diagnostic_d1r2 as d1r2

    notes: List[str] = []
    classifications = d1r2.ledger_classifications(events) if ledger_present else None
    by_seed: Dict[int, Dict[str, object]] = {}
    for row in rows:
        seed = row.get("seed") if isinstance(row, Mapping) else None
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed in seeds):
            notes.append(f"row for undeclared seed {seed!r} discarded")
        elif seed in by_seed:
            notes.append(f"duplicate row for seed {seed}")
        else:
            by_seed[seed] = dict(row)  # type: ignore[arg-type]
    out: List[Dict[str, object]] = []
    for seed in seeds:
        row = by_seed.get(seed)
        problems = ["no row"] if row is None else d1r2.validate_row_schema(row, classifications)
        if row is not None and not problems and row.get("status", "ok") != "ok":
            problems = [f"job status {row.get('status')!r}: {row.get('error')}"]
        if problems:
            notes.append(f"seed {seed}: worker row replaced by coordinator_from_ledger ({'; '.join(map(str, problems))[:300]})")
            out.extend(d1r2.reconstruct_rows_from_ledger(events, "worker_lost", [seed], torn=torn, stop_text="worker_lost: worker returned no schema-valid row"))
            continue
        row["ledger_torn_tail"] = torn  # type: ignore[index]
        if ledger_present and d1r2.validate_ledger_match(row, events):  # type: ignore[arg-type]
            row["status"] = "error"  # type: ignore[index]
            notes.append(f"seed {seed}: worker row warnings differ from the durable ledger")
        out.append(row)  # type: ignore[arg-type]
    out.sort(key=base._row_key)
    return out, notes


# ------------------------------------------------------------- shakedown
def run_shakedown(out: Path) -> Dict[str, object]:
    """Spec section 8 step 1 and section 11 item 15, on seed 4242 only.

    Full section 2 lengths (200,000 / 12,000). Only the pipeline decoder is
    fitted and scored; the network decoders are never fitted or scored here.
    Not a D1R2 datum; sets no threshold.
    """
    from snn import stage2_diagnostic_d1r2 as d1r2

    t0 = time.time()
    row = d1r2.run_d1r2_seed(d1r2.FIXTURE_SEED, decoders=("pipeline",))
    network_convergence = d1r2.run_fixture_network_convergence()
    wall = time.time() - t0
    expected = d1r2.object_hashes(d1r2.FIXTURE_SEED)
    violations = d1r2.validate_row(row, expected, decoders=("pipeline",))
    params = validated_effective_parameters()
    pipe = row["decoders"]["pipeline"]  # type: ignore[index]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    result = {
        "mode": "shakedown",
        "seed": d1r2.FIXTURE_SEED,
        "d1r2_seed": False,
        "lengths": {"train_stream": d1r2.PHASE_STEPS, "test_stream": d1r2.EVAL_LEN, "train_rows": row["train_rows"], "test_rows": row["test_rows"]},
        "decoders_fitted": sorted(row["decoders"]),  # type: ignore[arg-type]
        "network_decoders_fitted": sorted(network_convergence),
        "item16_network_convergence_only": network_convergence,
        "item16_no_accuracy_or_prediction_fields": all(
            set(record) == {"n_iter", "converged", "warnings"} for record in network_convergence.values()
        ),
        "item16_both_converged": all(record["converged"] is True for record in network_convergence.values()),
        "row_violations": violations,
        "row_valid_for_pipeline_only": not violations,
        "nesting_ok": row["nesting_ok"],
        "weights_constant": row["weights_constant"],
        "reward_computed": row["reward_computed"],
        "network_feature_dim": row["network_feature_dim"],
        "pipeline_feature_dim": row["pipeline_feature_dim"],
        "pipeline_heldout_accuracy": pipe["accuracy"],
        "pipeline_n_iter": pipe["n_iter"],
        "pipeline_converged": pipe["converged"],
        "item15_pipeline_reaches_0_99": bool(pipe["accuracy"] >= d1r2.THRESHOLDS["pipeline_pass_median_accuracy"]),
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "wall_seconds": wall,
        "peak_rss_single_process_bytes": peak_bytes,
        "note": "Construction checks on non-diagnostic fixture seed 4242 only. Pipeline is scored for item 15. The two network decoders are fit for convergence only for item 16; predict is never called and no network accuracy, correctness vector, or bound is computed or recorded.",
    }
    result["pass"] = bool(
        result["row_valid_for_pipeline_only"]
        and result["item15_pipeline_reaches_0_99"]
        and result["item16_no_accuracy_or_prediction_fields"]
        and result["item16_both_converged"]
        and result["effective_parameters_match_spec"]
    )
    (out / "shakedown_row.json").write_text(json.dumps(row, indent=2, sort_keys=True, default=_jsonable) + "\n")
    (out / "shakedown.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_jsonable) + "\n")
    manifest(out, ["provenance.json", "shakedown.json", "shakedown_row.json"])
    print(json.dumps({k: result[k] for k in ("pass", "pipeline_heldout_accuracy", "pipeline_n_iter", "network_decoders_fitted", "nesting_ok", "weights_constant", "effective_parameters_match_spec", "wall_seconds", "peak_rss_single_process_bytes")}, sort_keys=True))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.ss_probe:
        out = Path(args.out) if args.out else RESULTS
        try:
            result = shakedown_collision_check(out / "ss_probe.json")
        except ShakedownProbeFailure as exc:
            print(f"SHAKEDOWN COLLISION PROBE FAILED (implementation failure, nothing written): {exc}", file=sys.stderr)
            raise SystemExit(5)
        print(json.dumps({k: result[k] for k in ("decision", "counts", "legacy_only_duplicate_count", "identity_state_sha256", "reproduces_spec_section_3_3", "probe_code_sha256", "output_sha256")}, indent=1))
        return
    mode = "d1r2" if args.run_d1r2 else "shakedown"
    out = Path(args.out) if args.out else default_run_dir(mode)
    if out.exists() or out.is_symlink():
        raise StaleOutputError(f"refusing package: output location {out} already exists; packages require a fresh run-unique directory")
    provenance = preflight(args, mode, out)
    if args.shakedown:
        result = run_shakedown(out)
        if not result["pass"]:
            raise SystemExit(4)
    else:
        summary = run_d1r2(args, out, provenance)
        if summary["branch"]["code"] == "STOP_D1R2_INVALID":  # type: ignore[index]
            raise SystemExit(3)


if __name__ == "__main__":
    main()
