#!/usr/bin/env python3
"""Stage 2 diagnostic D1R runner (docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md at 2d9efd8).

The implementation card may use only ``--shakedown`` (non-diagnostic fixture
seed 4242, pipeline decoder only, spec section 11 item 15) or ``--ss-probe``
(section 3.3). ``--run-d1r`` is fail-closed behind a separate Nora approval
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
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
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
D1R_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_D1R_SPEC.md"
DIAG_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_SPEC.md"
STAGE2_SPEC = ROOT / "docs" / "STAGE2_SPEC.md"
RESULTS = ROOT / "results_stage2_diagnostic_d1r"
SCI_FILES = (
    "snn/core.py",
    "snn/stage2.py",
    "snn/stage2_r3.py",
    "snn/stage2_diagnostic.py",
    "snn/stage2_diagnostic_stats.py",
    "snn/stage2_diagnostic_d1r.py",
    "run_stage2.py",
    "run_stage2_diagnostic.py",
    "run_stage2_diagnostic_d1r.py",
    "tools/stage2_diagnostic_ss_probe.py",
    "tools/stage2_diagnostic_d1r_ss_probe.py",
    "tests/test_stage2_diagnostic_d1r.py",
    "docs/STAGE2_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_D1R_IMPLEMENTATION_NOTES.md",
)
STAGE = "D1R"

NO_ORACLE = (
    "The target is used only as an offline decoder label after feature capture. It is never a network input, "
    "hidden-state label, plasticity feature, state reset, initialization signal, or internal policy or world-model "
    "component. No reward is computed and no weight changes. No LLM logic enters the network, the feature path, "
    "the decoder, or the simulator."
)
CLAIM_BOUNDARY = (
    "Passing D1R does not pass Stage 2. D1R is only linear decodability under one frozen spike-only feature map, "
    "one split, and one decoder. REPRESENTATION_LIMITED is a bounded operational label: the strongest spike-only "
    "linear readout this spec permits did not clear the unchanged bar; it is not a proof that no readout could. "
    "None of D1R's outcomes establishes general learning, continual learning, developmental progress, biological "
    "equivalence, or an r4 result, and none changes a Stage 2 gate, control, validity rule, or the Stage 1 rule."
)
PIPELINE_CLAIM = (
    "Pipeline: under the fixed split, the pipeline-control map, preprocessing, and L2 logistic decoder, x_(t-1) was "
    "or was not recovered from the input-layer spike encoding at the D1R timing (validity of the code path only)."
)
NETWORK_CLAIM = (
    "Network positive control and A decoder: under the fixed split, the single D1R exact-lag spike feature map, "
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
    mode.add_argument("--run-d1r", action="store_true", help="locked: requires a Nora approval and run authorization")
    p.add_argument("--approval-file", help="separate implementation-review APPROVE + run authorization record")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None, help="package directory; must not exist")
    p.add_argument("--tests", default="not recorded")
    return p


# ------------------------------------------------------- effective parameters
def expected_parameters_from_spec() -> Dict[str, object]:
    """Transcribed by hand from docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md sections 2, 3.1, 5, 6, 10.

    Deliberately independent of snn/stage2_diagnostic_d1r.py constants; the
    runner refuses a D1R run on any difference from the constructed objects.
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
        "seeds": list(range(2100, 2120)),
        "namespace": 31,
        "root_entropy": 20261001,
        "K": 19,
        "network_feature_dim": 460,
        "pipeline_feature_dim": 46,
        "trace20_decay": math.exp(-1 / 20),
        "trace25_decay": math.exp(-1 / 25),
        "count_window_steps": 20,
        "decoder": {"penalty": "l2", "C": 1.0, "fit_intercept": True, "solver": "lbfgs", "tol": 1e-8, "max_iter": 2000, "class_weight": None},
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
    from snn import stage2_diagnostic_d1r as d1r

    effective = d1r.effective_parameters()
    mismatches = parameter_mismatches(effective, expected_parameters_from_spec())
    return {"effective": effective, "matches_spec": not mismatches, "mismatches": mismatches}


# ------------------------------------------------------------- preflight
def _approval(path: Optional[str], head: str, spec_sha: str) -> Dict[str, object]:
    if not path:
        raise SystemExit("refusing D1R run: --approval-file is required")
    record = json.loads(Path(path).read_text())
    required = {
        "decision": "APPROVE",
        "reviewer": "nora",
        "implementation_commit": head,
        "d1r_spec_sha256": spec_sha,
        "run_authorized": True,
    }
    mismatches = {k: (record.get(k), v) for k, v in required.items() if record.get(k) != v}
    if not isinstance(record.get("run_authorization_card"), str) or not record.get("run_authorization_card"):
        mismatches["run_authorization_card"] = (record.get("run_authorization_card"), "<separate run card id>")
    if mismatches:
        raise SystemExit(f"refusing D1R run: approval record mismatch {mismatches}")
    return record


def verified_probe() -> Dict[str, object]:
    """The retained section 3.3 probe, verified against its manifest and frozen counts."""
    path = RESULTS / "ss_probe.json"
    manifest_path = RESULTS / "SHA256SUMS"
    code_path = ROOT / "tools" / "stage2_diagnostic_d1r_ss_probe.py"
    if not path.exists() or not manifest_path.exists():
        return {"pass": False, "reason": "missing retained D1R ss_probe output or manifest"}
    record = json.loads(path.read_text())
    output_sha, code_sha = file_sha(path), file_sha(code_path)
    text = manifest_path.read_text()
    passed = bool(
        record.get("decision") == "PASS"
        and record.get("reproduces_spec_section_3_3") is True
        and record.get("numpy_version") == np.__version__
        and record.get("probe_code_sha256") == code_sha
        and f"{code_sha}  tools/stage2_diagnostic_d1r_ss_probe.py" in text
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
    from snn import stage2_diagnostic_d1r as d1r

    if not 1 <= args.workers <= d1r.MAX_WORKERS:
        raise SystemExit(f"workers must be 1..{d1r.MAX_WORKERS} (spec section 10)")
    head = git("rev-parse", "HEAD")
    try:
        upstream: Optional[str] = git("rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = None
    dirty = git("status", "--porcelain", "--", *SCI_FILES)
    d1r_sha, diag_sha, stage2_sha = file_sha(D1R_SPEC), file_sha(DIAG_SPEC), file_sha(STAGE2_SPEC)
    probe = verified_probe()
    approval = _approval(args.approval_file, head, d1r_sha) if mode == "d1r" else None
    params = validated_effective_parameters()
    prov: Dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "code_commit": head,
        "upstream_tracking_ref_sha": upstream,
        "code_commit_equals_upstream": head == upstream,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "d1r_spec_commit": d1r.SPEC_COMMIT,
        "d1r_spec_sha256_frozen": d1r.SPEC_SHA256,
        "d1r_spec_sha256_at_run": d1r_sha,
        "d1r_spec_unchanged": d1r_sha == d1r.SPEC_SHA256,
        "ebb90e74_sha256_frozen": d1r.EBB90E74_SHA256,
        "ebb90e74_sha256_at_run": diag_sha,
        "ebb90e74_unchanged": diag_sha == d1r.EBB90E74_SHA256,
        "stage2_spec_sha256_frozen": d1r.FROZEN_STAGE2_SHA256,
        "stage2_spec_sha256_at_run": stage2_sha,
        "stage2_spec_unchanged": stage2_sha == d1r.FROZEN_STAGE2_SHA256,
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
        "seeds": list(d1r.D1R_SEEDS) if mode == "d1r" else [d1r.FIXTURE_SEED],
        "maintained_tests": args.tests,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out),
    }
    claim_fresh_run_dir(out)
    (out / "provenance.json").write_text(_provenance_text(prov))
    if mode == "d1r":
        checks = (
            prov["code_commit_equals_upstream"],
            prov["d1r_spec_unchanged"],
            prov["ebb90e74_unchanged"],
            prov["stage2_spec_unchanged"],
            prov["scientific_files_clean"],
            probe["pass"],
            approval is not None,
            params["matches_spec"],
        )
        if not all(checks):
            raise SystemExit(
                "D1R preflight failed: pushed={0} d1r_spec={1} ebb90e74={2} stage2_spec={3} clean={4} ss_probe={5} approval={6} params={7}".format(*checks)
            )
    return prov


def _provenance_text(provenance: Mapping[str, object]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True, default=_jsonable) + "\n"


# ------------------------------------------------------------- job + pool
def _d1r_job(seed: int) -> Dict[str, object]:
    import torch

    from snn.stage2_diagnostic_d1r import run_d1r_seed

    torch.set_num_threads(1)
    try:
        row = run_d1r_seed(seed, authorized_run=True)
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:
        return {"diagnostic": "D1R", "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


class D1RRSSWatch(RSSWatch):
    """RSSWatch at the D1R cap: abort above 12 GiB total RSS rather than swap (spec section 10)."""

    def __init__(self, period: float = 2.0):
        from snn.stage2_diagnostic_d1r import RSS_LIMIT_BYTES

        super().__init__(limit=RSS_LIMIT_BYTES, period=period)

    def _run(self):  # same as RSSWatch._run with the D1R limit in the message
        while not self._stop.is_set():
            self.sample()
            if self.exceeded:
                print(f"ABORT: RSS {self.peak_total / 2**30:.2f} GiB > 12 GiB", flush=True)
                for c in self.proc.children(recursive=True):
                    try:
                        c.kill()
                    except self.psutil.Error:
                        pass
                os._exit(3)
            self._stop.wait(self.period)


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
    is ``D1R.progress.jsonl`` with exactly ``PROGRESS_FIELDS`` per finished
    job (identity, status, resources). An exception leaves only that ledger.
    """
    from snn.stage2_diagnostic_d1r import WALL_LIMIT_SECONDS

    wall_limit = WALL_LIMIT_SECONDS if wall_limit is None else wall_limit
    watch = watch if watch is not None else D1RRSSWatch()
    watch.start()  # type: ignore[attr-defined]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rows: List[Dict[str, object]] = []
    ledger = out / f"{STAGE}.progress.jsonl"
    ledger.open("x").close()
    try:
        with executor_factory(workers) as pool:
            futures = [pool.submit(fn, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                rec = progress_record(STAGE, i, len(jobs), row, time.time() - t0, watch.peak_total)  # type: ignore[attr-defined]
                with ledger.open("a") as handle:
                    handle.write(json.dumps(rec, sort_keys=True) + "\n")
                print(" ".join(f"{k}={rec[k]}" for k in PROGRESS_FIELDS), flush=True)
                if watch.exceeded:  # type: ignore[attr-defined]
                    raise RuntimeError("RSS exceeded 12 GiB; aborting rather than swapping")
                if time.time() - t0 > wall_limit:
                    for f in futures:
                        f.cancel()
                    raise RuntimeError("wall time exceeded 6 h; stopped and reported, nothing changed")
    finally:
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
    """Fail-closed default before the stage starts: a hard kill leaves STOP_D1R_INVALID on disk."""
    from snn import stage2_diagnostic_d1r as d1r

    record = {**d1r.branch("INVALID"), "status": None, "incomplete_reason": "sequence has not reached finalize_package"}
    (out / "branch_outcome.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def plain_language(summary: Mapping[str, object]) -> Dict[str, object]:
    code = summary["branch"]["code"]  # type: ignore[index]
    readouts = summary.get("readouts") or {}
    supported: List[str] = []
    if code == "STOP_D1R_INVALID":
        supported.append("Nothing beyond the invalidity itself: the sequence stopped before any decision-bearing readout could be inspected.")
    elif code == "STOP_D1R_INVALID_PIPELINE":
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
    d1r_summary: Optional[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    run_meta: Sequence[Mapping[str, object]],
    artifacts: Sequence[str],
    incomplete_reason: Optional[str] = None,
) -> Dict[str, object]:
    """The single exit for every D1R branch (spec sections 6.3, 8, 9, 12)."""
    from snn import stage2_diagnostic_d1r as d1r

    params = validated_effective_parameters()
    if d1r_summary is None or incomplete_reason is not None:
        d1r_summary = {
            "status": "INVALID",
            "valid": False,
            "violations": [incomplete_reason or "no D1R summary"],
            "readouts": {"pipeline": None, "network_positive": None, "A": None},
            "branch": d1r.branch("INVALID"),
        }
    per_seed: Dict[str, object] = {}
    if d1r_summary.get("valid"):
        # Under INVALID_PIPELINE (section 9 row 2) only pipeline fields are packaged.
        names = ("pipeline",) if d1r_summary["status"] == "INVALID_PIPELINE" else d1r.DECODERS
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
        "status": d1r_summary["status"],
        "violations": list(d1r_summary.get("violations", [])),
        "readouts": d1r_summary["readouts"],
        "branch_outcome": d1r_summary["branch"],
        "branch": d1r_summary["branch"],
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
        summary["branch_outcome"] = summary["branch"] = d1r.branch("INVALID")
        summary["per_seed"] = {}
        summary["run_metadata"]["incomplete_reason"] = reason  # type: ignore[index]
        summary["unexpected_artifacts"] = unexpected
        summary["missing_artifacts"] = missing
        summary["artifacts"] = sorted((declared - set(missing)) | {"summary.json"})
    summary["package_summary"] = plain_language(summary)
    absent = [k for k in REQUIRED_PACKAGE_FIELDS if k not in summary]
    if absent or summary["branch"]["code"] not in d1r.BRANCH_INDEX:  # type: ignore[index]
        raise AssertionError(f"package summary incomplete: {absent}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    (out / "branch_outcome.json").write_text(json.dumps({"status": summary["status"], **summary["branch"]}, indent=2, sort_keys=True) + "\n")  # type: ignore[arg-type]
    manifest(out, summary["artifacts"])  # type: ignore[arg-type]
    print(f"PACKAGE FROZEN d1r_status={summary['status']} branch={summary['branch']['code']} table_row={summary['branch']['table_row']}", flush=True)  # type: ignore[index]
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


def run_d1r(
    args: argparse.Namespace,
    out: Path,
    provenance: Mapping[str, object],
    execute: Optional[Executer] = None,
    seeds: Optional[Sequence[int]] = None,
    identity: Optional[Callable[[int], Mapping[str, str]]] = None,
) -> Dict[str, object]:
    """Spec section 8 step 2-3. Injection points exist for construction tests only."""
    from snn import stage2_diagnostic_d1r as d1r
    from snn.stage2_diagnostic_stats import redact_outcomes

    prepare_package_dir(out, provenance)
    seeds = tuple(d1r.D1R_SEEDS if seeds is None else seeds)
    identity = identity or d1r.object_hashes
    artifacts: List[str] = []
    run_meta: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    def _default_execute(fn, jobs):
        artifacts.append(f"{STAGE}.progress.jsonl")
        return run_pool(fn, jobs, args.workers, out)

    run_stage: Executer = execute or _default_execute  # type: ignore[assignment]
    probe_pass = bool((provenance.get("collision_probe") or {}).get("pass"))  # type: ignore[union-attr]
    write_incomplete_marker(out)
    try:
        expected = {s: identity(s) for s in seeds}  # coordinator identities before any worker transition
        rows, meta = run_stage(_d1r_job, list(seeds))
        run_meta.append(meta)
        rows.sort(key=base._row_key)
        summary = d1r.summarize(rows, expected, probe_pass, seeds)
        if summary["rows_schema_valid"] and summary["status"] == "INVALID_PIPELINE":
            # Section 9 row 2: no science reading of the network decoders, so
            # their outcome fields never reach disk on this branch.
            write_jsonl(out / "d1r_rows.network_outcomes_redacted.jsonl", [redact_network_outcomes(r) for r in rows])
            artifacts.append("d1r_rows.network_outcomes_redacted.jsonl")
        elif summary["rows_schema_valid"]:
            write_jsonl(out / "d1r_rows.jsonl", rows)
            artifacts.append("d1r_rows.jsonl")
        else:
            write_jsonl(out / "d1r_rows.invalid_redacted.jsonl", [redact_outcomes(r) for r in rows])  # type: ignore[misc]
            artifacts.append("d1r_rows.invalid_redacted.jsonl")
        return finalize_package(out, provenance, summary, rows, run_meta, artifacts)
    except Exception as exc:  # any unplanned stop freezes an outcome-free INVALID package
        return finalize_package(out, provenance, None, [], run_meta, artifacts, incomplete_reason=f"exception during sequence: {type(exc).__name__}")


# ------------------------------------------------------------- shakedown
def run_shakedown(out: Path) -> Dict[str, object]:
    """Spec section 8 step 1 and section 11 item 15, on seed 4242 only.

    Full section 2 lengths (200,000 / 12,000). Only the pipeline decoder is
    fitted and scored; the network decoders are never fitted or scored here.
    Not a D1R datum; sets no threshold.
    """
    from snn import stage2_diagnostic_d1r as d1r

    t0 = time.time()
    row = d1r.run_d1r_seed(d1r.FIXTURE_SEED, decoders=("pipeline",))
    wall = time.time() - t0
    expected = d1r.object_hashes(d1r.FIXTURE_SEED)
    violations = d1r.validate_row(row, expected, decoders=("pipeline",))
    params = validated_effective_parameters()
    pipe = row["decoders"]["pipeline"]  # type: ignore[index]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    result = {
        "mode": "shakedown",
        "seed": d1r.FIXTURE_SEED,
        "d1r_seed": False,
        "lengths": {"train_stream": d1r.PHASE_STEPS, "test_stream": d1r.EVAL_LEN, "train_rows": row["train_rows"], "test_rows": row["test_rows"]},
        "decoders_fitted": sorted(row["decoders"]),  # type: ignore[arg-type]
        "network_decoders_fitted": False,
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
        "item15_pipeline_reaches_0_99": bool(pipe["accuracy"] >= d1r.THRESHOLDS["pipeline_pass_median_accuracy"]),
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "wall_seconds": wall,
        "peak_rss_single_process_bytes": peak_bytes,
        "note": "Construction check on the non-diagnostic fixture seed only; network features captured but no network decoder fitted or scored; not a D1R datum.",
    }
    result["pass"] = bool(result["row_valid_for_pipeline_only"] and result["item15_pipeline_reaches_0_99"] and result["effective_parameters_match_spec"])
    (out / "shakedown_row.json").write_text(json.dumps(row, indent=2, sort_keys=True, default=_jsonable) + "\n")
    (out / "shakedown.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_jsonable) + "\n")
    manifest(out, ["provenance.json", "shakedown.json", "shakedown_row.json"])
    print(json.dumps({k: result[k] for k in ("pass", "pipeline_heldout_accuracy", "pipeline_n_iter", "network_decoders_fitted", "nesting_ok", "weights_constant", "effective_parameters_match_spec", "wall_seconds", "peak_rss_single_process_bytes")}, sort_keys=True))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.ss_probe:
        out = Path(args.out) if args.out else RESULTS
        cmd = [sys.executable, str(ROOT / "tools" / "stage2_diagnostic_d1r_ss_probe.py"), "--out", str(out / "ss_probe.json")]
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    mode = "d1r" if args.run_d1r else "shakedown"
    out = Path(args.out) if args.out else default_run_dir(mode)
    if out.exists() or out.is_symlink():
        raise StaleOutputError(f"refusing package: output location {out} already exists; packages require a fresh run-unique directory")
    provenance = preflight(args, mode, out)
    if args.shakedown:
        result = run_shakedown(out)
        if not result["pass"]:
            raise SystemExit(4)
    else:
        summary = run_d1r(args, out, provenance)
        if summary["branch"]["code"] == "STOP_D1R_INVALID":  # type: ignore[index]
            raise SystemExit(3)


if __name__ == "__main__":
    main()
