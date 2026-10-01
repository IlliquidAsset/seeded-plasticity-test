#!/usr/bin/env python3
"""Stage 2 D1-D4 diagnostic runner.

The implementation card may use only ``--shakedown`` or ``--ss-probe``.
``--run-diagnostic`` is fail-closed behind a separate Nora approval record tied
to the exact pushed implementation commit.

Outcome-exposure contract (spec section 8): while a stage executes, the only
file written is an outcome-free progress ledger. Seed rows stay in coordinator
memory until every predeclared row of the stage is present, then the strict
package validator runs. A schema-valid stage is written in full; an invalid
stage is written with outcome-bearing keys redacted. Every predeclared stop
branch, including an exception, goes through ``finalize_package``, which
writes the section 13 package summary and the SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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
    "tests/test_stage2_diagnostic_package.py",
    "tests/test_stage2_diagnostic_round3.py",
    "docs/STAGE2_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_SPEC.md",
    "docs/STAGE2_DIAGNOSTIC_IMPLEMENTATION_NOTES.md",
)
PROGRESS_FIELDS = ("stage", "index", "total", "seed", "condition", "status", "elapsed_s", "rss_peak_total_bytes")

# Spec section 13 claim boundary and no-oracle statement, carried verbatim in
# substance into every package, whatever branch stops the sequence.
NO_ORACLE = (
    "The target is used only to grade already-emitted output spikes and, in D1 only, as an offline decoder "
    "label after feature capture. It is never an input, hidden-state label, plasticity feature, state reset, "
    "initialization signal, or internal policy/world-model component. D4 drive is generated without task or "
    "target input. No LLM logic enters the network, policy, memory, reward rule, decoder feature path, or simulator."
)
CLAIM_BOUNDARY = (
    "Passing any diagnostic does not pass Stage 2. D1 is only linear decodability under one frozen specification. "
    "D2 is localization. D3 is a simpler-task contrast. D4 is one anti-silence perturbation at one frozen amplitude. "
    "None establishes general learning, continual learning, developmental progress, biological equivalence, Abe "
    "transfer, or an r4 result. Positive findings remain simulated bench findings and require Nora review before use."
)
D1_CLAIM = (
    "Under the fixed split, feature map, preprocessing, and L2 logistic decoder, the noisy A target was or was not "
    "linearly decodable from the hidden activity of the frozen random r3 network."
)
NOT_SUPPORTED = (
    "the network can or cannot represent A at all",
    "any nonlinear-decoder, trained-weight, architecture-wide, or learning-rule claim from D1",
    "passing Stage 2 or re-scoring r2/r3",
    "general or continual learning, developmental progress, biological equivalence, or Abe transfer",
    "any r4 result, drive retuning, or post-data constant change",
)

# Spec section 10, ordered; earlier rows take precedence.
BRANCH_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("STOP_INVALID", "Any D1-D4 INVALID, collision-probe failure, or incomplete package",
     "STOP. Return to spec/implementation review. No r4."),
    ("STOP_D1_INVALID_HARNESS", "D1 positive control not PASS",
     "D1 is INVALID_HARNESS. Repair only under a new reviewed spec; no downstream diagnostic."),
    ("STOP_D1_FAIL", "D1 FAIL",
     "The specified linear decoder cannot recover A under the fixed split. Task/architecture pairing needs roadmap revision. Return to Kendrick. No r4 readout or reward change."),
    ("STOP_D1_INCONCLUSIVE", "D1 INCONCLUSIVE",
     "Evidence is not decision-complete. Return to Kendrick/spec review. No downstream diagnostic and no r4."),
    ("RETURN_D3_FAIL", "D1 PASS and D3 FAIL",
     "Frozen activity is linearly decodable but the unchanged rule/harness does not establish lag-1 competence. Treat as a rule-limit or harness question; return to Kendrick. D4 observations cannot independently authorize r4."),
    ("RETURN_D3_INCONCLUSIVE", "D1 PASS and D3 INCONCLUSIVE",
     "Rule capacity is unresolved. Return to Kendrick. D4 may be reported if already completed in the fixed sequence, but cannot authorize r4."),
    ("RETURN_D4_SILENCE_FAIL", "D1 PASS, D3 PASS, D4(ii) FAIL",
     "The frozen drive did not prevent silence at its predeclared constants. No constant retuning inside this diagnostic. No r4 with this drive. D2 localization is reported."),
    ("RETURN_D4_COMPETENCE_FAIL", "D1 PASS, D3 PASS, D4(ii) PASS, D4(iii) FAIL",
     "Silence was prevented but competence did not improve decisively. Silence alone is not shown to be the bottleneck. Drive alone does not justify r4. D2 localization is reported."),
    ("RETURN_D4_COMPETENCE_INCONCLUSIVE", "D1 PASS, D3 PASS, D4(ii) PASS, D4(iii) INCONCLUSIVE",
     "Silence was prevented but learning evidence is unresolved. Return to Kendrick; no r4 and no retuning."),
    ("CONSIDER_DOC_ONLY_R4_SPEC", "D1 PASS, D3 PASS, D4(ii) PASS, D4(iii) PASS",
     "An r4 with this exact frozen drive is warranted for consideration. Create a new doc-only r4 spec card, inform Kendrick, retain all Stage 2 gates/controls/rule, and require Nora review before execution."),
    ("RETURN_D4_SILENCE_FAIL_COMPETENCE_PASS", "D1 PASS, D3 PASS, D4(ii) FAIL, D4(iii) PASS",
     "Activity criterion still failed, so the anti-silence mechanism is not validated even if competence happened to pass. No r4 with this drive; return to Kendrick."),
    ("RETURN_D4_OTHER_INCONCLUSIVE", "D1 PASS, D3 PASS, D4 overall INCONCLUSIVE for any other valid combination",
     "Return to Kendrick. No r4 and no post-data constant change."),
)
BRANCH_INDEX = {code: i + 1 for i, (code, _, _) in enumerate(BRANCH_TABLE)}


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
    p.add_argument(
        "--out",
        default=None,
        help="package directory; must not exist. Default: results_stage2_diagnostic/runs/<mode>-<utc>-<head>-pid<pid>",
    )
    p.add_argument("--tests", default="not recorded")
    return p


# ------------------------------------------------------- effective parameters
def expected_parameters_from_spec() -> Dict[str, object]:
    """Values transcribed by hand from docs/STAGE2_DIAGNOSTIC_SPEC.md sections 2-9.

    Deliberately independent of snn/stage2_diagnostic.py constants: the runner
    compares this block with the effective block read from constructed objects
    and refuses a diagnostic run on any difference.
    """
    return {
        "layer_sizes": [2, 20, 2],
        "dt_ms": 1.0,
        "tau_m_ms": [20.0, 20.0],
        "v_rest_mv": [-70.0, -70.0],
        "v_reset_mv": [-70.0, -70.0],
        "v_thresh_mv": [-54.0, -54.0],
        "tau_syn_ms": [0.0, 0.0],
        "hidden_tau_a_ms": 200.0,
        "hidden_beta_a_mv": 1.12,
        "w1_bounds_mv": [-10.0, 10.0],
        "w2_bounds_mv": [0.0, 10.0],
        "plasticity_credit": ["eligibility", "eligibility"],
        "gamma_mv": [0.25, 0.25],
        "tau_plus_ms": [20.0, 20.0],
        "tau_minus_ms": [20.0, 20.0],
        "tau_elig_ms": [25.0, 25.0],
        "a_plus": [1 / 25, 1 / 25],
        "a_minus": [1 / 25, 1 / 25],
        "noise_p": 0.10,
        "training_steps": 200_000,
        "eval_len": 12_000,
        "eval_warmup": 2_000,
        "eval_scored": 10_000,
        "seeds": list(range(2000, 2020)),
        "namespace": 30,
        "root_entropy": 20261001,
        "d1": {
            "warmup_rows": 2_000,
            "train_rows": 198_000,
            "test_rows": 10_000,
            "feature_dim": 60,
            "count_window_steps": 20,
            "trace20_decay": math.exp(-1 / 20),
            "trace25_decay": math.exp(-1 / 25),
            "decoder": {"penalty": "l2", "C": 1.0, "fit_intercept": True, "solver": "lbfgs", "tol": 1e-8, "max_iter": 2000, "class_weight": None},
            "prediction_threshold": 0.5,
            "block_length": 100,
            "block_resamples": 10_000,
            "block_percentile": 5,
        },
        "d2": {"checkpoint_steps": 1_000, "checkpoint_count": 200, "onset_ratio": 0.50, "lead_steps": 5_000, "localized_min_seeds": 14, "bound_fraction_denominator": 40},
        "d3": {"train_draws": 200_001, "eval_draws": 12_001},
        "d4_drive": {
            "targets": 22,
            "sources_per_target": 8,
            "sources_total": 176,
            "rate_hz": 25.0,
            "weight_mv": 2.0,
            "spike_probability_per_ms": 1 - math.exp(-25 / 1000),
            "realization_steps": 212_000,
            "random_call_shape": [212_000, 22, 8],
            "train_indices": [0, 199_999],
            "eval_indices": [200_000, 211_999],
            "hidden_targets": [0, 19],
            "O1_target": 20,
            "O0_target": 21,
            "plastic": False,
            "background_plasticity_objects": 0,
            "seed_sequence_component": 40,
        },
        "bootstrap": {"resamples": 100_000, "percentile_method": "inverted_cdf", "metric_ids": list(range(9))},
        "thresholds": {
            "d1_pass_median_accuracy": 0.70,
            "d1_pass_min_lower_bounds_gt_half": 15,
            "d1_lower_bound_reference": 0.50,
            "d1_fail_median_below": 0.55,
            "d2_onset_ratio": 0.50,
            "d2_lead_steps": 5_000,
            "d2_localized_min_seeds": 14,
            "d3_chance": 0.50,
            "d3_pass_median_accuracy": 0.70,
            "d3_pass_lower_bound_gt": 0.55,
            "d4_chance": 0.50,
            "d4_hidden_rate_ratio_min": 0.50,
            "d4_silent_end_max": 0.80,
            "d4_pass_median_accuracy": 0.70,
            "d4_pass_median_delta": 0.05,
            "d4_pass_delta_lower_gt": 0.0,
        },
    }


def parameter_mismatches(effective: Mapping[str, object], expected: Mapping[str, object], prefix: str = "") -> List[str]:
    """Every expected key must be present in ``effective`` with an identical value."""
    out: List[str] = []
    for key, want in expected.items():
        path = f"{prefix}{key}"
        if key not in effective:
            out.append(f"{path}: missing")
            continue
        have = effective[key]
        if isinstance(want, Mapping):
            if not isinstance(have, Mapping):
                out.append(f"{path}: expected mapping")
            else:
                out.extend(parameter_mismatches(have, want, path + "."))
        elif have != want:
            out.append(f"{path}: effective {have!r} != spec {want!r}")
    return out


def validated_effective_parameters() -> Dict[str, object]:
    from snn import stage2_diagnostic as diag

    effective = diag.effective_parameters()
    mismatches = parameter_mismatches(effective, expected_parameters_from_spec())
    return {"effective": effective, "matches_spec": not mismatches, "mismatches": mismatches}


# ------------------------------------------------------------- preflight
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
    params = validated_effective_parameters()
    prov: Dict[str, object] = {
        "schema_version": 2,
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
        "effective_parameters": params["effective"],
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "argv": [sys.executable, *sys.argv],
        "argv_display": shlex.join([sys.executable, *sys.argv]),
        "environment": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "VIRTUAL_ENV")},
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
        "output_dir": str(out),
    }
    # Fresh, exclusively created location: a prior full or partial package can
    # never be reused, overwritten, or manifested into this run.
    claim_fresh_run_dir(out)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True, default=_jsonable) + "\n")
    if mode == "diagnostic":
        checks = (
            prov["code_commit_equals_upstream"],
            prov["diagnostic_spec_unchanged"],
            prov["stage2_spec_unchanged"],
            prov["scientific_files_clean"],
            probe["pass"],
            approval is not None,
            params["matches_spec"],
        )
        if not all(checks):
            raise SystemExit(
                "diagnostic preflight failed: pushed={0} diag_spec={1} stage2_spec={2} clean={3} ss_probe={4} approval={5} params={6}".format(*checks)
            )
    return prov


# ------------------------------------------------------------- job wrappers
def _d1_job(seed: int) -> Dict[str, object]:
    import torch

    from snn.stage2_diagnostic import run_d1_seed

    torch.set_num_threads(1)
    try:
        row = run_d1_seed(seed)
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:
        return {"diagnostic": "D1", "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


def _condition_job(job: Tuple[int, str, bool, bool, Mapping[str, object]]) -> Dict[str, object]:
    import torch

    from snn.stage2_diagnostic import run_condition_seed

    torch.set_num_threads(1)
    seed, task, plastic, drive, expected = job
    condition = ("P" if plastic else "F0") + "_" + task + ("+drive" if drive else "-no")
    try:
        row = run_condition_seed(seed, task, plastic, drive, expected_hashes=expected)
        row["status"] = "ok"
        row["worker_pid"] = os.getpid()
        return row
    except Exception as exc:
        return {"condition": condition, "seed": seed, "status": "error", "error": repr(exc), "worker_pid": os.getpid()}


# ------------------------------------------------------------- stage pool
def progress_record(stage: str, index: int, total: int, row: Mapping[str, object], elapsed: float, rss: int) -> Dict[str, object]:
    """Outcome-free ledger line: identity, status, and resources only."""
    return {
        "stage": stage,
        "index": index,
        "total": total,
        "seed": row.get("seed"),
        "condition": row.get("condition", row.get("diagnostic")),
        "status": "ok" if row.get("status") == "ok" else "error",
        "elapsed_s": round(elapsed, 1),
        "rss_peak_total_bytes": int(rss),
    }


def run_pool(
    fn: Callable[[object], Dict[str, object]],
    jobs: Sequence[object],
    workers: int,
    stage: str,
    out: Path,
    executor_factory: Callable[[int], Executor] = lambda n: ProcessPoolExecutor(max_workers=n),
    watch: Optional[object] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Complete a whole predeclared stage before any outcome is persisted.

    Rows are held only in coordinator memory. The sole file written during the
    stage is ``<stage>.progress.jsonl`` with ``PROGRESS_FIELDS`` per finished
    job; it never contains an accuracy, rate, bound, interval, or decoder field.
    An exception leaves only that ledger behind.
    """
    watch = watch if watch is not None else RSSWatch()
    watch.start()  # type: ignore[attr-defined]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rows: List[Dict[str, object]] = []
    ledger = out / f"{stage}.progress.jsonl"
    ledger.open("x").close()  # exclusive: never reuse a prior run's ledger
    try:
        with executor_factory(workers) as pool:
            futures = [pool.submit(fn, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                rec = progress_record(stage, i, len(jobs), row, time.time() - t0, watch.peak_total)  # type: ignore[attr-defined]
                with ledger.open("a") as handle:
                    handle.write(json.dumps(rec, sort_keys=True) + "\n")
                print(" ".join(f"{k}={rec[k]}" for k in PROGRESS_FIELDS), flush=True)
                if watch.exceeded:  # type: ignore[attr-defined]
                    raise RuntimeError("RSS exceeded 8 GiB; aborting rather than swapping")
    finally:
        watch.sample()  # type: ignore[attr-defined]
        watch.stop()  # type: ignore[attr-defined]
    meta = {
        "stage": stage,
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
        "rss_limit_bytes": RSS_LIMIT,
        "rss_exceeded": watch.exceeded,  # type: ignore[attr-defined]
        "interruptions": 0,
        "retries": 0,
    }
    return rows, meta


def _row_key(row: Mapping[str, object]) -> Tuple[str, int, str]:
    """Spec section 2 order: diagnostic/condition, then seed; tolerant of malformed rows."""
    seed = row.get("seed")
    return (str(row.get("condition", row.get("diagnostic"))), seed if isinstance(seed, int) and not isinstance(seed, bool) else -1, repr(seed))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Exclusive create: a pre-existing file of the same name (another writer,
    or a remnant) raises instead of being silently overwritten."""
    with path.open("x") as handle:
        handle.write("".join(json.dumps(r, sort_keys=True, default=_jsonable) + "\n" for r in rows))


def write_stage_rows(out: Path, name: str, rows: Sequence[Mapping[str, object]], valid: bool) -> str:
    """Write a completed stage. Invalid stages are redacted (spec section 8)."""
    from snn.stage2_diagnostic_stats import redact_outcomes

    if valid:
        path = out / f"{name}.jsonl"
        write_jsonl(path, rows)
    else:
        path = out / f"{name}.invalid_redacted.jsonl"
        write_jsonl(path, [redact_outcomes(r) for r in rows])  # type: ignore[misc]
    return path.name


class StaleOutputError(SystemExit):
    """Refusal to write a package into a location that already holds artifacts."""


def default_run_dir(mode: str) -> Path:
    """Run-unique package location; never the retained ss_probe directory itself."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    head = git("rev-parse", "--short=12", "HEAD")
    return ROOT / "results_stage2_diagnostic" / "runs" / f"{mode}-{stamp}-{head}-pid{os.getpid()}"


def claim_fresh_run_dir(out: Path) -> Path:
    """Create ``out`` exclusively. Any existing path (prior full package, a
    hard-killed partial package, or the retained probe directory) is refused
    before any provenance, marker, or stage file is written."""
    if out.exists() or out.is_symlink():
        raise StaleOutputError(f"refusing package: output location {out} already exists; packages require a fresh run-unique directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.mkdir(exist_ok=False)
    except FileExistsError as exc:  # lost a race with another writer
        raise StaleOutputError(f"refusing package: output location {out} appeared concurrently") from exc
    return out


def package_files(out: Path) -> List[str]:
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() or p.is_symlink())


def manifest(out: Path, names: Sequence[str]) -> None:
    """SHA-256 manifest of exactly the declared artifacts, never a directory glob."""
    names = sorted(set(names) - {"SHA256SUMS"})
    for name in names:
        if not (out / name).is_file():
            raise AssertionError(f"declared artifact missing: {name}")
    (out / "SHA256SUMS").write_text("".join(f"{file_sha(out / n)}  {n}\n" for n in names))


# ------------------------------------------------------------- branch table
def branch_outcome(
    d1: Optional[str],
    d2: Optional[str] = None,
    d3: Optional[str] = None,
    d4: Optional[Mapping[str, object]] = None,
    incomplete: bool = False,
) -> Dict[str, object]:
    """Ordered spec section 10 branch table over whatever stages were reached."""
    d4_status = None if d4 is None else d4.get("status")
    invalid = (
        incomplete
        or d1 in (None, "INVALID")
        or d2 == "D2_INVALID"
        or d3 == "INVALID"
        or d4_status == "INVALID"
    )
    if invalid:
        code = "STOP_INVALID"
    elif d1 == "INVALID_HARNESS":
        code = "STOP_D1_INVALID_HARNESS"
    elif d1 == "FAIL":
        code = "STOP_D1_FAIL"
    elif d1 == "INCONCLUSIVE":
        code = "STOP_D1_INCONCLUSIVE"
    elif d1 != "PASS" or d2 is None or d3 is None or d4 is None:
        code = "STOP_INVALID"  # a PASS path that did not reach every predeclared stage is incomplete
    elif d3 == "FAIL":
        code = "RETURN_D3_FAIL"
    elif d3 == "INCONCLUSIVE":
        code = "RETURN_D3_INCONCLUSIVE"
    elif d4.get("silence_prevention") == "FAIL":
        code = "RETURN_D4_SILENCE_FAIL"
    elif d4.get("competence") == "FAIL":
        code = "RETURN_D4_COMPETENCE_FAIL"
    elif d4.get("competence") == "INCONCLUSIVE":
        code = "RETURN_D4_COMPETENCE_INCONCLUSIVE"
    elif d4.get("competence") == "PASS":
        code = "CONSIDER_DOC_ONLY_R4_SPEC"
    else:
        code = "RETURN_D4_OTHER_INCONCLUSIVE"
    _, condition, action = BRANCH_TABLE[BRANCH_INDEX[code] - 1]
    return {"code": code, "table_row": BRANCH_INDEX[code], "condition": condition, "action": action}


def plain_language(branch: Mapping[str, object], statuses: Mapping[str, object]) -> Dict[str, object]:
    reached = {k: v for k, v in statuses.items() if v is not None}
    supported: List[str] = []
    if branch["code"] == "STOP_INVALID":
        supported.append("Nothing beyond the invalidity itself: the sequence stopped before any decision-bearing readout could be inspected.")
    else:
        d1 = statuses.get("D1")
        if d1 in ("PASS", "FAIL", "INCONCLUSIVE"):
            supported.append(f"D1 {d1}: {D1_CLAIM}")
        if d1 == "INVALID_HARNESS":
            supported.append("D1 positive control did not pass, so the D1 harness cannot support any decodability statement.")
        if statuses.get("D2"):
            supported.append(f"D2 {statuses['D2']}: a mechanical localization label only, not a competence result.")
        if statuses.get("D3"):
            supported.append(f"D3 {statuses['D3']}: a balanced lag-1 contrast under the unchanged rule and budget.")
        if statuses.get("D4"):
            supported.append(f"D4 {statuses['D4']}: one fixed anti-silence drive at one frozen amplitude.")
    return {
        "stages_reached": sorted(reached),
        "statuses": dict(statuses),
        "branch": branch["code"],
        "next_action": branch["action"],
        "supported": supported,
        "not_supported": list(NOT_SUPPORTED),
    }


REQUIRED_PACKAGE_FIELDS = (
    "provenance",
    "effective_parameters",
    "effective_parameters_match_spec",
    "collision_probe",
    "run_metadata",
    "statuses",
    "stages",
    "branch_outcome",
    "package_summary",
    "claim_boundary",
    "no_oracle",
    "d1_claim_boundary",
    "artifacts",
)


def validate_package_summary(summary: Mapping[str, object]) -> List[str]:
    missing = [k for k in REQUIRED_PACKAGE_FIELDS if k not in summary]
    problems = [f"missing {k}" for k in missing]
    branch = summary.get("branch_outcome")
    if isinstance(branch, Mapping) and branch.get("code") not in BRANCH_INDEX:
        problems.append("branch outcome not in spec section 10 table")
    params = summary.get("effective_parameters")
    if isinstance(params, Mapping):
        drive = params.get("d4_drive")
        if not isinstance(drive, Mapping) or drive.get("sources_total") != 176 or drive.get("rate_hz") != 25.0 or drive.get("weight_mv") != 2.0:
            problems.append("D4 drive constants missing from effective parameters")
    return problems


def finalize_package(
    out: Path,
    provenance: Mapping[str, object],
    stages: Mapping[str, object],
    statuses: Mapping[str, object],
    run_meta: Sequence[Mapping[str, object]],
    artifacts: Sequence[str],
    incomplete_reason: Optional[str] = None,
) -> Dict[str, object]:
    """The single exit for every predeclared stop branch (spec sections 8, 10, 13)."""
    params = validated_effective_parameters()
    branch = branch_outcome(
        statuses.get("D1"),  # type: ignore[arg-type]
        statuses.get("D2"),  # type: ignore[arg-type]
        statuses.get("D3"),  # type: ignore[arg-type]
        stages.get("D4"),  # type: ignore[arg-type]
        incomplete=incomplete_reason is not None,
    )
    summary: Dict[str, object] = {
        "schema_version": 2,
        "provenance": provenance,
        "effective_parameters": params["effective"],
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "collision_probe": provenance.get("ss_probe"),
        "run_metadata": {
            "stages": list(run_meta),
            "workers": provenance.get("workers"),
            "total_wall_seconds": sum(float(m.get("wall_seconds", 0.0)) for m in run_meta),
            "peak_rss_total_bytes": max([int(m.get("peak_rss_total_bytes", 0)) for m in run_meta] or [0]),
            "interruptions": sum(int(m.get("interruptions", 0)) for m in run_meta),
            "retries": sum(int(m.get("retries", 0)) for m in run_meta),
            "incomplete_reason": incomplete_reason,
        },
        "statuses": dict(statuses),
        "stages": dict(stages),
        "branch_outcome": branch,
        "package_summary": plain_language(branch, statuses),
        "claim_boundary": CLAIM_BOUNDARY,
        "no_oracle": NO_ORACLE,
        "d1_claim_boundary": D1_CLAIM,
        "artifacts": sorted(set(artifacts) | {"summary.json", "branch_outcome.json", "provenance.json"}),
    }
    # Exact-contents gate: the directory may hold only what this sequence
    # declared. Anything else (written by another process mid-run) makes the
    # package STOP_INVALID, is named but never manifested, and every stage
    # record is redacted because the location can no longer be trusted.
    present = set(package_files(out))
    declared = set(summary["artifacts"]) - {"summary.json"}  # type: ignore[arg-type]
    unexpected = sorted(present - declared - {"summary.json", "SHA256SUMS"})
    missing = sorted(declared - present)
    if unexpected or missing:
        reason = f"package directory contents differ from declared artifacts: unexpected={unexpected} missing={missing}"
        branch = branch_outcome(None, incomplete=True)
        summary["branch_outcome"] = branch
        summary["package_summary"] = plain_language(branch, statuses)
        summary["run_metadata"]["incomplete_reason"] = reason  # type: ignore[index]
        summary["unexpected_artifacts"] = unexpected
        summary["missing_artifacts"] = missing
        summary["artifacts"] = sorted((declared - set(missing)) | {"summary.json"})
    problems = validate_package_summary(summary)
    if problems:
        raise AssertionError(f"package summary incomplete: {problems}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    (out / "branch_outcome.json").write_text(json.dumps({"statuses": dict(statuses), **branch}, indent=2, sort_keys=True) + "\n")
    manifest(out, summary["artifacts"])  # type: ignore[arg-type]
    print(f"PACKAGE FROZEN branch_outcome={branch['code']} table_row={branch['table_row']}", flush=True)
    return summary


# ------------------------------------------------------------- diagnostic
Executer = Callable[[str, Callable[..., Dict[str, object]], Sequence[object]], Tuple[List[Dict[str, object]], Dict[str, object]]]


def _provenance_text(provenance: Mapping[str, object]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True, default=_jsonable) + "\n"


def prepare_package_dir(out: Path, provenance: Mapping[str, object]) -> None:
    """Accept only a fresh location, or one holding exactly this run's provenance.

    - ``out`` absent: created exclusively, then this run's provenance written.
    - ``out`` present: its contents must be exactly ``provenance.json`` and that
      file must be byte-identical to this run's provenance (which carries a
      unique ``written_at``). Any other content, such as a prior full package,
      a hard-killed partial package, or a stray outcome file, is refused before
      any marker, ledger, or stage file is written, and nothing is modified.
    """
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
    """Fail-closed default written before D1 starts.

    If the process dies without reaching ``finalize_package`` (for example the
    RSS watchdog's hard ``os._exit``), the machine-readable outcome on disk is
    already STOP_INVALID / incomplete package, and nothing outcome-bearing.
    """
    code = "STOP_INVALID"
    _, condition, action = BRANCH_TABLE[BRANCH_INDEX[code] - 1]
    record = {
        "code": code,
        "table_row": BRANCH_INDEX[code],
        "condition": condition,
        "action": action,
        "statuses": {"D1": None, "D2": None, "D3": None, "D4": None},
        "incomplete_reason": "sequence has not reached finalize_package",
    }
    (out / "branch_outcome.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def run_diagnostic(
    args: argparse.Namespace,
    out: Path,
    provenance: Mapping[str, object],
    execute: Optional[Executer] = None,
    seeds: Optional[Sequence[int]] = None,
    d1_identity: Optional[Callable[[int], Mapping[str, str]]] = None,
    condition_identity: Optional[Callable[[int, str, bool], Mapping[str, Optional[str]]]] = None,
    n_boot: Optional[int] = None,
) -> Dict[str, object]:
    """Spec section 8 sequence. Injection points exist for construction tests only."""
    from snn import stage2_diagnostic as diag
    from snn import stage2_diagnostic_stats as dstat

    # Fail closed before anything is written: the package location must be
    # fresh (or hold only this run's own provenance from preflight).
    prepare_package_dir(out, provenance)

    seeds = tuple(diag.DIAGNOSTIC_SEEDS if seeds is None else seeds)
    n_boot = dstat.N_BOOT if n_boot is None else n_boot
    d1_identity = d1_identity or diag.d1_object_hashes
    condition_identity = condition_identity or diag.condition_object_hashes
    artifacts: List[str] = []

    def _default_execute(stage, fn, jobs):
        artifacts.append(f"{stage}.progress.jsonl")  # declared before the ledger exists
        return run_pool(fn, jobs, args.workers, stage, out)

    run_stage: Executer = execute or _default_execute  # type: ignore[assignment]

    run_meta: List[Dict[str, object]] = []
    stages: Dict[str, object] = {}
    statuses: Dict[str, object] = {"D1": None, "D2": None, "D3": None, "D4": None}
    identities: Dict[Tuple[str, bool], Dict[int, Mapping[str, Optional[str]]]] = {}

    def ids(task: str, drive: bool) -> Dict[int, Mapping[str, Optional[str]]]:
        if (task, drive) not in identities:
            identities[(task, drive)] = {s: condition_identity(s, task, drive) for s in seeds}  # type: ignore[misc]
        return identities[(task, drive)]

    def condition_jobs(task: str, plastics: Sequence[bool], drive: bool):
        # Every shared object is constructed and hashed in the coordinator
        # before a worker executes a transition; each worker reasserts it.
        expected = ids(task, drive)
        return [(seed, task, plastic, drive, expected[seed]) for plastic in plastics for seed in seeds]

    def stop(reason: Optional[str] = None) -> Dict[str, object]:
        return finalize_package(out, provenance, stages, statuses, run_meta, artifacts, incomplete_reason=reason)

    write_incomplete_marker(out)
    try:
        # ---- D1
        d1_expected = {s: d1_identity(s) for s in seeds}
        d1_rows, meta = run_stage("D1", _d1_job, list(seeds))
        run_meta.append(meta)
        d1_rows.sort(key=_row_key)
        d1 = diag.summarize_d1(d1_rows, d1_expected, seeds)
        artifacts.append(write_stage_rows(out, "d1_rows", d1_rows, d1["rows_schema_valid"] is True))
        if d1["valid"]:
            d1["normalization_hashes"] = {
                str(r["seed"]): {n: r["decoders"][n]["normalization_sha256"] for n in diag.D1_DECODERS} for r in d1_rows  # type: ignore[index]
            }
            d1["coefficient_hashes"] = {
                str(r["seed"]): {n: r["decoders"][n]["coefficients_sha256"] for n in diag.D1_DECODERS} for r in d1_rows  # type: ignore[index]
            }
            d1["convergence"] = {str(r["seed"]): {n: r["decoders"][n]["converged"] for n in diag.D1_DECODERS} for r in d1_rows}  # type: ignore[index]
        stages["D1"] = d1
        statuses["D1"] = d1["status"]
        if d1["status"] != "PASS":
            return stop()

        # ---- D2 (shared no-drive A rows, reused by D4)
        d2_rows, meta = run_stage("D2_A_no_drive", _condition_job, condition_jobs("A", (True, False), False))
        run_meta.append(meta)
        d2_rows.sort(key=_row_key)
        d2 = dstat.summarize_d2(d2_rows, {("A", False): ids("A", False)}, seeds)
        artifacts.append(write_stage_rows(out, "d2_rows", d2_rows, d2["rows_schema_valid"] is True))
        stages["D2"] = d2
        statuses["D2"] = d2["status"]
        if d2["status"] == "D2_INVALID":
            return stop()

        # ---- D3
        d3_rows, meta = run_stage("D3_lag1_no_drive", _condition_job, condition_jobs("lag1", (True, False), False))
        run_meta.append(meta)
        d3_rows.sort(key=_row_key)
        d3 = dstat.summarize_d3(d3_rows, {("lag1", False): ids("lag1", False)}, seeds, n_boot=n_boot)
        artifacts.append(write_stage_rows(out, "d3_rows", d3_rows, d3["rows_schema_valid"] is True))
        stages["D3"] = d3
        statuses["D3"] = d3["status"]
        if d3["status"] == "INVALID":
            return stop()

        # ---- D4 (only the predeclared drive rows)
        drive_jobs = condition_jobs("A", (True, False), True) + condition_jobs("lag1", (True,), True)
        drive_rows, meta = run_stage("D4_drive", _condition_job, drive_jobs)
        run_meta.append(meta)
        drive_rows.sort(key=_row_key)
        d4_input = d2_rows + [r for r in d3_rows if r.get("condition") == "P_lag1-no"] + drive_rows
        d4 = dstat.summarize_d4(
            d4_input,
            {k: ids(*k) for k in (("A", False), ("lag1", False), ("A", True), ("lag1", True))},
            seeds,
            n_boot=n_boot,
        )
        artifacts.append(write_stage_rows(out, "d4_drive_rows", drive_rows, d4["rows_schema_valid"] is True))
        stages["D4"] = d4
        statuses["D4"] = d4["status"]
        return stop()
    except Exception as exc:  # any unplanned stop still freezes an outcome-free INVALID package
        stages["exception"] = {"type": type(exc).__name__}
        return stop(reason=f"exception during sequence: {type(exc).__name__}")


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
    params = validated_effective_parameters()
    result = {
        "mode": "shakedown",
        "seed": seed,
        "full_diagnostic_seed": False,
        "checkpoint_rows": len(rows),
        "checkpoint_schema": "PASS",
        "evaluation_scored": ev["scored"],
        "evaluation_nonmutating": ev["weights_bitwise_constant"],
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
    }
    (out / "shakedown.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest(out, ["provenance.json", "shakedown.json"])
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = parser().parse_args()
    if args.ss_probe:
        # The retained collision probe lives at its own fixed, manifest-verified
        # location and is read back by ``verified_probe``; it is never a package.
        out = Path(args.out) if args.out else ROOT / "results_stage2_diagnostic"
        command = [sys.executable, str(ROOT / "tools" / "stage2_diagnostic_ss_probe.py"), "--out", str(out / "ss_probe.json")]
        subprocess.run(command, cwd=ROOT, check=True)
        return
    mode = "diagnostic" if args.run_diagnostic else "shakedown"
    out = Path(args.out) if args.out else default_run_dir(mode)
    if out.exists() or out.is_symlink():  # refuse before preflight writes anything
        raise StaleOutputError(f"refusing package: output location {out} already exists; packages require a fresh run-unique directory")
    provenance = preflight(args, mode, out)
    if args.shakedown:
        run_shakedown(out)
    else:
        summary = run_diagnostic(args, out, provenance)
        if summary["branch_outcome"]["code"] == "STOP_INVALID":  # type: ignore[index]
            raise SystemExit(3)


if __name__ == "__main__":
    main()
