#!/usr/bin/env python3
"""Stage 2 diagnostic D2-D4 entry with D1 = PASS by D1R2.

Governing text, docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md (f3d52ee) section 9 row 6:
"D1 is PASS (by D1R2). D2-D4 only per ebb90e74 section 8 steps 3-6 under a
separate run authorization; outcomes governed by ebb90e74 section 10 row 1 and
rows 5 onward. No r4 from D1R2 alone."

This runner executes ebb90e74 (docs/STAGE2_DIAGNOSTIC_SPEC.md) section 8 steps
3-6 only: the shared no-drive D2 rows, D3 and its frozen companion, the
predeclared D4 drive rows, and the package freeze. It never runs, re-scores,
reinterprets, or pools D1, D1R, or D1R2. D1 status is taken from the committed,
Nora-approved D1R2 package (results commit 3077119), verified file by file
against pinned SHA-256 values and against Git, and refused unless its recorded
branch is D1R2_PASS_D2_D4_ELIGIBLE (section 9 row 6).

All D2-D4 science is the approved ebb90e74 implementation, imported unchanged:
the worker job (``run_stage2_diagnostic._condition_job`` ->
``snn.stage2_diagnostic.run_condition_seed``), the stage pool, the validators
and readouts (``snn.stage2_diagnostic_stats.summarize_d2/d3/d4``), the
bootstrap, the effective-parameter transcription, and the ordered section 10
branch table (``run_stage2_diagnostic.branch_outcome``). With D1 fixed to PASS
only section 10 row 1 and rows 5 onward are reachable.

D1R2 spec hard limit 7: no D1R2 decoder may enter a training or evaluation
path of D2-D4. This module imports no D1R2 module and no decoder library. The
D1R2 package is read as bytes for hashing and as two JSON records (summary and
branch outcome) for the recorded status and claim text; the D1R2 row file,
which holds the decoder coefficients, is hashed and never parsed.

Modes:
  --shakedown   construction shakedown on non-diagnostic fixture seed 4242 only
  --ss-probe    re-executes the approved ebb90e74 section 3.2 probe and checks
                it against the retained output (identity enumeration only)
  --run-d2d4    LOCKED: requires a Nora implementation APPROVE record naming the
                exact pushed commit plus a separate run authorization card
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_stage2_diagnostic as base
from run_stage2 import _jsonable

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_stage2_diagnostic_d2d4"
DIAG_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_SPEC.md"
STAGE2_SPEC = ROOT / "docs" / "STAGE2_SPEC.md"
D1R_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_D1R_SPEC.md"
D1R2_SPEC = ROOT / "docs" / "STAGE2_DIAGNOSTIC_D1R2_SPEC.md"

FROZEN_SPEC_SHA256 = {
    "docs/STAGE2_SPEC.md": "695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd",
    "docs/STAGE2_DIAGNOSTIC_SPEC.md": "7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf",
    "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md": "c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743",
    "docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md": "e5a6a5f4eb7a7adafd9255ac6504593e45be51eefbfa92ce24c63b70fafb82a2",
}

# ---------------------------------------------------------------- D1R2 gate
D1R2_RESULTS_COMMIT = "3077119f4dd5710245e0b49a652a067ed58d2bff"
D1R2_SOURCE_COMMIT = "e8e014913c002639bcd1c453e397d2f5f5e9c07c"
D1R2_PACKAGE_REL = "results_stage2_diagnostic_d1r2/runs/d1r2-20261002T103331160786Z-e8e014913c00-pid75833"
D1R2_PACKAGE_SHA256: Dict[str, str] = {
    "D1R2.events.jsonl": "98086b9e02ac00012ec864df01b37759e178d7009da360f47bf8ef7fe67bbeec",
    "D1R2.progress.jsonl": "cdf9e42a8cfc5a1a9c00169eab42e6b4b08ab87a42f044c50618d7de2dacde63",
    "SHA256SUMS": "235334300172362cebfa76fbf23276b9bcbf5fa581b90b7470650ddca05a9737",
    "branch_outcome.json": "afeb2bc371774f80eb9f764c3722b8e406ed884e4cf80f0de59473ea6dce78dc",
    "d1r2_rows.jsonl": "00c1929f5f503432d967f4ff6fdfb1a1a417a241f964b9ed97f03b4f0ffec1cb",
    "provenance.json": "8067aa809ab39b52dcdd4229b3bd4d64993bfe7f56ef35dc9f6d4b951f6f718c",
    "summary.json": "95f465ad7b0e6302a9f2f02df3d3111cd8b8ccc58f2f54cf823798ab3c1bc9d0",
}
D1R2_REQUIRED_CODE = "D1R2_PASS_D2_D4_ELIGIBLE"
D1R2_REQUIRED_ROW = 6
D1R2_DECODERS = ("pipeline", "network_positive", "A")
# Verbatim from docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md section 9 row 6 (test-pinned).
GOVERNING_BRANCH_TEXT = (
    "D1 is `PASS` (by D1R2). D2-D4 only per ebb90e74 section 8 steps 3-6 under a separate run "
    "authorization; outcomes governed by ebb90e74 section 10 row 1 and rows 5 onward. No r4 from D1R2 alone."
)
# ebb90e74 section 10 rows that a D1 = PASS entry can never select.
D1_ONLY_BRANCHES = ("STOP_D1_INVALID_HARNESS", "STOP_D1_FAIL", "STOP_D1_INCONCLUSIVE")

STAGES = ("D2_A_no_drive", "D3_lag1_no_drive", "D4_drive")

SCI_FILES = tuple(
    sorted(
        set(base.SCI_FILES)
        | {
            "run_stage2.py",
            "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md",
            "docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md",
            "run_stage2_diagnostic_d2d4.py",
            "tests/test_stage2_diagnostic_d2d4.py",
            "docs/STAGE2_DIAGNOSTIC_D2D4_IMPLEMENTATION_NOTES.md",
            "results_stage2_diagnostic/ss_probe.json",
            "results_stage2_diagnostic/SHA256SUMS",
        }
        | {f"{D1R2_PACKAGE_REL}/{name}" for name in D1R2_PACKAGE_SHA256}
    )
)


class D1R2GateFailure(SystemExit):
    """The D1R2 package is missing, altered, uncommitted, or not section 9 row 6."""


class UnauthorizedSeed(ValueError):
    """A construction path was asked to simulate a diagnostic or D1R/D1R2 seed."""


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def _git_checks(rel_dir: str, names: Sequence[str]) -> List[str]:
    problems: List[str] = []
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", D1R2_RESULTS_COMMIT, "HEAD"], cwd=ROOT, check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        problems.append(f"results commit {D1R2_RESULTS_COMMIT} is not an ancestor of HEAD")
        return problems
    for name in names:
        rel = f"{rel_dir}/{name}"
        try:
            committed = git("rev-parse", f"{D1R2_RESULTS_COMMIT}:{rel}")
            working = git("hash-object", rel)
        except (subprocess.CalledProcessError, OSError):
            problems.append(f"{rel}: not present in results commit {D1R2_RESULTS_COMMIT[:12]}")
            continue
        if committed != working:
            problems.append(f"{rel}: working-tree blob differs from results commit")
    try:
        if git("status", "--porcelain", "--", rel_dir):
            problems.append(f"{rel_dir}: uncommitted change or untracked file")
    except (subprocess.CalledProcessError, OSError):
        problems.append("git status unavailable")
    return problems


def verify_d1r2_package(
    package_dir: Optional[Path] = None,
    pinned: Optional[Mapping[str, str]] = None,
    require_git: bool = True,
) -> Dict[str, object]:
    """Fail-closed D1 = PASS gate. Reads the recorded outcome only; never re-scores.

    Returns a record with ``pass`` and ``problems``. The overrides exist only for
    tests on tmp copies; the runner always uses the canonical path, the pinned
    hashes, and the Git check.
    """
    rel_dir = D1R2_PACKAGE_REL
    package_dir = ROOT / rel_dir if package_dir is None else Path(package_dir)
    pinned = dict(D1R2_PACKAGE_SHA256 if pinned is None else pinned)
    problems: List[str] = []
    record: Dict[str, object] = {
        "package_path": rel_dir,
        "results_commit": D1R2_RESULTS_COMMIT,
        "source_commit_required": D1R2_SOURCE_COMMIT,
        "required_branch_code": D1R2_REQUIRED_CODE,
        "required_table_row": D1R2_REQUIRED_ROW,
        "governing_text": GOVERNING_BRANCH_TEXT,
        "file_sha256": {},
    }

    def done() -> Dict[str, object]:
        record["problems"] = problems
        record["pass"] = not problems
        return record

    if package_dir.is_symlink() or not package_dir.is_dir():
        problems.append(f"D1R2 package directory missing: {package_dir}")
        return done()
    present = sorted(p.name for p in package_dir.iterdir())
    if present != sorted(pinned):
        problems.append(f"D1R2 package contents {present} differ from pinned {sorted(pinned)}")
    for name, want in sorted(pinned.items()):
        path = package_dir / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"{name}: missing or not a regular file")
            continue
        have = file_sha(path)
        record["file_sha256"][name] = have  # type: ignore[index]
        if have != want:
            problems.append(f"{name}: SHA-256 {have} != pinned {want}")
    if problems:
        return done()

    # Manifest: exactly the six declared artifacts, each matching.
    lines = (package_dir / "SHA256SUMS").read_text().splitlines()
    listed = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        listed[name] = digest
    if sorted(listed) != sorted(n for n in pinned if n != "SHA256SUMS"):
        problems.append(f"SHA256SUMS lists {sorted(listed)}")
    for name, digest in listed.items():
        if (package_dir / name).is_file() and file_sha(package_dir / name) != digest:
            problems.append(f"SHA256SUMS mismatch for {name}")

    # Recorded outcome (read, not recomputed).
    try:
        branch = json.loads((package_dir / "branch_outcome.json").read_text())
        summary = json.loads((package_dir / "summary.json").read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"D1R2 JSON unreadable: {type(exc).__name__}")
        return done()
    if branch.get("code") != D1R2_REQUIRED_CODE or branch.get("table_row") != D1R2_REQUIRED_ROW or branch.get("status") != "PASS":
        problems.append(f"branch_outcome.json is {branch.get('code')!r} row {branch.get('table_row')!r} status {branch.get('status')!r}")
    sbranch = summary.get("branch") or {}
    if summary.get("status") != "PASS" or sbranch.get("code") != D1R2_REQUIRED_CODE or sbranch.get("table_row") != D1R2_REQUIRED_ROW:
        problems.append("summary.json status/branch is not PASS / D1R2_PASS_D2_D4_ELIGIBLE / row 6")
    if (summary.get("branch_outcome") or {}).get("code") != D1R2_REQUIRED_CODE:
        problems.append("summary.json branch_outcome is not D1R2_PASS_D2_D4_ELIGIBLE")
    readouts = summary.get("readouts") or {}
    statuses = {k: (readouts.get(k) or {}).get("status") for k in D1R2_DECODERS}
    if any(v != "PASS" for v in statuses.values()):
        problems.append(f"summary.json readout statuses {statuses}")
    if summary.get("violations"):
        problems.append("summary.json carries violations")
    if summary.get("effective_parameters_match_spec") is not True:
        problems.append("summary.json effective parameters did not match the D1R2 spec")
    prov = summary.get("provenance") or {}
    if prov.get("code_commit") != D1R2_SOURCE_COMMIT:
        problems.append(f"D1R2 source commit {prov.get('code_commit')!r} != {D1R2_SOURCE_COMMIT}")
    if prov.get("d1r2_spec_sha256_at_run") != FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md"]:
        problems.append("D1R2 run did not record the frozen D1R2 spec hash")
    if prov.get("ebb90e74_sha256_at_run") != FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_SPEC.md"]:
        problems.append("D1R2 run did not record the frozen ebb90e74 hash")
    claim = summary.get("network_claim")
    if not isinstance(claim, str) or not claim:
        problems.append("summary.json network_claim missing")
    record.update(
        {
            "recorded_status": summary.get("status"),
            "recorded_branch": {k: sbranch.get(k) for k in ("code", "table_row", "condition", "action")},
            "recorded_readout_statuses": statuses,
            "recorded_source_commit": prov.get("code_commit"),
            "network_claim": claim,
        }
    )
    if require_git and not problems:
        problems.extend(_git_checks(rel_dir, sorted(pinned)))
    record["git_verified"] = bool(require_git and not problems)
    return done()


# ---------------------------------------------------------------- probe check
def load_base_probe():
    path = ROOT / "tools" / "stage2_diagnostic_ss_probe.py"
    spec = importlib.util.spec_from_file_location("stage2_diagnostic_ss_probe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def probe_check() -> Dict[str, object]:
    """ebb90e74 section 3.2: re-execute the approved probe and match the retained output.

    Identity enumeration only (SeedSequence.generate_state(8)); no network,
    stream, or drive realization is constructed. The cross-namespace checks
    against namespaces 31 and 32 are the retained, approved D1R and D1R2 probes,
    which enumerated every namespace-30 identity as legacy.
    """
    retained = base.verified_probe()
    mod = load_base_probe()
    fresh = mod.probe()
    keys = ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions")
    retained_json = json.loads((ROOT / "results_stage2_diagnostic" / "ss_probe.json").read_text())
    later = {}
    for name in ("results_stage2_diagnostic_d1r", "results_stage2_diagnostic_d1r2"):
        rec = json.loads((ROOT / name / "ss_probe.json").read_text())
        later[name] = {
            "decision": rec.get("decision"),
            "identity_state_sha256": rec.get("identity_state_sha256"),
            "new_vs_legacy_collisions": [len(rec.get(k) or []) for k in keys[2:]],
            "output_sha256": file_sha(ROOT / name / "ss_probe.json"),
        }
    passed = bool(
        retained["pass"]
        and fresh["decision"] == "PASS"
        and fresh["identity_state_sha256"] == retained_json["identity_state_sha256"]
        and fresh["counts"] == retained_json["counts"]
        and fresh["known_legacy_only_duplicate_count"] == retained_json["known_legacy_only_duplicate_count"]
        and not any(fresh[k] for k in keys)
        and all(v["decision"] == "PASS" and v["new_vs_legacy_collisions"] == [0, 0] for v in later.values())
    )
    return {
        "execution": "ebb90e74 section 3.2 probe re-executed in memory",
        "pass": passed,
        "decision": fresh["decision"],
        "numpy_version": np.__version__,
        "counts": fresh["counts"],
        "known_legacy_only_duplicate_count": fresh["known_legacy_only_duplicate_count"],
        "new_collisions": [len(fresh[k]) for k in keys],
        "identity_state_sha256": fresh["identity_state_sha256"],
        "retained": retained,
        "retained_identity_state_sha256": retained_json["identity_state_sha256"],
        "probe_code_sha256": file_sha(ROOT / "tools" / "stage2_diagnostic_ss_probe.py"),
        "namespace_31_32_probes_with_namespace_30_as_legacy": later,
    }


# ---------------------------------------------------------------- preflight
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shakedown", action="store_true", help="fixture seed 4242 only, short lengths, no readout")
    mode.add_argument("--ss-probe", action="store_true", help="ebb90e74 section 3.2 probe check (identities only)")
    mode.add_argument("--run-d2d4", action="store_true", help="LOCKED: Nora APPROVE + separate run authorization")
    p.add_argument("--approval-file", help="implementation APPROVE + run authorization record (run only)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", default=None, help="package directory; must not exist")
    p.add_argument("--tests", default="not recorded")
    return p


def _approval(path: Optional[str], head: str) -> Dict[str, object]:
    if not path:
        raise SystemExit("refusing D2-D4 run: --approval-file is required")
    record = json.loads(Path(path).read_text())
    required = {
        "decision": "APPROVE",
        "reviewer": "nora",
        "implementation_commit": head,
        "diagnostic_spec_sha256": FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_SPEC.md"],
        "d1r2_spec_sha256": FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md"],
        "d1r2_results_commit": D1R2_RESULTS_COMMIT,
        "run_authorized": True,
    }
    mismatches = {k: (record.get(k), v) for k, v in required.items() if record.get(k) != v}
    if not isinstance(record.get("run_authorization_card"), str) or not record.get("run_authorization_card"):
        mismatches["run_authorization_card"] = (record.get("run_authorization_card"), "<separate run card id>")
    if mismatches:
        raise SystemExit(f"refusing D2-D4 run: approval record mismatch {mismatches}")
    return record


def spec_hashes() -> Dict[str, Dict[str, object]]:
    out = {}
    for rel, want in FROZEN_SPEC_SHA256.items():
        have = file_sha(ROOT / rel)
        out[rel] = {"frozen": want, "at_run": have, "unchanged": have == want}
    return out


def default_run_dir(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    head = git("rev-parse", "--short=12", "HEAD")
    return RESULTS / "runs" / f"{mode}-{stamp}-{head}-pid{os.getpid()}"


def preflight(args: argparse.Namespace, mode: str, out: Path) -> Dict[str, object]:
    """Every check runs before anything is written. A run refuses on any failure."""
    from snn import stage2_diagnostic as diag

    if not 1 <= args.workers <= 5:
        raise SystemExit("workers must be 1..5 (ebb90e74 section 12)")
    head = git("rev-parse", "HEAD")
    try:
        upstream: Optional[str] = git("rev-parse", "@{u}")
    except subprocess.CalledProcessError:
        upstream = None
    approval = _approval(args.approval_file, head) if mode == "d2d4" else None
    gate = verify_d1r2_package()
    specs = spec_hashes()
    dirty = git("status", "--porcelain", "--", *SCI_FILES)
    probe = probe_check()
    params = base.validated_effective_parameters()
    prov: Dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "entry": "ebb90e74 section 8 steps 3-6 with D1 = PASS by D1R2 (D1R2 spec section 9 row 6)",
        "code_commit": head,
        "upstream_tracking_ref_sha": upstream,
        "code_commit_equals_upstream": head == upstream,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "diagnostic_spec_commit": diag.SPEC_COMMIT,
        "frozen_specs": specs,
        "frozen_specs_unchanged": all(v["unchanged"] for v in specs.values()),
        "d1_by_d1r2_gate": gate,
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
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "python": sys.version,
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "psutil": __import__("psutil").__version__,
        "workers": args.workers,
        "seeds": list(diag.DIAGNOSTIC_SEEDS) if mode == "d2d4" else [FIXTURE_SEED],
        "maintained_tests": args.tests,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out),
    }
    checks = {
        "d1r2_gate": gate["pass"],
        "frozen_specs": prov["frozen_specs_unchanged"],
        "ss_probe": probe["pass"],
        "params": params["matches_spec"],
    }
    if mode == "d2d4":
        checks.update({"pushed": prov["code_commit_equals_upstream"], "clean": prov["scientific_files_clean"], "approval": approval is not None})
    if not all(checks.values()):
        failed = sorted(k for k, v in checks.items() if not v)
        if mode == "d2d4" or not gate["pass"]:
            raise D1R2GateFailure(f"refusing {mode}: preflight failed {failed}; gate problems {gate.get('problems')}")
        raise SystemExit(f"refusing {mode}: preflight failed {failed}")
    base.claim_fresh_run_dir(out)
    (out / "provenance.json").write_text(_provenance_text(prov))
    return prov


def _provenance_text(provenance: Mapping[str, object]) -> str:
    return json.dumps(provenance, indent=2, sort_keys=True, default=_jsonable) + "\n"


# ---------------------------------------------------------------- package
def d1_stage_record(gate: Mapping[str, object]) -> Dict[str, object]:
    """The D1 stage entry of the package: a reference to the D1R2 package, never a re-score."""
    return {
        "status": "PASS",
        "source": "D1R2 (docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md section 9 row 6)",
        "executed_here": False,
        "rescored_here": False,
        "package_path": gate.get("package_path"),
        "results_commit": gate.get("results_commit"),
        "source_commit": gate.get("recorded_source_commit"),
        "file_sha256": gate.get("file_sha256"),
        "recorded_branch": gate.get("recorded_branch"),
        "recorded_readout_statuses": gate.get("recorded_readout_statuses"),
        "governing_text": GOVERNING_BRANCH_TEXT,
        "claim_boundary": gate.get("network_claim"),
    }


def plain_language(branch: Mapping[str, object], statuses: Mapping[str, object], d1_claim: str) -> Dict[str, object]:
    """base.plain_language with the D1 line sourced from the D1R2 package."""
    reached = {k: v for k, v in statuses.items() if v is not None}
    supported: List[str] = []
    if branch["code"] == "STOP_INVALID":
        supported.append("Nothing beyond the invalidity itself: the sequence stopped before any decision-bearing readout could be inspected.")
    else:
        supported.append(f"D1 PASS by D1R2 (results commit {D1R2_RESULTS_COMMIT[:7]}, {D1R2_REQUIRED_CODE}): {d1_claim}")
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
        "not_supported": list(base.NOT_SUPPORTED),
    }


def finalize_package(
    out: Path,
    provenance: Mapping[str, object],
    stages: Mapping[str, object],
    statuses: Mapping[str, object],
    run_meta: Sequence[Mapping[str, object]],
    artifacts: Sequence[str],
    d1_claim: str,
    incomplete_reason: Optional[str] = None,
) -> Dict[str, object]:
    """base.finalize_package, with the D1 claim line taken from the D1R2 package."""
    params = base.validated_effective_parameters()
    branch = base.branch_outcome(
        statuses.get("D1"),  # type: ignore[arg-type]
        statuses.get("D2"),  # type: ignore[arg-type]
        statuses.get("D3"),  # type: ignore[arg-type]
        stages.get("D4"),  # type: ignore[arg-type]
        incomplete=incomplete_reason is not None,
    )
    summary: Dict[str, object] = {
        "schema_version": 1,
        "entry": "D2-D4 with D1 = PASS by D1R2",
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
        "package_summary": plain_language(branch, statuses, d1_claim),
        "claim_boundary": base.CLAIM_BOUNDARY,
        "no_oracle": base.NO_ORACLE,
        "d1_claim_boundary": d1_claim,
        "governing_text": GOVERNING_BRANCH_TEXT,
        "artifacts": sorted(set(artifacts) | {"summary.json", "branch_outcome.json", "provenance.json"}),
    }
    present = set(base.package_files(out))
    declared = set(summary["artifacts"]) - {"summary.json"}  # type: ignore[arg-type]
    unexpected = sorted(present - declared - {"summary.json", "SHA256SUMS"})
    missing = sorted(declared - present)
    if unexpected or missing:
        reason = f"package directory contents differ from declared artifacts: unexpected={unexpected} missing={missing}"
        branch = base.branch_outcome(None, incomplete=True)
        summary["branch_outcome"] = branch
        summary["package_summary"] = plain_language(branch, statuses, d1_claim)
        summary["run_metadata"]["incomplete_reason"] = reason  # type: ignore[index]
        summary["unexpected_artifacts"] = unexpected
        summary["missing_artifacts"] = missing
        summary["artifacts"] = sorted((declared - set(missing)) | {"summary.json"})
    if branch["code"] in D1_ONLY_BRANCHES:  # unreachable with D1 = PASS; refuse rather than mislabel
        raise AssertionError(f"D1-only branch {branch['code']} selected on a D1 = PASS entry")
    problems = base.validate_package_summary(summary)
    if problems:
        raise AssertionError(f"package summary incomplete: {problems}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable) + "\n")
    (out / "branch_outcome.json").write_text(json.dumps({"statuses": dict(statuses), **branch}, indent=2, sort_keys=True) + "\n")
    base.manifest(out, summary["artifacts"])  # type: ignore[arg-type]
    print(f"PACKAGE FROZEN branch_outcome={branch['code']} table_row={branch['table_row']}", flush=True)
    return summary


Executer = Callable[[str, Callable[..., Dict[str, object]], Sequence[object]], Tuple[List[Dict[str, object]], Dict[str, object]]]


def run_d2_d4(
    args: argparse.Namespace,
    out: Path,
    provenance: Mapping[str, object],
    execute: Optional[Executer] = None,
    seeds: Optional[Sequence[int]] = None,
    condition_identity: Optional[Callable[[int, str, bool], Mapping[str, Optional[str]]]] = None,
    n_boot: Optional[int] = None,
    gate: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """ebb90e74 section 8 steps 3-6. Injection points exist for construction tests only.

    Steps 3-6 below are the approved ``run_stage2_diagnostic.run_diagnostic``
    body after its D1 stage, carried line for line; the only difference is that
    ``statuses['D1']`` comes from the verified D1R2 gate instead of a D1 stage.
    """
    from snn import stage2_diagnostic as diag
    from snn import stage2_diagnostic_stats as dstat

    # The real worker pool is reachable only from an approved, authorized run.
    if execute is None and not (provenance.get("mode") == "d2d4" and provenance.get("approval")):
        raise SystemExit("refusing D2-D4: the real stage pool requires an approved --run-d2d4 preflight")
    # Re-verify the gate at entry (defence in depth); refuse before writing anything.
    gate = verify_d1r2_package() if gate is None else gate
    if not gate.get("pass"):
        raise D1R2GateFailure(f"refusing D2-D4: D1R2 gate failed {gate.get('problems')}")
    d1_claim = str(gate.get("network_claim"))

    base.prepare_package_dir(out, provenance)

    seeds = tuple(diag.DIAGNOSTIC_SEEDS if seeds is None else seeds)
    n_boot = dstat.N_BOOT if n_boot is None else n_boot
    condition_identity = condition_identity or diag.condition_object_hashes
    artifacts: List[str] = []

    def _default_execute(stage, fn, jobs):
        artifacts.append(f"{stage}.progress.jsonl")  # declared before the ledger exists
        return base.run_pool(fn, jobs, args.workers, stage, out)

    run_stage: Executer = execute or _default_execute  # type: ignore[assignment]

    run_meta: List[Dict[str, object]] = []
    stages: Dict[str, object] = {"D1": d1_stage_record(gate)}
    statuses: Dict[str, object] = {"D1": "PASS", "D2": None, "D3": None, "D4": None}
    identities: Dict[Tuple[str, bool], Dict[int, Mapping[str, Optional[str]]]] = {}

    def ids(task: str, drive: bool) -> Dict[int, Mapping[str, Optional[str]]]:
        if (task, drive) not in identities:
            identities[(task, drive)] = {s: condition_identity(s, task, drive) for s in seeds}  # type: ignore[misc]
        return identities[(task, drive)]

    def condition_jobs(task: str, plastics: Sequence[bool], drive: bool):
        expected = ids(task, drive)
        return [(seed, task, plastic, drive, expected[seed]) for plastic in plastics for seed in seeds]

    def stop(reason: Optional[str] = None) -> Dict[str, object]:
        return finalize_package(out, provenance, stages, statuses, run_meta, artifacts, d1_claim, incomplete_reason=reason)

    (out / "d1_reference.json").write_text(json.dumps(stages["D1"], indent=2, sort_keys=True) + "\n")
    artifacts.append("d1_reference.json")
    base.write_incomplete_marker(out)
    try:
        # ---- step 3: D2 (shared no-drive A rows, reused by D4)
        d2_rows, meta = run_stage("D2_A_no_drive", base._condition_job, condition_jobs("A", (True, False), False))
        run_meta.append(meta)
        d2_rows.sort(key=base._row_key)
        d2 = dstat.summarize_d2(d2_rows, {("A", False): ids("A", False)}, seeds)
        artifacts.append(base.write_stage_rows(out, "d2_rows", d2_rows, d2["rows_schema_valid"] is True))
        stages["D2"] = d2
        statuses["D2"] = d2["status"]
        if d2["status"] == "D2_INVALID":
            return stop()

        # ---- step 4: D3 and its frozen companion
        d3_rows, meta = run_stage("D3_lag1_no_drive", base._condition_job, condition_jobs("lag1", (True, False), False))
        run_meta.append(meta)
        d3_rows.sort(key=base._row_key)
        d3 = dstat.summarize_d3(d3_rows, {("lag1", False): ids("lag1", False)}, seeds, n_boot=n_boot)
        artifacts.append(base.write_stage_rows(out, "d3_rows", d3_rows, d3["rows_schema_valid"] is True))
        stages["D3"] = d3
        statuses["D3"] = d3["status"]
        if d3["status"] == "INVALID":
            return stop()

        # ---- step 5: D4 (only the predeclared drive rows)
        drive_jobs = condition_jobs("A", (True, False), True) + condition_jobs("lag1", (True,), True)
        drive_rows, meta = run_stage("D4_drive", base._condition_job, drive_jobs)
        run_meta.append(meta)
        drive_rows.sort(key=base._row_key)
        d4_input = d2_rows + [r for r in d3_rows if r.get("condition") == "P_lag1-no"] + drive_rows
        d4 = dstat.summarize_d4(
            d4_input,
            {k: ids(*k) for k in (("A", False), ("lag1", False), ("A", True), ("lag1", True))},
            seeds,
            n_boot=n_boot,
        )
        artifacts.append(base.write_stage_rows(out, "d4_drive_rows", drive_rows, d4["rows_schema_valid"] is True))
        stages["D4"] = d4
        statuses["D4"] = d4["status"]
        # ---- step 6: freeze the package; no section 10 branch action before review
        return stop()
    except Exception as exc:  # any unplanned stop still freezes an outcome-free INVALID package
        stages["exception"] = {"type": type(exc).__name__}
        return stop(reason=f"exception during sequence: {type(exc).__name__}")


# ---------------------------------------------------------------- shakedown
FIXTURE_SEED = 4242
SHAKEDOWN_TRAIN = 2_000  # two whole 1,000-step checkpoint windows
SHAKEDOWN_EVAL = 2_300
SHAKEDOWN_WARMUP = 2_000
FORBIDDEN_SEEDS = frozenset(range(0, 20)) | frozenset(range(1000, 1020)) | frozenset(range(2000, 2020)) | frozenset(range(2100, 2120)) | frozenset(range(2200, 2220))
SHAKEDOWN_CONDITIONS = (
    ("A", True, False), ("A", False, False),
    ("lag1", True, False), ("lag1", False, False),
    ("A", True, True), ("A", False, True), ("lag1", True, True),
)


def shakedown_rows(seed: int = FIXTURE_SEED) -> List[Dict[str, object]]:
    """Every D2-D4 condition through the approved worker body, short lengths, fixture seed only."""
    from snn import stage2_diagnostic as diag

    if seed in FORBIDDEN_SEEDS:
        raise UnauthorizedSeed(f"seed {seed} is an experimental, qualification, or diagnostic seed")
    rows = []
    for task, plastic, drive in SHAKEDOWN_CONDITIONS:
        expected = diag.condition_object_hashes(seed, task, drive, n_train=SHAKEDOWN_TRAIN, n_eval=SHAKEDOWN_EVAL)
        row = diag.run_condition_seed(seed, task, plastic, drive, expected, n_train=SHAKEDOWN_TRAIN, n_eval=SHAKEDOWN_EVAL, warmup=SHAKEDOWN_WARMUP)
        row["status"] = "ok"
        rows.append(row)
    return rows


def shakedown_checks(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Construction checks only. No accuracy, rate, interval, or readout is recorded."""
    from snn import stage2_diagnostic as diag
    from snn import stage2_diagnostic_stats as dstat

    seed = rows[0]["seed"]
    expected = {(r["task"], r["drive"]): diag.condition_object_hashes(seed, r["task"], r["drive"], n_train=SHAKEDOWN_TRAIN, n_eval=SHAKEDOWN_EVAL) for r in rows}  # type: ignore[index]
    per_row = {}
    for r in rows:
        problems = dstat.validate_condition_row(
            r, expected[(r["task"], r["drive"])], n_checkpoints=SHAKEDOWN_TRAIN // diag.CHECKPOINT_STEPS,
            phase_steps=SHAKEDOWN_TRAIN, eval_len=SHAKEDOWN_EVAL, eval_scored=SHAKEDOWN_EVAL - SHAKEDOWN_WARMUP,
        )
        per_row[str(r["condition"])] = {
            "schema_problems_at_short_lengths": problems,
            "rejected_at_spec_lengths": bool(dstat.validate_condition_row(r, expected[(r["task"], r["drive"])])),
            "checkpoints": len(r["checkpoints"]),  # type: ignore[arg-type]
            "weights_bitwise_constant": r["weights_bitwise_constant"],
            "evaluation_nonmutating": r["evaluation"]["weights_bitwise_constant"],  # type: ignore[index]
            "drive_sha256": r["drive_sha256"],
            "train_stream_sha256": r["train_stream_sha256"],
            "initial_weights_sha256": r["initial_weights_sha256"],
        }
    by = {str(r["condition"]): r for r in rows}
    drive_hashes = {by[c]["drive_sha256"] for c in dstat.DRIVE_CONDITIONS}
    same_task = all(
        len({by[c][f] for c in by if by[c]["task"] == task}) == 1
        for task in ("A", "lag1") for f in dstat.PAIR_HASH_FIELDS
    )
    onset = diag.onset_record(by["P_A-no"]["checkpoints"], by["F0_A-no"]["checkpoints"])  # type: ignore[arg-type]
    checks = {
        "seven_conditions": sorted(by) == sorted(set(dstat.D4_CONDITIONS) | set(dstat.D3_CONDITIONS)),
        "all_rows_schema_valid_at_short_lengths": all(not v["schema_problems_at_short_lengths"] for v in per_row.values()),
        "all_rows_refused_at_spec_lengths": all(v["rejected_at_spec_lengths"] for v in per_row.values()),
        "F0_weights_constant": all(by[c]["weights_bitwise_constant"] is True for c in by if c.startswith("F0")),
        "evaluation_nonmutating": all(v["evaluation_nonmutating"] is True for v in per_row.values()),
        "common_drive_across_with_drive_conditions": len(drive_hashes) == 1 and None not in drive_hashes,
        "drive_absent_without_drive": all(by[c]["drive_sha256"] is None for c in by if c.endswith("-no")),
        "paired_objects_identical_within_task": same_task,
        "initial_weights_identical_across_conditions": len({by[c]["initial_weights_sha256"] for c in by}) == 1,
        "d2_onset_mechanically_derivable": onset.get("label") in ("HIDDEN_FIRST", "OUTPUT_FIRST", "CO_ONSET"),
    }
    return {"checks": checks, "per_row": per_row, "pass": all(checks.values())}


def run_shakedown(out: Path, provenance: Mapping[str, object]) -> Dict[str, object]:
    import time

    t0 = time.time()
    rows = shakedown_rows(FIXTURE_SEED)
    checked = shakedown_checks(rows)
    params = base.validated_effective_parameters()
    gate = provenance["d1_by_d1r2_gate"]
    probe = provenance["ss_probe"]
    modules = sorted(m for m in sys.modules if "d1r2" in m or m == "sklearn" or m.startswith("sklearn."))
    result = {
        "mode": "shakedown",
        "seed": FIXTURE_SEED,
        "diagnostic_seed": False,
        "lengths": {"train": SHAKEDOWN_TRAIN, "eval": SHAKEDOWN_EVAL, "eval_warmup": SHAKEDOWN_WARMUP},
        "d1r2_gate_pass": gate["pass"],  # type: ignore[index]
        "ss_probe_pass": probe["pass"],  # type: ignore[index]
        "effective_parameters_match_spec": params["matches_spec"],
        "effective_parameter_mismatches": params["mismatches"],
        "construction": checked,
        "d1r2_or_decoder_modules_loaded": modules,
        "wall_seconds": time.time() - t0,
        "note": "Construction checks on non-diagnostic fixture seed 4242 at short lengths through the approved worker body. No accuracy, rate, interval, or readout is recorded. Not a D2-D4 datum.",
    }
    result["pass"] = bool(gate["pass"] and probe["pass"] and params["matches_spec"] and checked["pass"] and not modules)  # type: ignore[index]
    (out / "shakedown.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_jsonable) + "\n")
    base.manifest(out, ["provenance.json", "shakedown.json"])
    print(json.dumps({"pass": result["pass"], "checks": checked["checks"], "modules": modules}, sort_keys=True))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.ss_probe:
        record = probe_check()
        out = Path(args.out) if args.out else RESULTS / "ss_probe_check.json"
        if out.exists():
            raise SystemExit(f"refusing: {out} exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True, default=_jsonable) + "\n")
        (out.parent / "SHA256SUMS").write_text(
            f"{record['probe_code_sha256']}  tools/stage2_diagnostic_ss_probe.py\n{file_sha(out)}  {out.name}\n"
        )
        print(json.dumps({k: record[k] for k in ("pass", "decision", "counts", "new_collisions", "identity_state_sha256")}, indent=1))
        if not record["pass"]:
            raise SystemExit(5)
        return
    mode = "d2d4" if args.run_d2d4 else "shakedown"
    out = Path(args.out) if args.out else default_run_dir(mode)
    if out.exists() or out.is_symlink():
        raise base.StaleOutputError(f"refusing package: output location {out} already exists")
    provenance = preflight(args, mode, out)
    if args.shakedown:
        if not run_shakedown(out, provenance)["pass"]:
            raise SystemExit(4)
        return
    summary = run_d2_d4(args, out, provenance)  # re-verifies the D1R2 gate at entry
    if summary["branch_outcome"]["code"] == "STOP_INVALID":  # type: ignore[index]
        raise SystemExit(3)


if __name__ == "__main__":
    main()
