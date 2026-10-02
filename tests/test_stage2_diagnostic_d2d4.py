"""Construction tests for the D2-D4 entry with D1 = PASS by D1R2 (run_stage2_diagnostic_d2d4.py).

No diagnostic seed (2000..2019) is simulated, streamed, or scored here, and no
D1, D1R, or D1R2 row is run or re-scored. Sequence tests inject synthetic rows
built in memory under the synthetic IDs 1000..1019 (shared fixtures from
tests/test_stage2_diagnostic_package.py, never simulated). Network simulation
runs only on the non-diagnostic fixture seed 4242 at short lengths. The D1R2
package is read only through the gate under test.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_stage2_diagnostic as base
import run_stage2_diagnostic_d2d4 as d2d4
from snn import stage2_diagnostic as diag
from snn import stage2_diagnostic_stats as stats

import test_stage2_diagnostic_package as pkg  # shared synthetic fixtures (IDs 1000..1019)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / d2d4.D1R2_PACKAGE_REL
N_BOOT = pkg.N_BOOT
REAL_GATE = d2d4.verify_d1r2_package()

# Approved files the entry must leave byte-identical (pins carried from
# tests/test_stage2_diagnostic_d1r2.py plus the D1R2 implementation itself).
APPROVED_UNCHANGED_SHA256 = {
    **{k: v for k, v in __import__("test_stage2_diagnostic_d1r2").APPROVED_D1_PATH_SHA256.items()},
    "snn/stage2_diagnostic_d1r2.py": "89a39c081a26969d04439eee2dcde1ad40491bedf7d25e88e9b2cbda9f97dedf",
    "run_stage2_diagnostic_d1r2.py": "74abf92173be27e5d639a3bd6e9f52f192290a79cfdd18513c3590c3f696dd17",
    "tools/stage2_diagnostic_d1r2_ss_probe.py": "cf0bb3b1adf5a9e34f48b5901e92f6e75d6a979cd91d042436f79450172693b3",
    "tests/test_stage2_diagnostic_d1r2.py": "811c26960b32d8c64ea7f03d751ad9e1cda9e5c01785024a370f50a924c9e668",
    "results_stage2_diagnostic_d1r2/ss_probe.json": "5a293367a61fcfd0c62e7f46bd3e316151c420fa3b019386ea60e5e130cd14fb",
    "results_stage2_diagnostic_d1r2/SHA256SUMS": "5fb06d4bfc5a3118a0210cbf8859949e8775011707d08225e23d33f2509446df",
    "docs/STAGE2_DIAGNOSTIC_IMPLEMENTATION_NOTES.md": "20805e57e96c9c96970c1941690df5ef8f3c4df27567acfa8a03be119acca2b4",
    "docs/STAGE2_DIAGNOSTIC_D1R2_IMPLEMENTATION_NOTES.md": "e7dff552ca1c67d7e9225c345a24cbe1161ff0ced4fd483f14b71b34939fc14c",
}


# ------------------------------------------------------------------ helpers
class RecordingExecute(pkg.FakeExecute):
    """Synthetic stage executor that also records the worker function used."""

    def __init__(self, plan):
        super().__init__(plan)
        self.fns = []

    def __call__(self, stage, fn, jobs):
        self.fns.append(fn)
        self.jobs = getattr(self, "jobs", {})
        self.jobs[stage] = list(jobs)
        return super().__call__(stage, fn, jobs)


def run_entry(tmp_path, plan, gate=None, name="pkg"):
    ex = RecordingExecute(plan)
    out = tmp_path / name
    prov = {"mode": "test", "workers": 1, "ss_probe": {"pass": True, "decision": "PASS"}}
    summary = d2d4.run_d2_d4(
        SimpleNamespace(workers=1), out, prov, execute=ex, seeds=pkg.SEEDS,
        condition_identity=pkg.cond_identity, n_boot=N_BOOT, gate=REAL_GATE if gate is None else gate,
    )
    return summary, ex, out


def full_plan(d3_centre=0.90, **d4_kw):
    return {
        "D2_A_no_drive": pkg._d2_rows(),
        "D3_lag1_no_drive": pkg._d3_rows(p_centre=d3_centre),
        "D4_drive": pkg._d4_rows(**d4_kw),
    }


def assert_frozen(out, summary, code):
    listed = pkg.assert_frozen_package(out, summary, code)
    assert "d1_reference.json" in listed
    assert summary["statuses"]["D1"] == "PASS"
    assert summary["stages"]["D1"]["executed_here"] is False and summary["stages"]["D1"]["rescored_here"] is False
    assert not any(n.startswith("d1_rows") or n.startswith("D1.") for n in listed)
    return listed


def copy_package(tmp_path):
    dst = tmp_path / "d1r2_copy"
    shutil.copytree(PACKAGE, dst)
    return dst


def pins_of(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def rewrite_manifest(directory):
    names = sorted(n for n in pins_of(directory) if n != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((directory / n).read_bytes()).hexdigest()}  {n}\n" for n in names))


# ============================================ 1. lock behind approval file
def test_d2d4_01_run_is_locked_behind_approval_file_and_writes_nothing(tmp_path):
    script = str(ROOT / "run_stage2_diagnostic_d2d4.py")
    out = tmp_path / "forbidden"
    proc = subprocess.run([sys.executable, script, "--run-d2d4", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "approval-file is required" in proc.stderr
    assert not out.exists()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    bogus = {
        "missing_run_authorization": {"decision": "APPROVE", "reviewer": "nora", "implementation_commit": head},
        "wrong_commit": {"decision": "APPROVE", "reviewer": "nora", "implementation_commit": "0" * 40, "run_authorized": True, "run_authorization_card": "t_x"},
        "wrong_reviewer": {"decision": "APPROVE", "reviewer": "ethan", "implementation_commit": head, "run_authorized": True, "run_authorization_card": "t_x"},
        "not_authorized": {"decision": "APPROVE", "reviewer": "nora", "implementation_commit": head, "run_authorized": False, "run_authorization_card": "t_x"},
    }
    for name, record in bogus.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(record))
        out_n = tmp_path / f"out_{name}"
        proc = subprocess.run([sys.executable, script, "--run-d2d4", "--approval-file", str(path), "--out", str(out_n)], cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode != 0 and "approval record mismatch" in proc.stderr, name
        assert not out_n.exists(), name
    proc = subprocess.run([sys.executable, script, "--shakedown", "--workers", "6", "--out", str(tmp_path / "w")], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "workers must be 1..5" in proc.stderr


def test_d2d4_02_approval_record_requires_every_field():
    head = "a" * 40
    good = {
        "decision": "APPROVE", "reviewer": "nora", "implementation_commit": head,
        "diagnostic_spec_sha256": d2d4.FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_SPEC.md"],
        "d1r2_spec_sha256": d2d4.FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md"],
        "d1r2_results_commit": d2d4.D1R2_RESULTS_COMMIT, "run_authorized": True, "run_authorization_card": "t_run",
    }
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.json"
        path.write_text(json.dumps(good))
        assert d2d4._approval(str(path), head)["run_authorization_card"] == "t_run"  # pure parser check, no run
        for key in good:
            bad = dict(good)
            bad.pop(key)
            path.write_text(json.dumps(bad))
            with pytest.raises(SystemExit, match="approval record mismatch"):
                d2d4._approval(str(path), head)


def test_d2d4_03_real_stage_pool_unreachable_without_approved_run_provenance(tmp_path):
    for prov in ({"mode": "shakedown"}, {"mode": "d2d4", "approval": None}, {"mode": "test"}):
        out = tmp_path / f"o_{prov['mode']}_{prov.get('approval')}"
        with pytest.raises(SystemExit, match="requires an approved --run-d2d4 preflight"):
            d2d4.run_d2_d4(SimpleNamespace(workers=1), out, prov, gate=REAL_GATE)
        assert not out.exists()


# ================================================ 2. D1R2 package-hash gate
def test_d2d4_04_committed_d1r2_package_passes_gate_with_git_identity():
    gate = REAL_GATE
    assert gate["pass"] is True, gate["problems"]
    assert gate["git_verified"] is True
    assert gate["recorded_branch"]["code"] == "D1R2_PASS_D2_D4_ELIGIBLE" and gate["recorded_branch"]["table_row"] == 6
    assert gate["recorded_readout_statuses"] == {"pipeline": "PASS", "network_positive": "PASS", "A": "PASS"}
    assert gate["recorded_source_commit"] == d2d4.D1R2_SOURCE_COMMIT
    assert gate["file_sha256"] == d2d4.D1R2_PACKAGE_SHA256
    # The pinned hashes are the committed, Nora-approved bytes at 3077119.
    for name, want in d2d4.D1R2_PACKAGE_SHA256.items():
        blob = subprocess.check_output(["git", "show", f"{d2d4.D1R2_RESULTS_COMMIT}:{d2d4.D1R2_PACKAGE_REL}/{name}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == want, name


def test_d2d4_05_gate_text_is_verbatim_d1r2_spec_section_9_row_6():
    spec = (ROOT / "docs" / "STAGE2_DIAGNOSTIC_D1R2_SPEC.md").read_text()
    row6 = [line for line in spec.splitlines() if line.startswith("| 6 |")]
    assert len(row6) == 1 and "`D1R2_PASS_D2_D4_ELIGIBLE`" in row6[0]
    assert d2d4.GOVERNING_BRANCH_TEXT in row6[0]


def test_d2d4_06_gate_fails_closed_on_missing_altered_extra_or_non_row6_package(tmp_path):
    # Missing directory.
    assert not d2d4.verify_d1r2_package(tmp_path / "absent", require_git=False)["pass"]
    # A faithful copy passes without Git (the runner always requires Git as well).
    good = copy_package(tmp_path)
    assert d2d4.verify_d1r2_package(good, require_git=False)["pass"]
    # Missing file.
    c = tmp_path / "missing"
    shutil.copytree(good, c)
    (c / "d1r2_rows.jsonl").unlink()
    g = d2d4.verify_d1r2_package(c, require_git=False)
    assert not g["pass"] and any("d1r2_rows.jsonl" in p for p in g["problems"])
    # One altered byte anywhere.
    for name in sorted(d2d4.D1R2_PACKAGE_SHA256):
        c = tmp_path / f"alt_{name}"
        shutil.copytree(good, c)
        data = bytearray((c / name).read_bytes())
        data[len(data) // 2] ^= 0x01
        (c / name).write_bytes(bytes(data))
        g = d2d4.verify_d1r2_package(c, require_git=False)
        assert not g["pass"] and any(name in p for p in g["problems"]), name
    # Extra file.
    c = tmp_path / "extra"
    shutil.copytree(good, c)
    (c / "note.txt").write_text("x")
    assert not d2d4.verify_d1r2_package(c, require_git=False)["pass"]
    # Symlinked package directory is refused.
    link = tmp_path / "link"
    link.symlink_to(good, target_is_directory=True)
    assert not d2d4.verify_d1r2_package(link, require_git=False)["pass"]


@pytest.mark.parametrize(
    "code,row,status",
    [
        ("STOP_D1R2_INVALID", 1, "INVALID"),
        ("STOP_D1R2_INVALID_PIPELINE", 2, "INVALID_PIPELINE"),
        ("CLOSE_D1_REPRESENTATION_LIMITED", 3, "REPRESENTATION_LIMITED"),
        ("D1R2_A_FAIL_EBB90E74_ROW3", 4, "FAIL"),
        ("D1R2_A_INCONCLUSIVE_EBB90E74_ROW4", 5, "INCONCLUSIVE"),
    ],
)
def test_d2d4_07_gate_refuses_every_non_row6_branch_even_with_self_consistent_hashes(tmp_path, code, row, status):
    c = copy_package(tmp_path)
    b = json.loads((c / "branch_outcome.json").read_text())
    b.update(code=code, table_row=row, status=status)
    (c / "branch_outcome.json").write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")
    s = json.loads((c / "summary.json").read_text())
    s["status"] = status
    s["branch"] = dict(s["branch"], code=code, table_row=row)
    s["branch_outcome"] = dict(s["branch_outcome"], code=code, table_row=row)
    (c / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
    rewrite_manifest(c)
    g = d2d4.verify_d1r2_package(c, pinned=pins_of(c), require_git=False)  # hashes made self-consistent on purpose
    assert not g["pass"]
    assert any("branch_outcome.json is" in p for p in g["problems"])
    assert any("summary.json status/branch" in p for p in g["problems"])


def test_d2d4_08_gate_refuses_summary_with_non_PASS_readout_or_wrong_source_commit(tmp_path):
    for mutate, needle in (
        (lambda s: s["readouts"]["A"].update(status="INCONCLUSIVE"), "readout statuses"),
        (lambda s: s["provenance"].update(code_commit="0" * 40), "source commit"),
        (lambda s: s.update(violations=["x"]), "violations"),
    ):
        c = tmp_path / f"c_{needle.replace(' ', '_')}"
        shutil.copytree(PACKAGE, c)
        s = json.loads((c / "summary.json").read_text())
        mutate(s)
        (c / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
        rewrite_manifest(c)
        g = d2d4.verify_d1r2_package(c, pinned=pins_of(c), require_git=False)
        assert not g["pass"] and any(needle in p for p in g["problems"]), needle


def test_d2d4_09_failed_gate_refuses_before_any_package_file_and_no_stage_runs(tmp_path):
    bad_gate = dict(REAL_GATE, **{"pass": False, "problems": ["synthetic"]})
    ex = RecordingExecute(full_plan())
    out = tmp_path / "pkg"
    with pytest.raises(d2d4.D1R2GateFailure):
        d2d4.run_d2_d4(SimpleNamespace(workers=1), out, {"mode": "test"}, execute=ex, seeds=pkg.SEEDS,
                       condition_identity=pkg.cond_identity, n_boot=N_BOOT, gate=bad_gate)
    assert ex.called == [] and not out.exists()


def test_d2d4_10_git_check_refuses_working_tree_bytes_that_differ_from_results_commit(tmp_path, monkeypatch):
    # Simulate a byte change by pointing the Git comparison at a different committed blob.
    real = d2d4.git

    def fake_git(*args):
        if args[:1] == ("hash-object",) and args[1].endswith("summary.json"):
            return "0" * 40
        return real(*args)

    monkeypatch.setattr(d2d4, "git", fake_git)
    g = d2d4.verify_d1r2_package()
    assert not g["pass"] and any("working-tree blob differs" in p for p in g["problems"])


# ============================== 3. hard limit 7: no D1R2 decoder in D2-D4
def test_d2d4_11_entry_imports_no_d1r2_module_or_decoder_library():
    tree = ast.parse((ROOT / "run_stage2_diagnostic_d2d4.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    assert not any("d1r" in m or "sklearn" in m for m in imported), imported
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(ROOT / 'tests')!r})
        import run_stage2_diagnostic_d2d4 as d2d4
        from snn import stage2_diagnostic as diag
        gate = d2d4.verify_d1r2_package()
        assert gate['pass'], gate['problems']
        row = diag.run_condition_seed(4242, 'A', True, True,
            diag.condition_object_hashes(4242, 'A', True, n_train=1000, n_eval=1100), n_train=1000, n_eval=1100, warmup=1000)
        print(json.dumps(sorted(m for m in sys.modules if 'd1r' in m or m.split('.')[0] == 'sklearn')))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def test_d2d4_12_gate_never_parses_the_d1r2_row_file_that_holds_decoder_coefficients(monkeypatch):
    opened_as_text = []
    real_read_text = Path.read_text

    def spy(self, *a, **k):
        opened_as_text.append(self.name)
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    assert d2d4.verify_d1r2_package()["pass"]
    assert "d1r2_rows.jsonl" not in opened_as_text and "D1R2.events.jsonl" not in opened_as_text
    assert set(opened_as_text) <= {"SHA256SUMS", "branch_outcome.json", "summary.json"}


def test_d2d4_13_worker_and_readouts_are_the_approved_ebb90e74_objects(tmp_path):
    summary, ex, out = run_entry(tmp_path, full_plan())
    assert all(fn is base._condition_job for fn in ex.fns)
    assert ex.called == ["D2_A_no_drive", "D3_lag1_no_drive", "D4_drive"]
    # Jobs carry only (seed, task, plastic, drive, expected hashes): no decoder object reaches a worker.
    for stage, jobs in ex.jobs.items():
        for job in jobs:
            assert len(job) == 5 and isinstance(job[0], int) and job[1] in ("A", "lag1")
            assert isinstance(job[2], bool) and isinstance(job[3], bool) and isinstance(job[4], dict)
    assert {(j[1], j[2], j[3]) for j in ex.jobs["D2_A_no_drive"]} == {("A", True, False), ("A", False, False)}
    assert {(j[1], j[2], j[3]) for j in ex.jobs["D3_lag1_no_drive"]} == {("lag1", True, False), ("lag1", False, False)}
    assert {(j[1], j[2], j[3]) for j in ex.jobs["D4_drive"]} == {("A", True, True), ("A", False, True), ("lag1", True, True)}
    assert len(ex.jobs["D2_A_no_drive"]) == len(ex.jobs["D3_lag1_no_drive"]) == 40 and len(ex.jobs["D4_drive"]) == 60


# ============================ 4. steps 3-6 carried byte-for-byte from ebb90e74
@pytest.mark.parametrize(
    "plan_kw",
    [
        {"d3_centre": 0.50},
        {"d3_centre": 0.90, "p_drive": 0.80},
        {"d3_centre": 0.90, "p_drive": 0.50},
        {"d3_centre": 0.90, "p_drive": 0.80, "silent": 0.95},
        {"d3_centre": 0.90, "f0_drive": 0.70},
    ],
)
def test_d2d4_14_steps_3_to_6_reproduce_the_approved_runner_after_a_D1_PASS(tmp_path, plan_kw):
    plan = full_plan(**plan_kw)
    approved_summary, approved_ex, approved_out = pkg.run_package(tmp_path / "approved", {"D1": pkg.d1_rows(), **copy.deepcopy(plan)})
    assert approved_summary["statuses"]["D1"] == "PASS"
    summary, ex, out = run_entry(tmp_path, copy.deepcopy(plan))
    assert approved_ex.called[1:] == ex.called
    for key in ("D2", "D3", "D4"):
        assert summary["stages"][key] == approved_summary["stages"][key], key
        assert summary["statuses"][key] == approved_summary["statuses"][key], key
    assert summary["branch_outcome"] == approved_summary["branch_outcome"]
    for name in ("d2_rows.jsonl", "d3_rows.jsonl", "d4_drive_rows.jsonl", "d2_rows.invalid_redacted.jsonl",
                 "d3_rows.invalid_redacted.jsonl", "d4_drive_rows.invalid_redacted.jsonl"):
        a, b = approved_out / name, out / name
        assert a.exists() == b.exists(), name
        if a.exists():
            assert a.read_bytes() == b.read_bytes(), name
    assert summary["effective_parameters"] == approved_summary["effective_parameters"]
    assert summary["claim_boundary"] == base.CLAIM_BOUNDARY and summary["no_oracle"] == base.NO_ORACLE


# ===================== 5. branch-table reachability with D1 = PASS by D1R2
REACHABLE_WITH_D1_PASS = ("STOP_INVALID", "RETURN_D3_FAIL", "RETURN_D3_INCONCLUSIVE", "RETURN_D4_SILENCE_FAIL",
                          "RETURN_D4_COMPETENCE_FAIL", "RETURN_D4_COMPETENCE_INCONCLUSIVE", "CONSIDER_DOC_ONLY_R4_SPEC")


@pytest.mark.parametrize(
    "plan,code,called",
    [
        ({"D2_A_no_drive": "bad_d2"}, "STOP_INVALID", ["D2_A_no_drive"]),
        ({"D3_lag1_no_drive": "bad_d3"}, "STOP_INVALID", ["D2_A_no_drive", "D3_lag1_no_drive"]),
        ({"d3_centre": 0.90, "f0_drive": 0.70}, "STOP_INVALID", None),
        ({"D2_A_no_drive": RuntimeError("worker pool died")}, "STOP_INVALID", ["D2_A_no_drive"]),
        ({"d3_centre": 0.50}, "RETURN_D3_FAIL", None),
        ({"d3_centre": 0.60}, "RETURN_D3_INCONCLUSIVE", None),
        ({"d3_centre": 0.90, "p_drive": 0.80, "silent": 0.95}, "RETURN_D4_SILENCE_FAIL", None),
        ({"d3_centre": 0.90, "p_drive": 0.50}, "RETURN_D4_COMPETENCE_FAIL", None),
        ({"d3_centre": 0.90, "p_drive": 0.60}, "RETURN_D4_COMPETENCE_INCONCLUSIVE", None),
        ({"d3_centre": 0.90, "p_drive": 0.80}, "CONSIDER_DOC_ONLY_R4_SPEC", None),
    ],
    ids=["D2_invalid", "D3_invalid", "D4_drive_invalid", "exception", "D3_FAIL", "D3_INCONCLUSIVE",
         "D4ii_FAIL", "D4iii_FAIL", "D4iii_INCONCLUSIVE", "all_PASS"],
)
def test_d2d4_15_branch_table_reachability_with_D1_PASS_by_D1R2(tmp_path, plan, code, called):
    full = full_plan(**{k: v for k, v in plan.items() if k in ("d3_centre", "p_drive", "f0_drive", "silent")})
    if plan.get("D2_A_no_drive") == "bad_d2":
        bad = pkg._d2_rows()
        bad[4]["checkpoints"] = bad[4]["checkpoints"][:-1]
        full["D2_A_no_drive"] = bad
    elif isinstance(plan.get("D2_A_no_drive"), Exception):
        full["D2_A_no_drive"] = plan["D2_A_no_drive"]
    if plan.get("D3_lag1_no_drive") == "bad_d3":
        bad = pkg._d3_rows(0.9)
        bad[25]["weights_bitwise_constant"] = False
        full["D3_lag1_no_drive"] = bad
    summary, ex, out = run_entry(tmp_path, full)
    assert_frozen(out, summary, code)
    assert ex.called == (called or ["D2_A_no_drive", "D3_lag1_no_drive", "D4_drive"])
    assert "D1" not in ex.called
    row = base.BRANCH_INDEX[code]
    assert row == 1 or row >= 5  # ebb90e74 section 10 row 1 and rows 5 onward only
    if code == "STOP_INVALID":
        for key in ("D2", "D3", "D4"):
            stage = summary["stages"].get(key)
            if stage is not None and stage.get("status") in ("INVALID", "D2_INVALID"):
                assert not stats.contains_outcome(stage)


def test_d2d4_16_D1_only_rows_2_to_4_are_unreachable_and_rows_11_12_follow_spec_precedence():
    d4_vals = [None]
    for status in ("PASS", "FAIL", "INCONCLUSIVE", "INVALID"):
        for silence in ("PASS", "FAIL", "NOT_EVALUATED"):
            for competence in ("PASS", "FAIL", "INCONCLUSIVE", "INVALID"):
                d4_vals.append({"status": status, "silence_prevention": silence, "competence": competence})
    reached = set()
    for d2 in (None, "D2_INVALID", "D2_PASS_LOCALIZED_HIDDEN", "D2_PASS_LOCALIZED_OUTPUT", "D2_FAIL_CO_ONSET"):
        for d3 in (None, "INVALID", "PASS", "FAIL", "INCONCLUSIVE"):
            for d4 in d4_vals:
                for incomplete in (False, True):
                    reached.add(base.branch_outcome("PASS", d2, d3, d4, incomplete=incomplete)["code"])
    assert not reached & set(d2d4.D1_ONLY_BRANCHES)
    assert all(base.BRANCH_INDEX[c] == 1 or base.BRANCH_INDEX[c] >= 5 for c in reached)
    assert set(REACHABLE_WITH_D1_PASS) <= reached
    # ebb90e74 section 10 is ordered: row 7 (D4(ii) FAIL) precedes and shadows row 11;
    # row 12 is the residual "any other valid combination". Both are carried unchanged.
    assert base.branch_outcome("PASS", "D2_FAIL_CO_ONSET", "PASS", {"status": "FAIL", "silence_prevention": "FAIL", "competence": "PASS"})["code"] == "RETURN_D4_SILENCE_FAIL"
    assert len(base.BRANCH_TABLE) == 12


def test_d2d4_17_finalizer_refuses_rather_than_mislabel_a_D1_only_branch(tmp_path):
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "provenance.json").write_text("{}\n")
    d2d4.base.write_incomplete_marker(out)
    with pytest.raises(AssertionError, match="D1-only branch"):
        d2d4.finalize_package(out, {}, {}, {"D1": "FAIL", "D2": None, "D3": None, "D4": None}, [], ["provenance.json"], "claim")
    assert not (out / "summary.json").exists() and not (out / "SHA256SUMS").exists()


def test_d2d4_18_package_records_D1_by_reference_with_hashes_and_plain_language(tmp_path):
    summary, _, out = run_entry(tmp_path, full_plan(d3_centre=0.50))
    ref = json.loads((out / "d1_reference.json").read_text())
    assert ref["status"] == "PASS" and ref["results_commit"] == d2d4.D1R2_RESULTS_COMMIT
    assert ref["file_sha256"] == d2d4.D1R2_PACKAGE_SHA256
    assert ref["recorded_branch"]["code"] == "D1R2_PASS_D2_D4_ELIGIBLE"
    assert ref["governing_text"] == d2d4.GOVERNING_BRANCH_TEXT
    assert summary["d1_claim_boundary"] == REAL_GATE["network_claim"]
    first = summary["package_summary"]["supported"][0]
    assert first.startswith("D1 PASS by D1R2") and "D1R2_PASS_D2_D4_ELIGIBLE" in first
    assert summary["package_summary"]["not_supported"] == list(base.NOT_SUPPORTED)
    marker_free = json.loads((out / "branch_outcome.json").read_text())
    assert marker_free["code"] == "RETURN_D3_FAIL"


def test_d2d4_19_hard_kill_leaves_fail_closed_marker_and_fresh_dir_is_required(tmp_path):
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "stale.json").write_text("{}")
    with pytest.raises(base.StaleOutputError):
        d2d4.run_d2_d4(SimpleNamespace(workers=1), out, {"mode": "test"}, execute=RecordingExecute(full_plan()),
                       seeds=pkg.SEEDS, condition_identity=pkg.cond_identity, n_boot=N_BOOT, gate=REAL_GATE)
    seen = {}

    def kill_mid_d2(stage, fn, jobs):
        seen["files"] = sorted(p.name for p in (tmp_path / "pkg2").iterdir())
        seen["marker"] = json.loads((tmp_path / "pkg2" / "branch_outcome.json").read_text())
        raise KeyboardInterrupt  # not an Exception: simulates a hard stop that skips the finalizer

    with pytest.raises(KeyboardInterrupt):
        d2d4.run_d2_d4(SimpleNamespace(workers=1), tmp_path / "pkg2", {"mode": "test"}, execute=kill_mid_d2,
                       seeds=pkg.SEEDS, condition_identity=pkg.cond_identity, n_boot=N_BOOT, gate=REAL_GATE)
    assert seen["marker"]["code"] == "STOP_INVALID" and not stats.contains_outcome(seen["marker"])
    assert seen["files"] == ["branch_outcome.json", "d1_reference.json", "provenance.json"]


# ============================ 6. frozen science, constants, and pins unchanged
def test_d2d4_20_frozen_specs_and_approved_code_unchanged_and_parameters_match_spec():
    for rel, want in {**d2d4.FROZEN_SPEC_SHA256, **APPROVED_UNCHANGED_SHA256}.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want, rel
    assert d2d4.FROZEN_SPEC_SHA256["docs/STAGE2_DIAGNOSTIC_SPEC.md"] == diag.SPEC_SHA256
    assert d2d4.FROZEN_SPEC_SHA256["docs/STAGE2_SPEC.md"] == diag.FROZEN_STAGE2_SHA256
    checked = base.validated_effective_parameters()
    assert checked["matches_spec"] is True, checked["mismatches"]
    assert diag.DIAGNOSTIC_SEEDS == tuple(range(2000, 2020)) and diag.NAMESPACE == 30
    assert stats.N_BOOT == 100_000 and sorted(stats.METRICS) == list(range(9))
    assert (diag.DRIVE_RATE_HZ, diag.DRIVE_WEIGHT_MV, diag.DRIVE_SOURCES_PER_TARGET, diag.DRIVE_TARGETS) == (25.0, 2.0, 8, 22)
    assert (diag.CHECKPOINT_STEPS, diag.N_CHECKPOINTS) == (1_000, 200)


def test_d2d4_21_ss_probe_check_reproduces_retained_ebb90e74_probe():
    record = d2d4.probe_check()
    assert record["pass"] is True
    assert record["new_collisions"] == [0, 0, 0, 0]
    assert record["known_legacy_only_duplicate_count"] == 6
    assert record["identity_state_sha256"] == "2f6b54fb03e2a6378cd203a3f84e21ef62d4967e2e8fa5008104fc3c8abbdeef"
    assert record["counts"]["new_records"] == 229 and record["counts"]["legacy_records"] == 475
    later = record["namespace_31_32_probes_with_namespace_30_as_legacy"]
    assert all(v["decision"] == "PASS" and v["new_vs_legacy_collisions"] == [0, 0] for v in later.values())


# ===================================================== 7. shakedown boundary
@pytest.mark.parametrize("seed", [0, 19, 1000, 2000, 2019, 2100, 2200, 2219])
def test_d2d4_22_shakedown_refuses_every_experimental_and_diagnostic_seed(seed):
    with pytest.raises(d2d4.UnauthorizedSeed):
        d2d4.shakedown_rows(seed)


def test_d2d4_23_shakedown_constructs_all_seven_conditions_on_fixture_seed_without_readouts():
    rows = d2d4.shakedown_rows(d2d4.FIXTURE_SEED)
    checked = d2d4.shakedown_checks(rows)
    assert checked["pass"] is True, checked["checks"]
    text = json.dumps(checked)
    for key in ("accuracy", "hidden_rate_hz", "both_silent_fraction", "lower_95", "median"):
        assert f'"{key}"' not in text, key
