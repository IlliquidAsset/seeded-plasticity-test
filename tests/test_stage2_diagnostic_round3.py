"""Review round 3 (Nora, t_53bf2d59 comment of 2026-10-01 12:14).

1. Stale result-directory contamination: a package must be built in a fresh,
   exclusively created location; any prior full or partial package (including
   one left by a hard kill) is refused before anything is written, and the
   manifest covers exactly the declared artifacts.
2. D1 label identity: stored A and positive-control label hashes must equal
   coordinator-derived hashes of x_t and x_(t-1) on scored rows.
3. Condition-row schema semantics: the top-level diagnostic label and every
   evaluation field (including coin-read type/range) are validated.

No diagnostic seed is simulated. Package-path tests inject synthetic rows under
non-diagnostic IDs 1000..1019; the producer agreement test uses short lengths
on fixture seed 4242 only.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_stage2_diagnostic as runner
from snn import stage2_diagnostic as diag
from snn import stage2_diagnostic_stats as stats

import test_stage2_diagnostic_package as pkg  # shared synthetic fixtures

ROOT = Path(runner.__file__).resolve().parent
SEEDS = pkg.SEEDS
PROV = {"mode": "test", "workers": 1, "ss_probe": {"pass": True, "decision": "PASS"}}


def run_into(out, plan, provenance=PROV):
    ex = pkg.FakeExecute(plan)
    summary = runner.run_diagnostic(
        SimpleNamespace(workers=1), out, provenance, execute=ex, seeds=SEEDS,
        d1_identity=pkg.d1_identity, condition_identity=pkg.cond_identity, n_boot=pkg.N_BOOT,
    )
    return summary, ex


def snapshot(out: Path):
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def full_plan():
    return {
        "D1": pkg.d1_rows(),
        "D2_A_no_drive": pkg._d2_rows(),
        "D3_lag1_no_drive": pkg._d3_rows(p_centre=0.9),
        "D4_drive": pkg._d4_rows(),
    }


D1_FAIL_PLAN = {"D1": pkg.d1_rows(a_acc=0.52, a_low=0.45)}


# ============================================ finding 1: stale artifacts
def test_r3_01_nora_probe_stale_d4_file_refuses_and_is_untouched(tmp_path):
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "d4_drive_rows.jsonl").write_text('{"accuracy": 0.99}\n')
    before = snapshot(out)
    ex = pkg.FakeExecute(D1_FAIL_PLAN)
    with pytest.raises(runner.StaleOutputError):
        runner.run_diagnostic(SimpleNamespace(workers=1), out, PROV, execute=ex, seeds=SEEDS,
                              d1_identity=pkg.d1_identity, condition_identity=pkg.cond_identity, n_boot=pkg.N_BOOT)
    assert ex.called == [], "no stage may run against a contaminated location"
    assert snapshot(out) == before, "refusal must not write, remove, or manifest anything"
    assert not (out / "SHA256SUMS").exists() and not (out / "branch_outcome.json").exists()


def test_r3_02_prior_full_package_then_D1_stop_rerun_is_refused(tmp_path):
    out = tmp_path / "pkg"
    summary, _ = run_into(out, full_plan())
    assert summary["branch_outcome"]["code"] == "CONSIDER_DOC_ONLY_R4_SPEC"
    before = snapshot(out)
    assert "d4_drive_rows.jsonl" in before
    with pytest.raises(runner.StaleOutputError):
        run_into(out, D1_FAIL_PLAN)
    assert snapshot(out) == before


def test_r3_03_hard_killed_partial_package_then_rerun_is_refused(tmp_path):
    """A BaseException (stand-in for SIGKILL / os._exit) escapes mid-D1 and leaves
    the fail-closed marker plus whatever the dead run wrote; a rerun into that
    location must refuse rather than mix the remnants into a new package."""
    out = tmp_path / "pkg"

    def killed(stage, fn, jobs):
        (out / "D1.progress.jsonl").write_text('{"stage": "D1", "index": 1}\n')
        (out / "d4_drive_rows.jsonl").write_text('{"accuracy": 0.99}\n')  # stale outcome from an older layout
        raise KeyboardInterrupt("hard kill")

    with pytest.raises(KeyboardInterrupt):
        runner.run_diagnostic(SimpleNamespace(workers=1), out, PROV, execute=killed, seeds=SEEDS,
                              d1_identity=pkg.d1_identity, condition_identity=pkg.cond_identity, n_boot=pkg.N_BOOT)
    marker = json.loads((out / "branch_outcome.json").read_text())
    assert marker["code"] == "STOP_INVALID" and not stats.contains_outcome(marker)
    before = snapshot(out)
    with pytest.raises(runner.StaleOutputError):
        run_into(out, D1_FAIL_PLAN)
    assert snapshot(out) == before


def test_r3_04_real_process_hard_kill_then_cli_rerun_refused(tmp_path):
    """Real os._exit mid-stage in a child process, then the CLI refuses the same --out."""
    out = tmp_path / "pkg"
    script = textwrap.dedent(
        f"""
        import os, sys
        from types import SimpleNamespace
        sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tests')!r}]
        import run_stage2_diagnostic as runner
        import test_stage2_diagnostic_package as pkg
        out = __import__('pathlib').Path({str(out)!r})
        def killed(stage, fn, jobs):
            (out / 'D1.progress.jsonl').write_text('{{"stage": "D1"}}\\n')
            os._exit(9)
        runner.run_diagnostic(SimpleNamespace(workers=1), out, {{'mode': 'test', 'workers': 1}}, execute=killed,
                              seeds=pkg.SEEDS, d1_identity=pkg.d1_identity, condition_identity=pkg.cond_identity, n_boot=10)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 9, proc.stderr
    assert sorted(snapshot(out)) == ["D1.progress.jsonl", "branch_outcome.json", "provenance.json"]
    assert json.loads((out / "branch_outcome.json").read_text())["code"] == "STOP_INVALID"
    before = snapshot(out)
    for mode in ("--shakedown", "--run-diagnostic"):
        cli = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic.py"), mode, "--out", str(out)],
                             cwd=ROOT, capture_output=True, text=True)
        assert cli.returncode != 0 and "already exists" in cli.stderr, (mode, cli.stderr)
    assert snapshot(out) == before


def test_r3_05_fresh_location_D1_stop_manifests_exactly_declared_artifacts(tmp_path):
    out = tmp_path / "fresh"
    summary, ex = run_into(out, D1_FAIL_PLAN)
    assert ex.called == ["D1"]
    listed = pkg.assert_frozen_package(out, summary, "STOP_D1_FAIL")
    assert set(listed) == set(summary["artifacts"]) == {"provenance.json", "branch_outcome.json", "summary.json", "d1_rows.jsonl"}
    assert json.loads((out / "provenance.json").read_text()) == PROV


@pytest.mark.parametrize(
    "stray",
    ["d4_drive_rows.jsonl", "old_d4_outcome.jsonl", "D2_A_no_drive.progress.jsonl"],
    ids=["collides_with_later_stage_file", "unrelated_name", "collides_with_later_ledger"],
)
def test_r3_06_artifact_injected_mid_run_forces_STOP_INVALID_and_is_never_manifested(tmp_path, stray):
    """Another writer drops a file into the package directory during D1. It is
    never overwritten into, adopted, or manifested; the package is STOP_INVALID."""
    out = tmp_path / "pkg"
    plan = full_plan()
    rows_d1 = plan["D1"]

    def d1_then_inject():
        (out / stray).write_text('{"accuracy": 0.99}\n')
        return rows_d1

    plan["D1"] = d1_then_inject
    summary, _ = run_into(out, plan)
    assert summary["branch_outcome"]["code"] == "STOP_INVALID"
    assert summary["unexpected_artifacts"] == [stray]
    assert stray not in summary["artifacts"]
    assert (out / stray).read_text() == '{"accuracy": 0.99}\n', "the stray file is never overwritten"
    manifest = (out / "SHA256SUMS").read_text()
    names = {line.split("  ", 1)[1] for line in manifest.splitlines()}
    assert stray not in names and names == set(summary["artifacts"])
    for name in names:
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() in manifest
    on_disk = json.loads((out / "branch_outcome.json").read_text())
    assert on_disk["code"] == "STOP_INVALID"


def test_r3_06b_run_pool_refuses_existing_ledger(tmp_path):
    (tmp_path / "D1.progress.jsonl").write_text('{"stale": true}\n')
    with pytest.raises(FileExistsError):
        runner.run_pool(lambda s: {"seed": s, "status": "ok"}, [1], 1, "D1", tmp_path, watch=pkg._Watch())
    assert (tmp_path / "D1.progress.jsonl").read_text() == '{"stale": true}\n'


def test_r3_07_preflight_provenance_is_the_only_accepted_preexisting_content(tmp_path):
    out = tmp_path / "pkg"
    runner.claim_fresh_run_dir(out)
    (out / "provenance.json").write_text(runner._provenance_text(PROV))
    summary, _ = run_into(out, D1_FAIL_PLAN)  # this run's own preflight output is accepted
    assert summary["branch_outcome"]["code"] == "STOP_D1_FAIL"
    other = tmp_path / "other"
    runner.claim_fresh_run_dir(other)
    (other / "provenance.json").write_text(runner._provenance_text({**PROV, "written_at": "a different run"}))
    with pytest.raises(runner.StaleOutputError, match="different run"):
        run_into(other, D1_FAIL_PLAN)
    with pytest.raises(runner.StaleOutputError):
        runner.claim_fresh_run_dir(out)


def test_r3_08_default_package_location_is_run_unique_and_separate_from_retained_probe():
    a = runner.default_run_dir("diagnostic")
    b = runner.default_run_dir("diagnostic")
    probe_dir = ROOT / "results_stage2_diagnostic"
    assert a != b and a.parent == b.parent == probe_dir / "runs"
    assert not a.exists() and runner.git("rev-parse", "--short=12", "HEAD") in a.name
    # The retained collision probe remains an explicit, manifest-verified input.
    probe = runner.verified_probe()
    assert probe["pass"] is True
    assert probe["output_sha256"] == hashlib.sha256((probe_dir / "ss_probe.json").read_bytes()).hexdigest()
    # The probe directory itself can never be a package location.
    with pytest.raises(runner.StaleOutputError):
        runner.claim_fresh_run_dir(probe_dir)


def test_r3_09_manifest_refuses_undeclared_missing_artifact(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "stray.jsonl").write_text("{}")
    runner.manifest(tmp_path, ["a.json"])
    assert (tmp_path / "SHA256SUMS").read_text().splitlines() == [
        f"{hashlib.sha256(b'{}').hexdigest()}  a.json"
    ]
    with pytest.raises(AssertionError, match="declared artifact missing"):
        runner.manifest(tmp_path, ["a.json", "never_written.json"])


# ========================================== finding 2: D1 label identity
LABEL_MUTATIONS = {
    "A_train_arbitrary": ("A", "train_label_sha256", "b" * 64),
    "A_test_arbitrary": ("A", "test_label_sha256", "c" * 64),
    "prior_train_arbitrary": ("positive_prior", "train_label_sha256", "a" * 64),
    "prior_test_arbitrary": ("positive_prior", "test_label_sha256", "d" * 64),
}


@pytest.mark.parametrize("name", sorted(LABEL_MUTATIONS))
def test_r3_10_arbitrary_distinct_label_hash_forces_INVALID_and_stops(tmp_path, name):
    decoder, key, value = LABEL_MUTATIONS[name]
    rows = pkg.d1_rows()
    rows[0]["decoders"][decoder][key] = value
    assert rows[0]["decoders"]["A"]["train_label_sha256"] != rows[0]["decoders"]["positive_prior"]["train_label_sha256"]
    result = diag.summarize_d1(rows, pkg.D1_EXPECTED, SEEDS)
    assert result["status"] == "INVALID" and result["A"] is None
    assert any("label identity" in v for v in result["violations"])
    summary, ex = run_into(tmp_path / "pkg", {"D1": rows})
    assert ex.called == ["D1"]
    pkg.assert_frozen_package(tmp_path / "pkg", summary, "STOP_INVALID")


def test_r3_11_swapped_A_and_prior_labels_force_INVALID():
    rows = pkg.d1_rows()
    dec = rows[2]["decoders"]
    for key in ("train_label_sha256", "test_label_sha256"):
        dec["A"][key], dec["positive_prior"][key] = dec["positive_prior"][key], dec["A"][key]
    assert diag.summarize_d1(rows, pkg.D1_EXPECTED, SEEDS)["status"] == "INVALID"


def test_r3_12_coordinator_identity_without_label_hashes_is_INVALID():
    expected = {s: {k: v for k, v in pkg.d1_identity(s).items() if "label" not in k} for s in SEEDS}
    result = diag.summarize_d1(pkg.d1_rows(), expected, SEEDS)
    assert result["status"] == "INVALID"
    assert any("coordinator identity lacks" in v for v in result["violations"])


def test_r3_13_coordinator_label_hashes_are_x_t_and_x_t_minus_1_independently_of_d1_labels():
    n_train, n_test = 2_600, 2_300
    ident = diag.d1_object_hashes(4242, n_train=n_train, n_test=n_test)
    x_tr = diag.a_stream(4242, True, n_train)
    x_te = diag.a_stream(4242, False, n_test)
    # Hand-built labels, element by element, with no call into d1_labels.
    a_tr = np.array([x_tr[t] for t in range(2000, n_train)], dtype=np.int8)
    p_tr = np.array([x_tr[t - 1] for t in range(2000, n_train)], dtype=np.int8)
    a_te = np.array([x_te[t] for t in range(2000, n_test)], dtype=np.int8)
    p_te = np.array([x_te[t - 1] for t in range(2000, n_test)], dtype=np.int8)
    assert ident["A_train_label_sha256"] == diag.sha256(a_tr)
    assert ident["positive_prior_train_label_sha256"] == diag.sha256(p_tr)
    assert ident["A_test_label_sha256"] == diag.sha256(a_te)
    assert ident["positive_prior_test_label_sha256"] == diag.sha256(p_te)
    assert ident["A_train_label_sha256"] != ident["positive_prior_train_label_sha256"]
    # The real producer emits exactly these, and a producer that used the
    # current input as the "prior" label is refused.
    row = diag.run_d1_seed(4242, n_train=n_train, n_test=n_test)
    row["status"] = "ok"
    kw = dict(train_rows=n_train - 2000, test_rows=n_test - 2000)
    assert diag.validate_d1_row(row, ident, **kw) == []
    leaked = copy.deepcopy(row)
    leaked["decoders"]["positive_prior"]["train_label_sha256"] = diag.sha256(a_tr)
    leaked["decoders"]["positive_prior"]["test_label_sha256"] = diag.sha256(a_te)
    assert any("x_(t-1)" in v for v in diag.validate_d1_row(leaked, ident, **kw))


# =================================== finding 3: condition schema semantics
def _set_eval(key, value):
    return lambda r: r["evaluation"].__setitem__(key, value)


CONDITION_MUTATIONS = {
    "diagnostic_WRONG": lambda r: r.__setitem__("diagnostic", "WRONG"),
    "diagnostic_D1": lambda r: r.__setitem__("diagnostic", "D1"),
    "coin_reads_negative": _set_eval("coin_reads", -100),
    "coin_reads_above_eval_len": _set_eval("coin_reads", diag.EVAL_LEN + 1),
    "coin_reads_float": _set_eval("coin_reads", 10.0),
    "coin_reads_bool": _set_eval("coin_reads", True),
    "coin_reads_none": _set_eval("coin_reads", None),
    "scored_float": _set_eval("scored", float(diag.EVAL_SCORED)),
    "accuracy_not_a_count": _set_eval("accuracy", 0.50001234),
    "accuracy_int": _set_eval("accuracy", 1),
    "eval_weights_truthy_int": _set_eval("weights_bitwise_constant", 1),
    "eval_extra_field": _set_eval("oracle_peek", 0.9),
    "no_drive_source_index_zero": _set_eval("source_start_index", 0),
    "row_extra_field": lambda r: r.__setitem__("target_hint", 1),
    "plastic_int": lambda r: r.__setitem__("plastic", int(r["plastic"])),
    "condition_not_str": lambda r: r.__setitem__("condition", 7),
    "worker_pid_str": lambda r: r.__setitem__("worker_pid", "1"),
    "checkpoint_diagnostic_label": lambda r: r["checkpoints"][3].__setitem__("diagnostic", "D3"),
    "checkpoint_negative_rate": lambda r: r["checkpoints"][3].__setitem__("hidden_rate_hz", -1.0),
    "checkpoint_rate_above_1kHz": lambda r: r["checkpoints"][3].__setitem__("output_layer_rate_hz", 1000.5),
    "checkpoint_extra_field": lambda r: r["checkpoints"][3].__setitem__("label", 1),
    "checkpoint_mean_outside_minmax": lambda r: r["checkpoints"][3].__setitem__("w1_mean", 50.0),
}


@pytest.mark.parametrize("name", sorted(CONDITION_MUTATIONS))
def test_r3_14_each_D3_schema_mutation_forces_INVALID_and_stops_before_D4(tmp_path, name):
    rows = pkg._d3_rows(p_centre=0.9)
    clean = stats.summarize_d3(rows, {("lag1", False): {s: pkg.cond_identity(s, "lag1", False) for s in SEEDS}}, SEEDS, n_boot=pkg.N_BOOT)
    assert clean["status"] == "PASS"
    CONDITION_MUTATIONS[name](rows[0])
    expected = {("lag1", False): {s: pkg.cond_identity(s, "lag1", False) for s in SEEDS}}
    result = stats.summarize_d3(rows, expected, SEEDS, n_boot=pkg.N_BOOT)
    assert result["status"] == "INVALID" and result["violations"], name
    assert not stats.contains_outcome({k: v for k, v in result.items() if k != "violations"})
    plan = {"D1": pkg.d1_rows(), "D2_A_no_drive": pkg._d2_rows(), "D3_lag1_no_drive": rows, "D4_drive": pkg._d4_rows()}
    summary, ex = run_into(tmp_path / "pkg", plan)
    assert ex.called == ["D1", "D2_A_no_drive", "D3_lag1_no_drive"], "D4 must not run"
    pkg.assert_frozen_package(tmp_path / "pkg", summary, "STOP_INVALID")


@pytest.mark.parametrize("name", ["diagnostic_WRONG", "coin_reads_negative", "checkpoint_diagnostic_label"])
def test_r3_15_D2_schema_mutation_stops_before_D3(tmp_path, name):
    rows = pkg._d2_rows()
    CONDITION_MUTATIONS[name](rows[7])
    summary, ex = run_into(tmp_path / "pkg", {"D1": pkg.d1_rows(), "D2_A_no_drive": rows})
    assert ex.called == ["D1", "D2_A_no_drive"]
    assert summary["statuses"]["D2"] == "D2_INVALID"
    pkg.assert_frozen_package(tmp_path / "pkg", summary, "STOP_INVALID")


@pytest.mark.parametrize("name", ["diagnostic_WRONG", "coin_reads_negative", "coin_reads_above_eval_len"])
def test_r3_16_D4_drive_row_schema_mutation_is_INVALID(tmp_path, name):
    plan = full_plan()
    CONDITION_MUTATIONS[name](plan["D4_drive"][5])
    summary, ex = run_into(tmp_path / "pkg", plan)
    assert summary["stages"]["D4"]["status"] == "INVALID"
    pkg.assert_frozen_package(tmp_path / "pkg", summary, "STOP_INVALID")


def test_r3_17_drive_source_index_must_be_int(tmp_path):
    plan = full_plan()
    plan["D4_drive"][0]["evaluation"]["source_start_index"] = float(diag.PHASE_STEPS)
    summary, _ = run_into(tmp_path / "pkg", plan)
    assert summary["branch_outcome"]["code"] == "STOP_INVALID"


def test_r3_18_real_short_rows_still_satisfy_strict_validator_on_fixture_seed():
    """Producer/validator agreement after tightening: both tasks, P and F0, with and without drive."""
    n_train, n_eval = 2_000, 2_300
    for task in ("A", "lag1"):
        for drive in (False, True):
            expected = diag.condition_object_hashes(4242, task, drive, n_train=n_train, n_eval=n_eval)
            for plastic in (True, False):
                row = diag.run_condition_seed(4242, task, plastic, drive, expected, n_train=n_train, n_eval=n_eval, warmup=2_000)
                row["status"] = "ok"
                row["worker_pid"] = os.getpid()
                problems = stats.validate_condition_row(
                    row, expected, n_checkpoints=2, phase_steps=n_train, eval_len=n_eval, eval_scored=n_eval - 2_000
                )
                assert problems == [], (task, drive, plastic, problems)
                assert 0 <= row["evaluation"]["coin_reads"] <= n_eval
