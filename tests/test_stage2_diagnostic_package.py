"""Review round 2 (Nora, t_53bf2d59): outcome exposure, D1 fail-closed validation,
early-stop package freezing, and section 13 effective parameters.

No diagnostic seed is simulated. Package-path tests inject synthetic rows built
in memory (no network transition) under non-diagnostic seed IDs 1000..1019.
Producer/validator agreement tests simulate only short lengths on the
non-diagnostic fixture seed 4242.
"""

import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_stage2_diagnostic as runner
from snn import stage2_diagnostic as diag
from snn import stage2_diagnostic_stats as stats

SEEDS = tuple(range(1000, 1020))  # synthetic IDs only; never simulated
FIXTURE_SEED = 4242
N_BOOT = 500


def h(*parts) -> str:
    return hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()


# ------------------------------------------------------------ D1 fixtures
def d1_identity(seed):
    return {
        "initial_weights_sha256": h("w", seed),
        "train_input_hash": h("trainA", seed),
        "test_input_hash": h("testA", seed),
    }


def d1_decoder(seed, name, accuracy, lower):
    rng = np.random.default_rng(seed * 10 + (name == "A"))
    mean = rng.normal(size=60).tolist()
    scale = (rng.random(60) + 0.5).tolist()
    coef = [rng.normal(size=60).tolist()]
    icpt = [float(rng.normal())]
    return {
        "accuracy": float(accuracy),
        "lower_95": float(lower),
        "scored": diag.D1_TEST_ROWS,
        "decoder_config": dict(diag.D1_DECODER_CONFIG),
        "train_label_sha256": h("label", name, seed),
        "mean": mean,
        "scale": scale,
        "normalization_sha256": diag.normalization_hash(mean, scale),
        "coefficients": coef,
        "intercept": icpt,
        "coefficients_sha256": diag.coefficients_hash(coef, icpt),
        "n_iter": [37],
        "converged": True,
    }


def d1_row(seed, a_acc=0.80, a_low=0.60, p_acc=0.90, p_low=0.85):
    ident = d1_identity(seed)
    return {
        "diagnostic": "D1",
        "seed": seed,
        "status": "ok",
        "train_input_hash": ident["train_input_hash"],
        "test_input_hash": ident["test_input_hash"],
        "train_feature_hash": h("trainF", seed),
        "test_feature_hash": h("testF", seed),
        "initial_weights_sha256": ident["initial_weights_sha256"],
        "train_rows": diag.D1_TRAIN_ROWS,
        "test_rows": diag.D1_TEST_ROWS,
        "warmup_rows": diag.D1_WARMUP,
        "feature_dim": diag.D1_FEATURE_DIM,
        "weights_constant": True,
        "decoders": {
            "A": d1_decoder(seed, "A", a_acc, a_low),
            "positive_prior": d1_decoder(seed, "positive_prior", p_acc, p_low),
        },
    }


def d1_rows(**kw):
    return [d1_row(s, **kw) for s in SEEDS]


D1_EXPECTED = {s: d1_identity(s) for s in SEEDS}


# ----------------------------------------------------- condition fixtures
def cond_identity(seed, task, drive):
    return {
        "initial_weights_sha256": h("w", seed),
        "train_stream_sha256": h("train", task, seed),
        "eval_stream_sha256": h("eval", task, seed),
        "tie_coin_sha256": h("coin", task, seed),
        "drive_sha256": h("drive", seed) if drive else None,
    }


def checkpoint(condition, seed, i, ident, weights_hash, hidden, output, silent):
    row = {
        "diagnostic": "D2",
        "condition": condition,
        "seed": seed,
        "checkpoint_index": i,
        "step_index": (i + 1) * diag.CHECKPOINT_STEPS - 1,
        "hidden_rate_hz": float(hidden),
        "output_rate_hz_O1": float(output),
        "output_rate_hz_O0": float(output),
        "output_layer_rate_hz": float(output),
        "both_silent_fraction": float(silent),
        "w1_mean": 0.0,
        "w1_min": -9.0,
        "w1_max": 9.0,
        "w1_lower_hits": 0,
        "w1_upper_hits": 0,
        "w2_mean": 5.0,
        "w2_min": 1.0,
        "w2_max": 9.0,
        "w2_lower_hits": 2,
        "w2_upper_hits": 0,
        "w1_lower_fraction": 0.0,
        "w2_lower_fraction": 2 / 40.0,
        "stream_sha256": ident["train_stream_sha256"],
        "initial_weights_sha256": ident["initial_weights_sha256"],
        "weights_sha256": weights_hash(i),
        "drive_sha256": ident["drive_sha256"],
        "finite": True,
        "bounds_ok": True,
    }
    diag.validate_checkpoint_row(row)
    return row


def cond_row(condition, seed, accuracy, hidden=10.0, output=8.0, silent=0.5):
    plastic, task, drive = stats.parse_condition(condition)
    ident = cond_identity(seed, task, drive)
    if plastic:
        weights_hash = lambda i: h("wP", condition, seed, i)  # noqa: E731
    else:
        weights_hash = lambda i: ident["initial_weights_sha256"]  # noqa: E731
    cps = [checkpoint(condition, seed, i, ident, weights_hash, hidden, output, silent) for i in range(diag.N_CHECKPOINTS)]
    return {
        "diagnostic": "D2/D3/D4",
        "condition": condition,
        "seed": seed,
        "status": "ok",
        "task": task,
        "plastic": plastic,
        "drive": drive,
        "initial_weights_sha256": ident["initial_weights_sha256"],
        "final_weights_sha256": cps[-1]["weights_sha256"],
        "weights_bitwise_constant": not plastic,
        "train_stream_sha256": ident["train_stream_sha256"],
        "eval_stream_sha256": ident["eval_stream_sha256"],
        "tie_coin_sha256": ident["tie_coin_sha256"],
        "drive_sha256": ident["drive_sha256"],
        "paired_hash_assertion_passed": True,
        "checkpoints": cps,
        "evaluation": {
            "accuracy": float(accuracy),
            "scored": diag.EVAL_SCORED,
            "weights_bitwise_constant": True,
            "coin_reads": 10,
            "source_start_index": diag.PHASE_STEPS if drive else None,
            "source_end_index": diag.PHASE_STEPS + diag.EVAL_LEN - 1 if drive else None,
        },
    }


def chance(seed, centre=0.50):
    return centre + 0.0004 * ((seed % 7) - 3)


def stage_rows(conditions, acc_fn, **kw):
    return [cond_row(c, s, acc_fn(c, s), **kw) for c in conditions for s in SEEDS]


class FakeExecute:
    """Stage executor returning synthetic rows; records which stages ran."""

    def __init__(self, plan):
        self.plan = plan
        self.called = []

    def __call__(self, stage, fn, jobs):
        self.called.append(stage)
        result = self.plan[stage]
        if isinstance(result, Exception):
            raise result
        rows = result() if callable(result) else result
        return copy.deepcopy(rows), {"stage": stage, "wall_seconds": 1.0, "peak_rss_total_bytes": 1, "interruptions": 0, "retries": 0}


def run_package(tmp_path, plan):
    ex = FakeExecute(plan)
    out = tmp_path / "pkg"
    prov = {"mode": "test", "workers": 1, "ss_probe": {"pass": True, "decision": "PASS"}}
    summary = runner.run_diagnostic(
        SimpleNamespace(workers=1),
        out,
        prov,
        execute=ex,
        seeds=SEEDS,
        d1_identity=d1_identity,
        condition_identity=cond_identity,
        n_boot=N_BOOT,
    )
    return summary, ex, out


def assert_frozen_package(out: Path, summary, code):
    for key in runner.REQUIRED_PACKAGE_FIELDS:
        assert key in summary, key
    assert runner.validate_package_summary(summary) == []
    assert summary["branch_outcome"]["code"] == code
    assert summary["no_oracle"] == runner.NO_ORACLE
    assert summary["claim_boundary"] == runner.CLAIM_BOUNDARY
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["branch_outcome"]["code"] == code
    branch = json.loads((out / "branch_outcome.json").read_text())
    assert branch["code"] == code and branch["table_row"] == runner.BRANCH_INDEX[code]
    manifest = (out / "SHA256SUMS").read_text().splitlines()
    listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in manifest}
    files = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    assert set(listed) == files
    for name, digest in listed.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    return listed


# ===================================================== finding 1: exposure
class _Watch:
    peak_total = peak_single = max_children = 0
    exceeded = False

    def start(self):
        pass

    def sample(self):
        pass

    def stop(self):
        pass


def _outcome_job_factory(out: Path, snapshots, fail_at=None):
    def job(seed):
        files = sorted(p.name for p in out.iterdir())
        texts = {p.name: p.read_text() for p in out.iterdir() if p.is_file()}
        snapshots.append((seed, files, texts))
        if seed == fail_at:
            raise RuntimeError("synthetic crash mid-stage")
        return d1_row(seed)

    return job


def test_r2_01_incomplete_stage_persists_only_outcome_free_ledger(tmp_path):
    out = tmp_path / "stage"
    out.mkdir()
    snapshots = []
    job = _outcome_job_factory(out, snapshots, fail_at=SEEDS[7])
    with pytest.raises(RuntimeError, match="synthetic crash"):
        runner.run_pool(job, list(SEEDS), 1, "D1", out, executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch())
    # During the stage and after the crash, the only file is the progress ledger.
    for _, files, texts in snapshots:
        assert files == ["D1.progress.jsonl"]
        for text in texts.values():
            for line in filter(None, text.splitlines()):
                rec = json.loads(line)
                assert set(rec) == set(runner.PROGRESS_FIELDS)
                assert not stats.contains_outcome(rec)
                assert "accuracy" not in line and "lower_95" not in line
    assert sorted(p.name for p in out.iterdir()) == ["D1.progress.jsonl"]
    ledger = (out / "D1.progress.jsonl").read_text().splitlines()
    assert len(ledger) >= 1 and all(set(json.loads(l)) == set(runner.PROGRESS_FIELDS) for l in ledger)
    assert not list(out.glob("*.partial.jsonl"))


def test_r2_02_completed_stage_rows_stay_in_memory_until_returned(tmp_path):
    out = tmp_path / "stage"
    out.mkdir()
    snapshots = []
    rows, meta = runner.run_pool(
        _outcome_job_factory(out, snapshots), list(SEEDS), 1, "D1", out,
        executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch(),
    )
    assert len(rows) == 20 and meta["rows_ok"] == 20
    assert sorted(p.name for p in out.iterdir()) == ["D1.progress.jsonl"]
    assert not stats.contains_outcome([json.loads(l) for l in (out / "D1.progress.jsonl").read_text().splitlines()])
    assert not stats.contains_outcome(meta)


def test_r2_03_exception_inside_sequence_freezes_outcome_free_invalid_package(tmp_path):
    good_d1 = d1_rows()
    summary, ex, out = run_package(tmp_path, {"D1": good_d1, "D2_A_no_drive": RuntimeError("worker pool died")})
    assert ex.called == ["D1", "D2_A_no_drive"]
    assert_frozen_package(out, summary, "STOP_INVALID")
    assert summary["run_metadata"]["incomplete_reason"].startswith("exception")
    assert summary["statuses"]["D2"] is None and "D2" not in summary["stages"]
    assert not (out / "d2_rows.jsonl").exists()


def test_r2_04_invalid_stage_rows_are_written_redacted(tmp_path):
    bad = d1_rows()
    bad[3]["weights_constant"] = False
    summary, _, out = run_package(tmp_path, {"D1": bad})
    assert not (out / "d1_rows.jsonl").exists()
    redacted = [json.loads(l) for l in (out / "d1_rows.invalid_redacted.jsonl").read_text().splitlines()]
    assert len(redacted) == 20 and not stats.contains_outcome(redacted)
    assert not stats.contains_outcome(summary["stages"]["D1"])


# ================================================ finding 2: D1 fail-open
def test_r2_05_nora_probe_bare_summarize_d1_is_INVALID_not_PASS():
    rows = []
    for seed in diag.DIAGNOSTIC_SEEDS:
        rows.append({
            "seed": seed,
            "weights_constant": False,
            "train_input_hash": "same",
            "test_input_hash": "same",
            "decoders": {
                "A": {"accuracy": 0.8, "lower_95": 0.6, "converged": False},
                "positive_prior": {"accuracy": 0.8, "lower_95": 0.6, "converged": False},
            },
        })
    result = diag.summarize_d1(rows)
    assert result["status"] == "INVALID" and result["A"] is None and result["positive_prior"] is None
    assert not stats.contains_outcome({k: v for k, v in result.items() if k != "violations"})


def test_r2_06_valid_synthetic_D1_package_reaches_predicates():
    assert diag.validate_d1_package(d1_rows(), D1_EXPECTED, SEEDS) == []
    assert diag.summarize_d1(d1_rows(), D1_EXPECTED, SEEDS)["status"] == "PASS"
    fail = diag.summarize_d1(d1_rows(a_acc=0.52, a_low=0.45), D1_EXPECTED, SEEDS)
    assert fail["status"] == "FAIL"
    harness = diag.summarize_d1(d1_rows(p_acc=0.60, p_low=0.55), D1_EXPECTED, SEEDS)
    assert harness["status"] == "INVALID_HARNESS"


def _set(path, value):
    def mutate(row):
        target = row
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _delete(path):
    def mutate(row):
        target = row
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return mutate


D1_ADVERSARIAL = {
    "weights_changed": _set(("weights_constant",), False),
    "identical_train_test_input_hash": lambda r: r.update(test_input_hash=r["train_input_hash"]),
    "identical_train_test_feature_hash": lambda r: r.update(test_feature_hash=r["train_feature_hash"]),
    "coordinator_identity_mismatch": _set(("initial_weights_sha256",), h("other")),
    "A_not_converged": _set(("decoders", "A", "converged"), False),
    "positive_not_converged": _set(("decoders", "positive_prior", "converged"), False),
    "iterations_hit_max_iter": _set(("decoders", "A", "n_iter"), [diag.D1_MAX_ITER]),
    "nan_accuracy": _set(("decoders", "A", "accuracy"), float("nan")),
    "inf_lower_bound": _set(("decoders", "positive_prior", "lower_95"), float("inf")),
    "accuracy_not_a_count": _set(("decoders", "A", "accuracy"), 0.80001234),
    "nonfinite_coefficient": lambda r: r["decoders"]["A"]["coefficients"][0].__setitem__(3, float("nan")),
    "coefficient_hash_mismatch": lambda r: r["decoders"]["A"]["coefficients"][0].__setitem__(3, 123.0),
    "normalization_hash_mismatch": lambda r: r["decoders"]["A"]["mean"].__setitem__(0, 9.0),
    "zero_scale": lambda r: r["decoders"]["A"]["scale"].__setitem__(0, 0.0),
    "wrong_decoder_C": lambda r: r["decoders"]["A"]["decoder_config"].__setitem__("C", 2.0),
    "train_row_count": _set(("train_rows",), 197_999),
    "test_row_count": _set(("test_rows",), 9_999),
    "scored_count": _set(("decoders", "A", "scored"), 9_999),
    "feature_dim": _set(("feature_dim",), 59),
    "warmup_rows": _set(("warmup_rows",), 0),
    "missing_field": _delete(("test_feature_hash",)),
    "missing_decoder_field": _delete(("decoders", "A", "normalization_sha256")),
    "missing_decoder": _delete(("decoders", "positive_prior")),
    "positive_label_equals_A_label": lambda r: r["decoders"]["positive_prior"].update(train_label_sha256=r["decoders"]["A"]["train_label_sha256"]),
    "job_error_status": _set(("status",), "error"),
    "bad_hash_format": _set(("train_input_hash",), "same"),
}


@pytest.mark.parametrize("name", sorted(D1_ADVERSARIAL))
def test_r2_07_each_D1_invariant_forces_INVALID_and_stops_downstream(tmp_path, name):
    rows = d1_rows()
    D1_ADVERSARIAL[name](rows[5])
    result = diag.summarize_d1(rows, D1_EXPECTED, SEEDS)
    assert result["status"] == "INVALID", name
    assert result["violations"] and result["A"] is None and result["positive_prior"] is None
    summary, ex, out = run_package(tmp_path, {"D1": rows})
    assert ex.called == ["D1"], "a D1 invariant failure must stop downstream execution"
    assert_frozen_package(out, summary, "STOP_INVALID")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),  # missing seed / row count
        lambda rows: rows.append(copy.deepcopy(rows[0])),  # duplicate seed
        lambda rows: rows[0].update(seed=999),  # unexpected seed
    ],
    ids=["missing_seed", "duplicate_seed", "unexpected_seed"],
)
def test_r2_08_D1_package_row_set_invariants(mutate):
    rows = d1_rows()
    mutate(rows)
    assert diag.summarize_d1(rows, D1_EXPECTED, SEEDS)["status"] == "INVALID"


def test_r2_09_real_short_D1_row_satisfies_validator_on_fixture_seed():
    n_train, n_test = 2_600, 2_300
    row = diag.run_d1_seed(FIXTURE_SEED, n_train=n_train, n_test=n_test)
    row["status"] = "ok"
    ident = {
        "initial_weights_sha256": diag.sha256(*diag.initial_weights(FIXTURE_SEED)),
        "train_input_hash": diag.sha256(diag.a_stream(FIXTURE_SEED, True, n_train)),
        "test_input_hash": diag.sha256(diag.a_stream(FIXTURE_SEED, False, n_test)),
    }
    assert diag.validate_d1_row(row, ident, train_rows=n_train - 2_000, test_rows=n_test - 2_000) == []
    # The same real row is refused at spec lengths: short rows can never pass the package validator.
    assert any("train_rows" in v for v in diag.validate_d1_row(row, ident))


# =========================================== finding 3: early-stop package
EARLY_BRANCHES = {
    "D1_FAIL": ({"D1": d1_rows(a_acc=0.52, a_low=0.45)}, "STOP_D1_FAIL", ["D1"]),
    "D1_INCONCLUSIVE": ({"D1": d1_rows(a_acc=0.62, a_low=0.52)}, "STOP_D1_INCONCLUSIVE", ["D1"]),
    "D1_INVALID_HARNESS": ({"D1": d1_rows(p_acc=0.60, p_low=0.55)}, "STOP_D1_INVALID_HARNESS", ["D1"]),
}


@pytest.mark.parametrize("name", sorted(EARLY_BRANCHES))
def test_r2_10_every_D1_stop_branch_freezes_section13_package(tmp_path, name):
    plan, code, called = EARLY_BRANCHES[name]
    summary, ex, out = run_package(tmp_path, plan)
    assert ex.called == called
    listed = assert_frozen_package(out, summary, code)
    assert {"summary.json", "branch_outcome.json", "d1_rows.jsonl"} <= set(listed)
    assert summary["d1_claim_boundary"] == runner.D1_CLAIM
    assert summary["effective_parameters_match_spec"] is True
    assert summary["package_summary"]["branch"] == code and summary["package_summary"]["not_supported"]
    if name != "D1_INVALID_HARNESS":
        assert set(summary["stages"]["D1"]["normalization_hashes"]) == {str(s) for s in SEEDS}


def _d2_rows():
    return stage_rows(stats.D2_CONDITIONS, lambda c, s: chance(s))


def _d3_rows(p_centre=0.50, f0_centre=0.50):
    return stage_rows(stats.D3_CONDITIONS, lambda c, s: chance(s, p_centre if c.startswith("P") else f0_centre))


def _d4_rows(p_drive=0.80, f0_drive=0.50, silent=0.5):
    rows = stage_rows(("P_A+drive",), lambda c, s: chance(s, p_drive), silent=silent)
    rows += stage_rows(("F0_A+drive",), lambda c, s: chance(s, f0_drive))
    rows += stage_rows(("P_lag1+drive",), lambda c, s: chance(s, 0.5))
    return rows


def test_r2_11_D2_invalid_stops_and_freezes(tmp_path):
    bad = _d2_rows()
    bad[4]["checkpoints"] = bad[4]["checkpoints"][:-1]
    summary, ex, out = run_package(tmp_path, {"D1": d1_rows(), "D2_A_no_drive": bad})
    assert ex.called == ["D1", "D2_A_no_drive"]
    assert_frozen_package(out, summary, "STOP_INVALID")
    assert summary["statuses"]["D2"] == "D2_INVALID"
    assert (out / "d2_rows.invalid_redacted.jsonl").exists() and not (out / "d2_rows.jsonl").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[25].update(weights_bitwise_constant=False),  # frozen F0 changed
        lambda rows: rows[25].update(drive_sha256=h("drive", 0)),  # drive in a no-drive row
        lambda rows: rows[3]["evaluation"].update(weights_bitwise_constant=False),
        lambda rows: rows[3]["checkpoints"][10].update(hidden_rate_hz=float("nan")),
        lambda rows: rows[3].update(tie_coin_sha256=h("x")),  # pairing broken
    ],
    ids=["F0_weights_changed", "drive_present_without_drive", "eval_mutated", "nonfinite_checkpoint", "pairing_hash"],
)
def test_r2_12_D3_integrity_failures_stop_before_D4(tmp_path, mutate):
    bad = _d3_rows()
    mutate(bad)
    summary, ex, out = run_package(tmp_path, {"D1": d1_rows(), "D2_A_no_drive": _d2_rows(), "D3_lag1_no_drive": bad})
    assert ex.called == ["D1", "D2_A_no_drive", "D3_lag1_no_drive"]
    assert_frozen_package(out, summary, "STOP_INVALID")
    assert summary["statuses"]["D3"] == "INVALID"
    assert not stats.contains_outcome(summary["stages"]["D3"])


def test_r2_13_D3_frozen_chance_interval_invalid_stops_before_D4(tmp_path):
    plan = {"D1": d1_rows(), "D2_A_no_drive": _d2_rows(), "D3_lag1_no_drive": _d3_rows(f0_centre=0.60)}
    summary, ex, out = run_package(tmp_path, plan)
    assert ex.called == ["D1", "D2_A_no_drive", "D3_lag1_no_drive"]
    assert_frozen_package(out, summary, "STOP_INVALID")


@pytest.mark.parametrize(
    "d3_centre,d4_kw,code",
    [
        (0.50, {}, "RETURN_D3_FAIL"),
        (0.90, {"p_drive": 0.80}, "CONSIDER_DOC_ONLY_R4_SPEC"),
        (0.90, {"p_drive": 0.50}, "RETURN_D4_COMPETENCE_FAIL"),
        (0.90, {"p_drive": 0.80, "silent": 0.95}, "RETURN_D4_SILENCE_FAIL"),
        (0.90, {"f0_drive": 0.70}, "STOP_INVALID"),
    ],
)
def test_r2_14_full_sequence_branches_freeze_with_D4_fields(tmp_path, d3_centre, d4_kw, code):
    plan = {
        "D1": d1_rows(),
        "D2_A_no_drive": _d2_rows(),
        "D3_lag1_no_drive": _d3_rows(p_centre=d3_centre),
        "D4_drive": _d4_rows(**d4_kw),
    }
    summary, ex, out = run_package(tmp_path, plan)
    assert ex.called == ["D1", "D2_A_no_drive", "D3_lag1_no_drive", "D4_drive"]
    listed = assert_frozen_package(out, summary, code)
    assert {"d1_rows.jsonl", "d2_rows.jsonl", "d3_rows.jsonl", "d4_drive_rows.jsonl"} <= set(listed)
    d4 = summary["stages"]["D4"]
    assert set(d4["drive_hashes_by_seed"]) == {str(s) for s in SEEDS}
    assert d4["lag1_report_only"]["report_only"] is True
    assert set(summary["stages"]["D3"]["generator_hashes"]) == {str(s) for s in SEEDS}


def test_r2_15_D4_drive_identity_mismatch_is_INVALID(tmp_path):
    rows = _d4_rows()
    rows[45]["drive_sha256"] = h("other-drive")  # P_lag1+drive seed reads a different realization
    for cp in rows[45]["checkpoints"]:
        cp["drive_sha256"] = rows[45]["drive_sha256"]
    plan = {"D1": d1_rows(), "D2_A_no_drive": _d2_rows(), "D3_lag1_no_drive": _d3_rows(0.9), "D4_drive": rows}
    summary, _, out = run_package(tmp_path, plan)
    assert_frozen_package(out, summary, "STOP_INVALID")
    assert summary["stages"]["D4"]["status"] == "INVALID"
    assert not stats.contains_outcome(summary["stages"]["D4"])


def test_r2_16_branch_table_is_exhaustive_ordered_and_unknown_codes_rejected():
    assert len(runner.BRANCH_TABLE) == 12
    assert runner.branch_outcome("INVALID")["code"] == "STOP_INVALID"
    assert runner.branch_outcome(None)["code"] == "STOP_INVALID"
    assert runner.branch_outcome("PASS")["code"] == "STOP_INVALID"  # PASS that never reached D2-D4 is incomplete
    assert runner.branch_outcome("INVALID_HARNESS")["code"] == "STOP_D1_INVALID_HARNESS"
    # Earlier rows take precedence: D4(ii) FAIL shadows the row-11 combination.
    d4 = {"status": "FAIL", "silence_prevention": "FAIL", "competence": "PASS"}
    assert runner.branch_outcome("PASS", "D2_FAIL_CO_ONSET", "PASS", d4)["code"] == "RETURN_D4_SILENCE_FAIL"
    assert runner.validate_package_summary({"branch_outcome": {"code": "MADE_UP"}})


def test_r2_17_hard_kill_leaves_fail_closed_branch_marker(tmp_path):
    out = tmp_path / "pkg"
    out.mkdir()
    runner.write_incomplete_marker(out)
    record = json.loads((out / "branch_outcome.json").read_text())
    assert record["code"] == "STOP_INVALID" and not stats.contains_outcome(record)


# ========================================= finding 4: section 13 fields
def test_r2_18_effective_parameters_read_from_objects_match_spec_transcription():
    checked = runner.validated_effective_parameters()
    assert checked["matches_spec"], checked["mismatches"]
    eff = checked["effective"]
    drive = eff["d4_drive"]
    assert drive["sources_total"] == 176 and drive["sources_per_target"] == 8
    assert drive["rate_hz"] == 25.0 and drive["weight_mv"] == 2.0
    assert drive["train_indices"] == [0, 199_999] and drive["eval_indices"] == [200_000, 211_999]
    assert drive["background_plasticity_objects"] == 0
    assert eff["d1"]["decoder"] == diag.D1_DECODER_CONFIG
    assert eff["thresholds"] == diag.THRESHOLDS
    assert eff["spec_sha256"] == diag.SPEC_SHA256


def test_r2_19_parameter_or_threshold_drift_is_detected():
    eff = copy.deepcopy(diag.effective_parameters())
    eff["d4_drive"]["rate_hz"] = 30.0
    eff["thresholds"]["d3_pass_lower_bound_gt"] = 0.50
    eff["gamma_mv"] = [0.25, 0.30]
    del eff["d1"]["block_length"]
    problems = runner.parameter_mismatches(eff, runner.expected_parameters_from_spec())
    assert any("d4_drive.rate_hz" in p for p in problems)
    assert any("thresholds.d3_pass_lower_bound_gt" in p for p in problems)
    assert any("gamma_mv" in p for p in problems)
    assert any("d1.block_length: missing" in p for p in problems)


def test_r2_20_D1_package_serializes_training_normalization_and_coefficient_hashes(tmp_path):
    summary, _, _ = run_package(tmp_path, {"D1": d1_rows(a_acc=0.52, a_low=0.45)})
    d1 = summary["stages"]["D1"]
    rows = {r["seed"]: r for r in d1_rows()}
    for seed in SEEDS:
        for name in diag.D1_DECODERS:
            dec = rows[seed]["decoders"][name]
            assert d1["normalization_hashes"][str(seed)][name] == diag.normalization_hash(dec["mean"], dec["scale"])
            assert d1["coefficient_hashes"][str(seed)][name] == dec["coefficients_sha256"]
            assert d1["convergence"][str(seed)][name] is True
    assert summary["effective_parameters"]["d4_drive"]["sources_total"] == 176


def test_r2_21_real_short_drive_row_satisfies_condition_validator_on_fixture_seed():
    n_train, n_eval = 2_000, 2_300
    expected = diag.condition_object_hashes(FIXTURE_SEED, "A", True, n_train=n_train, n_eval=n_eval)
    row = diag.run_condition_seed(FIXTURE_SEED, "A", False, True, expected, n_train=n_train, n_eval=n_eval, warmup=2_000)
    row["status"] = "ok"
    problems = stats.validate_condition_row(
        row, expected, n_checkpoints=2, phase_steps=n_train, eval_len=n_eval, eval_scored=n_eval - 2_000
    )
    assert problems == []
    assert row["evaluation"]["source_start_index"] == n_train
    # At spec lengths the same short row is refused.
    assert stats.validate_condition_row(row, expected)
