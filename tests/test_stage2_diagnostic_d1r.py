"""D1R section 11 construction tests (docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md at 2d9efd8).

One named test per section 11 item (``test_d1r_11_NN_*``), plus fail-closed
package tests carried from the approved ebb90e74 runner contract.

No D1R seed (2100..2119) is simulated, streamed, fitted, or scored here.
Network simulation runs only on the non-diagnostic fixture seed 4242, where
only the pipeline decoder is ever fitted (spec 11 item 15). Branch, validator,
and package tests use synthetic rows under the synthetic seed IDs 1000..1019
with synthetic identities; no network transition backs them. The producer /
validator agreement test for all three decoders uses an injected synthetic
capture (random features, no network) under the out-of-range ID 9001.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_stage2_diagnostic_d1r as runner
from snn import stage2_diagnostic as diag
from snn import stage2_diagnostic_d1r as d1r
from snn import stage2_diagnostic_stats as stats
from snn.stage2_r3 import ONEHOT

ROOT = Path(runner.__file__).resolve().parent
FIXTURE = 4242
SYN_SEEDS = tuple(range(1000, 1020))  # synthetic row IDs only, never simulated
SHORT_TRAIN, SHORT_TEST = 4_000, 3_000
SHORT_TRAIN_ROWS, SHORT_TEST_ROWS = SHORT_TRAIN - d1r.WARMUP, SHORT_TEST - d1r.WARMUP


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


# ---------------------------------------------------------------- synthetic rows
def syn_identity(seed: int) -> dict:
    prior_tr, prior_te = _h(f"prior-train-{seed}"), _h(f"prior-test-{seed}")
    return {
        "initial_weights_sha256": _h(f"w-{seed}"),
        "train_input_hash": _h(f"xtr-{seed}"),
        "test_input_hash": _h(f"xte-{seed}"),
        "pipeline_train_label_sha256": prior_tr,
        "pipeline_test_label_sha256": prior_te,
        "network_positive_train_label_sha256": prior_tr,
        "network_positive_test_label_sha256": prior_te,
        "A_train_label_sha256": _h(f"A-train-{seed}"),
        "A_test_label_sha256": _h(f"A-test-{seed}"),
    }


def _syn_decoder(name: str, seed: int, accuracy: float, lower: float) -> dict:
    width = d1r.DECODER_WIDTH[name]
    g = np.random.default_rng([seed, len(name)])
    mean = g.normal(size=width).tolist()
    scale = (1.0 + g.random(width)).tolist()
    coef = [g.normal(size=width).tolist()]
    icpt = [float(g.normal())]
    ident = syn_identity(seed)
    rec = {
        "accuracy": float(round(accuracy * d1r.TEST_ROWS) / d1r.TEST_ROWS),
        "scored": d1r.TEST_ROWS,
        "decoder_config": dict(d1r.DECODER_CONFIG),
        "train_label_sha256": ident[f"{name}_train_label_sha256"],
        "test_label_sha256": ident[f"{name}_test_label_sha256"],
        "mean": mean,
        "scale": scale,
        "normalization_sha256": diag.normalization_hash(mean, scale),
        "coefficients": coef,
        "intercept": icpt,
        "coefficients_sha256": diag.coefficients_hash(coef, icpt),
        "n_iter": [123],
        "converged": True,
    }
    if name in d1r.NETWORK_DECODERS:
        rec["lower_95"] = float(lower)
    return rec


def syn_row(seed: int, pipe=1.0, pos=(0.75, 0.6), a=(0.75, 0.6)) -> dict:
    ident = syn_identity(seed)
    return {
        "diagnostic": "D1R",
        "seed": seed,
        "train_input_hash": ident["train_input_hash"],
        "test_input_hash": ident["test_input_hash"],
        "train_network_feature_hash": _h(f"ftrn-{seed}"),
        "test_network_feature_hash": _h(f"ften-{seed}"),
        "train_pipeline_feature_hash": _h(f"ftrp-{seed}"),
        "test_pipeline_feature_hash": _h(f"ftep-{seed}"),
        "initial_weights_sha256": ident["initial_weights_sha256"],
        "train_rows": d1r.TRAIN_ROWS,
        "test_rows": d1r.TEST_ROWS,
        "warmup_rows": d1r.WARMUP,
        "network_feature_dim": d1r.NETWORK_FEATURE_DIM,
        "pipeline_feature_dim": d1r.PIPELINE_FEATURE_DIM,
        "weights_constant": True,
        "nesting_ok": True,
        "reward_computed": False,
        "decoders": {
            "pipeline": _syn_decoder("pipeline", seed, pipe, 0.0),
            "network_positive": _syn_decoder("network_positive", seed, *pos),
            "A": _syn_decoder("A", seed, *a),
        },
        "status": "ok",
        "worker_pid": 1,
    }


def syn_rows(**kw) -> list:
    return [syn_row(s, **kw) for s in SYN_SEEDS]


SYN_EXPECTED = {s: syn_identity(s) for s in SYN_SEEDS}


def summarize(rows, probe_pass=True, expected=SYN_EXPECTED):
    return d1r.summarize(rows, expected, probe_pass, SYN_SEEDS)


# ---------------------------------------------------------------- fixtures
def _fixture_capture(n: int, flip_at=None):
    w1, w2 = d1r.initial_weights(FIXTURE)
    x = d1r.a_stream(FIXTURE, True, n).copy()
    if flip_at is not None:
        x[flip_at] ^= 1
    return x, d1r.capture_stream(x, w1, w2)


@pytest.fixture(scope="module")
def short_pipeline_row():
    return d1r.run_d1r_seed(FIXTURE, decoders=("pipeline",), n_train=SHORT_TRAIN, n_test=SHORT_TEST)


# ================================================================ section 11
# 11.1
def test_d1r_11_01_namespace_31_identities_r3_draw_order_and_seeds_2100_2119():
    assert d1r.D1R_SEEDS == tuple(range(2100, 2120)) and d1r.NAMESPACE == 31
    for seed in (FIXTURE, 2100, 2119):  # tuples only for D1R seeds; no generator is drawn
        assert d1r.entropy_table(seed) == {
            "weights_w1_w2_o1": (20261001, seed, 31, 1),
            "weights_w2_o0": (20261001, seed, 31, 21),
            "A_train": (20261001, seed, 31, 2),
            "A_test": (20261001, seed, 31, 5),
            "D1R_block_bootstrap_A": (20261001, seed, 31, 60, 1),
            "D1R_block_bootstrap_network_positive_control": (20261001, seed, 31, 60, 2),
        }
    w1, w2 = d1r.initial_weights(FIXTURE)
    g = np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 31, 1]))
    assert np.array_equal(w1, g.uniform(-10.0, 10.0, size=(20, 2)))
    assert np.array_equal(w2[0:1], g.uniform(0.0, 10.0, size=(1, 20)))
    h = np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 31, 21]))
    assert np.array_equal(w2[1:2], h.uniform(0.0, 10.0, size=(1, 20)))
    assert w1.dtype == w2.dtype == np.float64 and w1.shape == (20, 2) and w2.shape == (2, 20)
    # Same r3 construction as ebb90e74, different namespace.
    assert not np.array_equal(w1, diag.initial_weights(FIXTURE)[0])
    for train, comp in ((True, 2), (False, 5)):
        want = diag._roll_a(np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 31, comp])), 500)
        assert np.array_equal(d1r.a_stream(FIXTURE, train, 500), want)
    assert d1r.sha256(d1r.a_stream(FIXTURE, True, 500)) != d1r.sha256(d1r.a_stream(FIXTURE, False, 500))
    # Every generator constructed by a D1R object goes through d1r.rng with a section 3.1 tuple.
    seen = []
    orig = d1r.rng
    d1r_rng = lambda e: (seen.append(tuple(e)), orig(e))[1]  # noqa: E731
    try:
        d1r.rng = d1r_rng
        d1r.initial_weights(FIXTURE)
        d1r.a_stream(FIXTURE, True, 10)
        d1r.a_stream(FIXTURE, False, 10)
        d1r.block_lower_bound(np.ones(200), FIXTURE, 1)
        d1r.block_lower_bound(np.ones(200), FIXTURE, 2)
    finally:
        d1r.rng = orig
    assert seen == list(d1r.entropy_table(FIXTURE).values())


# 11.2
def test_d1r_11_02_phiR_columns_0_59_bit_identical_to_approved_FeatureAccumulator():
    g = np.random.default_rng(20261001)
    spikes = (g.random((400, 20)) < 0.3).astype(np.float64)
    approved = diag.FeatureAccumulator()
    lag = d1r.LagFeatureAccumulator(20)
    for s in spikes:
        row, _ = lag.update(s)
        assert np.array_equal(row[:60], approved.update(s))
    # On the real frozen network: same weights and stream as the approved capture.
    n = 600
    approved_cap = diag.capture_d1_features(FIXTURE, train=True, n=n)
    w1, w2 = diag.initial_weights(FIXTURE)
    mine = d1r.capture_stream(diag.a_stream(FIXTURE, True, n), w1, w2)
    assert mine["network"].shape == (n, 460)
    assert np.array_equal(mine["network"][:, :60], approved_cap["features"])
    assert mine["network"][:, :60].tobytes() == approved_cap["features"].tobytes()


# 11.3
def test_d1r_11_03_hand_computed_lag_block_order_zero_fill_and_count20_nesting():
    n = 3
    hidden = np.zeros((45, n), dtype=np.float64)
    hidden[[0, 4, 19, 20, 39], 0] = 1
    hidden[[1, 2, 21, 40, 44], 1] = 1
    hidden[[10, 30, 31, 32], 2] = 1
    acc = d1r.LagFeatureAccumulator(n)
    assert acc.width == 3 * n + 20 * n
    for t, h in enumerate(hidden):
        row, nested = acc.update(h)
        lag = row[3 * n :].reshape(20, n)
        for k in range(20):
            want = hidden[t - k] if t - k >= 0 else np.zeros(n)
            assert np.array_equal(lag[k], want), (t, k)
        assert nested is True
        assert np.array_equal(row[:n], lag.sum(axis=0))
        assert np.array_equal(row[:n], hidden[max(0, t - 19) : t + 1].sum(axis=0))
    # Declared column order: lag 0 neurons 0..n-1, lag 1, ..., lag 19.
    acc2 = d1r.LagFeatureAccumulator(2)
    acc2.update(np.array([1.0, 0.0]))
    row, _ = acc2.update(np.array([0.0, 1.0]))
    assert row[6:12].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert d1r.NETWORK_FEATURE_DIM == 460 and d1r.PIPELINE_FEATURE_DIM == 46 and d1r.K == 19
    # A nesting violation is detected (section 7 item 7).
    bad = d1r.LagFeatureAccumulator(2)
    bad.update(np.array([1.0, 1.0]))
    bad.ring[:] = 0.0
    assert bad.update(np.array([0.0, 0.0]))[1] is False


# 11.4
def test_d1r_11_04_rows_cannot_depend_on_current_x_t_and_change_enters_at_t_plus_1():
    n = 400
    x0, base = _fixture_capture(n)
    network_changed_at_t1 = 0
    for t in (50, 120, 200, 260, 333):
        x1, alt = _fixture_capture(n, flip_at=t)
        assert x1[t] != x0[t] and np.array_equal(np.delete(x1, t), np.delete(x0, t))
        for key in ("network", "pipeline"):
            assert np.array_equal(base[key][: t + 1], alt[key][: t + 1]), (key, t)
        # psi lag-0 block at t+1 is exactly ONEHOT[x_t]: the change enters there and not earlier.
        assert not np.array_equal(base["pipeline"][t + 1], alt["pipeline"][t + 1])
        assert np.array_equal(alt["pipeline"][t + 1, 6:8], ONEHOT[int(x1[t])].numpy()[0])
        network_changed_at_t1 += int(not np.array_equal(base["network"][t + 1], alt["network"][t + 1]))
    assert network_changed_at_t1 >= 1  # hidden response to x_t first appears at row t+1
    # The accumulators take no input argument: x_t is not reachable from update().
    import inspect

    assert list(inspect.signature(d1r.LagFeatureAccumulator.update).parameters) == ["self", "spikes_t"]


# 11.5
def test_d1r_11_05_u_t_is_onehot_previous_input_and_psi_uses_identical_path_width_2():
    assert np.array_equal(d1r.input_layer_vector(None), np.zeros(2))
    assert np.array_equal(d1r.input_layer_vector(0), [1.0, 0.0])
    assert np.array_equal(d1r.input_layer_vector(1), [0.0, 1.0])
    n = 300
    x, cap = _fixture_capture(n)
    psi = cap["pipeline"]
    assert psi.shape == (n, 46)
    assert np.array_equal(psi[0], np.zeros(46))
    for t in range(1, n):
        assert np.array_equal(psi[t, 6:8], ONEHOT[int(x[t - 1])].numpy()[0])
    # Identical operator: an independent LagFeatureAccumulator(2) over u_t reproduces psi exactly,
    # and it is the same class (and approved base accumulator) as the width-20 network map.
    acc = d1r.LagFeatureAccumulator(2)
    u = [np.zeros(2)] + [ONEHOT[int(v)].numpy()[0] for v in x[:-1]]
    assert np.array_equal(np.stack([acc.update(v)[0] for v in u]), psi)
    assert type(acc.base) is diag.FeatureAccumulator and acc.base.n_hidden == 2


# 11.6
def test_d1r_11_06_label_hashes_coordinator_derived_x1999_and_x2000(monkeypatch, short_pipeline_row):
    xtr = d1r.a_stream(FIXTURE, True, SHORT_TRAIN)
    xte = d1r.a_stream(FIXTURE, False, SHORT_TEST)
    # Coordinator derivation never consults the worker's label code.
    monkeypatch.setattr(d1r, "worker_labels", lambda *_: (_ for _ in ()).throw(AssertionError("worker path used")))
    monkeypatch.setattr(diag, "d1_labels", lambda *_: (_ for _ in ()).throw(AssertionError("worker path used")))
    ident = d1r.object_hashes(FIXTURE, n_train=SHORT_TRAIN, n_test=SHORT_TEST)
    monkeypatch.undo()
    s = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype=np.int8).tobytes()).hexdigest()  # noqa: E731
    for name in ("pipeline", "network_positive"):
        assert ident[f"{name}_train_label_sha256"] == s(xtr[1999:-1])
        assert ident[f"{name}_test_label_sha256"] == s(xte[1999:-1])
    assert ident["A_train_label_sha256"] == s(xtr[2000:])
    assert ident["A_test_label_sha256"] == s(xte[2000:])
    # The real worker row (pipeline only, seed 4242) agrees exactly.
    pipe = short_pipeline_row["decoders"]["pipeline"]
    assert pipe["train_label_sha256"] == ident["pipeline_train_label_sha256"]
    assert pipe["test_label_sha256"] == ident["pipeline_test_label_sha256"]
    assert not d1r.validate_row(short_pipeline_row, ident, train_rows=SHORT_TRAIN_ROWS, test_rows=SHORT_TEST_ROWS, decoders=("pipeline",))
    # A worker that used the current input as the "prior" label is refused.
    bad = copy.deepcopy(short_pipeline_row)
    bad["decoders"]["pipeline"]["train_label_sha256"] = ident["A_train_label_sha256"]
    assert any("label identity" in v for v in d1r.validate_row(bad, ident, train_rows=SHORT_TRAIN_ROWS, test_rows=SHORT_TEST_ROWS, decoders=("pipeline",)))
    # On full synthetic rows: each of the 6 label slots, swapped A/prior, and missing identity all force INVALID.
    for name in d1r.DECODERS:
        for split in ("train", "test"):
            rows = syn_rows()
            rows[3]["decoders"][name][f"{split}_label_sha256"] = _h("arbitrary")
            assert summarize(rows)["status"] == "INVALID"
    rows = syn_rows()
    for split in ("train", "test"):
        k = f"{split}_label_sha256"
        d = rows[0]["decoders"]
        d["A"][k], d["network_positive"][k] = d["network_positive"][k], d["A"][k]
    assert summarize(rows)["status"] == "INVALID"
    partial = {s: {k: v for k, v in syn_identity(s).items() if "label" not in k} for s in SYN_SEEDS}
    assert summarize(syn_rows(), expected=partial)["status"] == "INVALID"
    assert summarize(syn_rows(), expected=None)["status"] == "INVALID"


# 11.7
def test_d1r_11_07_training_only_standardization_test_changes_cannot_alter_train_state():
    for width in (460, 46):
        g = np.random.default_rng(width)
        train = g.normal(size=(600, width))
        train[:, 5] = 2.0  # zero variance -> scale 1
        y = (train[:, 0] - 0.3 * train[:, 7] > 0).astype(np.int8)
        fitted = diag.fit_d1_decoder(train, y)
        before = [a.copy() for a in (fitted.mean, fitted.scale, fitted.coefficients, fitted.intercept)]
        fitted.predict(g.normal(size=(50, width)))
        fitted.predict(g.normal(size=(50, width)) * 1e6 + 1e6)
        assert all(np.array_equal(a, b) for a, b in zip(before, (fitted.mean, fitted.scale, fitted.coefficients, fitted.intercept)))
        assert np.array_equal(fitted.mean, train.mean(axis=0))
        want = train.std(axis=0, ddof=0)
        want[want == 0] = 1.0
        assert np.array_equal(fitted.scale, want)
    # Through the D1R producer: forcing a change in the TEST features leaves the stored
    # training transform and coefficients bit-identical (pipeline only, seed 4242).
    def perturbed(x, w1, w2, _calls=[]):
        cap = d1r.capture_stream(x, w1, w2)
        _calls.append(1)
        if len(_calls) == 2:  # the second capture is the held-out stream
            cap["pipeline"] = cap["pipeline"] + 1e6
            cap["network"] = cap["network"] - 1e6
        return cap

    clean = d1r.run_d1r_seed(FIXTURE, decoders=("pipeline",), n_train=SHORT_TRAIN, n_test=SHORT_TEST)
    forced = d1r.run_d1r_seed(FIXTURE, decoders=("pipeline",), n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=perturbed)
    for k in ("mean", "scale", "normalization_sha256", "coefficients", "intercept", "coefficients_sha256", "n_iter"):
        assert clean["decoders"]["pipeline"][k] == forced["decoders"]["pipeline"][k], k


# 11.8
def test_d1r_11_08_F0_weights_constant_no_reward_and_evaluation_nonmutating(monkeypatch):
    calls = []
    orig = diag.Stage2DiagnosticSNN.online_step_with_background

    def spy(self, inp, bg=None, reward_fn=None):
        calls.append((bg, reward_fn))
        return orig(self, inp, bg, reward_fn=reward_fn)

    def no_reward(*_a, **_k):
        raise AssertionError("reward applied under F0")

    monkeypatch.setattr(diag.Stage2DiagnosticSNN, "online_step_with_background", spy)
    from snn import stage2_r3

    monkeypatch.setattr(stage2_r3.CountingRSTDP, "apply_reward", no_reward)
    w1, w2 = d1r.initial_weights(FIXTURE)
    w1c, w2c = w1.copy(), w2.copy()
    x = d1r.a_stream(FIXTURE, False, 500)
    one = d1r.capture_stream(x, w1, w2)
    two = d1r.capture_stream(x, w1, w2)
    assert len(calls) == 1000 and all(bg is None and rf is None for bg, rf in calls)
    assert one["weights_constant"] is True and two["weights_constant"] is True
    assert np.array_equal(w1, w1c) and np.array_equal(w2, w2c)
    # Capture is a pure function of (stream, weights): re-running leaves nothing behind.
    for k in ("network", "pipeline"):
        assert np.array_equal(one[k], two[k])
    row = syn_row(SYN_SEEDS[0])
    row["reward_computed"] = True
    assert any("reward_computed" in v for v in d1r.validate_row(row, syn_identity(SYN_SEEDS[0])))
    row = syn_row(SYN_SEEDS[0])
    row["weights_constant"] = False
    assert any("bitwise constant" in v for v in d1r.validate_row(row, syn_identity(SYN_SEEDS[0])))


# 11.9
def test_d1r_11_09_block_bootstrap_uses_only_component_31_60_q():
    g = np.random.default_rng(7)
    v = (g.random(10_000) < 0.7).astype(np.float64)
    for q in (1, 2):
        seen = []
        orig = d1r.rng
        try:
            d1r.rng = lambda e: (seen.append(tuple(e)), orig(e))[1]  # noqa: E731
            got = d1r.block_lower_bound(v, FIXTURE, q)
        finally:
            d1r.rng = orig
        assert seen == [(20261001, FIXTURE, 31, 60, q)]
        want = d1r.block_lower_bound_from_generator(v, np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 31, 60, q])))
        assert got == want
    # Arithmetic is bit-identical to the approved ebb90e74 interval when given its generator.
    for q in (1, 2):
        approved = diag.circular_block_lower_bound(v, FIXTURE, q)
        mine = d1r.block_lower_bound_from_generator(v, np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 30, 60, q])))
        assert approved == mine
    with pytest.raises(ValueError):
        d1r.block_lower_bound(v, FIXTURE, 3)
    assert d1r.BOOTSTRAP_Q == {"A": 1, "network_positive": 2}
    # The pipeline decoder has no interval and no bootstrap identity.
    assert "lower_95" not in d1r.DECODER_FIELDS and "lower_95" in d1r.NETWORK_DECODER_FIELDS


# 11.10
@pytest.mark.parametrize(
    "kw, status, code, row",
    [
        (dict(), "PASS", "D1R_PASS_D2_D4_ELIGIBLE", 6),
        (dict(a=(0.65, 0.6)), "INCONCLUSIVE", "D1R_A_INCONCLUSIVE_EBB90E74_ROW4", 5),
        (dict(a=(0.52, 0.4)), "FAIL", "D1R_A_FAIL_EBB90E74_ROW3", 4),
        (dict(pos=(0.68835, 0.6)), "REPRESENTATION_LIMITED", "CLOSE_D1_REPRESENTATION_LIMITED", 3),
        (dict(pos=(0.52, 0.4)), "REPRESENTATION_LIMITED", "CLOSE_D1_REPRESENTATION_LIMITED", 3),
        (dict(pipe=0.98), "INVALID_PIPELINE", "STOP_D1R_INVALID_PIPELINE", 2),
    ],
)
def test_d1r_11_10_readouts_and_branch_rows_mechanical_on_synthetic_rows(kw, status, code, row):
    out = summarize(syn_rows(**kw))
    assert out["status"] == status and out["branch"]["code"] == code and out["branch"]["table_row"] == row
    if status == "INVALID_PIPELINE":
        assert out["readouts"]["network_positive"] is None and out["readouts"]["A"] is None


def test_d1r_11_10b_row_1_invalid_and_predicate_boundaries():
    for mutate in (
        lambda r: r[0].update(weights_constant=False),
        lambda r: r[0].update(nesting_ok=False),
        lambda r: r[0]["decoders"]["pipeline"].update(converged=False),
        lambda r: r[0]["decoders"]["A"].update(n_iter=[2000]),
        lambda r: r[0]["decoders"]["network_positive"].update(lower_95=float("nan")),
        lambda r: r[0].update(network_feature_dim=60),
        lambda r: r.pop(),
        lambda r: r.append(copy.deepcopy(r[0])),
        lambda r: r.reverse(),
        lambda r: r[0].update(status="error"),
        lambda r: r[0].update(diagnostic="D1"),
        lambda r: r[0].update(extra=1),
        lambda r: r[0]["decoders"].pop("pipeline"),
        lambda r: r[0].update(train_input_hash=_h("other")),
    ):
        rows = syn_rows()
        mutate(rows)
        out = summarize(rows)
        assert out["status"] == "INVALID" and out["branch"]["code"] == "STOP_D1R_INVALID" and out["branch"]["table_row"] == 1
        assert out["readouts"] == {"pipeline": None, "network_positive": None, "A": None}
    out = summarize(syn_rows(), probe_pass=False)
    assert out["branch"]["code"] == "STOP_D1R_INVALID"
    # Default seed set is exactly 2100..2119: synthetic IDs are refused there.
    assert d1r.summarize(syn_rows(), SYN_EXPECTED, True)["status"] == "INVALID"
    # Boundaries of the unchanged section 4.6 numbers and the 0.99 pipeline bar.
    assert summarize(syn_rows(pipe=0.99))["readouts"]["pipeline"]["status"] == "PASS"
    assert summarize(syn_rows(pipe=0.9899))["status"] == "INVALID_PIPELINE"
    rows = syn_rows(a=(0.70, 0.51))
    for r in rows[:5]:
        r["decoders"]["A"]["lower_95"] = 0.5
    assert summarize(rows)["readouts"]["A"] == {"median_accuracy": 0.70, "lower_bounds_gt_half": 15, "status": "PASS", "decision_bearing": True}
    rows[5]["decoders"]["A"]["lower_95"] = 0.5
    assert summarize(rows)["status"] == "INCONCLUSIVE"
    assert summarize(syn_rows(a=(0.5499, 0.6)))["status"] == "FAIL"
    assert summarize(syn_rows(a=(0.55, 0.6)))["status"] == "INCONCLUSIVE"
    assert set(d1r.BRANCH_INDEX) == set(d1r.STATUS_TO_BRANCH.values()) and sorted(d1r.BRANCH_INDEX.values()) == [1, 2, 3, 4, 5, 6]
    assert d1r.THRESHOLDS == {"pass_median_accuracy": 0.70, "pass_min_lower_bounds_gt_half": 15, "lower_bound_reference": 0.50, "fail_median_below": 0.55, "pipeline_pass_median_accuracy": 0.99}


# 11.11
class _Watch:
    peak_total = peak_single = max_children = 0
    exceeded = False
    limit = d1r.RSS_LIMIT_BYTES

    def start(self):
        pass

    def sample(self):
        pass

    def stop(self):
        pass


def test_d1r_11_11_progress_ledger_is_outcome_free_during_and_after_stage(tmp_path):
    out = tmp_path / "stage"
    out.mkdir()
    snaps = []
    by_seed = {r["seed"]: r for r in syn_rows()}

    def job(seed):
        snaps.append({p.name: p.read_text() for p in out.iterdir()})
        if seed == SYN_SEEDS[9]:
            raise RuntimeError("synthetic crash")
        return copy.deepcopy(by_seed[seed])

    with pytest.raises(RuntimeError, match="synthetic crash"):
        runner.run_pool(job, list(SYN_SEEDS), 1, out, executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch())
    assert sorted(p.name for p in out.iterdir()) == ["D1R.progress.jsonl"]
    lines = [l for s in snaps for t in s.values() for l in t.splitlines() if l]
    lines += (out / "D1R.progress.jsonl").read_text().splitlines()
    assert len(lines) >= 9
    forbidden = ("accuracy", "lower_95", "coefficients", "intercept", "decoders", "n_iter", "median", "converged", "normalization", "bound")
    for line in lines:
        rec = json.loads(line)
        assert set(rec) == set(runner.PROGRESS_FIELDS)
        assert not stats.contains_outcome(rec)
        assert not any(f in line for f in forbidden)
    out2 = tmp_path / "stage2"
    out2.mkdir()
    rows, meta = runner.run_pool(lambda s: copy.deepcopy(by_seed[s]), list(SYN_SEEDS), 1, out2, executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch())
    assert len(rows) == 20 and sorted(p.name for p in out2.iterdir()) == ["D1R.progress.jsonl"]
    assert not stats.contains_outcome(meta) and meta["rss_limit_bytes"] == 12 * 1024**3


# 11.12
def test_d1r_11_12_in_repo_collision_probe_reproduces_section_3_3_exactly():
    path = ROOT / "tools" / "stage2_diagnostic_d1r_ss_probe.py"
    spec = importlib.util.spec_from_file_location("d1r_ss_probe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    res = mod.probe()
    assert res["decision"] == "PASS" and res["reproduces_spec_section_3_3"] is True
    assert res["counts"] == {
        "legacy_records": 704,
        "legacy_unique_entropy_tuples": 698,
        "legacy_unique_states": 698,
        "new_records": 120,
        "new_unique_entropy_tuples": 120,
        "new_unique_states": 120,
    }
    assert res["legacy_only_duplicate_count"] == 6 and len(res["legacy_only_state_duplicate_groups"]) == 6
    for k in ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions"):
        assert res[k] == []
    assert res["seed_ranges_disjoint"] is True
    assert res["identity_state_sha256"] == "78fb3a3a604e5e5617cb5bb224a95e9cbf8ea0b45c77484b476fe05ae14c8706"
    assert np.__version__ == res["numpy_version"] == "1.26.4"
    # The probe enumerates exactly the identities the implementation uses.
    probe_ids = dict(mod.d1r_identities())
    for seed in d1r.D1R_SEEDS:
        for name, tup in d1r.entropy_table(seed).items():
            assert probe_ids[f"d1r/{name}/seed={seed}"] == tup
    assert len(probe_ids) == 6 * len(d1r.D1R_SEEDS)
    # Retained output and manifest.
    retained = json.loads((ROOT / "results_stage2_diagnostic_d1r" / "ss_probe.json").read_text())
    assert retained["identity_state_sha256"] == res["identity_state_sha256"] and retained["decision"] == "PASS"
    assert retained["probe_code_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    verified = runner.verified_probe()
    assert verified["pass"] is True and verified["legacy_only_duplicate_count"] == 6


# 11.13
APPROVED_D1_PATH_SHA256 = {
    "snn/stage2_diagnostic.py": "da7e4faa0f079de5ad1fd0945c2373398f0f05f5b26fee04dfdf16878a45731e",
    "snn/stage2_diagnostic_stats.py": "f4b84468baeb41efa6cf5d996cf6d81a1502f1be7bb1614334c862ea601a16e1",
    "run_stage2_diagnostic.py": "8096fb7a4b389e0ad61d13fd6b50a7833860178ed154dc4a10977d518403e8c1",
    "run_stage2.py": "930c93cb89693577e6cd05a8a026cd4577753807aea426483d6c1dead2956984",
    "snn/stage2_r3.py": "ad5e0fa1c2d85d067c33ef9992cc49755f4c51f413c1fd6dcd9a22ad4a4ec578",
    "snn/stage2.py": "3e48b09c18a38e0566402a44aec41be8d60ad1336937816355103e98e993432d",
    "snn/core.py": "fcc67519f168ef71dfd244475fbdee14cbbcc9d395e13bbf03e50728fc47b63c",
    "tools/stage2_diagnostic_ss_probe.py": "ab3dbac8a08fc7108b52a8963202145d4b8acb97ac6eba3720df5c4c21734f58",
    "tests/test_stage2_diagnostic.py": "bb7a119713e7fec0bc7377c5a94d1e02dab0d7c5572cf467c5aa2f6dd9fa4c6d",
    "tests/test_stage2_diagnostic_package.py": "9a04db593706e9db2d4c654e887f1f2b2be70411deaac65f1e12f40b9103ccc4",
    "tests/test_stage2_diagnostic_round3.py": "eaa3fc07464c703b774a274612f8ecb090fc69c466cff887f4eea190c4ba03cd",
    "results_stage2_diagnostic/ss_probe.json": "1d848fa5a1616e562cb8a48079cc9a412625726d702c323175dfec042827cc9e",
    "results_stage2_diagnostic/SHA256SUMS": "53e4de35616af0cd196a5ba3c52f4fdce129ff29776ad0b90094e0b34bae9fda",
}
FROZEN_SPECS = {
    "docs/STAGE2_SPEC.md": "695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd",
    "docs/STAGE2_DIAGNOSTIC_SPEC.md": "7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf",
    "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md": "c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743",
}


def test_d1r_11_13_approved_D1_path_and_tests_unchanged_and_frozen_spec_pins_green():
    for rel, want in {**APPROVED_D1_PATH_SHA256, **FROZEN_SPECS}.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want, rel
    assert diag.SPEC_SHA256 == FROZEN_SPECS["docs/STAGE2_DIAGNOSTIC_SPEC.md"] == d1r.EBB90E74_SHA256
    assert diag.FROZEN_STAGE2_SHA256 == FROZEN_SPECS["docs/STAGE2_SPEC.md"] == d1r.FROZEN_STAGE2_SHA256
    assert d1r.SPEC_SHA256 == FROZEN_SPECS["docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md"]
    assert diag.DIAGNOSTIC_SEEDS == tuple(range(2000, 2020)) and diag.NAMESPACE == 30 and diag.D1_FEATURE_DIM == 60
    # D1R reuses, never redefines, the approved D1 pieces.
    assert d1r.diag.FeatureAccumulator is diag.FeatureAccumulator and d1r.diag.fit_d1_decoder is diag.fit_d1_decoder


# 11.14
def test_d1r_11_14_maintained_suite_contract_and_d1r_run_locked(tmp_path):
    assert (ROOT / "pytest.ini").read_text().strip().splitlines() == ["[pytest]", "testpaths = tests"]
    assert Path(__file__).parent == ROOT / "tests"
    out = tmp_path / "forbidden"
    proc = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic_d1r.py"), "--run-d1r", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "approval-file is required" in proc.stderr
    bogus = tmp_path / "approval.json"
    bogus.write_text(json.dumps({"decision": "APPROVE", "reviewer": "nora", "implementation_commit": "0" * 40}))
    out2 = tmp_path / "forbidden2"
    proc = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic_d1r.py"), "--run-d1r", "--approval-file", str(bogus), "--out", str(out2)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "approval record mismatch" in proc.stderr
    proc = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic_d1r.py"), "--shakedown", "--workers", "4", "--out", str(tmp_path / "w")], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "workers must be 1..3" in proc.stderr
    for seed in (2100, 2109, 2119):
        with pytest.raises(d1r.UnauthorizedD1RSeed):
            d1r.run_d1r_seed(seed, n_train=10, n_test=10)
    params = runner.validated_effective_parameters()
    assert params["matches_spec"] is True, params["mismatches"]


# 11.15
def test_d1r_11_15_pipeline_reachability_on_fixture_seed_4242_full_lengths_pipeline_only():
    row = d1r.run_d1r_seed(FIXTURE, decoders=("pipeline",))
    assert set(row["decoders"]) == {"pipeline"}  # network decoders neither fitted nor scored
    assert row["train_rows"] == 198_000 and row["test_rows"] == 10_000 and row["warmup_rows"] == 2_000
    assert row["network_feature_dim"] == 460 and row["pipeline_feature_dim"] == 46
    assert row["nesting_ok"] is True and row["weights_constant"] is True and row["reward_computed"] is False
    ident = d1r.object_hashes(FIXTURE)
    assert d1r.validate_row(row, ident, decoders=("pipeline",)) == []
    pipe = row["decoders"]["pipeline"]
    assert pipe["converged"] is True and pipe["scored"] == 10_000
    assert pipe["accuracy"] >= 0.99


# ================================================================ package contract
def _fake_execute(rows_or_exc):
    def ex(fn, jobs):
        if isinstance(rows_or_exc, Exception):
            raise rows_or_exc
        return copy.deepcopy(rows_or_exc), {"stage": "D1R", "wall_seconds": 1.0, "peak_rss_total_bytes": 1, "interruptions": 0, "retries": 0}

    return ex


PROV = {"mode": "test", "workers": 1, "collision_probe": {"pass": True, "decision": "PASS"}}


def _run(tmp_path, rows_or_exc, prov=PROV, name="pkg"):
    out = tmp_path / name
    summary = runner.run_d1r(SimpleNamespace(workers=1), out, prov, execute=_fake_execute(rows_or_exc), seeds=SYN_SEEDS, identity=syn_identity)
    return summary, out


def _assert_manifest(out, summary):
    lines = (out / "SHA256SUMS").read_text().splitlines()
    names = sorted(l.split("  ", 1)[1] for l in lines)
    assert names == sorted(set(summary["artifacts"]) - {"SHA256SUMS"})
    for l in lines:
        digest, name = l.split("  ", 1)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    for k in runner.REQUIRED_PACKAGE_FIELDS:
        assert k in summary


@pytest.mark.parametrize(
    "kw, code",
    [
        (dict(), "D1R_PASS_D2_D4_ELIGIBLE"),
        (dict(a=(0.65, 0.6)), "D1R_A_INCONCLUSIVE_EBB90E74_ROW4"),
        (dict(a=(0.52, 0.4)), "D1R_A_FAIL_EBB90E74_ROW3"),
        (dict(pos=(0.68, 0.6)), "CLOSE_D1_REPRESENTATION_LIMITED"),
        (dict(pipe=0.98), "STOP_D1R_INVALID_PIPELINE"),
    ],
)
def test_d1r_pkg_01_every_branch_freezes_one_package_through_the_finalizer(tmp_path, kw, code):
    summary, out = _run(tmp_path, syn_rows(**kw))
    assert summary["branch"]["code"] == code
    _assert_manifest(out, summary)
    on_disk = json.loads((out / "branch_outcome.json").read_text())
    assert on_disk["code"] == code and on_disk["table_row"] == d1r.BRANCH_INDEX[code]
    if code == "STOP_D1R_INVALID_PIPELINE":
        assert (out / "d1r_rows.network_outcomes_redacted.jsonl").exists() and not (out / "d1r_rows.jsonl").exists()
        for line in (out / "d1r_rows.network_outcomes_redacted.jsonl").read_text().splitlines():
            r = json.loads(line)
            for name in d1r.NETWORK_DECODERS:
                assert not set(runner.NETWORK_OUTCOME_FIELDS) & set(r["decoders"][name])
        assert set(next(iter(summary["per_seed"].values()))["n_iter"]) == {"pipeline"}
        assert "network_positive" not in json.dumps(summary["readouts"]) or summary["readouts"]["network_positive"] is None
    else:
        assert (out / "d1r_rows.jsonl").exists()


def test_d1r_pkg_02_invalid_rows_redacted_and_exception_is_outcome_free_invalid(tmp_path):
    rows = syn_rows()
    rows[2]["weights_constant"] = False
    summary, out = _run(tmp_path, rows)
    assert summary["branch"]["code"] == "STOP_D1R_INVALID"
    assert (out / "d1r_rows.invalid_redacted.jsonl").exists()
    assert not stats.contains_outcome([json.loads(l) for l in (out / "d1r_rows.invalid_redacted.jsonl").read_text().splitlines()])
    _assert_manifest(out, summary)
    summary, out = _run(tmp_path, RuntimeError("pool died"), name="exc")
    assert summary["branch"]["code"] == "STOP_D1R_INVALID" and "exception" in summary["run_metadata"]["incomplete_reason"]
    assert sorted(p.name for p in out.iterdir()) == ["SHA256SUMS", "branch_outcome.json", "provenance.json", "summary.json"]
    assert not stats.contains_outcome(json.loads((out / "summary.json").read_text())["readouts"] or {})
    summary, out = _run(tmp_path, syn_rows(), prov={**PROV, "collision_probe": {"pass": False}}, name="probe")
    assert summary["branch"]["code"] == "STOP_D1R_INVALID"


def test_d1r_pkg_03_fresh_exclusive_directory_and_stray_artifacts(tmp_path):
    summary, out = _run(tmp_path, syn_rows())
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(runner.StaleOutputError):
        _run(tmp_path, syn_rows())
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "d1r_rows.jsonl").write_text("{}\n")
    with pytest.raises(runner.StaleOutputError):
        _run(tmp_path, syn_rows(), name="stale")
    assert sorted(p.name for p in stale.iterdir()) == ["d1r_rows.jsonl"]
    # A file injected mid-run forces STOP_D1R_INVALID and is never manifested.
    target = tmp_path / "inj"

    def ex(fn, jobs):
        (target / "stray_outcome.json").write_text("{}")
        return copy.deepcopy(syn_rows()), {"stage": "D1R", "wall_seconds": 1.0}

    summary = runner.run_d1r(SimpleNamespace(workers=1), target, PROV, execute=ex, seeds=SYN_SEEDS, identity=syn_identity)
    assert summary["branch"]["code"] == "STOP_D1R_INVALID" and summary["unexpected_artifacts"] == ["stray_outcome.json"]
    assert "stray_outcome.json" not in (target / "SHA256SUMS").read_text()


def test_d1r_pkg_04_producer_and_strict_validator_agree_on_synthetic_capture():
    """All three decoders through the real producer, with a synthetic capture (no network)."""
    n_train, n_test, seed = SHORT_TRAIN, SHORT_TEST, 9001  # outside every declared seed range

    def capture(x, w1, w2):
        x = np.asarray(x, dtype=np.int8)
        g = np.random.default_rng([seed, len(x)])
        net = g.random((len(x), 460))
        prev = np.concatenate(([0], x[:-1])).astype(np.float64)
        net[:, 60] = prev  # make all three fits well posed and fast
        net[:, 61] = x
        pipe = g.random((len(x), 46))
        pipe[:, 6] = prev
        return {
            "network": net, "pipeline": pipe, "nesting_ok": True, "weights_constant": True,
            "input_hash": d1r.sha256(x), "network_feature_hash": d1r.sha256(net), "pipeline_feature_hash": d1r.sha256(pipe),
        }

    row = d1r.run_d1r_seed(seed, n_train=n_train, n_test=n_test, capture=capture)
    ident = d1r.object_hashes(seed, n_train=n_train, n_test=n_test)
    assert d1r.validate_row(row, ident, train_rows=n_train - 2000, test_rows=n_test - 2000) == []
    assert set(row["decoders"]) == set(d1r.DECODERS)
    assert "lower_95" in row["decoders"]["A"] and "lower_95" not in row["decoders"]["pipeline"]
    # Full-length package validator refuses short rows.
    assert any("train_rows" in v for v in d1r.validate_row(row, ident))


# ================================================================ section 10 wall stop (round-1 review)
def test_d1r_wall_limit_bounded_return_and_no_queued_job_starts_after_deadline(tmp_path):
    """Nora's reproduction shape: one worker, three queued jobs, wall_limit 0.05 s.

    Jobs last 2.0 s (not 0.20 s) so the bound has a clear margin over host
    timer slack. The stop must fire at the deadline without waiting for a job
    to finish, and no queued job may start afterwards.
    """
    import threading
    import time

    events = []
    lock = threading.Lock()

    def job(seed):
        with lock:
            events.append(("start", seed, time.monotonic()))
        time.sleep(2.0)
        with lock:
            events.append(("end", seed, time.monotonic()))
        return {"seed": seed, "status": "ok"}

    out = tmp_path / "stage"
    out.mkdir()
    t0 = time.monotonic()
    with pytest.raises(runner.WallLimitExceeded):
        runner.run_pool(job, [1, 2, 3], 1, out, executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch(), wall_limit=0.05)
    returned = time.monotonic() - t0
    assert returned < 1.0, returned  # bounded: well before the first 2.0 s job could finish
    time.sleep(2.5)  # job 1 (unkillable thread) ends; jobs 2 and 3 would start here if still queued
    started = [s for kind, s, _ in events if kind == "start"]
    assert started == [1], events
    assert sorted(p.name for p in out.iterdir()) == ["D1R.progress.jsonl"]
    assert (out / "D1R.progress.jsonl").read_text() == ""  # no job finished before the stop


def test_d1r_wall_limit_terminates_running_worker_processes(tmp_path, monkeypatch):
    """Real ProcessPoolExecutor: running workers are killed, not waited for."""
    import time

    import psutil

    pids = []
    orig = runner.terminate_executor

    def capture_then_terminate(pool, *a, **k):
        pids.extend(p.pid for p in (pool._processes or {}).values())
        return orig(pool, *a, **k)

    monkeypatch.setattr(runner, "terminate_executor", capture_then_terminate)
    out = tmp_path / "stage"
    out.mkdir()
    t0 = time.monotonic()
    with pytest.raises(runner.WallLimitExceeded):
        # time.sleep is a picklable job that would run 60 s per job if not killed
        runner.run_pool(time.sleep, [60.0, 60.0, 60.0, 60.0], 2, out, executor_factory=lambda n: ProcessPoolExecutor(max_workers=n), watch=_Watch(), wall_limit=1.5)
    returned = time.monotonic() - t0
    assert returned < 20.0, returned
    assert pids, "worker processes were captured before termination"
    for pid in pids:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE, pid
    assert sorted(p.name for p in out.iterdir()) == ["D1R.progress.jsonl"]
    assert (out / "D1R.progress.jsonl").read_text() == ""


def test_d1r_wall_limit_freezes_outcome_free_invalid_package(tmp_path, monkeypatch):
    """The wall stop routes through run_d1r's single finalizer as outcome-free STOP_D1R_INVALID."""
    import time

    orig = runner.run_pool
    started = []

    def slow(seed):
        started.append(seed)
        time.sleep(2.0)
        return copy.deepcopy(syn_rows()[0])

    def bounded(fn, jobs, workers, out):  # never runs the real _d1r_job
        return orig(slow, jobs, 1, out, executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch(), wall_limit=0.05)

    monkeypatch.setattr(runner, "run_pool", bounded)
    out = tmp_path / "pkg"
    t0 = time.monotonic()
    summary = runner.run_d1r(SimpleNamespace(workers=1), out, PROV, seeds=SYN_SEEDS, identity=syn_identity)
    assert time.monotonic() - t0 < 1.0
    assert summary["branch"]["code"] == "STOP_D1R_INVALID"
    assert summary["run_metadata"]["incomplete_reason"] == "exception during sequence: WallLimitExceeded"
    assert summary["readouts"] == {"pipeline": None, "network_positive": None, "A": None} and summary["per_seed"] == {}
    _assert_manifest(out, summary)
    assert sorted(p.name for p in out.iterdir()) == ["D1R.progress.jsonl", "SHA256SUMS", "branch_outcome.json", "provenance.json", "summary.json"]
    assert (out / "D1R.progress.jsonl").read_text() == ""
    assert not stats.contains_outcome(json.loads((out / "summary.json").read_text())["readouts"] or {})
    time.sleep(2.5)
    assert started == [SYN_SEEDS[0]]
