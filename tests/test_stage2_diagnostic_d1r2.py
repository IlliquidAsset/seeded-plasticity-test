"""D1R2 section 11 construction tests (docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md at f3d52ee).

One named test per section 11 item: carried items 1-15 (``test_d1r2_11_NN_*``),
the D1R2 additions 11a-11f (``test_d1r2_11a_*`` .. ``test_d1r2_11f_*``), and
item 16 (``test_d1r2_11_16_*``), plus fail-closed package tests carried from
the approved runner contract.

No D1R2 seed (2200..2219) is simulated, streamed, fitted, or scored here.
Network simulation runs only on the non-diagnostic fixture seed 4242: the
pipeline decoder is fitted and scored (item 15) and, for item 16 only, the
two network decoders are fitted for convergence and never predicted. Branch,
validator, and package tests use synthetic rows under the synthetic seed IDs
1000..1019 with synthetic identities. The 11b/11e/11f producer tests stream
those IDs (or 9001) through an injected synthetic capture and synthetic
decoder models: no network transition and no LogisticRegression back them.
The 11d preflight finalizer writes the 20 D1R2 seed IDs as rows without
streaming, simulating, or fitting anything.
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

import run_stage2_diagnostic_d1r2 as runner
from snn import stage2_diagnostic as diag
from snn import stage2_diagnostic_d1r2 as d1r2
from snn import stage2_diagnostic_stats as stats
from snn.stage2_r3 import ONEHOT

ROOT = Path(runner.__file__).resolve().parent
FIXTURE = 4242
SYN_SEEDS = tuple(range(1000, 1020))  # synthetic row IDs only, never simulated
SHORT_TRAIN, SHORT_TEST = 4_000, 3_000
SHORT_TRAIN_ROWS, SHORT_TEST_ROWS = SHORT_TRAIN - d1r2.WARMUP, SHORT_TEST - d1r2.WARMUP


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
    width = d1r2.DECODER_WIDTH[name]
    g = np.random.default_rng([seed, len(name)])
    mean = g.normal(size=width).tolist()
    scale = (1.0 + g.random(width)).tolist()
    coef = [g.normal(size=width).tolist()]
    icpt = [float(g.normal())]
    ident = syn_identity(seed)
    rec = {
        "accuracy": float(round(accuracy * d1r2.TEST_ROWS) / d1r2.TEST_ROWS),
        "scored": d1r2.TEST_ROWS,
        "decoder_config": dict(d1r2.DECODER_CONFIG),
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
    if name in d1r2.NETWORK_DECODERS:
        rec["lower_95"] = float(lower)
    return rec


def syn_row(seed: int, pipe=1.0, pos=(0.75, 0.6), a=(0.75, 0.6)) -> dict:
    ident = syn_identity(seed)
    return {
        "diagnostic": "D1R2",
        "seed": seed,
        "row_source": "worker",
        "seed_error": None,
        "ledger_torn_tail": False,
        "non_decoder_warnings": [],
        "decoder_slots": {
            name: {"attempted": True, "completed": True, "stop_reason": "converged", "n_iter": 123,
                   "converged": True, "convergence_warning": False, "warnings": [], "fit_error": None}
            for name in d1r2.DECODERS
        },
        "train_input_hash": ident["train_input_hash"],
        "test_input_hash": ident["test_input_hash"],
        "train_network_feature_hash": _h(f"ftrn-{seed}"),
        "test_network_feature_hash": _h(f"ften-{seed}"),
        "train_pipeline_feature_hash": _h(f"ftrp-{seed}"),
        "test_pipeline_feature_hash": _h(f"ftep-{seed}"),
        "initial_weights_sha256": ident["initial_weights_sha256"],
        "train_rows": d1r2.TRAIN_ROWS,
        "test_rows": d1r2.TEST_ROWS,
        "warmup_rows": d1r2.WARMUP,
        "network_feature_dim": d1r2.NETWORK_FEATURE_DIM,
        "pipeline_feature_dim": d1r2.PIPELINE_FEATURE_DIM,
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
    return d1r2.summarize(rows, expected, probe_pass, SYN_SEEDS)


# ---------------------------------------------------------------- fixtures
def _fixture_capture(n: int, flip_at=None):
    w1, w2 = d1r2.initial_weights(FIXTURE)
    x = d1r2.a_stream(FIXTURE, True, n).copy()
    if flip_at is not None:
        x[flip_at] ^= 1
    return x, d1r2.capture_stream(x, w1, w2)


@pytest.fixture(scope="module")
def short_pipeline_row():
    return d1r2.run_d1r2_seed(FIXTURE, decoders=("pipeline",), n_train=SHORT_TRAIN, n_test=SHORT_TEST)


# ================================================================ section 11
# 11.1
def test_d1r2_11_01_namespace_32_identities_r3_draw_order_and_seeds_2200_2219():
    assert d1r2.D1R2_SEEDS == tuple(range(2200, 2220)) and d1r2.NAMESPACE == 32
    for seed in (FIXTURE, 2200, 2219):  # tuples only for D1R2 seeds; no generator is drawn
        assert d1r2.entropy_table(seed) == {
            "weights_w1_w2_o1": (20261001, seed, 32, 1),
            "weights_w2_o0": (20261001, seed, 32, 21),
            "A_train": (20261001, seed, 32, 2),
            "A_test": (20261001, seed, 32, 5),
            "D1R2_block_bootstrap_A": (20261001, seed, 32, 60, 1),
            "D1R2_block_bootstrap_network_positive_control": (20261001, seed, 32, 60, 2),
        }
    w1, w2 = d1r2.initial_weights(FIXTURE)
    g = np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 32, 1]))
    assert np.array_equal(w1, g.uniform(-10.0, 10.0, size=(20, 2)))
    assert np.array_equal(w2[0:1], g.uniform(0.0, 10.0, size=(1, 20)))
    h = np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 32, 21]))
    assert np.array_equal(w2[1:2], h.uniform(0.0, 10.0, size=(1, 20)))
    assert w1.dtype == w2.dtype == np.float64 and w1.shape == (20, 2) and w2.shape == (2, 20)
    # Same r3 construction as ebb90e74, different namespace.
    assert not np.array_equal(w1, diag.initial_weights(FIXTURE)[0])
    for train, comp in ((True, 2), (False, 5)):
        want = diag._roll_a(np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 32, comp])), 500)
        assert np.array_equal(d1r2.a_stream(FIXTURE, train, 500), want)
    assert d1r2.sha256(d1r2.a_stream(FIXTURE, True, 500)) != d1r2.sha256(d1r2.a_stream(FIXTURE, False, 500))
    # Every generator constructed by a D1R2 object goes through d1r2.rng with a section 3.1 tuple.
    seen = []
    orig = d1r2.rng
    d1r2_rng = lambda e: (seen.append(tuple(e)), orig(e))[1]  # noqa: E731
    try:
        d1r2.rng = d1r2_rng
        d1r2.initial_weights(FIXTURE)
        d1r2.a_stream(FIXTURE, True, 10)
        d1r2.a_stream(FIXTURE, False, 10)
        d1r2.block_lower_bound(np.ones(200), FIXTURE, 1)
        d1r2.block_lower_bound(np.ones(200), FIXTURE, 2)
    finally:
        d1r2.rng = orig
    assert seen == list(d1r2.entropy_table(FIXTURE).values())


# 11.2
def test_d1r2_11_02_phiR_columns_0_59_bit_identical_to_approved_FeatureAccumulator():
    g = np.random.default_rng(20261001)
    spikes = (g.random((400, 20)) < 0.3).astype(np.float64)
    approved = diag.FeatureAccumulator()
    lag = d1r2.LagFeatureAccumulator(20)
    for s in spikes:
        row, _ = lag.update(s)
        assert np.array_equal(row[:60], approved.update(s))
    # On the real frozen network: same weights and stream as the approved capture.
    n = 600
    approved_cap = diag.capture_d1_features(FIXTURE, train=True, n=n)
    w1, w2 = diag.initial_weights(FIXTURE)
    mine = d1r2.capture_stream(diag.a_stream(FIXTURE, True, n), w1, w2)
    assert mine["network"].shape == (n, 460)
    assert np.array_equal(mine["network"][:, :60], approved_cap["features"])
    assert mine["network"][:, :60].tobytes() == approved_cap["features"].tobytes()


# 11.3
def test_d1r2_11_03_hand_computed_lag_block_order_zero_fill_and_count20_nesting():
    n = 3
    hidden = np.zeros((45, n), dtype=np.float64)
    hidden[[0, 4, 19, 20, 39], 0] = 1
    hidden[[1, 2, 21, 40, 44], 1] = 1
    hidden[[10, 30, 31, 32], 2] = 1
    acc = d1r2.LagFeatureAccumulator(n)
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
    acc2 = d1r2.LagFeatureAccumulator(2)
    acc2.update(np.array([1.0, 0.0]))
    row, _ = acc2.update(np.array([0.0, 1.0]))
    assert row[6:12].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert d1r2.NETWORK_FEATURE_DIM == 460 and d1r2.PIPELINE_FEATURE_DIM == 46 and d1r2.K == 19
    # A nesting violation is detected (section 7 item 7).
    bad = d1r2.LagFeatureAccumulator(2)
    bad.update(np.array([1.0, 1.0]))
    bad.ring[:] = 0.0
    assert bad.update(np.array([0.0, 0.0]))[1] is False


# 11.4
def test_d1r2_11_04_rows_cannot_depend_on_current_x_t_and_change_enters_at_t_plus_1():
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

    assert list(inspect.signature(d1r2.LagFeatureAccumulator.update).parameters) == ["self", "spikes_t"]


# 11.5
def test_d1r2_11_05_u_t_is_onehot_previous_input_and_psi_uses_identical_path_width_2():
    assert np.array_equal(d1r2.input_layer_vector(None), np.zeros(2))
    assert np.array_equal(d1r2.input_layer_vector(0), [1.0, 0.0])
    assert np.array_equal(d1r2.input_layer_vector(1), [0.0, 1.0])
    n = 300
    x, cap = _fixture_capture(n)
    psi = cap["pipeline"]
    assert psi.shape == (n, 46)
    assert np.array_equal(psi[0], np.zeros(46))
    for t in range(1, n):
        assert np.array_equal(psi[t, 6:8], ONEHOT[int(x[t - 1])].numpy()[0])
    # Identical operator: an independent LagFeatureAccumulator(2) over u_t reproduces psi exactly,
    # and it is the same class (and approved base accumulator) as the width-20 network map.
    acc = d1r2.LagFeatureAccumulator(2)
    u = [np.zeros(2)] + [ONEHOT[int(v)].numpy()[0] for v in x[:-1]]
    assert np.array_equal(np.stack([acc.update(v)[0] for v in u]), psi)
    assert type(acc.base) is diag.FeatureAccumulator and acc.base.n_hidden == 2


# 11.6
def test_d1r2_11_06_label_hashes_coordinator_derived_x1999_and_x2000(monkeypatch, short_pipeline_row):
    xtr = d1r2.a_stream(FIXTURE, True, SHORT_TRAIN)
    xte = d1r2.a_stream(FIXTURE, False, SHORT_TEST)
    # Coordinator derivation never consults the worker's label code.
    monkeypatch.setattr(d1r2, "worker_labels", lambda *_: (_ for _ in ()).throw(AssertionError("worker path used")))
    monkeypatch.setattr(diag, "d1_labels", lambda *_: (_ for _ in ()).throw(AssertionError("worker path used")))
    ident = d1r2.object_hashes(FIXTURE, n_train=SHORT_TRAIN, n_test=SHORT_TEST)
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
    assert not d1r2.validate_row(short_pipeline_row, ident, train_rows=SHORT_TRAIN_ROWS, test_rows=SHORT_TEST_ROWS, decoders=("pipeline",))
    # A worker that used the current input as the "prior" label is refused.
    bad = copy.deepcopy(short_pipeline_row)
    bad["decoders"]["pipeline"]["train_label_sha256"] = ident["A_train_label_sha256"]
    assert any("label identity" in v for v in d1r2.validate_row(bad, ident, train_rows=SHORT_TRAIN_ROWS, test_rows=SHORT_TEST_ROWS, decoders=("pipeline",)))
    # On full synthetic rows: each of the 6 label slots, swapped A/prior, and missing identity all force INVALID.
    for name in d1r2.DECODERS:
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
def test_d1r2_11_07_training_only_standardization_test_changes_cannot_alter_train_state():
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
    # Through the D1R2 producer: forcing a change in the TEST features leaves the stored
    # training transform and coefficients bit-identical (pipeline only, seed 4242).
    def perturbed(x, w1, w2, _calls=[]):
        cap = d1r2.capture_stream(x, w1, w2)
        _calls.append(1)
        if len(_calls) == 2:  # the second capture is the held-out stream
            cap["pipeline"] = cap["pipeline"] + 1e6
            cap["network"] = cap["network"] - 1e6
        return cap

    clean = d1r2.run_d1r2_seed(FIXTURE, decoders=("pipeline",), n_train=SHORT_TRAIN, n_test=SHORT_TEST)
    forced = d1r2.run_d1r2_seed(FIXTURE, decoders=("pipeline",), n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=perturbed)
    for k in ("mean", "scale", "normalization_sha256", "coefficients", "intercept", "coefficients_sha256", "n_iter"):
        assert clean["decoders"]["pipeline"][k] == forced["decoders"]["pipeline"][k], k


# 11.8
def test_d1r2_11_08_F0_weights_constant_no_reward_and_evaluation_nonmutating(monkeypatch):
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
    w1, w2 = d1r2.initial_weights(FIXTURE)
    w1c, w2c = w1.copy(), w2.copy()
    x = d1r2.a_stream(FIXTURE, False, 500)
    one = d1r2.capture_stream(x, w1, w2)
    two = d1r2.capture_stream(x, w1, w2)
    assert len(calls) == 1000 and all(bg is None and rf is None for bg, rf in calls)
    assert one["weights_constant"] is True and two["weights_constant"] is True
    assert np.array_equal(w1, w1c) and np.array_equal(w2, w2c)
    # Capture is a pure function of (stream, weights): re-running leaves nothing behind.
    for k in ("network", "pipeline"):
        assert np.array_equal(one[k], two[k])
    row = syn_row(SYN_SEEDS[0])
    row["reward_computed"] = True
    assert any("reward_computed" in v for v in d1r2.validate_row(row, syn_identity(SYN_SEEDS[0])))
    row = syn_row(SYN_SEEDS[0])
    row["weights_constant"] = False
    assert any("bitwise constant" in v for v in d1r2.validate_row(row, syn_identity(SYN_SEEDS[0])))


# 11.9
def test_d1r2_11_09_block_bootstrap_uses_only_component_32_60_q():
    g = np.random.default_rng(7)
    v = (g.random(10_000) < 0.7).astype(np.float64)
    for q in (1, 2):
        seen = []
        orig = d1r2.rng
        try:
            d1r2.rng = lambda e: (seen.append(tuple(e)), orig(e))[1]  # noqa: E731
            got = d1r2.block_lower_bound(v, FIXTURE, q)
        finally:
            d1r2.rng = orig
        assert seen == [(20261001, FIXTURE, 32, 60, q)]
        want = d1r2.block_lower_bound_from_generator(v, np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 32, 60, q])))
        assert got == want
    # Arithmetic is bit-identical to the approved ebb90e74 interval when given its generator.
    for q in (1, 2):
        approved = diag.circular_block_lower_bound(v, FIXTURE, q)
        mine = d1r2.block_lower_bound_from_generator(v, np.random.default_rng(np.random.SeedSequence([20261001, FIXTURE, 30, 60, q])))
        assert approved == mine
    with pytest.raises(ValueError):
        d1r2.block_lower_bound(v, FIXTURE, 3)
    assert d1r2.BOOTSTRAP_Q == {"A": 1, "network_positive": 2}
    # The pipeline decoder has no interval and no bootstrap identity.
    assert "lower_95" not in d1r2.DECODER_FIELDS and "lower_95" in d1r2.NETWORK_DECODER_FIELDS


# 11.10
@pytest.mark.parametrize(
    "kw, status, code, row",
    [
        (dict(), "PASS", "D1R2_PASS_D2_D4_ELIGIBLE", 6),
        (dict(a=(0.65, 0.6)), "INCONCLUSIVE", "D1R2_A_INCONCLUSIVE_EBB90E74_ROW4", 5),
        (dict(a=(0.52, 0.4)), "FAIL", "D1R2_A_FAIL_EBB90E74_ROW3", 4),
        (dict(pos=(0.68835, 0.6)), "REPRESENTATION_LIMITED", "CLOSE_D1_REPRESENTATION_LIMITED", 3),
        (dict(pos=(0.52, 0.4)), "REPRESENTATION_LIMITED", "CLOSE_D1_REPRESENTATION_LIMITED", 3),
        (dict(pipe=0.98), "INVALID_PIPELINE", "STOP_D1R2_INVALID_PIPELINE", 2),
    ],
)
def test_d1r2_11_10_readouts_and_branch_rows_mechanical_on_synthetic_rows(kw, status, code, row):
    out = summarize(syn_rows(**kw))
    assert out["status"] == status and out["branch"]["code"] == code and out["branch"]["table_row"] == row
    if status == "INVALID_PIPELINE":
        assert out["readouts"]["network_positive"] is None and out["readouts"]["A"] is None


def test_d1r2_11_10b_row_1_invalid_and_predicate_boundaries():
    for mutate in (
        lambda r: r[0].update(weights_constant=False),
        lambda r: r[0].update(nesting_ok=False),
        lambda r: r[0]["decoders"]["pipeline"].update(converged=False),
        lambda r: r[0]["decoders"]["A"].update(n_iter=[20000]),
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
        assert out["status"] == "INVALID" and out["branch"]["code"] == "STOP_D1R2_INVALID" and out["branch"]["table_row"] == 1
        assert out["readouts"] == {"pipeline": None, "network_positive": None, "A": None}
    out = summarize(syn_rows(), probe_pass=False)
    assert out["branch"]["code"] == "STOP_D1R2_INVALID"
    # Default seed set is exactly 2200..2219: synthetic IDs are refused there.
    assert d1r2.summarize(syn_rows(), SYN_EXPECTED, True)["status"] == "INVALID"
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
    assert set(d1r2.BRANCH_INDEX) == set(d1r2.STATUS_TO_BRANCH.values()) and sorted(d1r2.BRANCH_INDEX.values()) == [1, 2, 3, 4, 5, 6]
    assert d1r2.THRESHOLDS == {"pass_median_accuracy": 0.70, "pass_min_lower_bounds_gt_half": 15, "lower_bound_reference": 0.50, "fail_median_below": 0.55, "pipeline_pass_median_accuracy": 0.99}


# 11.11
class _Watch:
    peak_total = peak_single = max_children = 0
    exceeded = False
    limit = d1r2.RSS_LIMIT_BYTES

    def start(self):
        pass

    def sample(self):
        pass

    def stop(self):
        pass


def test_d1r2_11_11_progress_ledger_is_outcome_free_during_and_after_stage(tmp_path):
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
    assert sorted(p.name for p in out.iterdir()) == ["D1R2.progress.jsonl"]
    lines = [l for s in snaps for t in s.values() for l in t.splitlines() if l]
    lines += (out / "D1R2.progress.jsonl").read_text().splitlines()
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
    assert len(rows) == 20 and sorted(p.name for p in out2.iterdir()) == ["D1R2.progress.jsonl"]
    assert not stats.contains_outcome(meta) and meta["rss_limit_bytes"] == 12 * 1024**3


# 11.12
def test_d1r2_11_12_in_repo_collision_probe_reproduces_section_3_3_exactly():
    path = ROOT / "tools" / "stage2_diagnostic_d1r2_ss_probe.py"
    spec = importlib.util.spec_from_file_location("d1r2_ss_probe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    res = mod.probe()
    assert res["decision"] == "PASS" and res["reproduces_spec_section_3_3"] is True
    assert res["counts"] == {
        "legacy_records": 824,
        "legacy_unique_entropy_tuples": 818,
        "legacy_unique_states": 818,
        "new_records": 120,
        "new_unique_entropy_tuples": 120,
        "new_unique_states": 120,
    }
    assert res["legacy_only_duplicate_count"] == 6 and len(res["legacy_only_state_duplicate_groups"]) == 6
    for k in ("new_entropy_collision_groups", "new_state_collision_groups", "new_vs_legacy_entropy_collisions", "new_vs_legacy_state_collisions"):
        assert res[k] == []
    assert res["seed_ranges_disjoint"] is True
    assert res["identity_state_sha256"] == "4d8dd8b477aa2615bc14f66ed11223d0995c10df0985919577bacc9648b6848e"  # spec 3.3
    assert np.__version__ == res["numpy_version"] == "1.26.4"
    # The probe enumerates exactly the identities the implementation uses.
    probe_ids = dict(mod.d1r2_identities())
    for seed in d1r2.D1R2_SEEDS:
        for name, tup in d1r2.entropy_table(seed).items():
            assert probe_ids[f"d1r2/{name}/seed={seed}"] == tup
    assert len(probe_ids) == 6 * len(d1r2.D1R2_SEEDS)
    # Retained output and manifest.
    retained = json.loads((ROOT / "results_stage2_diagnostic_d1r2" / "ss_probe.json").read_text())
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
# Section 11 item 13 (D1R2): the approved D1R implementation and its tests also remain unchanged.
APPROVED_D1R_PATH_SHA256 = {
    "snn/stage2_diagnostic_d1r.py": "52ff793ac84a6816ffcf7b355327a060572ae789bf247b2d314e5c2dc7b17dde",
    "run_stage2_diagnostic_d1r.py": "e339a1b7365f3f3f2b7b42175263b2cec11df24f18a5d6fcfeb5985a1a4e7c19",
    "tools/stage2_diagnostic_d1r_ss_probe.py": "67185d8cc33100157b93f4061ccec92bfe5f80e192c9b221d573c323f5e97b8f",
    "tests/test_stage2_diagnostic_d1r.py": "863763450cc121756001affaa2f3da083438a3dd66811c63dea5b25d585e95c0",
    "results_stage2_diagnostic_d1r/ss_probe.json": "5a4f7c3dc03b624d984add2c789f38a21ece7884cbc5604179caa9793411cf0c",
    "results_stage2_diagnostic_d1r/SHA256SUMS": "fdf1e196ec5d1a8c4ff774ca805075701859889c73ac11146438a89c5c9aa7ce",
    "docs/STAGE2_DIAGNOSTIC_D1R_IMPLEMENTATION_NOTES.md": "620ad14ab908c313dd561ee8ab1d0a90d6291c7ca5f4e6afdcd46a21cf471b7a",
}
FROZEN_SPECS = {
    "docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md": "c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743",
    "docs/STAGE2_SPEC.md": "695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd",
    "docs/STAGE2_DIAGNOSTIC_SPEC.md": "7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf",
    "docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md": "e5a6a5f4eb7a7adafd9255ac6504593e45be51eefbfa92ce24c63b70fafb82a2",
}


def test_d1r2_11_13_approved_D1_path_and_tests_unchanged_and_frozen_spec_pins_green():
    for rel, want in {**APPROVED_D1_PATH_SHA256, **APPROVED_D1R_PATH_SHA256, **FROZEN_SPECS}.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == want, rel
    assert diag.SPEC_SHA256 == FROZEN_SPECS["docs/STAGE2_DIAGNOSTIC_SPEC.md"] == d1r2.EBB90E74_SHA256
    assert diag.FROZEN_STAGE2_SHA256 == FROZEN_SPECS["docs/STAGE2_SPEC.md"] == d1r2.FROZEN_STAGE2_SHA256
    assert d1r2.SPEC_SHA256 == FROZEN_SPECS["docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md"]
    assert d1r2.CARRIED_D1R_SHA256 == FROZEN_SPECS["docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md"]
    from snn import stage2_diagnostic_d1r as d1r

    assert d1r.DECODER_CONFIG["max_iter"] == 2000 and d1r.NAMESPACE == 31  # approved D1R untouched
    assert diag.DIAGNOSTIC_SEEDS == tuple(range(2000, 2020)) and diag.NAMESPACE == 30 and diag.D1_FEATURE_DIM == 60
    # D1R2 reuses, never redefines, the approved D1 pieces.
    assert d1r2.diag.FeatureAccumulator is diag.FeatureAccumulator and d1r2.diag.fit_d1_decoder is diag.fit_d1_decoder


# 11.14
def test_d1r2_11_14_maintained_suite_contract_and_d1r2_run_locked(tmp_path):
    assert (ROOT / "pytest.ini").read_text().strip().splitlines() == ["[pytest]", "testpaths = tests"]
    assert Path(__file__).parent == ROOT / "tests"
    out = tmp_path / "forbidden"
    proc = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic_d1r2.py"), "--run-d1r2", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "approval-file is required" in proc.stderr
    bogus = tmp_path / "approval.json"
    bogus.write_text(json.dumps({"decision": "APPROVE", "reviewer": "nora", "implementation_commit": "0" * 40}))
    out2 = tmp_path / "forbidden2"
    proc = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic_d1r2.py"), "--run-d1r2", "--approval-file", str(bogus), "--out", str(out2)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "approval record mismatch" in proc.stderr
    proc = subprocess.run([sys.executable, str(ROOT / "run_stage2_diagnostic_d1r2.py"), "--shakedown", "--workers", "4", "--out", str(tmp_path / "w")], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "workers must be 1..3" in proc.stderr
    for seed in (2200, 2209, 2219):
        with pytest.raises(d1r2.UnauthorizedD1R2Seed):
            d1r2.run_d1r2_seed(seed, n_train=10, n_test=10)
    params = runner.validated_effective_parameters()
    assert params["matches_spec"] is True, params["mismatches"]


# 11.15
def test_d1r2_11_15_pipeline_reachability_on_fixture_seed_4242_full_lengths_pipeline_only():
    row = d1r2.run_d1r2_seed(FIXTURE, decoders=("pipeline",))
    assert set(row["decoders"]) == {"pipeline"}  # network decoders neither fitted nor scored
    assert row["train_rows"] == 198_000 and row["test_rows"] == 10_000 and row["warmup_rows"] == 2_000
    assert row["network_feature_dim"] == 460 and row["pipeline_feature_dim"] == 46
    assert row["nesting_ok"] is True and row["weights_constant"] is True and row["reward_computed"] is False
    ident = d1r2.object_hashes(FIXTURE)
    assert d1r2.validate_row(row, ident, decoders=("pipeline",)) == []
    pipe = row["decoders"]["pipeline"]
    assert pipe["converged"] is True and pipe["scored"] == 10_000
    assert pipe["accuracy"] >= 0.99


# ================================================================ package contract
def _fake_execute(rows_or_exc):
    def ex(fn, jobs):
        if isinstance(rows_or_exc, Exception):
            raise rows_or_exc
        return copy.deepcopy(rows_or_exc), {"stage": "D1R2", "wall_seconds": 1.0, "peak_rss_total_bytes": 1, "interruptions": 0, "retries": 0}

    return ex


PROV = {"mode": "test", "workers": 1, "collision_probe": {"pass": True, "decision": "PASS"}}


def _run(tmp_path, rows_or_exc, prov=PROV, name="pkg"):
    out = tmp_path / name
    summary = runner.run_d1r2(SimpleNamespace(workers=1), out, prov, execute=_fake_execute(rows_or_exc), seeds=SYN_SEEDS, identity=syn_identity)
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
        (dict(), "D1R2_PASS_D2_D4_ELIGIBLE"),
        (dict(a=(0.65, 0.6)), "D1R2_A_INCONCLUSIVE_EBB90E74_ROW4"),
        (dict(a=(0.52, 0.4)), "D1R2_A_FAIL_EBB90E74_ROW3"),
        (dict(pos=(0.68, 0.6)), "CLOSE_D1_REPRESENTATION_LIMITED"),
        (dict(pipe=0.98), "STOP_D1R2_INVALID_PIPELINE"),
    ],
)
def test_d1r2_pkg_01_every_branch_freezes_one_package_through_the_finalizer(tmp_path, kw, code):
    summary, out = _run(tmp_path, syn_rows(**kw))
    assert summary["branch"]["code"] == code
    _assert_manifest(out, summary)
    on_disk = json.loads((out / "branch_outcome.json").read_text())
    assert on_disk["code"] == code and on_disk["table_row"] == d1r2.BRANCH_INDEX[code]
    if code == "STOP_D1R2_INVALID_PIPELINE":
        assert (out / "d1r2_rows.network_outcomes_redacted.jsonl").exists() and not (out / "d1r2_rows.jsonl").exists()
        for line in (out / "d1r2_rows.network_outcomes_redacted.jsonl").read_text().splitlines():
            r = json.loads(line)
            for name in d1r2.NETWORK_DECODERS:
                assert not set(runner.NETWORK_OUTCOME_FIELDS) & set(r["decoders"][name])
        assert set(next(iter(summary["per_seed"].values()))["n_iter"]) == {"pipeline"}
        assert "network_positive" not in json.dumps(summary["readouts"]) or summary["readouts"]["network_positive"] is None
    else:
        assert (out / "d1r2_rows.jsonl").exists()


def test_d1r2_pkg_02_invalid_rows_redacted_and_exception_is_outcome_free_invalid(tmp_path):
    rows = syn_rows()
    rows[2]["weights_constant"] = False
    summary, out = _run(tmp_path, rows)
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID"
    assert (out / "d1r2_rows.invalid_redacted.jsonl").exists()
    assert not stats.contains_outcome([json.loads(l) for l in (out / "d1r2_rows.invalid_redacted.jsonl").read_text().splitlines()])
    _assert_manifest(out, summary)
    summary, out = _run(tmp_path, RuntimeError("pool died"), name="exc")
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID" and "exception" in summary["run_metadata"]["incomplete_reason"]
    # Section 6.3 totality: even a pool exception yields 20 rows with three slots each.
    assert sorted(p.name for p in out.iterdir()) == ["SHA256SUMS", "branch_outcome.json", "d1r2_rows.invalid_redacted.jsonl", "provenance.json", "summary.json"]
    exc_rows = [json.loads(l) for l in (out / "d1r2_rows.invalid_redacted.jsonl").read_text().splitlines()]
    assert len(exc_rows) == 20 and all(_disk_schema_ok(r) for r in exc_rows)
    assert {s["stop_reason"] for r in exc_rows for s in r["decoder_slots"].values()} == {"worker_lost"}
    assert not stats.contains_outcome(json.loads((out / "summary.json").read_text())["readouts"] or {})
    summary, out = _run(tmp_path, syn_rows(), prov={**PROV, "collision_probe": {"pass": False}}, name="probe")
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID"


def test_d1r2_pkg_03_fresh_exclusive_directory_and_stray_artifacts(tmp_path):
    summary, out = _run(tmp_path, syn_rows())
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(runner.StaleOutputError):
        _run(tmp_path, syn_rows())
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "d1r2_rows.jsonl").write_text("{}\n")
    with pytest.raises(runner.StaleOutputError):
        _run(tmp_path, syn_rows(), name="stale")
    assert sorted(p.name for p in stale.iterdir()) == ["d1r2_rows.jsonl"]
    # A file injected mid-run forces STOP_D1R2_INVALID and is never manifested.
    target = tmp_path / "inj"

    def ex(fn, jobs):
        (target / "stray_outcome.json").write_text("{}")
        return copy.deepcopy(syn_rows()), {"stage": "D1R2", "wall_seconds": 1.0}

    summary = runner.run_d1r2(SimpleNamespace(workers=1), target, PROV, execute=ex, seeds=SYN_SEEDS, identity=syn_identity)
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID" and summary["unexpected_artifacts"] == ["stray_outcome.json"]
    assert "stray_outcome.json" not in (target / "SHA256SUMS").read_text()


def test_d1r2_pkg_04_producer_and_strict_validator_agree_on_synthetic_capture():
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
            "input_hash": d1r2.sha256(x), "network_feature_hash": d1r2.sha256(net), "pipeline_feature_hash": d1r2.sha256(pipe),
        }

    row = d1r2.run_d1r2_seed(seed, n_train=n_train, n_test=n_test, capture=capture)
    ident = d1r2.object_hashes(seed, n_train=n_train, n_test=n_test)
    assert d1r2.validate_row(row, ident, train_rows=n_train - 2000, test_rows=n_test - 2000) == []
    assert set(row["decoders"]) == set(d1r2.DECODERS)
    assert "lower_95" in row["decoders"]["A"] and "lower_95" not in row["decoders"]["pipeline"]
    # Full-length package validator refuses short rows.
    assert any("train_rows" in v for v in d1r2.validate_row(row, ident))


# ================================================================ section 10 wall stop (round-1 review)
def test_d1r2_wall_limit_bounded_return_and_no_queued_job_starts_after_deadline(tmp_path):
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
    assert sorted(p.name for p in out.iterdir()) == ["D1R2.progress.jsonl"]
    assert (out / "D1R2.progress.jsonl").read_text() == ""  # no job finished before the stop


def test_d1r2_wall_limit_terminates_running_worker_processes(tmp_path, monkeypatch):
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
    assert sorted(p.name for p in out.iterdir()) == ["D1R2.progress.jsonl"]
    assert (out / "D1R2.progress.jsonl").read_text() == ""


def test_d1r2_wall_limit_freezes_outcome_free_invalid_package(tmp_path, monkeypatch):
    """The wall stop routes through run_d1r2's single finalizer as outcome-free STOP_D1R2_INVALID."""
    import time

    orig = runner.run_pool
    started = []

    def slow(job):
        started.append(job[0] if isinstance(job, tuple) else job)  # (seed, event-ledger path)
        time.sleep(2.0)
        return copy.deepcopy(syn_rows()[0])

    def bounded(fn, jobs, workers, out):  # never runs the real _d1r2_job
        return orig(slow, jobs, 1, out, executor_factory=lambda n: ThreadPoolExecutor(max_workers=1), watch=_Watch(), wall_limit=0.05)

    monkeypatch.setattr(runner, "run_pool", bounded)
    out = tmp_path / "pkg"
    t0 = time.monotonic()
    summary = runner.run_d1r2(SimpleNamespace(workers=1), out, PROV, seeds=SYN_SEEDS, identity=syn_identity)
    assert time.monotonic() - t0 < 1.0
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID"
    assert summary["run_metadata"]["incomplete_reason"] == "exception during sequence: WallLimitExceeded (terminated_wall_stop)"
    assert summary["readouts"] == {"pipeline": None, "network_positive": None, "A": None} and summary["per_seed"] == {}
    _assert_manifest(out, summary)
    assert sorted(p.name for p in out.iterdir()) == ["D1R2.progress.jsonl", "SHA256SUMS", "branch_outcome.json", "d1r2_rows.invalid_redacted.jsonl", "provenance.json", "summary.json"]
    wall_rows = [json.loads(l) for l in (out / "d1r2_rows.invalid_redacted.jsonl").read_text().splitlines()]
    assert len(wall_rows) == 20 and all(_disk_schema_ok(r) for r in wall_rows)
    assert {s["stop_reason"] for r in wall_rows for s in r["decoder_slots"].values()} == {"seed_not_started"}
    assert (out / "D1R2.progress.jsonl").read_text() == ""
    assert not stats.contains_outcome(json.loads((out / "summary.json").read_text())["readouts"] or {})
    time.sleep(2.5)
    assert started == [SYN_SEEDS[0]]


# ================================================================ D1R2 section 11 additions (11a-11f, item 16)
# Synthetic fixtures only. LEDGER_SEEDS are row IDs outside every declared seed
# range; their rows come from the real producer with an injected synthetic
# capture (no network) and synthetic decoder models (no LogisticRegression)
# unless a test says otherwise.
import os  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

LEDGER_SEEDS = tuple(range(9001, 9021))
STOP_LOSS = "Stop-loss: the D1 harness-repair path is closed. D1 returns to Kendrick as a NEEDS KENDRICK DECISION card. No D1R3."
ITER_TEXT = "lbfgs failed to converge after 20000 iteration(s) (status=1):\nSTOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n\nIncrease the number of iterations."
EVAL_TEXT = "lbfgs failed to converge after 13501 iteration(s) (status=1):\nSTOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT\n\nIncrease the number of iterations."
SECRET = "SECRET_SCORE_TEXT_row17"


def _disk_schema_ok(row) -> bool:
    """Section 6.3 schema of an on-disk INVALID row (the finalizer redacts the ``decoders`` outcome key)."""
    return "decoders" not in row and d1r2.validate_row_schema({**row, "decoders": {}}) == []


def _syn_capture(x, w1, w2):
    """Synthetic capture: random features plus the label columns; no network transition."""
    x = np.asarray(x, dtype=np.int8)
    g = np.random.default_rng([len(x), int(x[:64].sum())])
    net = g.random((len(x), 460))
    prev = np.concatenate(([0], x[:-1])).astype(np.float64)
    net[:, 60], net[:, 61] = prev, x
    pipe = g.random((len(x), 46))
    pipe[:, 6] = prev
    return {"network": net, "pipeline": pipe, "nesting_ok": True, "weights_constant": True,
            "input_hash": d1r2.sha256(x), "network_feature_hash": d1r2.sha256(net), "pipeline_feature_hash": d1r2.sha256(pipe)}


def _warning_capture(x, w1, w2):
    warnings.warn(RuntimeWarning("synthetic capture warning"))
    return _syn_capture(x, w1, w2)


class DerivedSolverWarning(ConvergenceWarning):
    """ConvergenceWarning subclass whose class name does not end in ``ConvergenceWarning`` (spec 6.3 subclass rule)."""


class FakeLR:
    """Synthetic decoder model with scripted solver outcomes (fixture only)."""

    def __init__(self, mode="ok", **config):
        self.mode, self.config = mode, config

    def fit(self, z, y):
        m = self.mode
        if m == "fit_raise":
            raise FloatingPointError("synthetic fit failure")
        if m in ("runtime_in_fit", "hang"):
            warnings.warn(RuntimeWarning("synthetic overflow in fit"))
        if m == "hang":
            warnings.warn(ConvergenceWarning(ITER_TEXT))
            time.sleep(600)
        n_iter = {"iter_warn": 20000, "iter_then_score_raise": 20000, "iter_nowarn": 20000, "eval": 13501, "eval_then_score_raise": 13501}.get(m, 37)
        if m in ("iter_warn", "iter_then_score_raise"):
            warnings.warn(ConvergenceWarning(ITER_TEXT))
        if m in ("eval", "eval_then_score_raise"):
            warnings.warn(ConvergenceWarning(EVAL_TEXT))
        if m == "other_cw":
            warnings.warn(ConvergenceWarning("synthetic other convergence warning"))
        if m == "derived_cw":
            warnings.warn(DerivedSolverWarning("synthetic derived convergence warning"))
        self.coef_ = np.zeros((1, z.shape[1]))
        self.coef_[0, 0] = 1.0
        self.intercept_ = np.zeros(1)
        self.n_iter_ = np.array([n_iter], dtype=np.int32)
        return self

    def predict_proba(self, z):
        if self.mode in ("score_raise", "iter_then_score_raise", "eval_then_score_raise"):
            raise ValueError(SECRET)
        if self.mode == "score_warn":
            warnings.warn(RuntimeWarning(SECRET))
        p = 1.0 / (1.0 + np.exp(-np.asarray(z)[:, 0]))
        return np.stack((1.0 - p, p), axis=1)


class Modes:
    """model_factory returning FakeLR with one scripted mode per decoder, in fit order."""

    def __init__(self, *modes):
        self.modes, self.calls = list(modes), []

    def __call__(self, **config):
        self.calls.append(dict(config))
        return FakeLR(self.modes[len(self.calls) - 1], **config)


def _produce(seed, *modes, ledger=None, capture=_syn_capture):
    return d1r2.run_d1r2_seed(seed, n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=capture, ledger_path=ledger, model_factory=Modes(*modes))


def _short_violations(row):
    return d1r2.validate_row(row, d1r2.object_hashes(row["seed"], n_train=SHORT_TRAIN, n_test=SHORT_TEST), train_rows=SHORT_TRAIN_ROWS, test_rows=SHORT_TEST_ROWS)


def _ledger_execute(ledger_name, produce_seeds, modes, exc=None):
    """Execute hook: run the producer for ``produce_seeds`` writing the run's event ledger, then return or raise."""

    def ex(fn, jobs):
        out = ex.out
        rows = []
        for seed in produce_seeds:
            row = _produce(seed, *modes.get(seed, ("ok", "ok", "ok")), ledger=out / ledger_name)
            row["status"], row["worker_pid"] = "ok", os.getpid()
            rows.append(row)
        if exc is not None:
            raise exc
        return rows, {"stage": "D1R2", "wall_seconds": 1.0}

    return ex


def _run_ledger_pkg(tmp_path, name, produce_seeds, modes=None, exc=None):
    out = tmp_path / name
    ex = _ledger_execute("D1R2.events.jsonl", produce_seeds, modes or {}, exc)
    ex.out = out
    summary = runner.run_d1r2(SimpleNamespace(workers=1), out, PROV, execute=ex, seeds=LEDGER_SEEDS, identity=lambda s: d1r2.object_hashes(s, n_train=SHORT_TRAIN, n_test=SHORT_TEST))
    rows = [json.loads(l) for l in (out / "d1r2_rows.invalid_redacted.jsonl").read_text().splitlines()] if (out / "d1r2_rows.invalid_redacted.jsonl").exists() else None
    return summary, out, rows


def _assert_no_outcome_on_disk(out):
    for p in out.iterdir():
        text = p.read_text()
        assert SECRET not in text, p.name
        if p.suffix == ".jsonl":
            for line in text.splitlines():
                assert not stats.contains_outcome(json.loads(line)), p.name


# 11a
def test_d1r2_11a_every_decoder_constructed_with_max_iter_20000_and_frozen_arguments():
    want = {"penalty": "l2", "C": 1.0, "fit_intercept": True, "solver": "lbfgs", "tol": 1e-8, "max_iter": 20000, "class_weight": None}
    assert d1r2.DECODER_CONFIG == want and d1r2.MAX_ITER == 20000
    assert {k: v for k, v in diag.D1_DECODER_CONFIG.items() if k != "max_iter"} == {k: v for k, v in want.items() if k != "max_iter"}
    row = _produce(9001, "ok", "ok", "ok")
    factory = Modes("ok", "ok", "ok")
    d1r2.run_d1r2_seed(9001, n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=_syn_capture, model_factory=factory)
    assert factory.calls == [want, want, want]  # pipeline, network_positive, A
    import inspect

    assert inspect.signature(d1r2.fit_decoder_slot).parameters["model_factory"].default is LogisticRegression
    assert inspect.signature(d1r2.run_d1r2_seed).parameters["model_factory"].default is LogisticRegression
    params = LogisticRegression(**d1r2.DECODER_CONFIG).get_params()
    assert {k: params[k] for k in want} == want
    for name in d1r2.DECODERS:
        assert row["decoders"][name]["decoder_config"] == want
    assert runner.expected_parameters_from_spec()["decoder"] == want


# 11b
def test_d1r2_11b_slot_schema_stop_reasons_attribution_reconstruction_and_redaction(tmp_path):
    # Iteration limit: a real lbfgs fit stopped by its iteration cap (fixture-only cap of 1).
    real_cap = lambda **kw: LogisticRegression(**{**kw, "max_iter": 1})  # noqa: E731
    row = d1r2.run_d1r2_seed(9001, n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=_syn_capture, model_factory=real_cap)
    for name in d1r2.DECODERS:  # all three still fitted and recorded
        s = row["decoder_slots"][name]
        assert s["attempted"] and s["completed"] and s["stop_reason"] == "iteration_limit" and s["n_iter"] == 1
        assert s["converged"] is False and s["convergence_warning"] is True
        assert s["warnings"][0]["category"] == "sklearn.exceptions.ConvergenceWarning" and "ITERATIONS REACHED LIMIT" in s["warnings"][0]["message_head"]
        assert d1r2.validate_slot(name, s) == []
    assert any("non-converged" in v for v in _short_violations(row))
    # n_iter >= 20000 without a warning, the F,G evaluation limit with n_iter < 20000, and another ConvergenceWarning.
    row = _produce(9001, "iter_nowarn", "eval", "other_cw")
    slots = row["decoder_slots"]
    assert [slots[n]["stop_reason"] for n in d1r2.DECODERS] == ["iteration_limit", "evaluation_limit", "other_convergence_warning"]
    assert slots["network_positive"]["n_iter"] == 13501 < 20000 and slots["network_positive"]["converged"] is False
    assert all(d1r2.validate_slot(n, slots[n]) == [] for n in d1r2.DECODERS)
    assert any("non-converged" in v for v in _short_violations(row))
    # A clean synthetic row validates (control for the checks above).
    clean = _produce(9001, "ok", "ok", "ok")
    assert all(clean["decoder_slots"][n]["stop_reason"] == "converged" for n in d1r2.DECODERS) and _short_violations(clean) == []
    # Fit exception in one decoder: fit_exception there, the other two still attempted.
    row = _produce(9001, "ok", "fit_raise", "ok")
    s = row["decoder_slots"]["network_positive"]
    assert s["stop_reason"] == "fit_exception" and s["attempted"] and not s["completed"] and s["n_iter"] is None and s["converged"] is False
    assert s["fit_error"]["exception_type"] == "builtins.FloatingPointError" and s["fit_error"]["phase"] == "fit"
    assert row["decoder_slots"]["pipeline"]["attempted"] and row["decoder_slots"]["A"]["attempted"]
    assert row["decoder_slots"]["A"]["stop_reason"] == "converged" and "network_positive" not in row["decoders"]
    assert any("non-converged" in v for v in _short_violations(row))
    # Upstream exception: three not_attempted_upstream_failure slots.
    def broken(x, w1, w2):
        raise MemoryError("synthetic capture failure")

    row = d1r2.run_d1r2_seed(9001, n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=broken)
    assert {row["decoder_slots"][n]["stop_reason"] for n in d1r2.DECODERS} == {"not_attempted_upstream_failure"}
    assert row["seed_error"]["phase"] == "capture" and d1r2.validate_row_schema(row) == [] and _short_violations(row)
    # Attribution: a RuntimeWarning in one decoder's fit is that decoder's only; a capture warning is non-decoder.
    ledger = tmp_path / "attr.jsonl"
    row = _produce(9001, "ok", "runtime_in_fit", "ok", ledger=ledger, capture=_warning_capture)
    assert [w["category"] for w in row["decoder_slots"]["network_positive"]["warnings"]] == ["builtins.RuntimeWarning"]
    assert row["decoder_slots"]["network_positive"]["warnings"][0]["phase"] == "fit"
    assert row["decoder_slots"]["network_positive"]["convergence_warning"] is False and row["decoder_slots"]["network_positive"]["converged"] is True
    assert row["decoder_slots"]["pipeline"]["warnings"] == [] and row["decoder_slots"]["A"]["warnings"] == []
    assert [w["phase"] for w in row["non_decoder_warnings"]] == ["capture", "capture"]
    assert d1r2.validate_ledger_match(row, d1r2.read_ledger(ledger)[0]) == []
    # A row missing any slot fails section 7 item 6.
    for name in d1r2.DECODERS:
        rows = syn_rows()
        del rows[4]["decoder_slots"][name]
        out = summarize(rows)
        assert out["status"] == "INVALID" and any("decoder_slots must be exactly" in v for v in out["violations"])
    # Worker loss, wall stop, RSS abort: 20 rows, three schema-valid slots each, via coordinator_from_ledger.
    for exc, reason in ((RuntimeError("worker died"), "worker_lost"), (runner.WallLimitExceeded("6 h"), "terminated_wall_stop"), (runner.RSSLimitExceeded("12 GiB"), "terminated_rss_abort")):
        summary, out, rows = _run_ledger_pkg(tmp_path, reason, LEDGER_SEEDS[:1], exc=exc)
        assert summary["branch"]["code"] == "STOP_D1R2_INVALID" and STOP_LOSS in summary["branch"]["action"]
        assert len(rows) == 20 and {r["row_source"] for r in rows} == {"coordinator_from_ledger"}
        assert all(_disk_schema_ok(r) for r in rows)
        assert {s["stop_reason"] for s in rows[0]["decoder_slots"].values()} == {"converged"}  # closed in the ledger
        assert rows[0]["decoder_slots"]["A"]["n_iter"] == 37
        later = {s["stop_reason"] for r in rows[1:] for s in r["decoder_slots"].values()}
        assert later == ({"worker_lost"} if reason == "worker_lost" else {"seed_not_started"})
        _assert_no_outcome_on_disk(out)
    # No outcome field and no predict_score text reaches disk on the INVALID branch.
    summary, out, rows = _run_ledger_pkg(tmp_path, "redact", LEDGER_SEEDS, modes={LEDGER_SEEDS[3]: ("ok", "score_warn", "iter_warn")})
    assert summary["status"] == "INVALID" and summary["readouts"] == {"pipeline": None, "network_positive": None, "A": None}
    assert sorted(p.name for p in out.iterdir()) == ["D1R2.events.jsonl", "SHA256SUMS", "branch_outcome.json", "d1r2_rows.invalid_redacted.jsonl", "provenance.json", "summary.json"]
    w = rows[3]["decoder_slots"]["network_positive"]["warnings"]
    assert w == [{"category": "builtins.RuntimeWarning", "phase": "predict_score", "message_head": None}]
    assert "decoders" not in rows[3] and rows[3]["decoder_slots"]["A"]["stop_reason"] == "iteration_limit"
    _assert_no_outcome_on_disk(out)


# 11c
@pytest.mark.parametrize(
    "kw, mutate, row_no, code",
    [
        (dict(), lambda r: r[0].update(nesting_ok=False), 1, "STOP_D1R2_INVALID"),
        (dict(pipe=0.98), None, 2, "STOP_D1R2_INVALID_PIPELINE"),
        (dict(pos=(0.60, 0.6)), None, 3, "CLOSE_D1_REPRESENTATION_LIMITED"),
        (dict(a=(0.52, 0.4)), None, 4, "D1R2_A_FAIL_EBB90E74_ROW3"),
        (dict(a=(0.65, 0.6)), None, 5, "D1R2_A_INCONCLUSIVE_EBB90E74_ROW4"),
        (dict(), None, 6, "D1R2_PASS_D2_D4_ELIGIBLE"),
    ],
)
def test_d1r2_11c_branch_table_reaches_every_row_and_rows_1_2_carry_stop_loss(kw, mutate, row_no, code):
    rows = syn_rows(**kw)
    if mutate:
        mutate(rows)
    out = summarize(rows)
    assert out["branch"]["code"] == code and out["branch"]["table_row"] == row_no
    assert (STOP_LOSS in out["branch"]["action"]) is (row_no in (1, 2))
    assert [c for c, _, _ in d1r2.BRANCH_TABLE][row_no - 1] == code
    if row_no == 1:
        assert "Run-preflight collision-probe failure" in out["branch"]["condition"] and "six-hour wall stop" in out["branch"]["condition"]


# 11d
def test_d1r2_11d_run_preflight_collision_finalizer_writes_20_preflight_rows(tmp_path):
    probe = runner.load_probe_module()
    new = probe.d1r2_identities()
    legacy_name, legacy_tuple = probe.legacy_identities()[0]
    new[7] = (new[7][0], legacy_tuple)  # injected new-vs-legacy collision (synthetic identity set)
    record = runner.run_preflight_probe(new=new)
    assert record["pass"] is False and record["decision"] == "INVALID" and record["execution"] == "run_preflight"
    assert record["new_vs_legacy_entropy_collisions"] and record["new_vs_legacy_state_collisions"]
    assert record["numpy_version"] == np.__version__ and len(record["code_sha256"]) == 64 and len(record["output_sha256"]) == 64

    def no_worker(fn, jobs):
        raise AssertionError("a worker started after a failed run preflight")

    def no_identity(seed):
        raise AssertionError("a D1R2 stream identity was built after a failed run preflight")

    out = tmp_path / "preflight"
    prov = {**PROV, "collision_probe": record}
    summary = runner.run_d1r2(SimpleNamespace(workers=1), out, prov, execute=no_worker, identity=no_identity)
    assert not (out / "D1R2.events.jsonl").exists() and not (out / "D1R2.progress.jsonl").exists()
    rows = [json.loads(l) for l in (out / "d1r2_rows.invalid_redacted.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in rows] == list(range(2200, 2220))
    for r in rows:
        assert r["row_source"] == "coordinator_preflight" and r["non_decoder_warnings"] == [] and "decoders" not in r
        assert r["seed_error"]["exception_type"] is None and r["seed_error"]["phase"] == "preflight_collision_probe"
        assert "collision probe failed" in r["seed_error"]["message_head"]
        assert _disk_schema_ok(r)  # section 7 item 6 presence and schema pass
        for name in d1r2.DECODERS:
            assert r["decoder_slots"][name] == {"attempted": False, "completed": False, "stop_reason": "preflight_failure", "n_iter": None,
                                                "converged": False, "convergence_warning": False, "warnings": [], "fit_error": None}
    violations = d1r2.validate_package([{**r, "decoders": {}} for r in rows], None)
    assert all("non-converged" in v or "coordinator" in v for v in violations) and len(violations) >= 20  # convergence fails
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID" and summary["branch"]["table_row"] == 1 and STOP_LOSS in summary["branch"]["action"]
    assert summary["collision_probe"]["new_vs_legacy_entropy_collisions"] == record["new_vs_legacy_entropy_collisions"]
    for k in ("code_sha256", "numpy_version", "output_sha256"):
        assert summary["collision_probe"][k] == record[k]
    _assert_manifest(out, summary)
    # The committed probe, re-executed as the run preflight, passes and matches section 3.3.
    clean = runner.run_preflight_probe()
    assert clean["pass"] is True and clean["identity_state_sha256"] == "4d8dd8b477aa2615bc14f66ed11223d0995c10df0985919577bacc9648b6848e"
    assert clean["legacy_only_duplicate_count"] == 6 and not clean["synthetic_override"]


# 11e
def _hang_fixture_job(job) -> dict:
    """Process-pool fixture job (synthetic only): capture warning, then a RuntimeWarning and a
    ConvergenceWarning inside network_positive's fit scope, then hang until terminated."""
    seed, ledger, pid_file = job
    Path(pid_file).write_text(str(os.getpid()))
    return d1r2.run_d1r2_seed(seed, n_train=SHORT_TRAIN, n_test=SHORT_TEST, capture=_warning_capture, ledger_path=Path(ledger), model_factory=Modes("ok", "hang", "ok"))


def _ledger_has_cw(ledger: Path) -> bool:
    if not ledger.exists():
        return False
    return any(e.get("event") == "warning" and str(e.get("category", "")).endswith("ConvergenceWarning") for e in d1r2.read_ledger(ledger)[0])


class _LedgerTriggeredWatch(_Watch):
    """RSS watch whose abort flag rises once the open scope's warnings are durable (RSS-abort path)."""

    def __init__(self, ledger):
        self.ledger = ledger

    @property
    def exceeded(self):
        return _ledger_has_cw(self.ledger)


class _JumpClock:
    """Real clock that jumps past the wall deadline once the warnings are durable (wall-stop path)."""

    def __init__(self, ledger):
        self.ledger, self.offset = ledger, 0.0

    def monotonic(self):
        if not self.offset and _ledger_has_cw(self.ledger):
            self.offset = 10.0 * 3600
        return time.monotonic() + self.offset

    def time(self):
        return time.time()

    def sleep(self, s):
        return time.sleep(s)


def test_d1r2_11e_warning_then_termination_reconstruction_by_sigkill_wall_and_rss(tmp_path, monkeypatch):
    import threading

    import psutil

    monkeypatch.setattr(runner, "POOL_POLL_SECONDS", 0.2)
    seed = LEDGER_SEEDS[0]
    for reason in ("worker_lost", "terminated_wall_stop", "terminated_rss_abort"):
        out = tmp_path / reason
        ledger = out / "D1R2.events.jsonl"
        pid_file = tmp_path / f"{reason}.pid"
        watch = _LedgerTriggeredWatch(ledger) if reason == "terminated_rss_abort" else _Watch()
        clock = _JumpClock(ledger)

        def ex(fn, jobs, ledger=ledger, pid_file=pid_file, watch=watch, reason=reason, clock=clock):
            def killer():  # worker loss: SIGKILL the worker once its warnings are on disk
                end = time.monotonic() + 120
                while time.monotonic() < end and not _ledger_has_cw(ledger):
                    time.sleep(0.05)
                os.kill(int(pid_file.read_text()), signal.SIGKILL)

            if reason == "worker_lost":
                threading.Thread(target=killer, daemon=True).start()
            if reason == "terminated_wall_stop":
                monkeypatch.setattr(runner, "time", clock)
            try:
                return runner.run_pool(_hang_fixture_job, [(s, str(ledger), str(pid_file)) for s in jobs], 1, ledger.parent,
                                       executor_factory=lambda n: ProcessPoolExecutor(max_workers=n), watch=watch, wall_limit=300.0)
            finally:
                monkeypatch.setattr(runner, "time", time)

        t0 = time.monotonic()
        summary = runner.run_d1r2(SimpleNamespace(workers=1), out, PROV, execute=ex, seeds=LEDGER_SEEDS, identity=syn_identity)
        assert time.monotonic() - t0 < 200
        pid = int(pid_file.read_text())
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE  # worker is gone
        assert reason.split("_")[-1] in summary["run_metadata"]["incomplete_reason"] or reason == "worker_lost"
        rows = [json.loads(l) for l in (out / "d1r2_rows.invalid_redacted.jsonl").read_text().splitlines()]
        assert len(rows) == 20 and all(_disk_schema_ok(r) for r in rows)
        assert {r["row_source"] for r in rows} == {"coordinator_from_ledger"}
        r0 = rows[0]
        open_slot = r0["decoder_slots"]["network_positive"]
        assert open_slot["stop_reason"] == reason and open_slot["attempted"] is True and open_slot["completed"] is False
        assert open_slot["n_iter"] is None and open_slot["converged"] is False and open_slot["convergence_warning"] is True
        assert [(w["category"], w["phase"]) for w in open_slot["warnings"]] == [("builtins.RuntimeWarning", "fit"), ("sklearn.exceptions.ConvergenceWarning", "fit")]
        assert open_slot["fit_error"]["exception_type"] is None and reason in open_slot["fit_error"]["message_head"]
        assert r0["decoder_slots"]["pipeline"]["stop_reason"] == "converged"  # closed before termination
        assert r0["decoder_slots"]["A"]["stop_reason"] == reason and r0["decoder_slots"]["A"]["attempted"] is False
        assert [w["phase"] for w in r0["non_decoder_warnings"]] == ["capture", "capture"]
        assert summary["branch"]["code"] == "STOP_D1R2_INVALID" and STOP_LOSS in summary["branch"]["action"]
    # A worker-written row whose warnings differ from the ledger is schema-invalid.
    ledger = tmp_path / "match.jsonl"
    row = _produce(seed, "ok", "runtime_in_fit", "score_warn", ledger=ledger)
    events, _ = d1r2.read_ledger(ledger)
    assert d1r2.validate_ledger_match(row, events) == []
    assert all(e["message_head"] is None for e in events if e.get("event") == "warning" and e["phase"] == "predict_score")
    assert SECRET not in ledger.read_text()
    bad = copy.deepcopy(row)
    bad["decoder_slots"]["network_positive"]["warnings"] = []
    assert d1r2.validate_ledger_match(bad, events)
    bad["status"] = "ok"
    completed, notes = runner.complete_rows([bad], events, False, [seed])
    assert completed[0]["status"] == "error" and notes and d1r2.validate_row(completed[0], None)
    # Torn final line: reconstructed with ledger_torn_tail = true.
    torn = tmp_path / "torn.jsonl"
    torn.write_bytes(ledger.read_bytes() + b'{"event":"warning","seed":9001,"deco')
    events, is_torn = d1r2.read_ledger(torn)
    assert is_torn is True and events == d1r2.read_ledger(ledger)[0]
    rows = d1r2.reconstruct_rows_from_ledger(events, "worker_lost", LEDGER_SEEDS, torn=is_torn)
    assert all(r["ledger_torn_tail"] is True and d1r2.validate_row_schema(r) == [] for r in rows)  # in-memory rows
    # Malformed interior line: INVALID.
    bad_ledger = tmp_path / "malformed.jsonl"
    lines = ledger.read_bytes().split(b"\n")
    bad_ledger.write_bytes(b"\n".join(lines[:2] + [b"{not json"] + lines[2:]))
    with pytest.raises(d1r2.MalformedLedger):
        d1r2.read_ledger(bad_ledger)
    out = tmp_path / "malformed_pkg"

    def ex_bad(fn, jobs):
        (out / "D1R2.events.jsonl").write_bytes(bad_ledger.read_bytes())
        r = copy.deepcopy(row)
        r["status"] = "ok"
        return [r], {"stage": "D1R2", "wall_seconds": 1.0}

    summary = runner.run_d1r2(SimpleNamespace(workers=1), out, PROV, execute=ex_bad, seeds=[seed], identity=syn_identity)
    assert summary["status"] == "INVALID" and any("malformed interior" in v for v in summary["violations"])


# 11f
def test_d1r2_11f_scoring_exception_slot_and_shakedown_probe_failure_writes_nothing(tmp_path):
    ledger = tmp_path / "score.jsonl"
    row = _produce(9001, "ok", "score_raise", "ok", ledger=ledger)
    s = row["decoder_slots"]["network_positive"]
    assert s["attempted"] is True and s["completed"] is False and s["stop_reason"] == "score_exception"
    assert s["n_iter"] == 37 < 20000 and s["converged"] is False and s["convergence_warning"] is False
    assert s["fit_error"]["exception_type"] == "builtins.ValueError" and s["fit_error"]["phase"] == "predict_score"
    assert d1r2.validate_slot("network_positive", s) == []
    assert row["decoder_slots"]["pipeline"]["attempted"] and row["decoder_slots"]["A"]["attempted"]
    assert row["decoder_slots"]["A"]["stop_reason"] == "converged"
    assert any("non-converged" in v for v in _short_violations(row))
    close = [e for e in d1r2.read_ledger(ledger)[0] if e.get("event") == "scope_close" and e["decoder"] == "network_positive"][0]
    assert close["slot"]["fit_error"] == {"exception_type": "builtins.ValueError", "phase": "predict_score", "message_head": None}
    assert SECRET not in ledger.read_text()
    summary, out, rows = _run_ledger_pkg(tmp_path, "score_pkg", LEDGER_SEEDS, modes={LEDGER_SEEDS[0]: ("ok", "score_raise", "ok")})
    assert summary["branch"]["code"] == "STOP_D1R2_INVALID" and summary["branch"]["table_row"] == 1 and STOP_LOSS in summary["branch"]["action"]
    assert rows[0]["decoder_slots"]["network_positive"]["fit_error"]["message_head"] is None
    _assert_no_outcome_on_disk(out)
    # A ConvergenceWarning fit followed by a scoring exception keeps iteration_limit / evaluation_limit.
    row = _produce(9001, "iter_then_score_raise", "eval_then_score_raise", "ok")
    for name, want in (("pipeline", "iteration_limit"), ("network_positive", "evaluation_limit")):
        s = row["decoder_slots"][name]
        assert s["stop_reason"] == want and s["fit_error"]["phase"] == "predict_score" and s["completed"] is False
        assert d1r2.validate_slot(name, s) == []
    # The validator enforces first-match order: relabelling that slot as score_exception is rejected.
    relabel = copy.deepcopy(row["decoder_slots"]["pipeline"])
    relabel["stop_reason"] = "score_exception"
    assert d1r2.validate_slot("pipeline", relabel)
    # Shakedown probe check (section 11 item 12) on a synthetic identity mismatch: fails and writes nothing.
    probe = runner.load_probe_module()
    mismatched = probe.d1r2_identities()[:-1]  # one identity missing: counts and identity hash differ from 3.3
    target = tmp_path / "shakedown_probe" / "ss_probe.json"
    with pytest.raises(runner.ShakedownProbeFailure):
        runner.shakedown_collision_check(target, new=mismatched)
    assert not target.parent.exists()


# 11.16 (accepted by Nora within its convergence-only fixture boundary)
def test_d1r2_11_16_convergence_only_network_fits_on_fixture_seed_never_predict(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("item 16 must never predict or score")

    monkeypatch.setattr(diag.FittedDecoder, "predict", forbidden)
    monkeypatch.setattr(LogisticRegression, "predict", forbidden)
    monkeypatch.setattr(LogisticRegression, "predict_proba", forbidden)
    monkeypatch.setattr(d1r2, "block_lower_bound", forbidden)
    seen = []
    orig_capture = d1r2.capture_stream
    monkeypatch.setattr(d1r2, "capture_stream", lambda x, w1, w2: (seen.append(len(x)), orig_capture(x, w1, w2))[1])
    import inspect

    assert inspect.signature(d1r2.run_fixture_network_convergence).parameters["n_train"].default == 200_000
    assert d1r2.FIXTURE_SEED == 4242 and 4242 not in d1r2.D1R2_SEEDS
    res = d1r2.run_fixture_network_convergence(n_train=SHORT_TRAIN)  # construction-scale; the shakedown runs full length
    assert seen == [SHORT_TRAIN]  # train stream only: no held-out stream exists
    assert set(res) == {"network_positive", "A"}
    for record in res.values():
        assert set(record) == {"n_iter", "converged", "warnings"}
        assert isinstance(record["n_iter"], int) and isinstance(record["converged"], bool)
        assert not stats.contains_outcome(record)


def test_d1r2_11_16_committed_shakedown_record_is_fixture_only_and_convergence_only():
    runs = sorted((ROOT / "results_stage2_diagnostic_d1r2" / "runs").glob("shakedown-*"))
    assert runs, "committed section 8 step 1 shakedown output missing"
    for run in runs:
        rec = json.loads((run / "shakedown.json").read_text())
        prov = json.loads((run / "provenance.json").read_text())
        assert rec["seed"] == 4242 and rec["d1r2_seed"] is False and prov["seeds"] == [4242] and prov["mode"] == "shakedown"
        assert rec["lengths"] == {"train_stream": 200_000, "test_stream": 12_000, "train_rows": 198_000, "test_rows": 10_000}
        assert rec["decoders_fitted"] == ["pipeline"] and rec["network_decoders_fitted"] == ["A", "network_positive"]
        assert rec["item16_no_accuracy_or_prediction_fields"] is True
        for record in rec["item16_network_convergence_only"].values():
            assert set(record) == {"n_iter", "converged", "warnings"}
        assert rec["pass"] is True and rec["item16_both_converged"] is True and rec["item15_pipeline_reaches_0_99"] is True
        row = json.loads((run / "shakedown_row.json").read_text())
        assert set(row["decoders"]) == {"pipeline"} and row["seed"] == 4242
        for line in (run / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest


# 11b (Nora round 1 correction): spec 6.3 "ConvergenceWarning or a subclass" uses real issubclass semantics.
def test_d1r2_11b_convergence_warning_subclass_with_other_name_is_non_converged_worker_and_ledger(tmp_path):
    assert issubclass(DerivedSolverWarning, ConvergenceWarning) and not DerivedSolverWarning.__name__.endswith("ConvergenceWarning")
    assert d1r2.is_convergence_category(DerivedSolverWarning) is True
    assert d1r2.is_convergence_category(RuntimeWarning) is False
    category = f"{DerivedSolverWarning.__module__}.{DerivedSolverWarning.__qualname__}"
    assert d1r2.classify_category_name(category) is True
    assert d1r2.classify_category_name("builtins.RuntimeWarning") is False
    assert d1r2.classify_category_name("sklearn.exceptions.ConvergenceWarning") is True
    assert d1r2.classify_category_name("no_such_module_x.Unknown") is None

    # Worker-written slot (Nora's reproduction: subclass warning, n_iter_ = [1], seed ID 9001).
    class OneIter(FakeLR):
        def fit(self, z, y):
            super().fit(z, y)
            self.n_iter_ = np.array([1], dtype=np.int32)
            return self

    ledger = tmp_path / "worker.jsonl"
    g = np.random.default_rng(0)
    xtr, ytr = g.random((200, 4)), (g.random(200) > 0.5).astype(np.int8)
    slot, _, _ = d1r2.fit_decoder_slot("network_positive", xtr, ytr, xtr[:50], ytr[:50], 9001, ledger_path=ledger,
                                       model_factory=lambda **kw: OneIter("derived_cw", **kw))
    assert slot["warnings"] == [{"category": category, "phase": "fit", "message_head": "synthetic derived convergence warning"}]
    assert slot["n_iter"] == 1 and slot["completed"] is True
    assert slot["convergence_warning"] is True and slot["converged"] is False
    assert slot["stop_reason"] == "other_convergence_warning"
    assert d1r2.validate_slot("network_positive", slot) == []
    # The exact mislabel Nora observed is rejected by the validator.
    mislabel = {**copy.deepcopy(slot), "stop_reason": "converged", "converged": True, "convergence_warning": False}
    assert d1r2.validate_slot("network_positive", mislabel)
    # The classification is durable: the ledger warning event carries the issubclass result.
    events = d1r2.read_ledger(ledger)[0]
    warn_events = [e for e in events if e.get("event") == "warning"]
    assert len(warn_events) == 1 and warn_events[0]["category"] == category and warn_events[0]["convergence_warning"] is True
    close = [e for e in events if e.get("event") == "scope_close"][0]
    assert close["slot"]["convergence_warning"] is True and close["slot"]["stop_reason"] == "other_convergence_warning"

    # Through the full producer and package: the subclass makes the seed and D1R2 INVALID.
    row = _produce(9001, "ok", "derived_cw", "ok", ledger=tmp_path / "row.jsonl")
    s = row["decoder_slots"]["network_positive"]
    assert s["stop_reason"] == "other_convergence_warning" and s["converged"] is False and s["convergence_warning"] is True
    assert d1r2.validate_row_schema(row) == []
    assert d1r2.validate_ledger_match(row, d1r2.read_ledger(tmp_path / "row.jsonl")[0]) == []
    assert any("non-converged" in v for v in _short_violations(row))
    summary, out, rows = _run_ledger_pkg(tmp_path, "derived_pkg", LEDGER_SEEDS, modes={LEDGER_SEEDS[0]: ("ok", "derived_cw", "ok")})
    assert summary["status"] == "INVALID" and summary["branch"]["code"] == "STOP_D1R2_INVALID" and summary["branch"]["table_row"] == 1
    assert rows[0]["row_source"] == "worker" and rows[0]["decoder_slots"]["network_positive"]["stop_reason"] == "other_convergence_warning"
    _assert_no_outcome_on_disk(out)

    # Coordinator reconstruction uses the durable classification, not the class name.
    # (a) Open slot at termination: scope_open + subclass warning event, no scope_close.
    # The category string is deliberately unresolvable in the coordinator process, so only the
    # durable ledger flag can classify it; with the flag, reconstruction reaches the worker's result.
    foreign = "__worker_only__.DerivedSolverWarning"
    assert d1r2.classify_category_name(foreign) is None
    seed = LEDGER_SEEDS[0]
    open_events = [
        {"event": "seed_started", "seed": seed},
        {"event": "scope_open", "seed": seed, "decoder": "pipeline"},
        {"event": "scope_close", "seed": seed, "decoder": "pipeline", "slot": copy.deepcopy(_produce(seed, "ok", "ok", "ok")["decoder_slots"]["pipeline"])},
        {"event": "scope_open", "seed": seed, "decoder": "network_positive"},
        {"event": "warning", "seed": seed, "decoder": "network_positive", "category": foreign, "phase": "fit",
         "message_head": "synthetic derived convergence warning", "convergence_warning": True},
    ]
    for reason in ("worker_lost", "terminated_wall_stop", "terminated_rss_abort"):
        rec = d1r2.reconstruct_rows_from_ledger(open_events, reason, [seed])[0]
        o = rec["decoder_slots"]["network_positive"]
        assert o["stop_reason"] == reason and o["attempted"] is True and o["completed"] is False and o["n_iter"] is None
        assert o["converged"] is False and o["convergence_warning"] is True
        assert o["warnings"] == [{"category": foreign, "phase": "fit", "message_head": "synthetic derived convergence warning"}]
        assert d1r2.validate_row_schema(rec) == [] and d1r2.validate_ledger_match(rec, open_events) == []
    # (b) Open slot whose real subclass warning was written by the real hook, then the worker died.
    ledger_open = tmp_path / "open.jsonl"

    class DiesAfterWarning(FakeLR):
        def fit(self, z, y):
            warnings.warn(DerivedSolverWarning("synthetic derived convergence warning"))
            raise KeyboardInterrupt  # BaseException: escapes the slot like a killed worker, no scope_close

    with pytest.raises(KeyboardInterrupt):
        d1r2.fit_decoder_slot("A", xtr, ytr, xtr[:50], ytr[:50], seed, ledger_path=ledger_open,
                              model_factory=lambda **kw: DiesAfterWarning("ok", **kw))
    durable = d1r2.read_ledger(ledger_open)[0]
    assert [e["event"] for e in durable] == ["scope_open", "warning"] and durable[1]["convergence_warning"] is True
    rec = d1r2.reconstruct_rows_from_ledger([{"event": "seed_started", "seed": seed}, *durable], "worker_lost", [seed])[0]
    o = rec["decoder_slots"]["A"]
    assert o["stop_reason"] == "worker_lost" and o["convergence_warning"] is True and o["converged"] is False
    assert o["warnings"][0]["category"] == category and d1r2.validate_slot("A", o) == []
    # (c) A ledger-closed slot: the reconstructed convergence_warning is re-derived from the durable events.
    closed_events = [{"event": "seed_started", "seed": 9001}, *d1r2.read_ledger(tmp_path / "row.jsonl")[0]]
    rec = d1r2.reconstruct_rows_from_ledger(closed_events, "worker_lost", [9001])[0]
    c = rec["decoder_slots"]["network_positive"]
    assert c["stop_reason"] == "other_convergence_warning" and c["convergence_warning"] is True and c["converged"] is False
    assert d1r2.validate_row_schema(rec) == []
    # Fail closed: a ledger warning event without its boolean classification is flagged by the ledger cross-check,
    # and an unresolvable category without a durable flag is classified as a convergence warning.
    stripped = [{k: v for k, v in e.items() if k != "convergence_warning"} for e in d1r2.read_ledger(tmp_path / "row.jsonl")[0]]
    assert any("convergence classification" in v for v in d1r2.validate_ledger_match(row, stripped))
    unflagged = [{k: v for k, v in e.items() if k != "convergence_warning"} for e in open_events]
    o = d1r2.reconstruct_rows_from_ledger(unflagged, "worker_lost", [seed])[0]["decoder_slots"]["network_positive"]
    assert o["convergence_warning"] is True and o["converged"] is False
    # A durable False flag from the hook is respected for an ordinary warning (no false invalidation).
    ordinary = [*open_events[:4], {**open_events[4], "category": "builtins.RuntimeWarning", "convergence_warning": False}]
    o = d1r2.reconstruct_rows_from_ledger(ordinary, "worker_lost", [seed])[0]["decoder_slots"]["network_positive"]
    assert o["convergence_warning"] is False
