"""Frozen Stage 2 diagnostic D1R construction (one-shot D1 harness repair).

Implements docs/STAGE2_DIAGNOSTIC_D1R_SPEC.md at 2d9efd8 without editing the
approved ebb90e74 D1 path. The approved network, A-stream roller, causal
``FeatureAccumulator``, decoder, and thresholds are imported unchanged from
``snn.stage2_diagnostic``; this module adds only what the D1R spec changes:
namespace 31 identities, the exact-lag spike block (K = 19), the width-2
input-layer pipeline-control map, three decoders, the strict D1R validators,
the section 6.3 status, and the section 9 branch table.

Importing this module executes nothing. ``run_d1r_seed`` refuses every D1R
seed (2100..2119) unless the runner passes ``authorized_run=True`` after its
fail-closed preflight.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from snn import stage2_diagnostic as diag
from snn.stage2 import ROOT_ENTROPY, sha256
from snn.stage2_r3 import EVAL_LEN, ONEHOT, PHASE_STEPS, init_state

SPEC_COMMIT = "2d9efd8e4269057620607725a3bd615db22796f2"
SPEC_SHA256 = "c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743"
EBB90E74_SHA256 = diag.SPEC_SHA256
FROZEN_STAGE2_SHA256 = diag.FROZEN_STAGE2_SHA256

NAMESPACE = 31
D1R_SEEDS = tuple(range(2100, 2120))
FIXTURE_SEED = 4242  # non-diagnostic shakedown seed (spec 3.2, 8 step 1, 11 item 15)
K = 19
N_LAGS = K + 1
N_HIDDEN = 20
N_INPUT = 2
NETWORK_FEATURE_DIM = 3 * N_HIDDEN + N_LAGS * N_HIDDEN  # 460
PIPELINE_FEATURE_DIM = 3 * N_INPUT + N_LAGS * N_INPUT  # 46
WARMUP = diag.D1_WARMUP
TRAIN_ROWS = diag.D1_TRAIN_ROWS
TEST_ROWS = diag.D1_TEST_ROWS
DECODER_CONFIG = dict(diag.D1_DECODER_CONFIG)
MAX_ITER = diag.D1_MAX_ITER
DECODERS = ("pipeline", "network_positive", "A")
NETWORK_DECODERS = ("network_positive", "A")
BOOTSTRAP_Q = {"A": 1, "network_positive": 2}
MAX_WORKERS = 3
RSS_LIMIT_BYTES = 12 * 1024**3
WALL_LIMIT_SECONDS = 6 * 3600

# Section 6.1 numbers are ebb90e74 section 4.6 exactly (read from the approved
# THRESHOLDS); section 6.2 adds only the pipeline bar.
THRESHOLDS: Dict[str, float] = {
    "pass_median_accuracy": diag.THRESHOLDS["d1_pass_median_accuracy"],
    "pass_min_lower_bounds_gt_half": diag.THRESHOLDS["d1_pass_min_lower_bounds_gt_half"],
    "lower_bound_reference": diag.THRESHOLDS["d1_lower_bound_reference"],
    "fail_median_below": diag.THRESHOLDS["d1_fail_median_below"],
    "pipeline_pass_median_accuracy": 0.99,
}

FEATURE_MAP = (
    "phiR_t = concat(count20_t, trace20_t, trace25_t, h_t, h_(t-1), ..., h_(t-19)) of hidden spikes "
    "present before x_t is integrated; 460 dims"
)
PIPELINE_MAP = (
    "psi_t = the same operator on u_t = ONEHOT[x_(t-1)] (u_0 = 0), width 2; 46 dims"
)


# ------------------------------------------------------------ identities
def rng(entropy: Sequence[int]) -> np.random.Generator:
    """Fresh generator; the single construction point for every D1R object."""
    return np.random.default_rng(np.random.SeedSequence(list(entropy)))


def entropy_table(seed: int) -> Dict[str, Tuple[int, ...]]:
    """Spec section 3.1, verbatim, in the section 3.3 probe's enumeration order.

    tools/stage2_diagnostic_d1r_ss_probe.py enumerates exactly these names and
    tuples (via this function), so the retained identity-state hash pins the
    implementation's identities."""
    return {
        "weights_w1_w2_o1": (ROOT_ENTROPY, seed, NAMESPACE, 1),
        "weights_w2_o0": (ROOT_ENTROPY, seed, NAMESPACE, 21),
        "A_train": (ROOT_ENTROPY, seed, NAMESPACE, 2),
        "A_test": (ROOT_ENTROPY, seed, NAMESPACE, 5),
        "D1R_block_bootstrap_A": (ROOT_ENTROPY, seed, NAMESPACE, 60, 1),
        "D1R_block_bootstrap_network_positive_control": (ROOT_ENTROPY, seed, NAMESPACE, 60, 2),
    }


def initial_weights(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """r3 distributions and draw order (as ebb90e74 ``initial_weights``) under namespace 31."""
    t = entropy_table(seed)
    g = rng(t["weights_w1_w2_o1"])
    w1 = g.uniform(-10.0, 10.0, size=(20, 2)).astype(np.float64)
    w2_o1 = g.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64)
    h = rng(t["weights_w2_o0"])
    w2_o0 = h.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64)
    return w1, np.concatenate((w2_o1, w2_o0), axis=0)


def a_stream(seed: int, train: bool, n: Optional[int] = None) -> np.ndarray:
    """r3 section 7.1 order-2 A stream (approved ``_roll_a``) with the namespace-31 component."""
    key = "A_train" if train else "A_test"
    default_n = PHASE_STEPS if train else EVAL_LEN
    return diag._roll_a(rng(entropy_table(seed)[key]), default_n if n is None else n)


# ------------------------------------------------------------ feature maps
class LagFeatureAccumulator:
    """The one D1R feature operator, parameterized only by width ``n``.

    Columns 0..3n-1 are the approved ebb90e74 ``FeatureAccumulator`` output
    (count20, trace20, trace25) computed by that exact class; columns 3n..
    are the exact-lag block lag 0 neurons 0..n-1, lag 1, ..., lag K, zero
    before index 0. The same class yields phiR_t (n = 20, hidden spikes h_t)
    and psi_t (n = 2, input-layer spikes u_t).
    """

    def __init__(self, n: int):
        self.n = n
        self.base = diag.FeatureAccumulator(n_hidden=n)
        self.ring = np.zeros((N_LAGS, n), dtype=np.float64)
        self.pos = -1  # index of the most recent entry in ``ring``
        # order[p] lists ring rows lag 0..K when the newest entry sits at p.
        self._order = [np.array([(p - k) % N_LAGS for k in range(N_LAGS)], dtype=np.int64) for p in range(N_LAGS)]
        self.width = 3 * n + N_LAGS * n

    def update(self, spikes_t: np.ndarray) -> Tuple[np.ndarray, bool]:
        s = np.asarray(spikes_t, dtype=np.float64).reshape(self.n)
        phi = self.base.update(s)
        self.pos = (self.pos + 1) % N_LAGS
        self.ring[self.pos] = s
        lag = self.ring[self._order[self.pos]].reshape(-1)
        # Section 7 item 7: count20 equals the sum of the 20 lag blocks exactly.
        nested = bool(np.array_equal(phi[: self.n], self.ring.sum(axis=0)))
        return np.concatenate((phi, lag)), nested


def input_layer_vector(previous_x: Optional[int]) -> np.ndarray:
    """u_t: ONEHOT[x_(t-1)] for t >= 1, zero at t = 0 (spec 5.1)."""
    if previous_x is None:
        return np.zeros(N_INPUT, dtype=np.float64)
    return ONEHOT[int(previous_x)].detach().cpu().numpy()[0].astype(np.float64)


def capture_stream(
    x: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> Dict[str, object]:
    """Causal frozen-network capture of phiR_t and psi_t for one stream.

    At index t both rows are formed from state already present (h_t and
    u_t), then x_t is integrated with no reward function. Fresh network and
    feature state per stream.
    """
    xs = np.asarray(x, dtype=np.int8)
    net = diag.build_network(w1, w2)
    init_state(net)
    network = np.empty((len(xs), NETWORK_FEATURE_DIM), dtype=np.float64)
    pipeline = np.empty((len(xs), PIPELINE_FEATURE_DIM), dtype=np.float64)
    hidden_acc = LagFeatureAccumulator(N_HIDDEN)
    input_acc = LagFeatureAccumulator(N_INPUT)
    nested = True
    previous: Optional[int] = None
    for t, xt in enumerate(xs):
        network[t], ok_h = hidden_acc.update(net._spikes[0].detach().cpu().numpy()[0])
        pipeline[t], ok_u = input_acc.update(input_layer_vector(previous))
        nested = nested and ok_h and ok_u
        net.online_step_with_background(ONEHOT[int(xt)], None, reward_fn=None)
        previous = int(xt)
    weights_constant = all(np.array_equal(a, s.weight.detach().cpu().numpy()) for a, s in zip((w1, w2), net.synapses))
    return {
        "network": network,
        "pipeline": pipeline,
        "nesting_ok": nested,
        "weights_constant": weights_constant,
        "input_hash": sha256(xs),
        "network_feature_hash": sha256(network),
        "pipeline_feature_hash": sha256(pipeline),
    }


def worker_labels(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Worker-side labels over all rows; index 0 of the prior label is unscorable."""
    current, prior = diag.d1_labels(np.asarray(x, dtype=np.int8))
    return {"A": current, "network_positive": prior, "pipeline": prior}


# ------------------------------------------------------------ bootstrap
def block_lower_bound(correctness: np.ndarray, seed: int, q: int) -> float:
    """ebb90e74 section 4.5 interval with the namespace-31 component [20261001, seed, 31, 60, q]."""
    if q not in (1, 2):
        raise ValueError("q must be 1 (A) or 2 (network positive control)")
    key = {1: "D1R_block_bootstrap_A", 2: "D1R_block_bootstrap_network_positive_control"}[q]
    return block_lower_bound_from_generator(correctness, rng(entropy_table(seed)[key]))


def block_lower_bound_from_generator(
    correctness: np.ndarray,
    g: np.random.Generator,
    n_resamples: int = diag.D1_BLOCK_RESAMPLES,
    block_length: int = diag.D1_BLOCK_LENGTH,
    chunk_size: int = 100,
) -> float:
    """Arithmetic identical to ``diag.circular_block_lower_bound`` (tested bit-for-bit)."""
    v = np.asarray(correctness, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("correctness must be one non-empty vector")
    n_blocks = math.ceil(len(v) / block_length)
    reps = np.empty(n_resamples, dtype=np.float64)
    offsets = np.arange(block_length, dtype=np.int64)
    done = 0
    while done < n_resamples:
        batch = min(chunk_size, n_resamples - done)
        starts = g.integers(0, len(v), size=(batch, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :]) % len(v)
        samples = v[idx].reshape(batch, -1)[:, : len(v)]
        reps[done : done + batch] = samples.mean(axis=1)
        done += batch
    return float(np.percentile(reps, 5, method="inverted_cdf"))


# ------------------------------------------------------------ one seed
class UnauthorizedD1RSeed(RuntimeError):
    """A D1R seed was requested outside the authorized runner path."""


def _decoder_record(name: str, fitted: diag.FittedDecoder, correctness: np.ndarray, train_y: np.ndarray, test_y: np.ndarray, seed: int) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "accuracy": float(correctness.mean()),
        "scored": int(correctness.size),
        "decoder_config": dict(DECODER_CONFIG),
        "train_label_sha256": sha256(np.ascontiguousarray(train_y, dtype=np.int8)),
        "test_label_sha256": sha256(np.ascontiguousarray(test_y, dtype=np.int8)),
        "mean": fitted.mean.tolist(),
        "scale": fitted.scale.tolist(),
        "normalization_sha256": diag.normalization_hash(fitted.mean, fitted.scale),
        "coefficients": fitted.coefficients.tolist(),
        "intercept": fitted.intercept.tolist(),
        "coefficients_sha256": diag.coefficients_hash(fitted.coefficients, fitted.intercept),
        "n_iter": [int(v) for v in fitted.n_iter.tolist()],
        "converged": bool(np.all(fitted.n_iter < MAX_ITER)),
    }
    if name in BOOTSTRAP_Q:
        rec["lower_95"] = block_lower_bound(correctness, seed, BOOTSTRAP_Q[name])
    return rec


def run_d1r_seed(
    seed: int,
    *,
    decoders: Sequence[str] = DECODERS,
    authorized_run: bool = False,
    n_train: int = PHASE_STEPS,
    n_test: int = EVAL_LEN,
    capture: Callable[..., Dict[str, object]] = capture_stream,
) -> Dict[str, object]:
    """One D1R seed row. The CLI owns run authorization.

    - Every seed in 2100..2119 is refused unless ``authorized_run`` is True,
      which only the runner's job wrapper passes after preflight.
    - ``decoders`` restricts which decoders are fitted; the seed-4242
      shakedown fits ``("pipeline",)`` only (spec 11 item 15).
    - Keyword lengths and ``capture`` exist for construction tests; the
      runner never passes them and the package validator rejects other lengths.
    """
    if seed in D1R_SEEDS and not authorized_run:
        raise UnauthorizedD1RSeed(f"seed {seed} is a D1R seed; only the authorized runner may execute it")
    unknown = [d for d in decoders if d not in DECODERS]
    if unknown or len(set(decoders)) != len(decoders) or not decoders:
        raise ValueError(f"decoders must be a non-empty subset of {DECODERS}, got {decoders}")
    w1, w2 = initial_weights(seed)
    x_train = a_stream(seed, True, n_train)
    x_test = a_stream(seed, False, n_test)
    train = capture(x_train, w1, w2)
    test = capture(x_test, w1, w2)
    tr = slice(WARMUP, None)
    if train["network"][tr].shape[0] != n_train - WARMUP or test["network"][tr].shape[0] != n_test - WARMUP:
        raise AssertionError("D1R row-count invariant failed")
    ltr, lte = worker_labels(x_train), worker_labels(x_test)
    out: Dict[str, object] = {
        "diagnostic": "D1R",
        "seed": seed,
        "train_input_hash": train["input_hash"],
        "test_input_hash": test["input_hash"],
        "train_network_feature_hash": train["network_feature_hash"],
        "test_network_feature_hash": test["network_feature_hash"],
        "train_pipeline_feature_hash": train["pipeline_feature_hash"],
        "test_pipeline_feature_hash": test["pipeline_feature_hash"],
        "initial_weights_sha256": sha256(w1, w2),
        "train_rows": int(train["network"][tr].shape[0]),
        "test_rows": int(test["network"][tr].shape[0]),
        "warmup_rows": WARMUP,
        "network_feature_dim": int(train["network"].shape[1]),
        "pipeline_feature_dim": int(train["pipeline"].shape[1]),
        "weights_constant": bool(train["weights_constant"] and test["weights_constant"]),
        "nesting_ok": bool(train["nesting_ok"] and test["nesting_ok"]),
        "reward_computed": False,
        "decoders": {},
    }
    for name in DECODERS:
        if name not in decoders:
            continue
        key = "pipeline" if name == "pipeline" else "network"
        fitted = diag.fit_d1_decoder(train[key][tr], ltr[name][tr])
        pred = fitted.predict(test[key][tr])
        correctness = pred == lte[name][tr]
        out["decoders"][name] = _decoder_record(name, fitted, correctness, ltr[name][tr], lte[name][tr], seed)  # type: ignore[index]
    return out


def object_hashes(seed: int, *, n_train: int = PHASE_STEPS, n_test: int = EVAL_LEN) -> Dict[str, str]:
    """Coordinator identities, constructed with no network transition.

    Label hashes are derived here by direct slicing of the regenerated
    namespace-31 streams, independently of ``worker_labels`` and the worker:
    ``pipeline`` and ``network_positive`` use x[1999:-1] (x_(t-1)), ``A``
    uses x[2000:] (x_t), for train and held-out (spec 6, 7 item 3, 11 item 6).
    """
    x_train = np.ascontiguousarray(a_stream(seed, True, n_train), dtype=np.int8)
    x_test = np.ascontiguousarray(a_stream(seed, False, n_test), dtype=np.int8)
    w = WARMUP
    prior_train = sha256(np.ascontiguousarray(x_train[w - 1 : -1]))
    prior_test = sha256(np.ascontiguousarray(x_test[w - 1 : -1]))
    return {
        "initial_weights_sha256": sha256(*initial_weights(seed)),
        "train_input_hash": sha256(x_train),
        "test_input_hash": sha256(x_test),
        "pipeline_train_label_sha256": prior_train,
        "pipeline_test_label_sha256": prior_test,
        "network_positive_train_label_sha256": prior_train,
        "network_positive_test_label_sha256": prior_test,
        "A_train_label_sha256": sha256(np.ascontiguousarray(x_train[w:])),
        "A_test_label_sha256": sha256(np.ascontiguousarray(x_test[w:])),
    }


IDENTITY_FIELDS = (
    "initial_weights_sha256",
    "train_input_hash",
    "test_input_hash",
    "pipeline_train_label_sha256",
    "pipeline_test_label_sha256",
    "network_positive_train_label_sha256",
    "network_positive_test_label_sha256",
    "A_train_label_sha256",
    "A_test_label_sha256",
)


# ------------------------------------------------------------ validation
ROW_FIELDS = (
    "diagnostic",
    "seed",
    "train_input_hash",
    "test_input_hash",
    "train_network_feature_hash",
    "test_network_feature_hash",
    "train_pipeline_feature_hash",
    "test_pipeline_feature_hash",
    "initial_weights_sha256",
    "train_rows",
    "test_rows",
    "warmup_rows",
    "network_feature_dim",
    "pipeline_feature_dim",
    "weights_constant",
    "nesting_ok",
    "reward_computed",
    "decoders",
)
DECODER_FIELDS = (
    "accuracy",
    "scored",
    "decoder_config",
    "train_label_sha256",
    "test_label_sha256",
    "mean",
    "scale",
    "normalization_sha256",
    "coefficients",
    "intercept",
    "coefficients_sha256",
    "n_iter",
    "converged",
)
NETWORK_DECODER_FIELDS = DECODER_FIELDS + ("lower_95",)
DECODER_WIDTH = {"pipeline": PIPELINE_FEATURE_DIM, "network_positive": NETWORK_FEATURE_DIM, "A": NETWORK_FEATURE_DIM}


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_decoder(
    name: str,
    d: object,
    expected: Optional[Mapping[str, str]],
    tag: str,
    *,
    test_rows: int = TEST_ROWS,
) -> List[str]:
    v: List[str] = []
    dtag = f"{tag} decoder={name}"
    if not isinstance(d, Mapping):
        return [f"{dtag}: not a mapping"]
    fields = NETWORK_DECODER_FIELDS if name in NETWORK_DECODERS else DECODER_FIELDS
    missing = [k for k in fields if k not in d]
    if missing:
        return [f"{dtag}: missing field(s) {missing}"]
    unknown = sorted(set(d) - set(fields), key=str)
    if unknown:
        v.append(f"{dtag}: undeclared field(s) {unknown}")
    if not diag._finite_unit(d["accuracy"]):
        v.append(f"{dtag}: accuracy is not a finite float in [0,1]")
    elif abs(d["accuracy"] * test_rows - round(d["accuracy"] * test_rows)) > 1e-6:
        v.append(f"{dtag}: accuracy is not a count over {test_rows} scored rows")
    if name in NETWORK_DECODERS and not diag._finite_unit(d["lower_95"]):
        v.append(f"{dtag}: lower_95 is not a finite float in [0,1]")
    if d["scored"] != test_rows or not _is_int(d["scored"]):
        v.append(f"{dtag}: scored={d['scored']!r}, expected {test_rows}")
    if not isinstance(d["decoder_config"], Mapping) or dict(d["decoder_config"]) != DECODER_CONFIG:
        v.append(f"{dtag}: decoder configuration differs from spec section 2")
    for split in ("train", "test"):
        key = f"{split}_label_sha256"
        if not diag._is_hash(d[key]):
            v.append(f"{dtag}: {key} is not a SHA-256 hex digest")
        elif expected is not None and d[key] != expected.get(f"{name}_{key}"):
            want = "x_t" if name == "A" else "x_(t-1)"
            v.append(f"{dtag}: {key} differs from coordinator-derived {want} label identity")
    width = DECODER_WIDTH[name]
    if not (diag._finite_vector(d["mean"], width) and diag._finite_vector(d["scale"], width)):
        v.append(f"{dtag}: training normalization mean/scale not finite length-{width}")
    else:
        if not (np.asarray(d["scale"], dtype=np.float64) > 0.0).all():
            v.append(f"{dtag}: non-positive training scale")
        if d["normalization_sha256"] != diag.normalization_hash(d["mean"], d["scale"]):
            v.append(f"{dtag}: normalization_sha256 does not match stored mean/scale")
    try:
        coef = np.asarray(d["coefficients"], dtype=np.float64)
        icpt = np.asarray(d["intercept"], dtype=np.float64)
        coef_ok = coef.shape == (1, width) and icpt.shape == (1,) and bool(np.isfinite(coef).all() and np.isfinite(icpt).all())
    except (TypeError, ValueError):
        coef_ok = False
    if not coef_ok:
        v.append(f"{dtag}: coefficients/intercept not finite (1,{width})/(1,)")
    elif d["coefficients_sha256"] != diag.coefficients_hash(d["coefficients"], d["intercept"]):
        v.append(f"{dtag}: coefficients_sha256 does not match stored coefficients")
    n_iter = d["n_iter"]
    ok = isinstance(n_iter, list) and len(n_iter) == 1 and all(_is_int(i) for i in n_iter)
    if not ok or not all(0 < i < MAX_ITER for i in n_iter):
        v.append(f"{dtag}: solver iterations {n_iter!r} not strictly below max_iter")
    if d["converged"] is not True:
        v.append(f"{dtag}: decoder did not converge")
    return v


def validate_row(
    row: Mapping[str, object],
    expected: Optional[Mapping[str, str]],
    *,
    train_rows: int = TRAIN_ROWS,
    test_rows: int = TEST_ROWS,
    decoders: Sequence[str] = DECODERS,
) -> List[str]:
    """Every section 6.1 / 7 INVALID invariant for one D1R seed row. Empty = valid.

    ``decoders`` may be narrowed only by the seed-4242 shakedown (pipeline
    only, spec 11 item 15) and length overrides only by short construction
    tests; ``validate_package`` always uses all three decoders and full lengths.
    """
    expected_decoders = tuple(decoders)
    if not isinstance(row, Mapping):
        return ["D1R: row is not a mapping"]
    seed = row.get("seed")
    tag = f"D1R seed={seed}"
    v: List[str] = []
    if row.get("status", "ok") != "ok":
        v.append(f"{tag}: job status {row.get('status')!r}")
    missing = [k for k in ROW_FIELDS if k not in row]
    if missing:
        return v + [f"{tag}: missing field(s) {missing}"]
    unknown = sorted(set(row) - set(ROW_FIELDS) - {"status", "worker_pid"}, key=str)
    if unknown:
        v.append(f"{tag}: undeclared field(s) {unknown}")
    if row["diagnostic"] != "D1R":
        v.append(f"{tag}: diagnostic label {row['diagnostic']!r}")
    if not _is_int(seed):
        v.append(f"{tag}: seed is not int")
    hashes = (
        "train_input_hash",
        "test_input_hash",
        "train_network_feature_hash",
        "test_network_feature_hash",
        "train_pipeline_feature_hash",
        "test_pipeline_feature_hash",
        "initial_weights_sha256",
    )
    for k in hashes:
        if not diag._is_hash(row[k]):
            v.append(f"{tag}: {k} is not a SHA-256 hex digest")
    for a, b in (("train_input_hash", "test_input_hash"), ("train_network_feature_hash", "test_network_feature_hash"), ("train_pipeline_feature_hash", "test_pipeline_feature_hash")):
        if row[a] == row[b]:
            v.append(f"{tag}: {a} equals {b} (split not disjoint)")
    if expected is None:
        v.append(f"{tag}: no coordinator identity supplied")
    else:
        absent = [k for k in IDENTITY_FIELDS if not diag._is_hash(expected.get(k))]
        if absent:
            v.append(f"{tag}: coordinator identity lacks {absent}")
        for k in ("initial_weights_sha256", "train_input_hash", "test_input_hash"):
            if row[k] != expected.get(k):
                v.append(f"{tag}: {k} differs from coordinator identity")
    if row["weights_constant"] is not True:
        v.append(f"{tag}: frozen W1/W2 were not bitwise constant")
    if row["nesting_ok"] is not True:
        v.append(f"{tag}: count20 != sum of lag blocks at some captured row (section 7 item 7)")
    if row["reward_computed"] is not False:
        v.append(f"{tag}: reward_computed must be False under F0")
    want = (
        ("train_rows", train_rows),
        ("test_rows", test_rows),
        ("warmup_rows", WARMUP),
        ("network_feature_dim", NETWORK_FEATURE_DIM),
        ("pipeline_feature_dim", PIPELINE_FEATURE_DIM),
    )
    for k, w in want:
        if row[k] != w or not _is_int(row[k]):
            v.append(f"{tag}: {k}={row[k]!r}, expected {w}")
    decs = row["decoders"]
    if not isinstance(decs, Mapping) or set(decs) != set(expected_decoders):
        return v + [f"{tag}: decoders must be exactly {list(expected_decoders)}"]
    for name in expected_decoders:
        v.extend(validate_decoder(name, decs[name], expected, tag, test_rows=test_rows))
    try:
        if set(expected_decoders) == set(DECODERS):
            if decs["A"]["train_label_sha256"] == decs["network_positive"]["train_label_sha256"]:
                v.append(f"{tag}: A and network-positive training labels are identical")
            for split in ("train", "test"):
                key = f"{split}_label_sha256"
                if decs["pipeline"][key] != decs["network_positive"][key]:
                    v.append(f"{tag}: pipeline and network-positive {split} labels differ (must be the identical x_(t-1) vector)")
    except (KeyError, TypeError):
        pass
    return v


def validate_package(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[Mapping[int, Mapping[str, str]]],
    seeds: Sequence[int] = D1R_SEEDS,
) -> List[str]:
    """Section 7 items 1-7 over the whole package: exact seed set, sorted, each row valid."""
    v: List[str] = []
    seen = [r.get("seed") if isinstance(r, Mapping) else None for r in rows]
    if len(rows) != len(seeds):
        v.append(f"D1R: {len(rows)} rows, expected {len(seeds)}")
    dupes = sorted({s for s in seen if seen.count(s) > 1}, key=str)
    if dupes:
        v.append(f"D1R: duplicate seed rows {dupes}")
    if set(seen) != set(seeds):
        v.append(f"D1R: seed set differs; missing {sorted(set(seeds) - set(seen), key=str)}, unexpected {sorted(set(seen) - set(seeds), key=str)}")
    if seen != sorted(seen, key=lambda s: (not _is_int(s), s if _is_int(s) else 0)):
        v.append("D1R: rows are not sorted by seed")
    if expected is None:
        v.append("D1R: coordinator identities not supplied")
    for row in rows:
        s = row.get("seed") if isinstance(row, Mapping) else None
        v.extend(validate_row(row, None if expected is None else expected.get(s)))  # type: ignore[arg-type]
    return v


# ------------------------------------------------------------ readouts and branch
def section46(acc: Sequence[float], lower: Sequence[float]) -> Dict[str, object]:
    """ebb90e74 section 4.6 predicate, restated in D1R section 6.1."""
    t = THRESHOLDS
    median = float(np.median(np.asarray(acc, dtype=np.float64)))
    count = int((np.asarray(lower, dtype=np.float64) > t["lower_bound_reference"]).sum())
    if median >= t["pass_median_accuracy"] and count >= t["pass_min_lower_bounds_gt_half"]:
        status = "PASS"
    elif median < t["fail_median_below"]:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    return {"median_accuracy": median, "lower_bounds_gt_half": count, "status": status}


def pipeline_readout(acc: Sequence[float]) -> Dict[str, object]:
    median = float(np.median(np.asarray(acc, dtype=np.float64)))
    return {"median_accuracy": median, "status": "PASS" if median >= THRESHOLDS["pipeline_pass_median_accuracy"] else "INVALID_PIPELINE"}


BRANCH_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("STOP_D1R_INVALID",
     "Collision-probe failure, any section 7 invariant failure, any INVALID decoder or seed, or incomplete package",
     "STOP. Return to implementation/spec review. No science reading. No D2-D4, no r4."),
    ("STOP_D1R_INVALID_PIPELINE",
     "Row 1 not met and pipeline median < 0.99",
     "STOP. Implementation review. A code defect, not science. No science reading of the network decoders. No D2-D4, no r4."),
    ("CLOSE_D1_REPRESENTATION_LIMITED",
     "Pipeline PASS, network_positive FAIL or INCONCLUSIVE",
     "D1 closes as REPRESENTATION_LIMITED (reading (b)). No further D1 harness repair is permitted. Return to Kendrick with a roadmap decision card on task presentation and encoding. No D2-D4, no r4."),
    ("D1R_A_FAIL_EBB90E74_ROW3",
     "Pipeline PASS, network_positive PASS, A FAIL",
     "ebb90e74 section 10 row 3: the specified linear decoder cannot recover A under the fixed split. Task/architecture pairing needs roadmap revision. Return to Kendrick. No r4 readout or reward change. No D2-D4."),
    ("D1R_A_INCONCLUSIVE_EBB90E74_ROW4",
     "Pipeline PASS, network_positive PASS, A INCONCLUSIVE",
     "ebb90e74 section 10 row 4: evidence is not decision-complete. Return to Kendrick/spec review. No downstream diagnostic and no r4."),
    ("D1R_PASS_D2_D4_ELIGIBLE",
     "Pipeline PASS, network_positive PASS, A PASS",
     "D1 is PASS (by D1R). D2-D4 may proceed only per ebb90e74 section 8 steps 3-6 under a separate run authorization; their outcomes are governed by ebb90e74 section 10 row 1 (any D2-D4 INVALID) and rows 5 onward, unchanged. No r4 from D1R alone."),
)
BRANCH_INDEX = {code: i + 1 for i, (code, _, _) in enumerate(BRANCH_TABLE)}
STATUS_TO_BRANCH = {
    "INVALID": "STOP_D1R_INVALID",
    "INVALID_PIPELINE": "STOP_D1R_INVALID_PIPELINE",
    "REPRESENTATION_LIMITED": "CLOSE_D1_REPRESENTATION_LIMITED",
    "FAIL": "D1R_A_FAIL_EBB90E74_ROW3",
    "INCONCLUSIVE": "D1R_A_INCONCLUSIVE_EBB90E74_ROW4",
    "PASS": "D1R_PASS_D2_D4_ELIGIBLE",
}


def branch(status: Optional[str]) -> Dict[str, object]:
    code = STATUS_TO_BRANCH.get(status or "INVALID", "STOP_D1R_INVALID")
    _, condition, action = BRANCH_TABLE[BRANCH_INDEX[code] - 1]
    return {"code": code, "table_row": BRANCH_INDEX[code], "condition": condition, "action": action}


def summarize(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[Mapping[int, Mapping[str, str]]],
    probe_pass: bool,
    seeds: Sequence[int] = D1R_SEEDS,
) -> Dict[str, object]:
    """Section 6.3 status and section 9 branch, mechanically and fail-closed.

    The strict package validator and the collision-probe gate run first; any
    finding returns INVALID with no readout computed. Under INVALID_PIPELINE
    the network readouts are not computed (section 9 row 2: no science reading).
    """
    violations = validate_package(rows, expected, seeds)
    if not probe_pass:
        violations = ["D1R: collision probe not verified PASS"] + violations
    empty = {"pipeline": None, "network_positive": None, "A": None}
    if violations:
        return {"status": "INVALID", "valid": False, "rows_schema_valid": False, "violations": violations, "readouts": empty, "branch": branch("INVALID")}
    pipe = pipeline_readout([r["decoders"]["pipeline"]["accuracy"] for r in rows])  # type: ignore[index]
    readouts: Dict[str, object] = {"pipeline": pipe, "network_positive": None, "A": None}
    if pipe["status"] != "PASS":
        status = "INVALID_PIPELINE"
    else:
        for name in NETWORK_DECODERS:
            readouts[name] = section46(
                [r["decoders"][name]["accuracy"] for r in rows],  # type: ignore[index]
                [r["decoders"][name]["lower_95"] for r in rows],  # type: ignore[index]
            )
        positive = readouts["network_positive"]["status"]  # type: ignore[index]
        status = "REPRESENTATION_LIMITED" if positive != "PASS" else readouts["A"]["status"]  # type: ignore[index]
        readouts["A"]["decision_bearing"] = positive == "PASS"  # type: ignore[index]
    return {"status": status, "valid": True, "rows_schema_valid": True, "violations": [], "readouts": readouts, "branch": branch(status)}


# ------------------------------------------------------------ effective parameters
def effective_parameters() -> Dict[str, object]:
    """D1R effective parameter block (spec 12). Network values are read back
    from a network constructed on the non-diagnostic fixture seed; no
    transition is simulated."""
    net = diag.build_network(*initial_weights(FIXTURE_SEED))
    hidden, output = net.neurons
    acc_h, acc_u = LagFeatureAccumulator(N_HIDDEN), LagFeatureAccumulator(N_INPUT)
    return {
        "spec_commit": SPEC_COMMIT,
        "spec_sha256": SPEC_SHA256,
        "ebb90e74_sha256": EBB90E74_SHA256,
        "stage2_spec_sha256": FROZEN_STAGE2_SHA256,
        "layer_sizes": list(net.layer_sizes),
        "dt_ms": float(net.dt),
        "tau_m_ms": [float(hidden.tau_m), float(output.tau_m)],
        "v_rest_mv": [float(hidden.v_rest), float(output.v_rest)],
        "v_reset_mv": [float(hidden.v_reset), float(output.v_reset)],
        "v_thresh_mv": [float(hidden.v_thresh), float(output.v_thresh)],
        "hidden_tau_a_ms": float(hidden.tau_a),
        "hidden_beta_a_mv": float(hidden.beta_a),
        "w1_init": "U(-10,10) mV, [20261001, seed, 31, 1] first draw",
        "w2_O1_init": "U(0,10) mV, [20261001, seed, 31, 1] second draw",
        "w2_O0_init": "U(0,10) mV, [20261001, seed, 31, 21]",
        "noise_p": float(diag.NOISE_P),
        "train_stream_len": PHASE_STEPS,
        "test_stream_len": EVAL_LEN,
        "warmup_rows": WARMUP,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "seeds": list(D1R_SEEDS),
        "namespace": NAMESPACE,
        "root_entropy": ROOT_ENTROPY,
        "K": K,
        "network_feature_dim": acc_h.width,
        "pipeline_feature_dim": acc_u.width,
        "feature_map": FEATURE_MAP,
        "pipeline_map": PIPELINE_MAP,
        "trace20_decay": diag.TRACE20_DECAY,
        "trace25_decay": diag.TRACE25_DECAY,
        "count_window_steps": diag.FeatureAccumulator().buffer.shape[0],
        "decoder": dict(DECODER_CONFIG),
        "standardization": "training mean and population std (ddof=0); zero-variance scale 1.0",
        "prediction_threshold": 0.5,
        "block_length": diag.D1_BLOCK_LENGTH,
        "block_resamples": diag.D1_BLOCK_RESAMPLES,
        "block_percentile": 5,
        "bootstrap_entropy": "[20261001, seed, 31, 60, q]; q=1 A, q=2 network_positive",
        "decoders": list(DECODERS),
        "labels": {"pipeline": "x_(t-1) = x[1999:-1]", "network_positive": "x_(t-1) = x[1999:-1]", "A": "x_t = x[2000:]"},
        "max_workers": MAX_WORKERS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "thresholds": dict(THRESHOLDS),
    }
