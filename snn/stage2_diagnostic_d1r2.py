"""Frozen Stage 2 diagnostic D1R2 construction (final D1 harness attempt).

Implements docs/STAGE2_DIAGNOSTIC_D1R2_SPEC.md at f3d52ee without editing the
approved D1 or D1R paths. Importing this module executes nothing, and every
D1R2 seed is refused unless the separately authorized runner calls it.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from snn import stage2_diagnostic as diag
from snn.stage2 import ROOT_ENTROPY, sha256
from snn.stage2_r3 import EVAL_LEN, ONEHOT, PHASE_STEPS, init_state

SPEC_COMMIT = "f3d52ee91b7ec87f4e665f6ce80208954c30c7a2"
SPEC_SHA256 = "e5a6a5f4eb7a7adafd9255ac6504593e45be51eefbfa92ce24c63b70fafb82a2"
CARRIED_D1R_SHA256 = "c1f03efefd203f65b0409793e1adc848ff22a3522d8692d634918efd0870a743"
EBB90E74_SHA256 = diag.SPEC_SHA256
FROZEN_STAGE2_SHA256 = diag.FROZEN_STAGE2_SHA256

NAMESPACE = 32
D1R2_SEEDS = tuple(range(2200, 2220))
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
DECODER_CONFIG = {**diag.D1_DECODER_CONFIG, "max_iter": 20_000}
MAX_ITER = 20_000
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
    """Fresh generator; the single construction point for every D1R2 object."""
    return np.random.default_rng(np.random.SeedSequence(list(entropy)))


def entropy_table(seed: int) -> Dict[str, Tuple[int, ...]]:
    """Spec section 3.1, verbatim, in the section 3.3 probe's enumeration order.

    tools/stage2_diagnostic_d1r2_ss_probe.py enumerates exactly these names and
    tuples (test_d1r2_11_12 checks equality), so the retained identity-state
    hash pins the implementation's identities."""
    return {
        "weights_w1_w2_o1": (ROOT_ENTROPY, seed, NAMESPACE, 1),
        "weights_w2_o0": (ROOT_ENTROPY, seed, NAMESPACE, 21),
        "A_train": (ROOT_ENTROPY, seed, NAMESPACE, 2),
        "A_test": (ROOT_ENTROPY, seed, NAMESPACE, 5),
        "D1R2_block_bootstrap_A": (ROOT_ENTROPY, seed, NAMESPACE, 60, 1),
        "D1R2_block_bootstrap_network_positive_control": (ROOT_ENTROPY, seed, NAMESPACE, 60, 2),
    }


def initial_weights(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """r3 distributions and draw order (as ebb90e74 ``initial_weights``) under namespace 32."""
    t = entropy_table(seed)
    g = rng(t["weights_w1_w2_o1"])
    w1 = g.uniform(-10.0, 10.0, size=(20, 2)).astype(np.float64)
    w2_o1 = g.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64)
    h = rng(t["weights_w2_o0"])
    w2_o0 = h.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64)
    return w1, np.concatenate((w2_o1, w2_o0), axis=0)


def a_stream(seed: int, train: bool, n: Optional[int] = None) -> np.ndarray:
    """r3 section 7.1 order-2 A stream (approved ``_roll_a``) with the namespace-32 component."""
    key = "A_train" if train else "A_test"
    default_n = PHASE_STEPS if train else EVAL_LEN
    return diag._roll_a(rng(entropy_table(seed)[key]), default_n if n is None else n)


# ------------------------------------------------------------ feature maps
class LagFeatureAccumulator:
    """The one D1R2 (= D1R) feature operator, parameterized only by width ``n``.

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
    """ebb90e74 section 4.5 interval with the namespace-32 component [20261001, seed, 32, 60, q]."""
    if q not in (1, 2):
        raise ValueError("q must be 1 (A) or 2 (network positive control)")
    key = {1: "D1R2_block_bootstrap_A", 2: "D1R2_block_bootstrap_network_positive_control"}[q]
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


# ------------------------------------------------------------ durable diagnostics
SLOT_FIELDS = (
    "attempted", "completed", "stop_reason", "n_iter", "converged",
    "convergence_warning", "warnings", "fit_error",
)
STOP_REASONS = {
    "preflight_failure", "seed_not_started", "not_attempted_upstream_failure",
    "worker_lost", "terminated_wall_stop", "terminated_rss_abort",
    "fit_exception", "iteration_limit", "evaluation_limit",
    "other_convergence_warning", "score_exception", "converged",
}
PHASES = {"standardize", "fit", "predict_score"}
NON_DECODER_PHASES = {"streams", "capture", "labels"}


def _message_head(value: object) -> str:
    return "\n".join(line[:300] for line in str(value).splitlines()[:2])


def _category_name(category: type) -> str:
    return f"{category.__module__}.{category.__qualname__}"


def append_ledger_event(path: Optional[Path], event: Mapping[str, object]) -> None:
    """One complete O_APPEND write followed by fsync, with no user-space buffer."""
    if path is None:
        return
    data = (json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError("short progress-ledger write")
        os.fsync(fd)
    finally:
        os.close(fd)


def is_convergence_category(category: object) -> bool:
    """Spec 6.3: ``sklearn.exceptions.ConvergenceWarning`` or any subclass (real issubclass)."""
    return isinstance(category, type) and issubclass(category, ConvergenceWarning)


def _resolve_category(name: object) -> Optional[type]:
    """Resolve a fully qualified category name to its class; None if it does not resolve exactly.

    A submodule is imported only when its top-level package is already loaded
    in this process (for example a scipy or sklearn submodule), so a benign
    library warning is never misclassified merely because its submodule was
    not yet imported by the coordinator, and no new top-level package is ever
    imported from a ledger string. Names that still do not resolve (classes
    defined in a local scope, a foreign ``__main__``) return None.
    """
    if not isinstance(name, str) or not name or "<" in name:
        return None
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:cut])
        obj: object = sys.modules.get(module_name)
        if obj is None and parts[0] in sys.modules and parts[0] != "__main__":
            try:
                obj = importlib.import_module(module_name)
            except Exception:
                obj = None
        if obj is None:
            continue
        try:
            for attr in parts[cut:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, type) and _category_name(obj) == name:
            return obj
    return None


def classify_category_name(name: object) -> Optional[bool]:
    """issubclass(resolved class, ConvergenceWarning); None when the name does not resolve."""
    cls = _resolve_category(name)
    return None if cls is None else is_convergence_category(cls)


def _entry_flags(entries: Sequence[Mapping[str, object]], recorded: Optional[Sequence[object]] = None) -> List[bool]:
    """Per-entry ConvergenceWarning classification (spec 6.3: the class or any subclass).

    ``recorded`` holds the exact ``issubclass(category, ConvergenceWarning)``
    result the emitting hook computed on the real class (worker memory, or the
    durable ledger ``warning`` event); when it is a boolean it is used as is.
    Without a recorded boolean the fully qualified name is resolved to its
    class and tested with ``issubclass``; a name that does not resolve is
    classified as a convergence warning, so an unclassifiable warning can
    only make a slot non-converged (fail closed).
    """
    if recorded is not None and len(recorded) != len(entries):
        raise ValueError("warning classifications do not align with warning entries")
    out: List[bool] = []
    for index, entry in enumerate(entries):
        flag = None if recorded is None else recorded[index]
        if isinstance(flag, bool):
            out.append(flag)
            continue
        resolved = classify_category_name(entry.get("category") if isinstance(entry, Mapping) else None)
        out.append(True if resolved is None else resolved)
    return out


def _warning_call(
    fn: Callable[[], object], *, seed: int, decoder: Optional[str], phase: str,
    ledger_path: Optional[Path], captured: List[Dict[str, object]],
    flags: Optional[List[bool]] = None,
) -> object:
    def showwarning(message, category, filename, lineno, file=None, line=None):
        entry = {"category": _category_name(category), "phase": phase, "message_head": _message_head(message)}
        is_cw = is_convergence_category(category)
        ledger_entry = dict(entry)
        if phase == "predict_score":
            ledger_entry["message_head"] = None
        append_ledger_event(ledger_path, {
            "event": "warning", "seed": seed, "decoder": decoder, **ledger_entry, "convergence_warning": is_cw,
        })
        captured.append(entry)
        if flags is not None:
            flags.append(is_cw)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        old = warnings.showwarning
        warnings.showwarning = showwarning
        try:
            return fn()
        finally:
            warnings.showwarning = old


def redact_slot_scoring_text(slot: Mapping[str, object]) -> Dict[str, object]:
    """Deep copy of a slot with every predict_score ``message_head`` set to null (spec 6.3)."""
    out = json.loads(json.dumps(dict(slot)))
    for warning in out.get("warnings") or []:
        if isinstance(warning, dict) and warning.get("phase") == "predict_score":
            warning["message_head"] = None
    error = out.get("fit_error")
    if isinstance(error, dict) and error.get("phase") == "predict_score":
        error["message_head"] = None
    return out


def empty_slot(stop_reason: str, *, attempted: bool = False, fit_error: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    return {
        "attempted": attempted,
        "completed": False,
        "stop_reason": stop_reason,
        "n_iter": None,
        "converged": False,
        "convergence_warning": False,
        "warnings": [],
        "fit_error": fit_error,
    }


def _convergence_warnings(
    entries: Sequence[Mapping[str, object]], recorded: Optional[Sequence[object]] = None,
) -> List[Mapping[str, object]]:
    """Entries whose category is ConvergenceWarning or any subclass (spec 6.3)."""
    return [w for w, flag in zip(entries, _entry_flags(entries, recorded)) if flag]


def _warning_stop(
    entries: Sequence[Mapping[str, object]], n_iter: Optional[int], recorded: Optional[Sequence[object]] = None,
) -> str:
    conv = _convergence_warnings(entries, recorded)
    messages = "\n".join(str(w.get("message_head") or "") for w in conv).upper()
    if n_iter is not None and n_iter >= MAX_ITER or "ITERATIONS REACHED LIMIT" in messages:
        return "iteration_limit"
    if "F,G EVALUATIONS EXCEEDS LIMIT" in messages:
        return "evaluation_limit"
    if conv:
        return "other_convergence_warning"
    return "converged"


def fit_decoder_slot(
    name: str,
    train_features: np.ndarray,
    train_y: np.ndarray,
    test_features: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    *,
    ledger_path: Optional[Path] = None,
    model_factory: Callable[..., object] = LogisticRegression,
    score: bool = True,
) -> Tuple[Dict[str, object], Optional[diag.FittedDecoder], Optional[np.ndarray]]:
    """Fit one fixed decoder and return its total section-6.3 slot."""
    append_ledger_event(ledger_path, {"event": "scope_open", "seed": seed, "decoder": name})
    captured: List[Dict[str, object]] = []
    cw_flags: List[bool] = []  # exact issubclass(category, ConvergenceWarning), aligned with ``captured``
    fitted: Optional[diag.FittedDecoder] = None
    correctness: Optional[np.ndarray] = None
    fit_error: Optional[Dict[str, object]] = None
    n_iter: Optional[int] = None
    fit_returned = False
    score_returned = False
    phase = "standardize"
    try:
        def standardize():
            x = np.asarray(train_features, dtype=np.float64)
            y = np.asarray(train_y, dtype=np.int8)
            mean = x.mean(axis=0)
            raw = x.std(axis=0, ddof=0)
            scale = np.where(raw == 0.0, 1.0, raw)
            return x, y, mean, scale, (x - mean) / scale

        x, y, mean, scale, z = _warning_call(
            standardize, seed=seed, decoder=name, phase=phase,
            ledger_path=ledger_path, captured=captured, flags=cw_flags,
        )
        phase = "fit"
        model = model_factory(**DECODER_CONFIG)
        _warning_call(
            lambda: model.fit(z, y), seed=seed, decoder=name, phase=phase,
            ledger_path=ledger_path, captured=captured, flags=cw_flags,
        )
        fit_returned = True
        n_iter = int(np.asarray(model.n_iter_).reshape(-1)[0])
        fitted = diag.FittedDecoder(
            mean=np.asarray(mean).copy(), scale=np.asarray(scale).copy(),
            coefficients=np.asarray(model.coef_).copy(), intercept=np.asarray(model.intercept_).copy(),
            n_iter=np.asarray(model.n_iter_).copy(), model=model,
        )
        if score:
            phase = "predict_score"
            pred = _warning_call(
                lambda: fitted.predict(test_features), seed=seed, decoder=name, phase=phase,
                ledger_path=ledger_path, captured=captured, flags=cw_flags,
            )
            correctness = np.asarray(pred) == np.asarray(test_y, dtype=np.int8)
            score_returned = True
        else:
            # Item 16 is convergence-only and never calls predict.
            score_returned = True
    except Exception as exc:
        fit_error = {"exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}", "phase": phase, "message_head": _message_head(exc)}

    conv_warning = bool(_convergence_warnings(captured, cw_flags))
    if not fit_returned:
        reason = "fit_exception"
    else:
        reason = _warning_stop(captured, n_iter, cw_flags)
        if reason == "converged" and not score_returned:
            reason = "score_exception"
    completed = bool(fit_returned and score_returned)
    converged = bool(completed and n_iter is not None and n_iter < MAX_ITER and not conv_warning)
    slot = {
        "attempted": True,
        "completed": completed,
        "stop_reason": reason,
        "n_iter": n_iter if fit_returned else None,
        "converged": converged,
        "convergence_warning": conv_warning,
        "warnings": captured,
        "fit_error": fit_error,
    }
    ledger_slot = redact_slot_scoring_text(slot)
    append_ledger_event(ledger_path, {"event": "scope_close", "seed": seed, "decoder": name, "slot": ledger_slot})
    return slot, fitted, correctness


def fit_convergence_only(name: str, features: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, object]:
    """Item 16 helper: fit only, no prediction, accuracy, correctness, or bound."""
    slot, _, _ = fit_decoder_slot(name, features, labels, np.empty((0, features.shape[1])), np.empty(0), seed, score=False)
    return {"n_iter": slot["n_iter"], "converged": slot["converged"], "warnings": slot["warnings"]}


def run_fixture_network_convergence(n_train: int = PHASE_STEPS) -> Dict[str, Dict[str, object]]:
    """Section 11 item 16: fixture 4242, full train length, no prediction or accuracy.

    ``n_train`` below the section 2 length exists only for construction tests;
    the shakedown always uses the default full length.
    """
    seed = FIXTURE_SEED
    w1, w2 = initial_weights(seed)
    x_train = a_stream(seed, True, n_train)
    captured = capture_stream(x_train, w1, w2)
    labels = worker_labels(x_train)
    features = captured["network"][WARMUP:]
    return {
        name: fit_convergence_only(name, features, labels[name][WARMUP:], seed)
        for name in NETWORK_DECODERS
    }


# ------------------------------------------------------------ one seed
class UnauthorizedD1R2Seed(RuntimeError):
    """A D1R2 seed was requested outside the authorized runner path."""


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


def run_d1r2_seed(
    seed: int,
    *,
    decoders: Sequence[str] = DECODERS,
    authorized_run: bool = False,
    n_train: int = PHASE_STEPS,
    n_test: int = EVAL_LEN,
    capture: Callable[..., Dict[str, object]] = capture_stream,
    ledger_path: Optional[Path] = None,
    model_factory: Callable[..., object] = LogisticRegression,
) -> Dict[str, object]:
    """One total D1R2 seed row. Diagnostic seeds require runner authorization."""
    if seed in D1R2_SEEDS and not authorized_run:
        raise UnauthorizedD1R2Seed(f"seed {seed} is a D1R2 seed; only the authorized runner may execute it")
    unknown = [d for d in decoders if d not in DECODERS]
    if unknown or len(set(decoders)) != len(decoders) or not decoders:
        raise ValueError(f"decoders must be a non-empty subset of {DECODERS}, got {decoders}")

    append_ledger_event(ledger_path, {"event": "seed_started", "seed": seed})
    non_decoder_warnings: List[Dict[str, object]] = []
    seed_error: Optional[Dict[str, object]] = None
    phase = "streams"
    try:
        w1, w2 = initial_weights(seed)
        x_train = _warning_call(lambda: a_stream(seed, True, n_train), seed=seed, decoder=None, phase="streams", ledger_path=ledger_path, captured=non_decoder_warnings)
        x_test = _warning_call(lambda: a_stream(seed, False, n_test), seed=seed, decoder=None, phase="streams", ledger_path=ledger_path, captured=non_decoder_warnings)
        phase = "capture"
        train = _warning_call(lambda: capture(x_train, w1, w2), seed=seed, decoder=None, phase=phase, ledger_path=ledger_path, captured=non_decoder_warnings)
        test = _warning_call(lambda: capture(x_test, w1, w2), seed=seed, decoder=None, phase=phase, ledger_path=ledger_path, captured=non_decoder_warnings)
        phase = "labels"
        ltr = _warning_call(lambda: worker_labels(x_train), seed=seed, decoder=None, phase=phase, ledger_path=ledger_path, captured=non_decoder_warnings)
        lte = _warning_call(lambda: worker_labels(x_test), seed=seed, decoder=None, phase=phase, ledger_path=ledger_path, captured=non_decoder_warnings)
    except Exception as exc:
        seed_error = {"exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}", "phase": phase, "message_head": _message_head(exc)}
        append_ledger_event(ledger_path, {"event": "upstream_failure", "seed": seed, "seed_error": seed_error})
        slots = {name: empty_slot("not_attempted_upstream_failure") for name in DECODERS}
        return {
            "diagnostic": "D1R2", "seed": seed, "row_source": "worker", "seed_error": seed_error,
            "ledger_torn_tail": False, "non_decoder_warnings": non_decoder_warnings,
            "train_input_hash": None, "test_input_hash": None, "train_network_feature_hash": None,
            "test_network_feature_hash": None, "train_pipeline_feature_hash": None,
            "test_pipeline_feature_hash": None, "initial_weights_sha256": None,
            "train_rows": None, "test_rows": None, "warmup_rows": None,
            "network_feature_dim": None, "pipeline_feature_dim": None,
            "weights_constant": None, "nesting_ok": None, "reward_computed": None,
            "decoder_slots": slots, "decoders": {},
        }

    tr = slice(WARMUP, None)
    out: Dict[str, object] = {
        "diagnostic": "D1R2", "seed": seed, "row_source": "worker", "seed_error": None,
        "ledger_torn_tail": False, "non_decoder_warnings": non_decoder_warnings,
        "train_input_hash": train["input_hash"], "test_input_hash": test["input_hash"],
        "train_network_feature_hash": train["network_feature_hash"], "test_network_feature_hash": test["network_feature_hash"],
        "train_pipeline_feature_hash": train["pipeline_feature_hash"], "test_pipeline_feature_hash": test["pipeline_feature_hash"],
        "initial_weights_sha256": sha256(w1, w2),
        "train_rows": int(train["network"][tr].shape[0]), "test_rows": int(test["network"][tr].shape[0]),
        "warmup_rows": WARMUP, "network_feature_dim": int(train["network"].shape[1]),
        "pipeline_feature_dim": int(train["pipeline"].shape[1]),
        "weights_constant": bool(train["weights_constant"] and test["weights_constant"]),
        "nesting_ok": bool(train["nesting_ok"] and test["nesting_ok"]), "reward_computed": False,
        "decoder_slots": {}, "decoders": {},
    }
    for name in DECODERS:
        if name not in decoders:
            out["decoder_slots"][name] = empty_slot("not_attempted_upstream_failure")  # type: ignore[index]
            continue
        key = "pipeline" if name == "pipeline" else "network"
        slot, fitted, correctness = fit_decoder_slot(
            name, train[key][tr], ltr[name][tr], test[key][tr], lte[name][tr], seed,
            ledger_path=ledger_path, model_factory=model_factory,
        )
        out["decoder_slots"][name] = slot  # type: ignore[index]
        if fitted is not None and correctness is not None:
            out["decoders"][name] = _decoder_record(name, fitted, correctness, ltr[name][tr], lte[name][tr], seed)  # type: ignore[index]
    return out


def object_hashes(seed: int, *, n_train: int = PHASE_STEPS, n_test: int = EVAL_LEN) -> Dict[str, str]:
    """Coordinator identities, constructed with no network transition.

    Label hashes are derived here by direct slicing of the regenerated
    namespace-32 streams, independently of ``worker_labels`` and the worker:
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


class MalformedLedger(ValueError):
    """A ledger line other than the final one is not one complete JSON object (spec 6.3: INVALID)."""


def read_ledger(path: Path, strict: bool = True) -> Tuple[List[Dict[str, object]], bool]:
    """Read durable events, discarding only a torn final line.

    A torn final line (no newline) is an event whose hook never returned; it
    is discarded and reported as ``torn``. Any malformed interior line makes
    the ledger INVALID: ``strict`` raises ``MalformedLedger``; non-strict
    skips it so the coordinator can still write 20 total rows for the INVALID
    package.
    """
    events, torn, bad = read_ledger_report(path)
    if bad and strict:
        raise MalformedLedger(f"malformed interior ledger line(s) {bad}")
    return events, torn


def read_ledger_report(path: Path) -> Tuple[List[Dict[str, object]], bool, List[int]]:
    """(events, torn_tail, malformed interior line numbers)."""
    data = path.read_bytes() if path.exists() else b""
    torn = bool(data and not data.endswith(b"\n"))
    lines = data.split(b"\n")[:-1]  # the last element is b"" (complete) or the torn tail
    events: List[Dict[str, object]] = []
    bad: List[int] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad.append(index + 1)
            continue
        if not isinstance(value, dict):
            bad.append(index + 1)
            continue
        events.append(value)
    return events, torn, bad


def _coordinator_base(seed: int, source: str, seed_error: Optional[Dict[str, object]], torn: bool) -> Dict[str, object]:
    return {
        "diagnostic": "D1R2", "seed": seed, "row_source": source,
        "seed_error": seed_error, "ledger_torn_tail": torn,
        "non_decoder_warnings": [], "decoder_slots": {}, "decoders": {},
        "train_input_hash": None, "test_input_hash": None,
        "train_network_feature_hash": None, "test_network_feature_hash": None,
        "train_pipeline_feature_hash": None, "test_pipeline_feature_hash": None,
        "initial_weights_sha256": None, "train_rows": None, "test_rows": None,
        "warmup_rows": None, "network_feature_dim": None, "pipeline_feature_dim": None,
        "weights_constant": None, "nesting_ok": None, "reward_computed": None,
    }


def preflight_failure_rows(summary: str, seeds: Sequence[int] = D1R2_SEEDS) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        row = _coordinator_base(seed, "coordinator_preflight", {
            "exception_type": None, "phase": "preflight_collision_probe", "message_head": _message_head(summary),
        }, False)
        row["decoder_slots"] = {name: empty_slot("preflight_failure") for name in DECODERS}
        rows.append(row)
    return rows


def reconstruct_rows_from_ledger(
    events: Sequence[Mapping[str, object]], reason: str, seeds: Sequence[int] = D1R2_SEEDS,
    *, torn: bool = False, stop_text: Optional[str] = None,
) -> List[Dict[str, object]]:
    if reason not in {"worker_lost", "terminated_wall_stop", "terminated_rss_abort"}:
        raise ValueError("invalid reconstruction reason")
    rows: List[Dict[str, object]] = []
    stop_text = stop_text or reason
    for seed in seeds:
        own = [event for event in events if event.get("seed") == seed]
        started = any(event.get("event") == "seed_started" for event in own)
        upstream = [event for event in own if event.get("event") == "upstream_failure"]
        seed_error = upstream[-1].get("seed_error") if upstream else None
        row = _coordinator_base(seed, "coordinator_from_ledger", seed_error if isinstance(seed_error, dict) else None, torn)
        row["non_decoder_warnings"] = [
            {key: event.get(key) for key in ("category", "phase", "message_head")}
            for event in own if event.get("event") == "warning" and event.get("decoder") is None
        ]
        slots: Dict[str, object] = {}
        for name in DECODERS:
            closed = [event for event in own if event.get("event") == "scope_close" and event.get("decoder") == name]
            slot_events = [event for event in own if event.get("event") == "warning" and event.get("decoder") == name]
            warnings_for_slot = [{key: event.get(key) for key in ("category", "phase", "message_head")} for event in slot_events]
            # Durable per-warning issubclass result written by the hook (spec 6.3 subclass rule).
            conv_warning = bool(_convergence_warnings(warnings_for_slot, [event.get("convergence_warning") for event in slot_events]))
            if closed and isinstance(closed[-1].get("slot"), Mapping):
                slot = dict(closed[-1]["slot"])  # type: ignore[arg-type]
                slot["warnings"] = warnings_for_slot
                slot["convergence_warning"] = conv_warning
            else:
                opened = any(event.get("event") == "scope_open" and event.get("decoder") == name for event in own)
                if upstream and not opened:
                    slot = empty_slot("not_attempted_upstream_failure")
                elif not started and reason in {"terminated_wall_stop", "terminated_rss_abort"}:
                    slot = empty_slot("seed_not_started")
                else:
                    slot = empty_slot(reason, attempted=opened, fit_error={
                        "exception_type": None, "phase": None, "message_head": _message_head(stop_text),
                    })
                    slot["warnings"] = warnings_for_slot
                    slot["convergence_warning"] = conv_warning
            slots[name] = slot
        row["decoder_slots"] = slots
        rows.append(row)
    return rows


# ------------------------------------------------------------ validation
ROW_FIELDS = (
    "diagnostic", "seed", "row_source", "seed_error", "ledger_torn_tail",
    "non_decoder_warnings", "decoder_slots", "train_input_hash", "test_input_hash",
    "train_network_feature_hash", "test_network_feature_hash", "train_pipeline_feature_hash",
    "test_pipeline_feature_hash", "initial_weights_sha256", "train_rows", "test_rows",
    "warmup_rows", "network_feature_dim", "pipeline_feature_dim", "weights_constant",
    "nesting_ok", "reward_computed", "decoders",
)
DECODER_FIELDS = ("accuracy", "scored", "decoder_config", "train_label_sha256", "test_label_sha256", "mean", "scale", "normalization_sha256", "coefficients", "intercept", "coefficients_sha256", "n_iter", "converged")
NETWORK_DECODER_FIELDS = DECODER_FIELDS + ("lower_95",)
DECODER_WIDTH = {"pipeline": PIPELINE_FEATURE_DIM, "network_positive": NETWORK_FEATURE_DIM, "A": NETWORK_FEATURE_DIM}
ROW_SOURCES = {"worker", "coordinator_from_ledger", "coordinator_preflight"}
WARNING_FIELDS = {"category", "phase", "message_head"}
ERROR_FIELDS = {"exception_type", "phase", "message_head"}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _head_ok(value: object, nullable: bool = False) -> bool:
    if value is None:
        return nullable
    return isinstance(value, str) and len(value.splitlines()) <= 2 and all(len(line) <= 300 for line in value.splitlines())


def validate_warning(value: object, decoder: bool = True) -> List[str]:
    if not isinstance(value, Mapping) or set(value) != WARNING_FIELDS:
        return ["warning fields differ from schema"]
    out: List[str] = []
    if not isinstance(value["category"], str) or not value["category"]:
        out.append("warning category invalid")
    if value["phase"] not in (PHASES if decoder else NON_DECODER_PHASES):
        out.append("warning phase invalid")
    if not _head_ok(value["message_head"], value["phase"] == "predict_score"):
        out.append("warning message_head invalid")
    return out


def validate_slot(name: str, value: object) -> List[str]:
    tag = f"decoder slot {name}"
    if not isinstance(value, Mapping):
        return [f"{tag}: not a mapping"]
    if set(value) != set(SLOT_FIELDS):
        return [f"{tag}: fields differ from schema"]
    out: List[str] = []
    attempted, completed, reason = value["attempted"], value["completed"], value["stop_reason"]
    n_iter, warns = value["n_iter"], value["warnings"]
    if not isinstance(attempted, bool) or not isinstance(completed, bool) or (completed and not attempted):
        out.append(f"{tag}: attempted/completed invalid")
    if reason not in STOP_REASONS:
        out.append(f"{tag}: stop_reason invalid")
    if n_iter is not None and (not _is_int(n_iter) or n_iter <= 0):
        out.append(f"{tag}: n_iter invalid")
    if not isinstance(warns, list):
        out.append(f"{tag}: warnings not a list")
        warns = []
    for warning in warns:
        out.extend(f"{tag}: {v}" for v in validate_warning(warning))
    derived_warning = bool(_convergence_warnings(warns))
    if not isinstance(value["converged"], bool) or not isinstance(value["convergence_warning"], bool) or value["convergence_warning"] != derived_warning:
        out.append(f"{tag}: convergence flags invalid")
    err = value["fit_error"]
    if err is not None:
        if not isinstance(err, Mapping) or set(err) != ERROR_FIELDS:
            out.append(f"{tag}: fit_error schema invalid")
        elif reason in {"worker_lost", "terminated_wall_stop", "terminated_rss_abort"}:
            if err["exception_type"] is not None or err["phase"] is not None or not _head_ok(err["message_head"]):
                out.append(f"{tag}: coordinator stop error invalid")
        elif not isinstance(err["exception_type"], str) or err["phase"] not in PHASES or not _head_ok(err["message_head"], err["phase"] == "predict_score"):
            out.append(f"{tag}: fit_error invalid")
    converged = bool(completed and _is_int(n_iter) and n_iter < MAX_ITER and not derived_warning)
    if value["converged"] != converged or (reason == "converged") != converged:
        out.append(f"{tag}: converged semantics invalid")
    if reason in {"preflight_failure", "seed_not_started", "not_attempted_upstream_failure"} and (attempted or completed or n_iter is not None or err is not None):
        out.append(f"{tag}: not-attempted semantics invalid")
    if reason in {"preflight_failure", "seed_not_started"} and warns:
        out.append(f"{tag}: warnings on a slot whose seed never began")
    if reason in {"worker_lost", "terminated_wall_stop", "terminated_rss_abort"} and (completed or n_iter is not None or err is None):
        out.append(f"{tag}: termination semantics invalid")
    if reason in {"fit_exception", "iteration_limit", "evaluation_limit", "other_convergence_warning", "score_exception", "converged"} and attempted is not True:
        out.append(f"{tag}: attempted slot semantics invalid")
    if reason == "fit_exception" and (completed or n_iter is not None):
        out.append(f"{tag}: fit_exception n_iter/completed invalid")
    if reason in {"iteration_limit", "evaluation_limit", "other_convergence_warning", "score_exception", "converged"} and not _is_int(n_iter):
        out.append(f"{tag}: fit returned but n_iter is not an integer")
    if reason in {"iteration_limit", "evaluation_limit", "other_convergence_warning"}:
        if not derived_warning and not (_is_int(n_iter) and n_iter >= MAX_ITER):
            out.append(f"{tag}: non-converged reason without evidence")
        if err is not None and (not isinstance(err, Mapping) or err.get("phase") != "predict_score"):
            out.append(f"{tag}: only a scoring exception may follow a returned fit")
        if err is not None and completed:
            out.append(f"{tag}: completed slot carries an exception")
    if reason == "converged" and err is not None:
        out.append(f"{tag}: converged slot carries an exception")
    if reason in {"iteration_limit", "evaluation_limit", "other_convergence_warning", "score_exception", "converged"} and _is_int(n_iter):
        want = _warning_stop(warns, n_iter)
        if want == "converged" and not completed:
            want = "score_exception"
        if want != reason:
            out.append(f"{tag}: stop_reason {reason} violates first-match order (expected {want})")
    if reason == "fit_exception" and (err is None or err.get("phase") not in {"standardize", "fit"}):
        out.append(f"{tag}: fit_exception semantics invalid")
    if reason == "score_exception" and (err is None or err.get("phase") != "predict_score" or n_iter is None or derived_warning or completed):
        out.append(f"{tag}: score_exception semantics invalid")
    messages = " ".join(str(w.get("message_head") or "").upper() for w in warns)
    if reason == "iteration_limit" and not ((_is_int(n_iter) and n_iter >= MAX_ITER) or "ITERATIONS REACHED LIMIT" in messages):
        out.append(f"{tag}: iteration-limit evidence absent")
    if reason == "evaluation_limit" and "F,G EVALUATIONS EXCEEDS LIMIT" not in messages:
        out.append(f"{tag}: evaluation-limit evidence absent")
    return out


def validate_decoder(name: str, value: object, expected: Optional[Mapping[str, str]], tag: str, test_rows: int = TEST_ROWS) -> List[str]:
    if not isinstance(value, Mapping):
        return [f"{tag} decoder={name}: not a mapping"]
    fields = NETWORK_DECODER_FIELDS if name in NETWORK_DECODERS else DECODER_FIELDS
    if set(value) != set(fields):
        return [f"{tag} decoder={name}: outcome fields differ"]
    out: List[str] = []
    if not diag._finite_unit(value["accuracy"]) or abs(value["accuracy"] * test_rows - round(value["accuracy"] * test_rows)) > 1e-6:
        out.append(f"{tag} decoder={name}: accuracy invalid")
    if name in NETWORK_DECODERS and not diag._finite_unit(value["lower_95"]):
        out.append(f"{tag} decoder={name}: lower_95 invalid")
    if value["scored"] != test_rows or not _is_int(value["scored"]):
        out.append(f"{tag} decoder={name}: scored invalid")
    if not isinstance(value["decoder_config"], Mapping) or dict(value["decoder_config"]) != DECODER_CONFIG:
        out.append(f"{tag} decoder={name}: decoder config differs")
    for split in ("train", "test"):
        key = f"{split}_label_sha256"
        if not diag._is_hash(value[key]) or (expected is not None and value[key] != expected.get(f"{name}_{key}")):
            out.append(f"{tag} decoder={name}: {key} label identity differs from coordinator identity")
    width = DECODER_WIDTH[name]
    if not (diag._finite_vector(value["mean"], width) and diag._finite_vector(value["scale"], width)):
        out.append(f"{tag} decoder={name}: normalization invalid")
    else:
        if not (np.asarray(value["scale"]) > 0).all() or value["normalization_sha256"] != diag.normalization_hash(value["mean"], value["scale"]):
            out.append(f"{tag} decoder={name}: normalization payload invalid")
    try:
        coef, intercept = np.asarray(value["coefficients"], dtype=np.float64), np.asarray(value["intercept"], dtype=np.float64)
        payload_ok = coef.shape == (1, width) and intercept.shape == (1,) and np.isfinite(coef).all() and np.isfinite(intercept).all()
    except (TypeError, ValueError):
        payload_ok = False
    if not payload_ok or value["coefficients_sha256"] != diag.coefficients_hash(value["coefficients"], value["intercept"]):
        out.append(f"{tag} decoder={name}: coefficients invalid")
    if not isinstance(value["n_iter"], list) or len(value["n_iter"]) != 1 or not _is_int(value["n_iter"][0]):
        out.append(f"{tag} decoder={name}: n_iter invalid")
    elif not 0 < value["n_iter"][0] < MAX_ITER:
        out.append(f"{tag} decoder={name}: n_iter reached max_iter (section 6.2 non-converged)")
    if value["converged"] is not True:
        out.append(f"{tag} decoder={name}: converged flag false (section 6.2)")
    return out


def validate_row_schema(row: object) -> List[str]:
    if not isinstance(row, Mapping):
        return ["D1R2 row is not a mapping"]
    tag = f"D1R2 seed={row.get('seed')}"
    missing = [key for key in ROW_FIELDS if key not in row]
    if missing:
        return [f"{tag}: missing field(s) {missing}"]
    unknown = sorted(set(row) - set(ROW_FIELDS) - {"status", "worker_pid"}, key=str)
    out = [f"{tag}: undeclared fields {unknown}"] if unknown else []
    if row["diagnostic"] != "D1R2" or not _is_int(row["seed"]):
        out.append(f"{tag}: diagnostic/seed invalid")
    if row["row_source"] not in ROW_SOURCES or not isinstance(row["ledger_torn_tail"], bool):
        out.append(f"{tag}: row source/torn-tail invalid")
    if not isinstance(row["non_decoder_warnings"], list):
        out.append(f"{tag}: non_decoder_warnings invalid")
    else:
        for warning in row["non_decoder_warnings"]:
            out.extend(f"{tag}: {v}" for v in validate_warning(warning, False))
    slots = row["decoder_slots"]
    if not isinstance(slots, Mapping) or set(slots) != set(DECODERS):
        out.append(f"{tag}: decoder_slots must be exactly {list(DECODERS)}")
    else:
        for name in DECODERS:
            out.extend(f"{tag}: {v}" for v in validate_slot(name, slots[name]))
    return out


def validate_row(
    row: Mapping[str, object], expected: Optional[Mapping[str, str]],
    train_rows: int = TRAIN_ROWS, test_rows: int = TEST_ROWS,
    decoders: Sequence[str] = DECODERS,
) -> List[str]:
    out = validate_row_schema(row)
    if out:
        return out
    tag = f"D1R2 seed={row['seed']}"
    if row.get("status", "ok") != "ok":
        out.append(f"{tag}: job status error")
    if row["row_source"] != "worker":
        out.append(f"{tag}: coordinator row has unavailable section 7 invariants")
        return out
    for key in ("train_input_hash", "test_input_hash", "train_network_feature_hash", "test_network_feature_hash", "train_pipeline_feature_hash", "test_pipeline_feature_hash", "initial_weights_sha256"):
        if not diag._is_hash(row[key]):
            out.append(f"{tag}: {key} invalid")
    if expected is None:
        out.append(f"{tag}: no coordinator identity")
    else:
        for key in ("initial_weights_sha256", "train_input_hash", "test_input_hash"):
            if row[key] != expected.get(key):
                out.append(f"{tag}: {key} differs from coordinator identity")
    for key, want in (("train_rows", train_rows), ("test_rows", test_rows), ("warmup_rows", WARMUP), ("network_feature_dim", NETWORK_FEATURE_DIM), ("pipeline_feature_dim", PIPELINE_FEATURE_DIM)):
        if row[key] != want or not _is_int(row[key]):
            out.append(f"{tag}: {key} invariant failed")
    if row["weights_constant"] is not True:
        out.append(f"{tag}: frozen W1/W2 were not bitwise constant")
    if row["reward_computed"] is not False:
        out.append(f"{tag}: reward_computed must be False under F0")
    if row["nesting_ok"] is not True:
        out.append(f"{tag}: count20 nesting invariant failed")
    if row["seed_error"] is not None:
        out.append(f"{tag}: upstream seed error")
    slots, outcomes = row["decoder_slots"], row["decoders"]
    all_converged = all(slots[name]["converged"] is True for name in decoders)
    if all_converged:
        if not isinstance(outcomes, Mapping) or set(outcomes) != set(decoders):
            out.append(f"{tag}: converged row requires selected outcomes")
        else:
            for name in decoders:
                found = validate_decoder(name, outcomes[name], expected, tag, test_rows)
                out.extend(found)
                if not found and outcomes[name]["n_iter"] != [slots[name]["n_iter"]]:
                    out.append(f"{tag} decoder={name}: record n_iter differs from its slot")
    elif not isinstance(outcomes, Mapping):
        out.append(f"{tag}: decoders must be mapping")
    if not all_converged:
        out.append(f"{tag}: one or more decoder slots non-converged")
    return out


def validate_ledger_match(row: Mapping[str, object], events: Sequence[Mapping[str, object]]) -> List[str]:
    out: List[str] = []
    warnings_by_seed = [event for event in events if event.get("event") == "warning" and event.get("seed") == row.get("seed")]
    for decoder in (*DECODERS, None):
        actual = row["non_decoder_warnings"] if decoder is None else row["decoder_slots"][decoder]["warnings"]
        actual = [{**warning, "message_head": None if warning["phase"] == "predict_score" else warning["message_head"]} for warning in actual]
        events_for = [event for event in warnings_by_seed if event.get("decoder") == decoder]
        durable = [{key: event.get(key) for key in ("category", "phase", "message_head")} for event in events_for]
        if actual != durable:
            out.append(f"seed={row.get('seed')} decoder={decoder}: warnings differ from ledger")
            continue
        flags = [event.get("convergence_warning") for event in events_for]
        if not all(isinstance(flag, bool) for flag in flags):
            out.append(f"seed={row.get('seed')} decoder={decoder}: ledger warning lacks its convergence classification")
        elif decoder is not None:
            durable_cw = bool(_convergence_warnings(durable, flags))
            if row["decoder_slots"][decoder]["convergence_warning"] != durable_cw:
                out.append(f"seed={row.get('seed')} decoder={decoder}: convergence_warning differs from the durable ledger classification")
    return out


def validate_package(rows: Sequence[Mapping[str, object]], expected: Optional[Mapping[int, Mapping[str, str]]], seeds: Sequence[int] = D1R2_SEEDS) -> List[str]:
    out: List[str] = []
    seen = [row.get("seed") if isinstance(row, Mapping) else None for row in rows]
    if len(rows) != len(seeds): out.append(f"D1R2: {len(rows)} rows, expected {len(seeds)}")
    if len(set(seen)) != len(seen): out.append("D1R2: duplicate seed rows")
    if set(seen) != set(seeds): out.append("D1R2: seed set differs")
    if seen != sorted(seen, key=lambda value: (not _is_int(value), value if _is_int(value) else 0)): out.append("D1R2: rows not sorted")
    if expected is None: out.append("D1R2: coordinator identities absent")
    for row in rows:
        seed = row.get("seed") if isinstance(row, Mapping) else None
        out.extend(validate_row(row, None if expected is None else expected.get(seed)))
    return out


def redact_invalid_row(row: Mapping[str, object]) -> Dict[str, object]:
    """Retain only outcome-free diagnostics and redact scoring text on INVALID."""
    import copy

    out = copy.deepcopy(dict(row))
    out.pop("decoders", None)
    slots = out.get("decoder_slots")
    if isinstance(slots, dict):
        for slot in slots.values():
            if not isinstance(slot, dict):
                continue
            for warning in slot.get("warnings", []):
                if isinstance(warning, dict) and warning.get("phase") == "predict_score":
                    warning["message_head"] = None
            error = slot.get("fit_error")
            if isinstance(error, dict) and error.get("phase") == "predict_score":
                error["message_head"] = None
    return out


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
    ("STOP_D1R2_INVALID",
     "Run-preflight collision-probe failure, any section 7 invariant failure, any INVALID decoder or seed, six-hour wall stop, RSS abort, or incomplete package",
     "STOP. No science reading. Stop-loss: the D1 harness-repair path is closed. D1 returns to Kendrick as a NEEDS KENDRICK DECISION card. No D1R3. No D2-D4, no r4."),
    ("STOP_D1R2_INVALID_PIPELINE",
     "Row 1 not met and pipeline median < 0.99",
     "STOP. Code defect, not science. No science reading of the network decoders. Stop-loss: the D1 harness-repair path is closed. D1 returns to Kendrick as a NEEDS KENDRICK DECISION card. No D1R3. No D2-D4, no r4."),
    ("CLOSE_D1_REPRESENTATION_LIMITED",
     "Pipeline PASS, network_positive FAIL or INCONCLUSIVE",
     "D1 closes as REPRESENTATION_LIMITED. No further D1 harness repair. Return to Kendrick with a roadmap decision card. No D2-D4, no r4."),
    ("D1R2_A_FAIL_EBB90E74_ROW3", "Pipeline PASS, network_positive PASS, A FAIL", "Return to Kendrick. No r4 readout or reward change. No D2-D4."),
    ("D1R2_A_INCONCLUSIVE_EBB90E74_ROW4", "Pipeline PASS, network_positive PASS, A INCONCLUSIVE", "Return to Kendrick/spec review. No downstream diagnostic, no r4."),
    ("D1R2_PASS_D2_D4_ELIGIBLE", "Pipeline PASS, network_positive PASS, A PASS", "D1 is PASS by D1R2. D2-D4 require a separate run authorization. No r4 from D1R2 alone."),
)
BRANCH_INDEX = {code: i + 1 for i, (code, _, _) in enumerate(BRANCH_TABLE)}
STATUS_TO_BRANCH = {
    "INVALID": "STOP_D1R2_INVALID", "INVALID_PIPELINE": "STOP_D1R2_INVALID_PIPELINE",
    "REPRESENTATION_LIMITED": "CLOSE_D1_REPRESENTATION_LIMITED",
    "FAIL": "D1R2_A_FAIL_EBB90E74_ROW3", "INCONCLUSIVE": "D1R2_A_INCONCLUSIVE_EBB90E74_ROW4",
    "PASS": "D1R2_PASS_D2_D4_ELIGIBLE",
}


def branch(status: Optional[str]) -> Dict[str, object]:
    code = STATUS_TO_BRANCH.get(status or "INVALID", "STOP_D1R2_INVALID")
    _, condition, action = BRANCH_TABLE[BRANCH_INDEX[code] - 1]
    return {"code": code, "table_row": BRANCH_INDEX[code], "condition": condition, "action": action}


def summarize(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[Mapping[int, Mapping[str, str]]],
    probe_pass: bool,
    seeds: Sequence[int] = D1R2_SEEDS,
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
    """D1R2 effective parameter block (spec 12). Network values are read back
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
        "w1_init": "U(-10,10) mV, [20261001, seed, 32, 1] first draw",
        "w2_O1_init": "U(0,10) mV, [20261001, seed, 32, 1] second draw",
        "w2_O0_init": "U(0,10) mV, [20261001, seed, 32, 21]",
        "noise_p": float(diag.NOISE_P),
        "train_stream_len": PHASE_STEPS,
        "test_stream_len": EVAL_LEN,
        "warmup_rows": WARMUP,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "seeds": list(D1R2_SEEDS),
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
        "bootstrap_entropy": "[20261001, seed, 32, 60, q]; q=1 A, q=2 network_positive",
        "decoders": list(DECODERS),
        "labels": {"pipeline": "x_(t-1) = x[1999:-1]", "network_positive": "x_(t-1) = x[1999:-1]", "A": "x_t = x[2000:]"},
        "max_workers": MAX_WORKERS,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "thresholds": dict(THRESHOLDS),
    }
