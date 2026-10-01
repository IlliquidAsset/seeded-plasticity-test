"""Frozen Stage 2 D1-D4 diagnostic construction.

Implements docs/STAGE2_DIAGNOSTIC_SPEC.md at ebb90e74 without changing the
Stage 2 r3 engine or frozen specification.  This module is construction and
scoring infrastructure; importing it executes no diagnostic.
"""

from __future__ import annotations

import copy
import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from snn.stage2 import NOISE_P, ROOT_ENTROPY, sha256
from snn.stage2_r3 import (
    BETA_A,
    EVAL_LEN,
    EVAL_WARMUP,
    O0,
    O1,
    ONEHOT,
    PHASE_STEPS,
    W1_BOUNDS,
    W2_BOUNDS,
    Stage2R3SNN,
    build_network as build_r3_network,
    init_state,
    make_reward_fn,
    predict,
    CoinReader,
)

SPEC_COMMIT = "ebb90e74d088057f1d37e08ac80ed7bab4e8851e"
SPEC_SHA256 = "7e6ada6109924df0c31bc49de97082699a032549887fd8b0411c45e2902981bf"
FROZEN_STAGE2_SHA256 = "695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd"
NAMESPACE = 30
DIAGNOSTIC_SEEDS = tuple(range(2000, 2020))
CHECKPOINT_STEPS = 1_000
N_CHECKPOINTS = 200
DRIVE_STEPS = PHASE_STEPS + EVAL_LEN
DRIVE_SOURCES_PER_TARGET = 8
DRIVE_TARGETS = 22
DRIVE_RATE_HZ = 25.0
DRIVE_WEIGHT_MV = 2.0
DRIVE_P = 1.0 - math.exp(-DRIVE_RATE_HZ / 1000.0)
D1_WARMUP = 2_000
D1_TRAIN_ROWS = 198_000
D1_TEST_ROWS = 10_000


def rng(entropy: Sequence[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(entropy)))


def _roll_a(g: np.random.Generator, n: int) -> np.ndarray:
    x = np.zeros(n, dtype=np.int8)
    x[: min(2, n)] = g.integers(0, 2, size=min(2, n)).astype(np.int8)
    if n <= 2:
        return x
    flips = g.random(n - 2) < NOISE_P
    for t, flip in enumerate(flips, start=2):
        x[t] = (int(x[t - 2]) ^ int(x[t - 1])) ^ int(flip)
    return x


def initial_weights(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """r3 distributions/draw order in namespace 30 (spec 3.1)."""
    g = rng([ROOT_ENTROPY, seed, NAMESPACE, 1])
    w1 = g.uniform(-10.0, 10.0, size=(20, 2)).astype(np.float64)
    w2_o1 = g.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64)
    h = rng([ROOT_ENTROPY, seed, NAMESPACE, 21])
    w2_o0 = h.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64)
    return w1, np.concatenate((w2_o1, w2_o0), axis=0)


def a_stream(seed: int, train: bool, n: Optional[int] = None) -> np.ndarray:
    component = 2 if train else 5
    default_n = PHASE_STEPS if train else EVAL_LEN
    return _roll_a(rng([ROOT_ENTROPY, seed, NAMESPACE, component]), default_n if n is None else n)


def tie_coin(seed: int, task: str, n: int = EVAL_LEN) -> np.ndarray:
    component = {"A": 1, "lag1": 2}[task]
    return rng([ROOT_ENTROPY, seed, NAMESPACE, 22, component]).integers(0, 2, size=n).astype(np.int8)


def lag1_stream(seed: int, train: bool, n: Optional[int] = None) -> Dict[str, object]:
    """One-call fair-bit generator with explicit x_-1 context (spec 6.1)."""
    component = 31 if train else 32
    default_n = PHASE_STEPS if train else EVAL_LEN
    n = default_n if n is None else n
    u = rng([ROOT_ENTROPY, seed, NAMESPACE, component]).integers(0, 2, size=n + 1).astype(np.int8)
    inputs = u[1:].copy()
    targets = u[:-1].copy()
    return {
        "context_x_minus_1": int(u[0]),
        "inputs": inputs,
        "targets": targets,
        "raw": u,
        "nominal_p_one": 0.5,
        "hash": sha256(u),
    }


def background_source_vectors(seed: int, n_steps: int = DRIVE_STEPS) -> np.ndarray:
    """Exactly one vectorized random call; bool shape (steps,22,8)."""
    g = rng([ROOT_ENTROPY, seed, NAMESPACE, 40])
    return g.random((n_steps, DRIVE_TARGETS, DRIVE_SOURCES_PER_TARGET)) < DRIVE_P


def background_hash(source: np.ndarray) -> str:
    return sha256(np.asarray(source, dtype=np.bool_))


def background_mv(source_step: np.ndarray) -> np.ndarray:
    a = np.asarray(source_step, dtype=np.bool_)
    if a.shape != (DRIVE_TARGETS, DRIVE_SOURCES_PER_TARGET):
        raise ValueError(f"background step shape {a.shape}, expected {(DRIVE_TARGETS, DRIVE_SOURCES_PER_TARGET)}")
    return a.sum(axis=1, dtype=np.int64).astype(np.float64) * DRIVE_WEIGHT_MV


def exchange_output_drive(source: np.ndarray, labels: Sequence[str] = ("O1", "O0")) -> Tuple[np.ndarray, Tuple[str, str]]:
    """Swap independent exchangeable O1/O0 columns and their labels."""
    if tuple(labels) != ("O1", "O0"):
        raise ValueError("labels must be ('O1','O0') before exchange")
    out = np.array(source, copy=True)
    out[:, [20, 21], :] = out[:, [21, 20], :]
    return out, ("O0", "O1")


class Stage2DiagnosticSNN(Stage2R3SNN):
    """r3 network with an optional non-plastic, direct background input."""

    background_plasticities: Tuple[()] = ()

    def online_step_with_background(
        self,
        input_spikes: torch.Tensor,
        background_input_mv: Optional[np.ndarray] = None,
        reward_fn: Optional[Callable[[torch.Tensor], float]] = None,
    ) -> Tuple[torch.Tensor, float]:
        if background_input_mv is None:
            return super().online_step(input_spikes, reward_fn=reward_fn)
        bg = np.asarray(background_input_mv, dtype=np.float64)
        if bg.shape != (DRIVE_TARGETS,):
            raise ValueError(f"background input shape {bg.shape}, expected {(DRIVE_TARGETS,)}")

        # This is PureSNN.online_step's frozen order. Background is added only
        # to ordinary postsynaptic input, never to ``pres`` or a plasticity.
        pres = [input_spikes] + self._spikes[:-1]
        for i, plasticity in enumerate(self.plasticities):
            plasticity.step(pres[i], self._spikes[i])
        out_t = self._spikes[-1]
        reward = 0.0
        if reward_fn is not None:
            reward = float(reward_fn(out_t))
            if reward != 0.0:
                r = torch.tensor(reward, dtype=out_t.dtype, device=out_t.device)
                for plasticity in self.plasticities:
                    plasticity.apply_reward(r)
        new_spikes = []
        slices = (slice(0, 20), slice(20, 22))
        for i in range(self.n_layers):
            current = self.synapses[i](pres[i], None)
            current = current + torch.as_tensor(bg[slices[i]], dtype=current.dtype, device=current.device).unsqueeze(0)
            spikes, self._v[i] = self.neurons[i](current, self._v[i])
            new_spikes.append(spikes)
        self._spikes = new_spikes
        return out_t, reward


def build_network(w1: np.ndarray, w2: np.ndarray, beta_a: float = BETA_A) -> Stage2DiagnosticSNN:
    base = build_r3_network(w1, w2, beta_a)
    base.__class__ = Stage2DiagnosticSNN
    return base  # type: ignore[return-value]


class FeatureAccumulator:
    """Causal D1 feature recurrence over already-present hidden spikes."""

    def __init__(self, n_hidden: int = 20):
        self.n_hidden = n_hidden
        self.buffer = np.zeros((20, n_hidden), dtype=np.float64)
        self.count20 = np.zeros(n_hidden, dtype=np.float64)
        self.trace20 = np.zeros(n_hidden, dtype=np.float64)
        self.trace25 = np.zeros(n_hidden, dtype=np.float64)
        self.index = 0

    def update(self, hidden_spikes_t: np.ndarray) -> np.ndarray:
        h = np.asarray(hidden_spikes_t, dtype=np.float64).reshape(self.n_hidden)
        old = self.buffer[self.index].copy()
        self.buffer[self.index] = h
        self.index = (self.index + 1) % 20
        self.count20 += h - old
        self.trace20 = math.exp(-1.0 / 20.0) * self.trace20 + h
        self.trace25 = math.exp(-1.0 / 25.0) * self.trace25 + h
        return np.concatenate((self.count20, self.trace20, self.trace25)).copy()


def d1_labels(inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs, dtype=np.int8)
    current = x.copy()
    prior = np.empty_like(x)
    prior[0] = -1  # deliberately unscorable; warm-up excludes it
    prior[1:] = x[:-1]
    return current, prior


def capture_d1_features(seed: int, train: bool, n: Optional[int] = None) -> Dict[str, object]:
    """Capture causal frozen-network features. No reward and no weight update."""
    x = a_stream(seed, train=train, n=n)
    w1, w2 = initial_weights(seed)
    net = build_network(w1, w2)
    init_state(net)
    features = np.empty((len(x), 60), dtype=np.float64)
    acc = FeatureAccumulator()
    for t, xt in enumerate(x):
        # phi_t is formed before x_t is integrated.
        features[t] = acc.update(net._spikes[0].detach().cpu().numpy()[0])
        net.online_step_with_background(ONEHOT[int(xt)], None, reward_fn=None)
    current, prior = d1_labels(x)
    return {
        "features": features,
        "label_A": current,
        "label_prior": prior,
        "input_hash": sha256(x),
        "feature_hash": sha256(features),
        "weights_constant": all(
            np.array_equal(a, b.weight.detach().cpu().numpy()) for a, b in zip((w1, w2), net.synapses)
        ),
    }


@dataclass(frozen=True)
class FittedDecoder:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    n_iter: np.ndarray
    model: object

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.scale

    def predict(self, features: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(self.transform(features))[:, 1]
        return (proba >= 0.5).astype(np.int8)


def fit_d1_decoder(train_features: np.ndarray, labels: np.ndarray) -> FittedDecoder:
    """The exact training-only transform and fixed sklearn decoder in spec 4.4."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    mean = x.mean(axis=0)
    raw_scale = x.std(axis=0, ddof=0)
    scale = np.where(raw_scale == 0.0, 1.0, raw_scale)
    z = (x - mean) / scale
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        solver="lbfgs",
        tol=1e-8,
        max_iter=2000,
        class_weight=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(z, y)
    return FittedDecoder(
        mean=mean.copy(),
        scale=scale.copy(),
        coefficients=model.coef_.copy(),
        intercept=model.intercept_.copy(),
        n_iter=model.n_iter_.copy(),
        model=model,
    )


def circular_block_lower_bound(
    correctness: np.ndarray,
    seed: int,
    target_q: int,
    n_resamples: int = 10_000,
    block_length: int = 100,
    chunk_size: int = 100,
) -> float:
    """One-sided 95% D1 circular moving-block lower interval."""
    v = np.asarray(correctness, dtype=np.float64)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("correctness must be one non-empty vector")
    if target_q not in (1, 2):
        raise ValueError("target_q must be 1 (A) or 2 (positive control)")
    n_blocks = math.ceil(len(v) / block_length)
    g = rng([ROOT_ENTROPY, seed, NAMESPACE, 60, target_q])
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


def run_d1_seed(seed: int) -> Dict[str, object]:
    """One complete D1 seed. The CLI, not this function, owns run authorization."""
    train = capture_d1_features(seed, train=True)
    test = capture_d1_features(seed, train=False)
    tr = slice(D1_WARMUP, None)
    te = slice(D1_WARMUP, None)
    if train["features"][tr].shape[0] != D1_TRAIN_ROWS or test["features"][te].shape[0] != D1_TEST_ROWS:
        raise AssertionError("D1 row-count invariant failed")
    out: Dict[str, object] = {
        "diagnostic": "D1",
        "seed": seed,
        "train_input_hash": train["input_hash"],
        "test_input_hash": test["input_hash"],
        "train_feature_hash": train["feature_hash"],
        "test_feature_hash": test["feature_hash"],
        "initial_weights_sha256": sha256(*initial_weights(seed)),
        "train_rows": D1_TRAIN_ROWS,
        "test_rows": D1_TEST_ROWS,
        "weights_constant": bool(train["weights_constant"] and test["weights_constant"]),
        "decoders": {},
    }
    for name, field, q in (("A", "label_A", 1), ("positive_prior", "label_prior", 2)):
        fitted = fit_d1_decoder(train["features"][tr], train[field][tr])
        pred = fitted.predict(test["features"][te])
        correctness = pred == test[field][te]
        out["decoders"][name] = {
            "accuracy": float(correctness.mean()),
            "lower_95": circular_block_lower_bound(correctness, seed, q),
            "mean": fitted.mean.tolist(),
            "scale": fitted.scale.tolist(),
            "coefficients": fitted.coefficients.tolist(),
            "intercept": fitted.intercept.tolist(),
            "n_iter": fitted.n_iter.tolist(),
            "converged": bool(np.all(fitted.n_iter < 2000)),
        }
    return out


CHECKPOINT_FIELDS = (
    "diagnostic",
    "condition",
    "seed",
    "checkpoint_index",
    "step_index",
    "hidden_rate_hz",
    "output_rate_hz_O1",
    "output_rate_hz_O0",
    "output_layer_rate_hz",
    "both_silent_fraction",
    "w1_mean",
    "w1_min",
    "w1_max",
    "w1_lower_hits",
    "w1_upper_hits",
    "w2_mean",
    "w2_min",
    "w2_max",
    "w2_lower_hits",
    "w2_upper_hits",
    "w1_lower_fraction",
    "w2_lower_fraction",
    "stream_sha256",
    "initial_weights_sha256",
    "weights_sha256",
    "drive_sha256",
    "finite",
    "bounds_ok",
)


class CheckpointAccumulator:
    def __init__(self, condition: str, seed: int, stream_hash: str, initial_hash: str, drive_hash: Optional[str]):
        self.condition = condition
        self.seed = seed
        self.stream_hash = stream_hash
        self.initial_hash = initial_hash
        self.drive_hash = drive_hash
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hidden = np.zeros(20, dtype=np.int64)
        self.outputs = np.zeros(2, dtype=np.int64)
        self.both_silent = 0

    def add(self, hidden_after: torch.Tensor, output_before: Tuple[int, int]) -> None:
        self.hidden += hidden_after.detach().cpu().numpy()[0].astype(np.int64)
        self.outputs += np.asarray(output_before, dtype=np.int64)
        self.both_silent += int(output_before == (0, 0))
        self.steps += 1

    def close(self, checkpoint_index: int, net: Stage2R3SNN) -> Dict[str, object]:
        if self.steps != CHECKPOINT_STEPS:
            raise ValueError(f"checkpoint window has {self.steps} steps, expected {CHECKPOINT_STEPS}")
        w1 = net.synapses[0].weight.detach().cpu().numpy()
        w2 = net.synapses[1].weight.detach().cpu().numpy()
        h_rates = 1000.0 * self.hidden / self.steps
        o_rates = 1000.0 * self.outputs / self.steps
        row: Dict[str, object] = {
            "diagnostic": "D2",
            "condition": self.condition,
            "seed": self.seed,
            "checkpoint_index": checkpoint_index,
            "step_index": (checkpoint_index + 1) * CHECKPOINT_STEPS - 1,
            "hidden_rate_hz": float(np.median(h_rates)),
            "output_rate_hz_O1": float(o_rates[O1]),
            "output_rate_hz_O0": float(o_rates[O0]),
            "output_layer_rate_hz": float(np.median(o_rates)),
            "both_silent_fraction": float(self.both_silent / self.steps),
            "w1_mean": float(w1.mean()),
            "w1_min": float(w1.min()),
            "w1_max": float(w1.max()),
            "w1_lower_hits": int((w1 == W1_BOUNDS[0]).sum()),
            "w1_upper_hits": int((w1 == W1_BOUNDS[1]).sum()),
            "w2_mean": float(w2.mean()),
            "w2_min": float(w2.min()),
            "w2_max": float(w2.max()),
            "w2_lower_hits": int((w2 == W2_BOUNDS[0]).sum()),
            "w2_upper_hits": int((w2 == W2_BOUNDS[1]).sum()),
            "w1_lower_fraction": float((w1 == W1_BOUNDS[0]).sum() / 40.0),
            "w2_lower_fraction": float((w2 == W2_BOUNDS[0]).sum() / 40.0),
            "stream_sha256": self.stream_hash,
            "initial_weights_sha256": self.initial_hash,
            "weights_sha256": sha256(w1, w2),
            "drive_sha256": self.drive_hash,
            "finite": bool(np.isfinite(w1).all() and np.isfinite(w2).all()),
            "bounds_ok": bool(
                w1.min() >= W1_BOUNDS[0]
                and w1.max() <= W1_BOUNDS[1]
                and w2.min() >= W2_BOUNDS[0]
                and w2.max() <= W2_BOUNDS[1]
            ),
        }
        validate_checkpoint_row(row)
        self.reset()
        return row


class CheckpointSchemaError(ValueError):
    pass


def _is_hash(v: object) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def validate_checkpoint_row(row: Mapping[str, object]) -> None:
    missing = [k for k in CHECKPOINT_FIELDS if k not in row]
    if missing:
        raise CheckpointSchemaError(f"missing field(s): {', '.join(missing)}")
    ints = ("seed", "checkpoint_index", "step_index", "w1_lower_hits", "w1_upper_hits", "w2_lower_hits", "w2_upper_hits")
    for k in ints:
        if not isinstance(row[k], int) or isinstance(row[k], bool):
            raise CheckpointSchemaError(f"{k} is not int")
    if row["step_index"] != (row["checkpoint_index"] + 1) * CHECKPOINT_STEPS - 1:
        raise CheckpointSchemaError("wrong checkpoint step index")
    floats = (
        "hidden_rate_hz",
        "output_rate_hz_O1",
        "output_rate_hz_O0",
        "output_layer_rate_hz",
        "both_silent_fraction",
        "w1_mean",
        "w1_min",
        "w1_max",
        "w2_mean",
        "w2_min",
        "w2_max",
        "w1_lower_fraction",
        "w2_lower_fraction",
    )
    if any(not isinstance(row[k], float) or not math.isfinite(row[k]) for k in floats):
        raise CheckpointSchemaError("non-finite or non-float numeric field")
    if not 0.0 <= row["both_silent_fraction"] <= 1.0:
        raise CheckpointSchemaError("both_silent_fraction out of range")
    limits = {"w1_lower_hits": 40, "w1_upper_hits": 40, "w2_lower_hits": 40, "w2_upper_hits": 40}
    if any(not 0 <= row[k] <= hi for k, hi in limits.items()):
        raise CheckpointSchemaError("bound hit count out of range")
    if row["w1_lower_fraction"] != row["w1_lower_hits"] / 40.0 or row["w2_lower_fraction"] != row["w2_lower_hits"] / 40.0:
        raise CheckpointSchemaError("bound fraction mismatch")
    if not (_is_hash(row["stream_sha256"]) and _is_hash(row["initial_weights_sha256"]) and _is_hash(row["weights_sha256"])):
        raise CheckpointSchemaError("invalid hash")
    if row["drive_sha256"] is not None and not _is_hash(row["drive_sha256"]):
        raise CheckpointSchemaError("invalid drive hash")
    if row["finite"] is not True or row["bounds_ok"] is not True:
        raise CheckpointSchemaError("finite/bounds invariant failed")


def validate_checkpoint_rows(
    rows: Sequence[Mapping[str, object]],
    seed: int,
    condition: str,
    stream_hash: str,
    initial_hash: str,
    expected_count: int = N_CHECKPOINTS,
) -> None:
    if len(rows) != expected_count:
        raise CheckpointSchemaError(f"expected {expected_count} checkpoints, got {len(rows)}")
    for i, row in enumerate(rows):
        validate_checkpoint_row(row)
        if (row["seed"], row["condition"], row["checkpoint_index"]) != (seed, condition, i):
            raise CheckpointSchemaError("seed/condition/checkpoint sequence mismatch")
        if row["stream_sha256"] != stream_hash or row["initial_weights_sha256"] != initial_hash:
            raise CheckpointSchemaError("stream/initial-weight hash mismatch")


def classify_onsets(hidden_step: Optional[int], output_step: Optional[int]) -> str:
    if hidden_step is not None and (output_step is None or output_step - hidden_step >= 5_000):
        return "HIDDEN_FIRST"
    if output_step is not None and (hidden_step is None or hidden_step - output_step >= 5_000):
        return "OUTPUT_FIRST"
    return "CO_ONSET"


def onset_record(primary: Sequence[Mapping[str, object]], frozen: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    p = {r["checkpoint_index"]: r for r in primary}
    f = {r["checkpoint_index"]: r for r in frozen}
    if set(p) != set(f):
        raise ValueError("P/F0 checkpoint indices differ")

    def first(rate_key: str) -> Optional[int]:
        for c in sorted(p):
            denominator = float(f[c][rate_key])
            if denominator != 0.0 and float(p[c][rate_key]) < 0.50 * denominator:
                return int(p[c]["step_index"])
        return None

    h = first("hidden_rate_hz")
    o = first("output_layer_rate_hz")

    def fractions(step: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
        if step is None:
            return None, None
        row = next(r for r in primary if int(r["step_index"]) == step)
        return float(row["w1_lower_fraction"]), float(row["w2_lower_fraction"])

    h_w1, h_w2 = fractions(h)
    o_w1, o_w2 = fractions(o)
    finite = [x for x in (h, o) if x is not None]
    earlier = min(finite) if finite else None
    e_w1, e_w2 = fractions(earlier)
    return {
        "hidden_onset_step": h,
        "hidden_onset_detected": h is not None,
        "hidden_onset_w1_lower_fraction": h_w1,
        "hidden_onset_w2_lower_fraction": h_w2,
        "output_onset_step": o,
        "output_onset_detected": o is not None,
        "output_onset_w1_lower_fraction": o_w1,
        "output_onset_w2_lower_fraction": o_w2,
        "earlier_onset_step": earlier,
        "earlier_onset_w1_lower_fraction": e_w1,
        "earlier_onset_w2_lower_fraction": e_w2,
        "label": classify_onsets(h, o),
    }


def _step_background(source: Optional[np.ndarray], index: int) -> Optional[np.ndarray]:
    return None if source is None else background_mv(source[index])


def train_condition(
    net: Stage2DiagnosticSNN,
    inputs: np.ndarray,
    targets: np.ndarray,
    plastic: bool,
    condition: str,
    seed: int,
    source: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    if len(inputs) != len(targets):
        raise ValueError("input/target lengths differ")
    if len(inputs) % CHECKPOINT_STEPS:
        raise ValueError("training length must be a whole checkpoint window")
    if source is not None and len(source) < len(inputs):
        raise ValueError("background source is too short")
    w_initial = tuple(s.weight.detach().cpu().numpy().copy() for s in net.synapses)
    stream_hash = sha256(np.asarray(inputs, dtype=np.int8), np.asarray(targets, dtype=np.int8))
    initial_hash = sha256(*w_initial)
    drive_hash = background_hash(source) if source is not None else None
    acc = CheckpointAccumulator(condition, seed, stream_hash, initial_hash, drive_hash)
    rows: List[Dict[str, object]] = []
    for t, (xt, yt) in enumerate(zip(inputs, targets)):
        current = net.current_output()
        z = (int(current[0, O1]), int(current[0, O0]))
        reward_fn = make_reward_fn(int(yt)) if plastic else None
        net.online_step_with_background(ONEHOT[int(xt)], _step_background(source, t), reward_fn=reward_fn)
        acc.add(net._spikes[0], z)
        if (t + 1) % CHECKPOINT_STEPS == 0:
            rows.append(acc.close(len(rows), net))
    validate_checkpoint_rows(rows, seed, condition, stream_hash, initial_hash, expected_count=len(inputs) // CHECKPOINT_STEPS)
    return rows


def evaluate_task(
    live: Stage2DiagnosticSNN,
    inputs: np.ndarray,
    targets: np.ndarray,
    coin: np.ndarray,
    source: Optional[np.ndarray] = None,
    source_offset: int = PHASE_STEPS,
    warmup: int = EVAL_WARMUP,
) -> Dict[str, object]:
    if not (len(inputs) == len(targets) == len(coin)):
        raise ValueError("evaluation vector lengths differ")
    if source is not None and len(source) < source_offset + len(inputs):
        raise ValueError("background source does not cover continuous evaluation indices")
    ev = copy.deepcopy(live)
    ev.reset_log = []
    init_state(ev)
    before = [s.weight.detach().clone() for s in ev.synapses]
    reader = CoinReader(coin)
    correct = scored = 0
    for i, (xt, yt) in enumerate(zip(inputs, targets)):
        out = ev.current_output()
        pred = predict(int(out[0, O1]), int(out[0, O0]), i, reader, i >= warmup)
        ev.online_step_with_background(ONEHOT[int(xt)], _step_background(source, source_offset + i), reward_fn=None)
        if i >= warmup:
            correct += int(pred == int(yt))
            scored += 1
    return {
        "accuracy": correct / scored,
        "scored": scored,
        "weights_bitwise_constant": all(torch.equal(a, s.weight) for a, s in zip(before, ev.synapses)),
        "coin_reads": reader.reads,
        "source_start_index": source_offset if source is not None else None,
        "source_end_index": source_offset + len(inputs) - 1 if source is not None else None,
    }


def condition_object_hashes(seed: int, task: str, drive: bool) -> Dict[str, Optional[str]]:
    """Construct and hash every paired object without simulating a transition."""
    w1, w2 = initial_weights(seed)
    if task == "A":
        train_x = a_stream(seed, True)
        train_y = train_x.copy()
        eval_x = a_stream(seed, False)
        eval_y = eval_x.copy()
    elif task == "lag1":
        tr = lag1_stream(seed, True)
        te = lag1_stream(seed, False)
        train_x, train_y = tr["inputs"], tr["targets"]
        eval_x, eval_y = te["inputs"], te["targets"]
    else:
        raise ValueError(task)
    source = background_source_vectors(seed) if drive else None
    return {
        "initial_weights_sha256": sha256(w1, w2),
        "train_stream_sha256": sha256(train_x, train_y),
        "eval_stream_sha256": sha256(eval_x, eval_y),
        "tie_coin_sha256": sha256(tie_coin(seed, task)),
        "drive_sha256": background_hash(source) if source is not None else None,
    }


def run_condition_seed(
    seed: int,
    task: str,
    plastic: bool,
    drive: bool,
    expected_hashes: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, object]:
    """One D2/D3/D4 condition row plus D2-style checkpoints."""
    w1, w2 = initial_weights(seed)
    if task == "A":
        train_x = a_stream(seed, True)
        train_y = train_x.copy()
        eval_x = a_stream(seed, False)
        eval_y = eval_x.copy()
    elif task == "lag1":
        tr = lag1_stream(seed, True)
        te = lag1_stream(seed, False)
        train_x, train_y = tr["inputs"], tr["targets"]
        eval_x, eval_y = te["inputs"], te["targets"]
    else:
        raise ValueError(task)
    source = background_source_vectors(seed) if drive else None
    coin = tie_coin(seed, task)
    local_hashes: Dict[str, Optional[str]] = {
        "initial_weights_sha256": sha256(w1, w2),
        "train_stream_sha256": sha256(train_x, train_y),
        "eval_stream_sha256": sha256(eval_x, eval_y),
        "tie_coin_sha256": sha256(coin),
        "drive_sha256": background_hash(source) if source is not None else None,
    }
    if expected_hashes is not None and dict(expected_hashes) != local_hashes:
        raise AssertionError(f"paired object hash mismatch task={task} seed={seed} drive={drive}")
    condition = ("P" if plastic else "F0") + "_" + task + ("+drive" if drive else "-no")
    net = build_network(w1, w2)
    init_state(net)
    checkpoints = train_condition(net, train_x, train_y, plastic, condition, seed, source)
    eval_result = evaluate_task(net, eval_x, eval_y, coin, source)
    final_weights = tuple(s.weight.detach().cpu().numpy().copy() for s in net.synapses)
    return {
        "diagnostic": "D2/D3/D4",
        "condition": condition,
        "seed": seed,
        "task": task,
        "plastic": plastic,
        "drive": drive,
        "initial_weights_sha256": local_hashes["initial_weights_sha256"],
        "final_weights_sha256": sha256(*final_weights),
        "weights_bitwise_constant": all(np.array_equal(a, b) for a, b in zip((w1, w2), final_weights)),
        "train_stream_sha256": local_hashes["train_stream_sha256"],
        "eval_stream_sha256": local_hashes["eval_stream_sha256"],
        "tie_coin_sha256": local_hashes["tie_coin_sha256"],
        "drive_sha256": local_hashes["drive_sha256"],
        "paired_hash_assertion_passed": expected_hashes is not None,
        "checkpoints": checkpoints,
        "evaluation": eval_result,
    }


def summarize_d1(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if sorted(int(r["seed"]) for r in rows) != list(DIAGNOSTIC_SEEDS):
        raise ValueError("D1 requires exactly seeds 2000..2019")

    def one(name: str) -> Dict[str, object]:
        acc = np.asarray([r["decoders"][name]["accuracy"] for r in rows], dtype=float)
        lower = np.asarray([r["decoders"][name]["lower_95"] for r in rows], dtype=float)
        median = float(np.median(acc))
        count = int((lower > 0.50).sum())
        status = "PASS" if median >= 0.70 and count >= 15 else "FAIL" if median < 0.55 else "INCONCLUSIVE"
        return {"median_accuracy": median, "lower_bounds_gt_half": count, "status": status}

    a = one("A")
    positive = one("positive_prior")
    status = "INVALID_HARNESS" if positive["status"] != "PASS" else a["status"]
    return {"A": a, "positive_prior": positive, "status": status}
