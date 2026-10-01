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
D1_FEATURE_DIM = 60
TRACE20_DECAY = math.exp(-1.0 / 20.0)
TRACE25_DECAY = math.exp(-1.0 / 25.0)
D1_MAX_ITER = 2000
D1_FEATURE_MAP = "phi_t = concat(count20_t, trace20_t, trace25_t) of hidden spikes h_t present before x_t is integrated"
D1_DECODER_CONFIG: Dict[str, object] = {
    "penalty": "l2",
    "C": 1.0,
    "fit_intercept": True,
    "solver": "lbfgs",
    "tol": 1e-8,
    "max_iter": D1_MAX_ITER,
    "class_weight": None,
}
D1_BLOCK_LENGTH = 100
D1_BLOCK_RESAMPLES = 10_000
EVAL_SCORED = EVAL_LEN - EVAL_WARMUP
D1_DECODERS = ("A", "positive_prior")

# Predeclared numeric readout constants (spec sections 4.6, 5.3, 6.3, 7.5).
# Every readout reads these. run_stage2_diagnostic.expected_parameters_from_spec
# transcribes them independently from the spec text and the runner refuses a
# mismatch before any diagnostic transition.
THRESHOLDS: Dict[str, float] = {
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
}


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


def effective_parameters() -> Dict[str, object]:
    """Section 13 effective parameter block, read from a constructed network.

    Network/plasticity values are read back from live objects (not copied from
    source constants), so a construction drift is visible in the package. No
    transition is simulated; the weights are the fixed non-diagnostic fixture.
    """
    net = build_network(*initial_weights(4242))
    hidden, output = net.neurons[0], net.neurons[1]
    p1, p2 = net.plasticities
    return {
        "spec_commit": SPEC_COMMIT,
        "spec_sha256": SPEC_SHA256,
        "stage2_spec_sha256": FROZEN_STAGE2_SHA256,
        "layer_sizes": list(net.layer_sizes),
        "dt_ms": float(net.dt),
        "hidden_neuron": type(hidden).__name__,
        "output_neuron": type(output).__name__,
        "tau_m_ms": [float(hidden.tau_m), float(output.tau_m)],
        "v_rest_mv": [float(hidden.v_rest), float(output.v_rest)],
        "v_reset_mv": [float(hidden.v_reset), float(output.v_reset)],
        "v_thresh_mv": [float(hidden.v_thresh), float(output.v_thresh)],
        "tau_syn_ms": [float(s.tau_syn) for s in net.synapses],
        "hidden_tau_a_ms": float(hidden.tau_a),
        "hidden_beta_a_mv": float(hidden.beta_a),
        "w1_init": "U(-10,10) mV, [20261001, seed, 30, 1] first draw",
        "w2_O1_init": "U(0,10) mV, [20261001, seed, 30, 1] second draw",
        "w2_O0_init": "U(0,10) mV, [20261001, seed, 30, 21]",
        "w1_bounds_mv": [float(p1.w_min), float(p1.w_max)],
        "w2_bounds_mv": [float(p2.w_min), float(p2.w_max)],
        "plasticity_credit": [p1.credit, p2.credit],
        "gamma_mv": [float(p1.lr), float(p2.lr)],
        "tau_plus_ms": [float(p1.tau_plus), float(p2.tau_plus)],
        "tau_minus_ms": [float(p1.tau_minus), float(p2.tau_minus)],
        "tau_elig_ms": [float(p1.tau_elig), float(p2.tau_elig)],
        "a_plus": [float(p1.a_plus), float(p2.a_plus)],
        "a_minus": [float(p1.a_minus), float(p2.a_minus)],
        "reward": "r_t = (2*y_t - 1) * (z1_t - z0_t), one global scalar, next transition",
        "readout": "O1 alone -> 1; O0 alone -> 0; tie -> paired fair tie-coin",
        "noise_p": float(NOISE_P),
        "training_steps": PHASE_STEPS,
        "eval_len": EVAL_LEN,
        "eval_warmup": EVAL_WARMUP,
        "eval_scored": EVAL_SCORED,
        "seeds": list(DIAGNOSTIC_SEEDS),
        "namespace": NAMESPACE,
        "root_entropy": ROOT_ENTROPY,
        "d1": {
            "warmup_rows": D1_WARMUP,
            "train_rows": D1_TRAIN_ROWS,
            "test_rows": D1_TEST_ROWS,
            "feature_dim": D1_FEATURE_DIM,
            "feature_map": D1_FEATURE_MAP,
            "trace20_decay": TRACE20_DECAY,
            "trace25_decay": TRACE25_DECAY,
            "count_window_steps": 20,
            "decoder": dict(D1_DECODER_CONFIG),
            "standardization": "training mean and population std (ddof=0); zero-variance scale 1.0",
            "prediction_threshold": 0.5,
            "block_length": D1_BLOCK_LENGTH,
            "block_resamples": D1_BLOCK_RESAMPLES,
            "block_percentile": 5,
            "positive_control_label": "x_(t-1)",
        },
        "d2": {
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_count": N_CHECKPOINTS,
            "onset_ratio": THRESHOLDS["d2_onset_ratio"],
            "lead_steps": THRESHOLDS["d2_lead_steps"],
            "localized_min_seeds": THRESHOLDS["d2_localized_min_seeds"],
            "bound_fraction_denominator": 40,
        },
        "d3": {
            "train_draws": PHASE_STEPS + 1,
            "eval_draws": EVAL_LEN + 1,
            "target": "y_t = x_(t-1), y_0 = x_-1",
        },
        "d4_drive": {
            "targets": DRIVE_TARGETS,
            "sources_per_target": DRIVE_SOURCES_PER_TARGET,
            "sources_total": DRIVE_TARGETS * DRIVE_SOURCES_PER_TARGET,
            "rate_hz": DRIVE_RATE_HZ,
            "weight_mv": DRIVE_WEIGHT_MV,
            "spike_probability_per_ms": DRIVE_P,
            "realization_steps": DRIVE_STEPS,
            "random_call_shape": [DRIVE_STEPS, DRIVE_TARGETS, DRIVE_SOURCES_PER_TARGET],
            "train_indices": [0, PHASE_STEPS - 1],
            "eval_indices": [PHASE_STEPS, DRIVE_STEPS - 1],
            "hidden_targets": [0, 19],
            "O1_target": 20,
            "O0_target": 21,
            "plastic": False,
            "background_plasticity_objects": len(getattr(net, "background_plasticities", ())),
            "seed_sequence_component": 40,
        },
        "bootstrap": {"resamples": 100_000, "percentile_method": "inverted_cdf", "entropy": "[20261001, 30, 7, k]", "metric_ids": list(range(9))},
        "thresholds": dict(THRESHOLDS),
    }


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
        self.trace20 = TRACE20_DECAY * self.trace20 + h
        self.trace25 = TRACE25_DECAY * self.trace25 + h
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
    model = LogisticRegression(**D1_DECODER_CONFIG)
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
    n_resamples: int = D1_BLOCK_RESAMPLES,
    block_length: int = D1_BLOCK_LENGTH,
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


def run_d1_seed(seed: int, *, n_train: int = PHASE_STEPS, n_test: int = EVAL_LEN) -> Dict[str, object]:
    """One complete D1 seed. The CLI, not this function, owns run authorization.

    Keyword lengths exist only for short construction tests on non-diagnostic
    seeds; the runner uses the spec defaults and the validator rejects others.
    """
    train = capture_d1_features(seed, train=True, n=n_train)
    test = capture_d1_features(seed, train=False, n=n_test)
    tr = slice(D1_WARMUP, None)
    te = slice(D1_WARMUP, None)
    if train["features"][tr].shape[0] != n_train - D1_WARMUP or test["features"][te].shape[0] != n_test - D1_WARMUP:
        raise AssertionError("D1 row-count invariant failed")
    out: Dict[str, object] = {
        "diagnostic": "D1",
        "seed": seed,
        "train_input_hash": train["input_hash"],
        "test_input_hash": test["input_hash"],
        "train_feature_hash": train["feature_hash"],
        "test_feature_hash": test["feature_hash"],
        "initial_weights_sha256": sha256(*initial_weights(seed)),
        "train_rows": int(train["features"][tr].shape[0]),
        "test_rows": int(test["features"][te].shape[0]),
        "warmup_rows": D1_WARMUP,
        "feature_dim": int(train["features"].shape[1]),
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
            "scored": int(correctness.size),
            "decoder_config": dict(D1_DECODER_CONFIG),
            "train_label_sha256": sha256(np.asarray(train[field][tr], dtype=np.int8)),
            "mean": fitted.mean.tolist(),
            "scale": fitted.scale.tolist(),
            "normalization_sha256": normalization_hash(fitted.mean, fitted.scale),
            "coefficients": fitted.coefficients.tolist(),
            "intercept": fitted.intercept.tolist(),
            "coefficients_sha256": coefficients_hash(fitted.coefficients, fitted.intercept),
            "n_iter": [int(v) for v in fitted.n_iter.tolist()],
            "converged": bool(np.all(fitted.n_iter < D1_MAX_ITER)),
        }
    return out


def normalization_hash(mean: Sequence[float], scale: Sequence[float]) -> str:
    """SHA-256 of the stored training-only float64 mean vector then scale vector."""
    return sha256(np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64))


def coefficients_hash(coefficients: Sequence[Sequence[float]], intercept: Sequence[float]) -> str:
    return sha256(np.asarray(coefficients, dtype=np.float64), np.asarray(intercept, dtype=np.float64))


def d1_object_hashes(seed: int) -> Dict[str, str]:
    """Coordinator-side D1 identities, constructed without a network transition."""
    return {
        "initial_weights_sha256": sha256(*initial_weights(seed)),
        "train_input_hash": sha256(a_stream(seed, True)),
        "test_input_hash": sha256(a_stream(seed, False)),
    }


# ----------------------------------------------------------- D1 validation
D1_ROW_FIELDS = (
    "diagnostic",
    "seed",
    "train_input_hash",
    "test_input_hash",
    "train_feature_hash",
    "test_feature_hash",
    "initial_weights_sha256",
    "train_rows",
    "test_rows",
    "warmup_rows",
    "feature_dim",
    "weights_constant",
    "decoders",
)
D1_DECODER_FIELDS = (
    "accuracy",
    "lower_95",
    "scored",
    "decoder_config",
    "train_label_sha256",
    "mean",
    "scale",
    "normalization_sha256",
    "coefficients",
    "intercept",
    "coefficients_sha256",
    "n_iter",
    "converged",
)


def _finite_unit(v: object) -> bool:
    return isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0


def _finite_vector(v: object, n: int) -> bool:
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return a.shape == (n,) and bool(np.isfinite(a).all())


def validate_d1_row(
    row: Mapping[str, object],
    expected: Optional[Mapping[str, str]],
    *,
    train_rows: int = D1_TRAIN_ROWS,
    test_rows: int = D1_TEST_ROWS,
) -> List[str]:
    """Every spec 0.4 / 4.6 INVALID invariant for one D1 seed row. Empty list = valid.

    ``train_rows``/``test_rows`` default to the spec and are overridden only
    by short construction tests; the package validator always uses the spec.
    """
    v: List[str] = []
    seed = row.get("seed") if isinstance(row, Mapping) else None
    tag = f"D1 seed={seed}"
    if not isinstance(row, Mapping):
        return [f"{tag}: row is not a mapping"]
    if row.get("status", "ok") != "ok":
        v.append(f"{tag}: job status {row.get('status')!r}")
    missing = [k for k in D1_ROW_FIELDS if k not in row]
    if missing:
        v.append(f"{tag}: missing field(s) {missing}")
        return v
    if row["diagnostic"] != "D1":
        v.append(f"{tag}: diagnostic label {row['diagnostic']!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        v.append(f"{tag}: seed is not int")
    for k in ("train_input_hash", "test_input_hash", "train_feature_hash", "test_feature_hash", "initial_weights_sha256"):
        if not _is_hash(row[k]):
            v.append(f"{tag}: {k} is not a SHA-256 hex digest")
    if row["train_input_hash"] == row["test_input_hash"]:
        v.append(f"{tag}: train and test input hashes are identical (split not disjoint)")
    if row["train_feature_hash"] == row["test_feature_hash"]:
        v.append(f"{tag}: train and test feature hashes are identical")
    if expected is None:
        v.append(f"{tag}: no coordinator identity supplied")
    else:
        for k in ("initial_weights_sha256", "train_input_hash", "test_input_hash"):
            if row[k] != expected.get(k):
                v.append(f"{tag}: {k} differs from coordinator identity")
    if row["weights_constant"] is not True:
        v.append(f"{tag}: frozen W1/W2 were not bitwise constant")
    for k, want in (("train_rows", train_rows), ("test_rows", test_rows), ("warmup_rows", D1_WARMUP), ("feature_dim", D1_FEATURE_DIM)):
        if row[k] != want or isinstance(row[k], bool):
            v.append(f"{tag}: {k}={row[k]!r}, expected {want}")
    decoders = row["decoders"]
    if not isinstance(decoders, Mapping) or set(decoders) != set(D1_DECODERS):
        v.append(f"{tag}: decoders must be exactly {list(D1_DECODERS)}")
        return v
    for name in D1_DECODERS:
        d = decoders[name]
        dtag = f"{tag} decoder={name}"
        if not isinstance(d, Mapping):
            v.append(f"{dtag}: not a mapping")
            continue
        dmissing = [k for k in D1_DECODER_FIELDS if k not in d]
        if dmissing:
            v.append(f"{dtag}: missing field(s) {dmissing}")
            continue
        if not _finite_unit(d["accuracy"]):
            v.append(f"{dtag}: accuracy is not a finite float in [0,1]")
        elif abs(d["accuracy"] * test_rows - round(d["accuracy"] * test_rows)) > 1e-6:
            v.append(f"{dtag}: accuracy is not a count over {test_rows} scored rows")
        if not _finite_unit(d["lower_95"]):
            v.append(f"{dtag}: lower_95 is not a finite float in [0,1]")
        if d["scored"] != test_rows or isinstance(d["scored"], bool):
            v.append(f"{dtag}: scored={d['scored']!r}, expected {test_rows}")
        config = d["decoder_config"]
        if not isinstance(config, Mapping) or dict(config) != D1_DECODER_CONFIG:
            v.append(f"{dtag}: decoder configuration differs from spec 4.4")
        if not _is_hash(d["train_label_sha256"]):
            v.append(f"{dtag}: train_label_sha256 is not a SHA-256 hex digest")
        mean_ok = _finite_vector(d["mean"], D1_FEATURE_DIM)
        scale_ok = _finite_vector(d["scale"], D1_FEATURE_DIM)
        if not (mean_ok and scale_ok):
            v.append(f"{dtag}: training normalization mean/scale not finite length-{D1_FEATURE_DIM}")
        else:
            if not (np.asarray(d["scale"], dtype=np.float64) > 0.0).all():
                v.append(f"{dtag}: non-positive training scale")
            if d["normalization_sha256"] != normalization_hash(d["mean"], d["scale"]):
                v.append(f"{dtag}: normalization_sha256 does not match stored mean/scale")
        try:
            coef = np.asarray(d["coefficients"], dtype=np.float64)
            icpt = np.asarray(d["intercept"], dtype=np.float64)
            coef_ok = coef.shape == (1, D1_FEATURE_DIM) and icpt.shape == (1,) and bool(np.isfinite(coef).all() and np.isfinite(icpt).all())
        except (TypeError, ValueError):
            coef_ok = False
        if not coef_ok:
            v.append(f"{dtag}: coefficients/intercept not finite (1,{D1_FEATURE_DIM})/(1,)")
        elif d["coefficients_sha256"] != coefficients_hash(d["coefficients"], d["intercept"]):
            v.append(f"{dtag}: coefficients_sha256 does not match stored coefficients")
        n_iter = d["n_iter"]
        iters_ok = isinstance(n_iter, list) and len(n_iter) == 1 and all(isinstance(i, int) and not isinstance(i, bool) for i in n_iter)
        if not iters_ok or not all(0 < i < D1_MAX_ITER for i in n_iter):
            v.append(f"{dtag}: solver iterations {n_iter!r} not strictly below max_iter")
        if d["converged"] is not True:
            v.append(f"{dtag}: decoder did not converge")
    if all(isinstance(decoders[n], Mapping) and "train_label_sha256" in decoders[n] for n in D1_DECODERS):
        if decoders["A"]["train_label_sha256"] == decoders["positive_prior"]["train_label_sha256"]:
            v.append(f"{tag}: A and positive-control training labels are identical")
    return v


def validate_d1_package(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[Mapping[int, Mapping[str, str]]],
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> List[str]:
    """Package-level D1 validity: exact seed set, no duplicates, every row valid."""
    v: List[str] = []
    seen = [r.get("seed") if isinstance(r, Mapping) else None for r in rows]
    if len(rows) != len(seeds):
        v.append(f"D1: {len(rows)} rows, expected {len(seeds)}")
    dupes = sorted({s for s in seen if seen.count(s) > 1}, key=str)
    if dupes:
        v.append(f"D1: duplicate seed rows {dupes}")
    if set(seen) != set(seeds):
        v.append(f"D1: seed set differs; missing {sorted(set(seeds) - set(seen))}, unexpected {sorted((set(seen) - set(seeds)), key=str)}")
    if expected is None:
        v.append("D1: coordinator identities not supplied")
    for row in rows:
        s = row.get("seed") if isinstance(row, Mapping) else None
        v.extend(validate_d1_row(row, None if expected is None else expected.get(s)))  # type: ignore[arg-type]
    return v


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
    lead = THRESHOLDS["d2_lead_steps"]
    if hidden_step is not None and (output_step is None or output_step - hidden_step >= lead):
        return "HIDDEN_FIRST"
    if output_step is not None and (hidden_step is None or hidden_step - output_step >= lead):
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
            if denominator != 0.0 and float(p[c][rate_key]) < THRESHOLDS["d2_onset_ratio"] * denominator:
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


def _condition_objects(seed: int, task: str, drive: bool, n_train: int, n_eval: int):
    w1, w2 = initial_weights(seed)
    if task == "A":
        train_x = a_stream(seed, True, n_train)
        train_y = train_x.copy()
        eval_x = a_stream(seed, False, n_eval)
        eval_y = eval_x.copy()
    elif task == "lag1":
        tr = lag1_stream(seed, True, n_train)
        te = lag1_stream(seed, False, n_eval)
        train_x, train_y = tr["inputs"], tr["targets"]
        eval_x, eval_y = te["inputs"], te["targets"]
    else:
        raise ValueError(task)
    source = background_source_vectors(seed, n_train + n_eval) if drive else None
    coin = tie_coin(seed, task, n_eval)
    return w1, w2, train_x, train_y, eval_x, eval_y, source, coin


def condition_object_hashes(
    seed: int,
    task: str,
    drive: bool,
    *,
    n_train: int = PHASE_STEPS,
    n_eval: int = EVAL_LEN,
) -> Dict[str, Optional[str]]:
    """Construct and hash every paired object without simulating a transition.

    The keyword lengths exist only for short construction tests; the runner
    always uses the spec defaults.
    """
    w1, w2, train_x, train_y, eval_x, eval_y, source, coin = _condition_objects(seed, task, drive, n_train, n_eval)
    return {
        "initial_weights_sha256": sha256(w1, w2),
        "train_stream_sha256": sha256(train_x, train_y),
        "eval_stream_sha256": sha256(eval_x, eval_y),
        "tie_coin_sha256": sha256(coin),
        "drive_sha256": background_hash(source) if source is not None else None,
    }


def run_condition_seed(
    seed: int,
    task: str,
    plastic: bool,
    drive: bool,
    expected_hashes: Optional[Mapping[str, Optional[str]]] = None,
    *,
    n_train: int = PHASE_STEPS,
    n_eval: int = EVAL_LEN,
    warmup: int = EVAL_WARMUP,
) -> Dict[str, object]:
    """One D2/D3/D4 condition row plus D2-style checkpoints.

    The keyword lengths exist only for short construction tests; the runner
    always uses the spec defaults and the package validator rejects any row
    whose checkpoint count or scored length differs from the spec.
    """
    w1, w2, train_x, train_y, eval_x, eval_y, source, coin = _condition_objects(seed, task, drive, n_train, n_eval)
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
    eval_result = evaluate_task(net, eval_x, eval_y, coin, source, source_offset=n_train, warmup=warmup)
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


def summarize_d1(
    rows: Sequence[Mapping[str, object]],
    expected: Optional[Mapping[int, Mapping[str, str]]] = None,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
) -> Dict[str, object]:
    """Spec 4.6 readout. Fail-closed: the strict package validator runs first.

    Any validator finding returns ``INVALID`` with no accuracy predicate
    evaluated and no accuracy statistic in the returned record. Omitting the
    coordinator identities is itself a finding, so a bare call cannot PASS.
    """
    violations = validate_d1_package(rows, expected, seeds)
    if violations:
        return {"status": "INVALID", "valid": False, "rows_schema_valid": False, "violations": violations, "A": None, "positive_prior": None}

    t = THRESHOLDS

    def one(name: str) -> Dict[str, object]:
        acc = np.asarray([r["decoders"][name]["accuracy"] for r in rows], dtype=float)  # type: ignore[index]
        lower = np.asarray([r["decoders"][name]["lower_95"] for r in rows], dtype=float)  # type: ignore[index]
        median = float(np.median(acc))
        count = int((lower > t["d1_lower_bound_reference"]).sum())
        if median >= t["d1_pass_median_accuracy"] and count >= t["d1_pass_min_lower_bounds_gt_half"]:
            status = "PASS"
        elif median < t["d1_fail_median_below"]:
            status = "FAIL"
        else:
            status = "INCONCLUSIVE"
        return {"median_accuracy": median, "lower_bounds_gt_half": count, "status": status}

    a = one("A")
    positive = one("positive_prior")
    status = "INVALID_HARNESS" if positive["status"] != "PASS" else a["status"]
    return {"status": status, "valid": True, "rows_schema_valid": True, "violations": [], "A": a, "positive_prior": positive}
