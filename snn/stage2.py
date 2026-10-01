"""Stage 2: continual hidden-structure learning (docs/STAGE2_SPEC.md @ ef85a15).

Engine: ``snn/core.py`` ``PureSNN.online_step`` with reward mode
``REWARD_PER_SPIKE_NEXT_STEP`` and the Stage 1 temporal MSTDPET parameters;
hidden layer is ``ALIFNeuron`` (sec. 4.3). No separate NumPy learner.

Every constant below is restated from the frozen spec; ``FROZEN_PARAMS`` is
serialized into provenance and checked by tests against the spec values.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from snn.core import REWARD_PER_SPIKE_NEXT_STEP, ALIFNeuron, PureSNN, RSTDPPlasticity

# ---------------------------------------------------------------- frozen spec
SPEC_COMMIT = "ef85a15b0382bdc86b8dba377671e4c2b2e4f79f"
SPEC_SHA256 = "0c1bbdba036b13955c81f2ce0ffee0557b09842ef7d56b0dec0834faf8d8ec87"
ROOT_ENTROPY = 20261001
EXPERIMENTAL_SEEDS = tuple(range(20))
ARMS = ("P", "F0", "FS", "SC", "NS")  # sec. 9.1 order
ARM_NAMES = {
    "P": "primary: SFA on, R-STDP live",
    "F0": "frozen from the start",
    "FS": "freeze at the A->B shift",
    "SC": "temporally scrambled B twin",
    "NS": "SFA off, R-STDP active",
}
CHECKPOINTS = ("A_pre", "B_pre", "B_post", "A_post")  # sec. 9.1 order

FROZEN_PARAMS: Dict[str, object] = {
    "layer_sizes": [2, 20, 1],
    "n_trainable_synapses": 60,
    "dt_ms": 1.0,
    "tau_m_ms": 20.0,
    "tau_syn_ms": 0.0,
    "v_rest_mv": -70.0,
    "v_reset_mv": -70.0,
    "v_thresh_mv": -54.0,
    "dtype": "float64",
    "batch_size": 1,
    "w1_init": "U(-10,10) mV",
    "w2_init": "U(0,10) mV",
    "w1_bounds_mv": [-10.0, 10.0],
    "w2_bounds_mv": [0.0, 10.0],
    "credit": "eligibility",
    "gamma_mv": 0.25,
    "tau_plus_ms": 20.0,
    "tau_minus_ms": 20.0,
    "tau_elig_ms": 25.0,
    "a_plus": 1.0 / 25.0,
    "a_minus": 1.0 / 25.0,
    "reward_mode": REWARD_PER_SPIKE_NEXT_STEP,
    "reward": "r_t = (2*x_t - 1) * z_t",
    "tau_a_ms": 200.0,
    "beta_a_mv": 1.12,
    "beta_a_mv_sfa_off": 0.0,
    "noise_p": 0.10,
    "phase_steps": 200_000,
    "total_steps": 400_000,
    "eval_len": 12_000,
    "eval_warmup": 2_000,
    "eval_scored": 10_000,
    "seeds": list(EXPERIMENTAL_SEEDS),
    "arms": list(ARMS),
    "checkpoints": list(CHECKPOINTS),
    "bootstrap_resamples": 100_000,
    "root_entropy": ROOT_ENTROPY,
}

PHASE_STEPS = 200_000
EVAL_LEN = 12_000
EVAL_WARMUP = 2_000
NOISE_P = 0.10
TAU_A = 200.0
BETA_A = 1.12
SAMPLE_EVERY = 1_000  # instrumentation sampling cadence (steps), declared in docs/STAGE2_IMPLEMENTATION_NOTES.md


# ------------------------------------------------------------------- streams
def rng(entropy) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(entropy)))


def _roll(x: np.ndarray, start: int, flips: np.ndarray, rule: str) -> None:
    """Fill x[start:start+len(flips)] in order; x[start-2], x[start-1] must be set."""
    for j, f in enumerate(flips):
        t = start + j
        ideal = int(x[t - 2]) ^ int(x[t - 1])
        if rule == "B":
            ideal = 1 - ideal
        x[t] = ideal ^ int(f)


def initial_weights(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    g = rng([ROOT_ENTROPY, seed, 1])
    w1 = g.uniform(-10.0, 10.0, size=(20, 2))
    w2 = g.uniform(0.0, 10.0, size=(1, 20))
    return w1.astype(np.float64), w2.astype(np.float64)


def a_training_stream(seed: int, n: int = PHASE_STEPS) -> np.ndarray:
    g = rng([ROOT_ENTROPY, seed, 2])
    x = np.zeros(n, dtype=np.int8)
    x[:2] = g.integers(0, 2, size=2).astype(np.int8)
    flips = g.random(n - 2) < NOISE_P
    _roll(x, 2, flips, "A")
    return x


def b_training_stream(seed: int, a_stream: np.ndarray, n: int = PHASE_STEPS) -> np.ndarray:
    """Structured B; context = last two realized A observations (sec. 3.3/7.1)."""
    g = rng([ROOT_ENTROPY, seed, 3])
    flips = g.random(n) < NOISE_P
    buf = np.zeros(n + 2, dtype=np.int8)
    buf[0], buf[1] = a_stream[-2], a_stream[-1]
    _roll(buf, 2, flips, "B")
    return buf[2:].copy()


def scrambled_b(seed: int, b_struct: np.ndarray) -> np.ndarray:
    g = rng([ROOT_ENTROPY, seed, 4])
    perm = g.permutation(len(b_struct))
    return b_struct[perm].copy()


def heldout_stream(seed: int, rule: str, n: int = EVAL_LEN) -> np.ndarray:
    comp = 5 if rule == "A" else 6
    g = rng([ROOT_ENTROPY, seed, comp])
    x = np.zeros(n, dtype=np.int8)
    x[:2] = g.integers(0, 2, size=2).astype(np.int8)
    flips = g.random(n - 2) < NOISE_P
    _roll(x, 2, flips, rule)
    return x


def sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def seed_objects(seed: int, phase_steps: int = PHASE_STEPS, eval_len: int = EVAL_LEN) -> Dict[str, np.ndarray]:
    w1, w2 = initial_weights(seed)
    a = a_training_stream(seed, phase_steps)
    b = b_training_stream(seed, a, phase_steps)
    return {
        "w1": w1,
        "w2": w2,
        "A_train": a,
        "B_train": b,
        "B_scrambled": scrambled_b(seed, b),
        "A_heldout": heldout_stream(seed, "A", eval_len),
        "B_heldout": heldout_stream(seed, "B", eval_len),
    }


def object_hashes(obj: Dict[str, np.ndarray]) -> Dict[str, str]:
    return {
        "initial_weights": sha256(obj["w1"], obj["w2"]),
        "A_train": sha256(obj["A_train"]),
        "B_train": sha256(obj["B_train"]),
        "B_scrambled": sha256(obj["B_scrambled"]),
        "A_heldout": sha256(obj["A_heldout"]),
        "B_heldout": sha256(obj["B_heldout"]),
    }


def arm_shared_hash_keys(arm: str) -> Tuple[str, ...]:
    keys = ("initial_weights", "A_train", "B_train", "A_heldout", "B_heldout")
    return keys + (("B_scrambled",) if arm == "SC" else ())


def order2_lookup_accuracy(x: np.ndarray) -> float:
    """Best fixed (x_{t-2}, x_{t-1}) -> majority-next-bit accuracy on the whole stream (sec. 6.4)."""
    x = np.asarray(x, dtype=np.int64)
    ctx = 2 * x[:-2] + x[1:-1]
    nxt = x[2:]
    correct = 0
    for c in range(4):
        m = ctx == c
        ones = int(nxt[m].sum())
        correct += max(ones, int(m.sum()) - ones)
    return correct / float(len(nxt))


# ------------------------------------------------------------------- network
class CountingRSTDP(RSTDPPlasticity):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reset_calls = 0

    def reset(self, batch_size, device, dtype=None):
        self.reset_calls += 1
        super().reset(batch_size, device, dtype)


class CountingALIF(ALIFNeuron):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reset_calls = 0

    def reset_adaptation(self, batch_size, device=None, dtype=None):
        self.reset_calls += 1
        super().reset_adaptation(batch_size, device, dtype)


class Stage2SNN(PureSNN):
    """PureSNN that logs every state reset / weight restore (Gate 4 item 2)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reset_log: List[str] = []

    def reset_online_state(self, batch_size=1, device=None, dtype=None):
        self.reset_log.append("reset_online_state")
        super().reset_online_state(batch_size, device, dtype)

    def restore_frozen_weights(self, frozen):
        self.reset_log.append("restore_frozen_weights")
        super().restore_frozen_weights(frozen)

    def reset_counts(self) -> Dict[str, int]:
        return {
            "reset_online_state": sum(e == "reset_online_state" for e in self.reset_log),
            "restore_frozen_weights": sum(e == "restore_frozen_weights" for e in self.reset_log),
            "plasticity_reset": sum(p.reset_calls for p in self.plasticities),
            "sfa_reset": sum(getattr(n, "reset_calls", 0) for n in self.neurons),
        }


W1_BOUNDS = (-10.0, 10.0)
W2_BOUNDS = (0.0, 10.0)


def build_network(w1: np.ndarray, w2: np.ndarray, beta_a: float) -> Stage2SNN:
    net = Stage2SNN(
        [2, 20, 1], dt=1.0, weight_scale=0.0, tau_m=20.0, tau_syn=0.0,
        v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0,
    ).double()
    net.neurons[0] = CountingALIF(20, tau_m=20.0, v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0,
                                  dt=1.0, tau_a=TAU_A, beta_a=beta_a)
    with torch.no_grad():
        net.synapses[0].weight.copy_(torch.as_tensor(w1, dtype=torch.float64))
        net.synapses[1].weight.copy_(torch.as_tensor(w2, dtype=torch.float64))
    # No autograd: R-STDP is the only weight writer. Without this, membrane state
    # would chain an autograd graph across all 400,000 steps (arithmetic is identical).
    for p in net.parameters():
        p.requires_grad_(False)
    amp = 1.0 / 25.0
    net.plasticities = [
        CountingRSTDP(syn, lr=0.25, tau_elig=25.0, dt=1.0, a_plus=amp, a_minus=amp,
                      tau_plus=20.0, tau_minus=20.0, w_min=lo, w_max=hi, credit="eligibility")
        for syn, (lo, hi) in zip(net.synapses, (W1_BOUNDS, W2_BOUNDS))
    ]
    return net


def init_state(net: Stage2SNN) -> None:
    """Sec. 5.1 start state: v_rest everywhere, zero spikes/traces/eligibility/adaptation."""
    net.reset_online_state(batch_size=1, dtype=torch.float64)


ONEHOT = (torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([[0.0, 1.0]], dtype=torch.float64))


def state_bytes(net: Stage2SNN) -> bytes:
    """Byte serialization of the full live state (weights + dynamic state + counters)."""
    parts: List[bytes] = []
    for s in net.synapses:
        parts.append(s.weight.data.numpy().tobytes())
    for v in net._v:
        parts.append(v.numpy().tobytes())
    for z in net._spikes:
        parts.append(z.numpy().tobytes())
    parts.append(net.neurons[0].a.numpy().tobytes())
    for p in net.plasticities:
        for t in (p.eligibility, p.pre_trace, p.post_trace, p.last_pairing):
            parts.append(t.numpy().tobytes())
    parts.append(repr(net.reset_counts()).encode())
    return b"".join(parts)


def start_state_ok(net: Stage2SNN) -> bool:
    ok = all(bool((v == -70.0).all()) for v in net._v)
    ok &= all(bool((z == 0).all()) for z in net._spikes)
    ok &= bool((net.neurons[0].a == 0).all())
    for p in net.plasticities:
        for t in (p.eligibility, p.pre_trace, p.post_trace, p.last_pairing):
            ok &= bool((t == 0).all())
    return ok


# ---------------------------------------------------------------- evaluation
class SourceLog:
    """Access log: how many observations each consumer read from each named stream."""

    def __init__(self):
        self.counts: Dict[str, int] = {}

    def add(self, consumer: str, stream: str, n: int) -> None:
        key = f"{consumer}:{stream}"
        self.counts[key] = self.counts.get(key, 0) + n


def evaluate(live: Stage2SNN, x: np.ndarray, log: Optional[SourceLog] = None, name: str = "",
             warmup: int = EVAL_WARMUP) -> Dict[str, object]:
    """Non-mutating held-out evaluation (sec. 5): deepcopy, sec. 5.1 start state, no reward."""
    ev = copy.deepcopy(live)
    ev.reset_log = []
    init_state(ev)
    start_ok = start_state_ok(ev)
    w_before = [s.weight.data.clone() for s in ev.synapses]
    correct = 0
    ones = 0
    scored = 0
    for i in range(len(x)):
        xi = int(x[i])
        z = int(ev.current_output()[0, 0].item())
        ev.online_step(ONEHOT[xi], reward_fn=None)
        if i >= warmup:
            scored += 1
            correct += int(z == xi)
            ones += z
    w_const = all(torch.equal(a, s.weight.data) for a, s in zip(w_before, ev.synapses))
    if log is not None:
        log.add("eval", name, len(x))
    return {
        "accuracy": correct / scored,
        "prediction_one_fraction": ones / scored,
        "scored": scored,
        "eval_copy_start_state_ok": bool(start_ok),
        "eval_copy_init_calls": ev.reset_counts()["reset_online_state"],
        "eval_weights_bitwise_constant": bool(w_const),
    }


# ------------------------------------------------------------------ training
class PhaseStats:
    def __init__(self):
        self.steps = 0
        self.steps_with_reward_fn = 0
        self.output_spikes = 0
        self.hidden_spikes = torch.zeros(1, 20, dtype=torch.float64)
        self.pos = 0
        self.neg = 0
        self.abs_dw = [0.0, 0.0]
        self.rewards: List[float] = []
        self.dw_out: List[float] = []
        self.a_sum = torch.zeros(1, 20, dtype=torch.float64)
        self.a_max = torch.zeros(1, 20, dtype=torch.float64)
        self.thr_samples_max_abs = 0.0
        self.samples = 0
        self.nonfinite = False
        self.out_of_bounds = False

    def summary(self, beta_a: float) -> Dict[str, object]:
        from snn.florian_bench import pearson

        n = max(self.steps, 1)
        a_mean = float(self.a_sum.sum()) / (20.0 * n)
        a_max = float(self.a_max.max())
        r = np.asarray(self.rewards, dtype=float)
        d = np.asarray(self.dw_out, dtype=float)
        return {
            "reward_dw_sufficient_stats": {
                "n": int(r.size), "sum_r": float(r.sum()), "sum_d": float(d.sum()),
                "sum_rr": float((r * r).sum()), "sum_dd": float((d * d).sum()), "sum_rd": float((r * d).sum()),
            },
            "steps": self.steps,
            "steps_with_reward_fn": self.steps_with_reward_fn,
            "output_spikes": self.output_spikes,
            "output_rate_hz": 1000.0 * self.output_spikes / n,
            "hidden_rate_hz": 1000.0 * float(self.hidden_spikes.sum()) / (20.0 * n),
            "reward_events_positive": self.pos,
            "reward_events_negative": self.neg,
            "abs_weight_change_w1_mv": self.abs_dw[0],
            "abs_weight_change_w2_mv": self.abs_dw[1],
            "total_abs_weight_change_mv": self.abs_dw[0] + self.abs_dw[1],
            "reward_signed_dw_out_corr": pearson(self.rewards, self.dw_out),
            "sfa_a_mean": a_mean,
            "sfa_a_max": a_max,
            "sfa_threshold_contribution_mean_mv": beta_a * a_mean,
            "sfa_threshold_contribution_max_mv": beta_a * a_max,
            "sfa_threshold_sampled_max_abs_mv": self.thr_samples_max_abs,
            "instrumentation_samples": self.samples,
            "nonfinite_detected": self.nonfinite,
            "out_of_bounds_detected": self.out_of_bounds,
        }


def _check(net: Stage2SNN, st: PhaseStats) -> None:
    st.samples += 1
    alif = net.neurons[0]
    st.thr_samples_max_abs = max(st.thr_samples_max_abs, float(alif.threshold_contribution().abs().max()))
    tensors = [s.weight.data for s in net.synapses] + list(net._v) + [alif.a]
    tensors += [p.eligibility for p in net.plasticities]
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        st.nonfinite = True
    w1, w2 = net.synapses[0].weight.data, net.synapses[1].weight.data
    if float(w1.min()) < W1_BOUNDS[0] or float(w1.max()) > W1_BOUNDS[1] or float(w2.min()) < W2_BOUNDS[0] or float(w2.max()) > W2_BOUNDS[1]:
        st.out_of_bounds = True


def train_phase(net: Stage2SNN, x: np.ndarray, plastic: bool, st: PhaseStats,
                log: Optional[SourceLog] = None, name: str = "") -> None:
    w = [s.weight.data for s in net.synapses]
    alif = net.neurons[0]
    for t in range(len(x)):
        xt = int(x[t])
        sign = 1.0 if xt else -1.0
        if plastic:
            will_reward = bool(net.current_output()[0, 0].item())
            before = [u.clone() for u in w] if will_reward else None
            out, reward = net.online_step(ONEHOT[xt], reward_fn=lambda f: sign * float(f[0, 0]))
            st.steps_with_reward_fn += 1
            if reward != 0.0:
                assert before is not None
                d1 = float((w[0] - before[0]).abs().sum())
                d2 = float((w[1] - before[1]).abs().sum())
                st.abs_dw[0] += d1
                st.abs_dw[1] += d2
                st.rewards.append(reward)
                st.dw_out.append(float((w[1] - before[1]).sum()))
                if reward > 0:
                    st.pos += 1
                else:
                    st.neg += 1
        else:
            out, reward = net.online_step(ONEHOT[xt], reward_fn=None)
        st.output_spikes += int(out[0, 0].item())
        st.hidden_spikes.add_(net._spikes[0])
        st.a_sum.add_(alif.a)
        torch.maximum(st.a_max, alif.a, out=st.a_max)
        st.steps += 1
        if st.steps % SAMPLE_EVERY == 0:
            _check(net, st)
    _check(net, st)
    if log is not None:
        log.add("train", name, len(x))


def weight_arrays(net: Stage2SNN) -> Tuple[np.ndarray, np.ndarray]:
    return tuple(s.weight.data.numpy().copy() for s in net.synapses)  # type: ignore[return-value]


def _bound_hits(w1: np.ndarray, w2: np.ndarray) -> Dict[str, int]:
    return {
        "w1_at_min": int((w1 == W1_BOUNDS[0]).sum()),
        "w1_at_max": int((w1 == W1_BOUNDS[1]).sum()),
        "w2_at_min": int((w2 == W2_BOUNDS[0]).sum()),
        "w2_at_max": int((w2 == W2_BOUNDS[1]).sum()),
    }


def run_arm_seed(arm: str, seed: int, expected_hashes: Optional[Dict[str, str]] = None,
                 phase_steps: int = PHASE_STEPS, eval_len: int = EVAL_LEN,
                 eval_warmup: int = EVAL_WARMUP) -> Dict[str, object]:
    """One continuous Stage 2 run (sec. 3-8) for one arm and seed."""
    if arm not in ARMS:
        raise ValueError(arm)
    torch.set_num_threads(1)
    t0 = time.time()
    obj = seed_objects(seed, phase_steps, eval_len)
    hashes = object_hashes(obj)
    shared = arm_shared_hash_keys(arm)
    hash_ok = True
    if expected_hashes is not None:
        hash_ok = all(hashes[k] == expected_hashes[k] for k in shared)
        if not hash_ok:
            raise AssertionError(f"stream hash mismatch arm={arm} seed={seed}")
    beta_a = 0.0 if arm == "NS" else BETA_A
    net = build_network(obj["w1"], obj["w2"], beta_a)
    n_weights_before = sum(s.weight.numel() for s in net.synapses)
    init_state(net)
    init_counts = net.reset_counts()
    init_start_ok = start_state_ok(net)
    w_init = weight_arrays(net)
    log = SourceLog()

    plastic_a = arm in ("P", "FS", "SC", "NS")
    plastic_b = arm in ("P", "SC", "NS")
    b_stream_name = "B_scrambled" if arm == "SC" else "B_train"
    b_stream = obj[b_stream_name]

    st_a = PhaseStats()
    train_phase(net, obj["A_train"], plastic_a, st_a, log, "A_train")
    w_shift = weight_arrays(net)
    counts_after_a = net.reset_counts()
    live_bytes_before_pre = state_bytes(net)
    evals: Dict[str, Dict[str, object]] = {}
    evals["A_pre"] = evaluate(net, obj["A_heldout"], log, "A_heldout", eval_warmup)
    evals["B_pre"] = evaluate(net, obj["B_heldout"], log, "B_heldout", eval_warmup)
    pre_isolated = state_bytes(net) == live_bytes_before_pre

    st_b = PhaseStats()
    train_phase(net, b_stream, plastic_b, st_b, log, b_stream_name)
    w_final = weight_arrays(net)
    live_bytes_before_post = state_bytes(net)
    evals["B_post"] = evaluate(net, obj["B_heldout"], log, "B_heldout", eval_warmup)
    evals["A_post"] = evaluate(net, obj["A_heldout"], log, "A_heldout", eval_warmup)
    post_isolated = state_bytes(net) == live_bytes_before_post
    counts_final = net.reset_counts()
    n_weights_after = sum(s.weight.numel() for s in net.synapses)

    def maxdiff(a, b):
        return float(max(np.max(np.abs(a[0] - b[0])), np.max(np.abs(a[1] - b[1]))))

    train_reads = {k: v for k, v in log.counts.items() if k.startswith("train:")}
    heldout_consumed_by_training = sum(v for k, v in train_reads.items() if "heldout" in k)
    w1f, w2f = w_final
    row: Dict[str, object] = {
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "seed": seed,
        "engine": "snn/core.py PureSNN.online_step + ALIFNeuron (reward_mode=per_spike_next_step)",
        "beta_a_mv": beta_a,
        "plastic_A": plastic_a,
        "plastic_B": plastic_b,
        "B_training_stream": b_stream_name,
        "phase_steps": phase_steps,
        "eval_len": eval_len,
        "eval_warmup": eval_warmup,
        "hashes": hashes,
        "shared_hash_keys": list(shared),
        "hash_assertion_passed": bool(hash_ok and expected_hashes is not None),
        "accuracy": {c: evals[c]["accuracy"] for c in CHECKPOINTS},
        "prediction_one_fraction": {c: evals[c]["prediction_one_fraction"] for c in CHECKPOINTS},
        "evaluations": evals,
        "n_evaluations": len(evals),
        "evaluation_isolation_byte_identical": bool(pre_isolated and post_isolated),
        "phase_A": st_a.summary(beta_a),
        "phase_B": st_b.summary(beta_a),
        "max_abs_dW_total": maxdiff(w_final, w_init),
        "max_abs_dW_A": maxdiff(w_shift, w_init),
        "max_abs_dW_B": maxdiff(w_final, w_shift),
        "final_w1_min": float(w1f.min()),
        "final_w1_max": float(w1f.max()),
        "final_w2_min": float(w2f.min()),
        "final_w2_max": float(w2f.max()),
        "final_bound_hits": _bound_hits(w1f, w2f),
        "n_trainable_weights_before": n_weights_before,
        "n_trainable_weights_after": n_weights_after,
        "live_init_counts": init_counts,
        "live_reset_counts_after_A": counts_after_a,
        "live_reset_counts_final": counts_final,
        "live_post_init_reset_calls": {k: counts_final[k] - init_counts[k] for k in counts_final},
        "live_init_start_state_ok": bool(init_start_ok),
        "access_log": dict(sorted(log.counts.items())),
        "heldout_observations_consumed_by_training": heldout_consumed_by_training,
        "replay_buffer": None,
        "B_stream_symbol_counts": [int((b_stream == 0).sum()), int((b_stream == 1).sum())],
        "B_struct_symbol_counts": [int((obj["B_train"] == 0).sum()), int((obj["B_train"] == 1).sum())],
        "B_stream_input_spikes": int(len(b_stream)),  # one-hot: exactly one input spike per step
        "B_struct_input_spikes": int(len(obj["B_train"])),
        "B_stream_order2_lookup_accuracy": order2_lookup_accuracy(b_stream),
        "training_hashes_differ_from_heldout": bool(
            {hashes["A_train"], hashes["B_train"], hashes["B_scrambled"]}.isdisjoint({hashes["A_heldout"], hashes["B_heldout"]})
        ),
        "wall_seconds": time.time() - t0,
    }
    return row
