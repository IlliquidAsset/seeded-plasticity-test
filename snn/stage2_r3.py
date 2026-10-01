"""Stage 2 r3: continual hidden-structure learning, two-output readout (docs/STAGE2_SPEC.md @ def4b36).

r3 changes exactly one mechanism relative to r2 (``snn/stage2.py``, kept unchanged
for r2 reproducibility): the single LIF output becomes two non-adaptive LIF outputs,
``O1`` (index 0, "predict 1") and ``O0`` (index 1, "predict 0"), network ``[2, 20, 2]``,
with ONE global scalar reward ``r_t = (2*x_t - 1) * (z1_t - z0_t)`` passed unchanged
to ``PureSNN.online_step`` (Stage 1 ``REWARD_PER_SPIKE_NEXT_STEP`` path). Evaluation
predictions use the sec. 3.1 rule with a fair tie-coin from its own SeedSequence
component. No learning code changes: plasticity, reward application, and clamp are
``snn/core.py`` unmodified.

Streams, W1, W2 row O1, the scramble, and the bootstrap are r2's procedures,
imported from ``snn/stage2.py`` so every r2 object is bitwise identical.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from array import array
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from snn.core import REWARD_PER_SPIKE_NEXT_STEP, ALIFNeuron, PureSNN, RSTDPPlasticity
from snn.stage2 import (  # r2 procedures, reused unchanged (sec. 7: every r2 draw bitwise identical)
    ARM_NAMES, ARMS, CHECKPOINTS, EXPERIMENTAL_SEEDS, ROOT_ENTROPY, SourceLog,
    a_training_stream, b_training_stream, heldout_stream, order2_lookup_accuracy, rng,
    scrambled_b, sha256,
)
from snn.stage2 import initial_weights as r2_initial_weights

# ---------------------------------------------------------------- frozen spec
SPEC_COMMIT = "def4b36220762a906f360ea44b9d82102c9cb4d2"
SPEC_SHA256 = "695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd"
QUALIFICATION_SEEDS = tuple(range(1000, 1020))  # sec. 11 item 15, non-experimental
O1, O0 = 0, 1  # output indices (sec. 3.1)
OUTPUT_NAMES = ("O1", "O0")
COIN_COMPONENT = 22
O0_COMPONENT = 21
CHECKPOINT_COIN_INDEX = {"A_pre": 1, "B_pre": 2, "B_post": 3, "A_post": 4}  # sec. 7 c = 1..4
COIN_LEN = 12_000

FROZEN_PARAMS: Dict[str, object] = {
    "spec_revision": "r3",
    "layer_sizes": [2, 20, 2],
    "n_trainable_synapses": 80,
    "outputs": ["O1 (index 0, predict 1)", "O0 (index 1, predict 0)"],
    "output_neuron": "LIF, non-adaptive, no lateral/recurrent/inhibitory connection",
    "dt_ms": 1.0,
    "tau_m_ms": 20.0,
    "tau_syn_ms": 0.0,
    "v_rest_mv": -70.0,
    "v_reset_mv": -70.0,
    "v_thresh_mv": -54.0,
    "dtype": "float64",
    "batch_size": 1,
    "w1_init": "U(-10,10) mV [20261001, seed, 1]",
    "w2_O1_init": "U(0,10) mV [20261001, seed, 1] (r2 W2 draw)",
    "w2_O0_init": "U(0,10) mV [20261001, seed, 21]",
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
    "reward": "r_t = (2*x_t - 1) * (z1_t - z0_t)",
    "prediction_rule": "O1 alone -> 1; O0 alone -> 0; tie -> coin[seed, c][i]",
    "tie_coin": "[20261001, seed, 22, c], c=1..4 (A_pre,B_pre,B_post,A_post); integers(0,2,size=12000) int8",
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
    "qualification_seeds": list(QUALIFICATION_SEEDS),
    "arms": list(ARMS),
    "checkpoints": list(CHECKPOINTS),
    "bootstrap_resamples": 100_000,
    "root_entropy": ROOT_ENTROPY,
}

PHASE_STEPS = 200_000
EVAL_LEN = 12_000
EVAL_WARMUP = 2_000
TAU_A = 200.0
BETA_A = 1.12
SAMPLE_EVERY = 1_000  # instrumentation sampling cadence (steps); docs/STAGE2_R3_IMPLEMENTATION_NOTES.md
W1_BOUNDS = (-10.0, 10.0)
W2_BOUNDS = (0.0, 10.0)
N_TRAINABLE = 80


# ------------------------------------------------------------- seed objects
def initial_weights(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sec. 7.1: W1 and W2 row O1 from component 1 (r2 draw), W2 row O0 from component 21."""
    w1, w2_o1 = r2_initial_weights(seed)
    h = rng([ROOT_ENTROPY, seed, O0_COMPONENT])
    w2_o0 = h.uniform(0.0, 10.0, size=(1, 20))
    w2 = np.concatenate([w2_o1, w2_o0], axis=0).astype(np.float64)
    return w1.astype(np.float64), w2


def tie_coin(seed: int, checkpoint: str) -> np.ndarray:
    """Sec. 7.1: fresh generator [20261001, seed, 22, c]; integers(0, 2, size=12000) -> int8."""
    c = CHECKPOINT_COIN_INDEX[checkpoint]
    g = rng([ROOT_ENTROPY, seed, COIN_COMPONENT, c])
    return g.integers(0, 2, size=COIN_LEN).astype(np.int8)


def seed_objects(seed: int, phase_steps: int = PHASE_STEPS, eval_len: int = EVAL_LEN) -> Dict[str, np.ndarray]:
    w1, w2 = initial_weights(seed)
    a = a_training_stream(seed, phase_steps)
    b = b_training_stream(seed, a, phase_steps)
    obj = {
        "w1": w1,
        "w2": w2,
        "A_train": a,
        "B_train": b,
        "B_scrambled": scrambled_b(seed, b),
        "A_heldout": heldout_stream(seed, "A", eval_len),
        "B_heldout": heldout_stream(seed, "B", eval_len),
    }
    for ck in CHECKPOINTS:
        obj[f"coin_{ck}"] = tie_coin(seed, ck)
    return obj


COIN_KEYS = tuple(f"coin_{ck}" for ck in CHECKPOINTS)


def object_hashes(obj: Dict[str, np.ndarray]) -> Dict[str, str]:
    h = {
        "initial_weights": sha256(obj["w1"], obj["w2"]),
        "A_train": sha256(obj["A_train"]),
        "B_train": sha256(obj["B_train"]),
        "B_scrambled": sha256(obj["B_scrambled"]),
        "A_heldout": sha256(obj["A_heldout"]),
        "B_heldout": sha256(obj["B_heldout"]),
    }
    for k in COIN_KEYS:
        h[k] = sha256(obj[k])
    return h


def arm_shared_hash_keys(arm: str) -> Tuple[str, ...]:
    keys = ("initial_weights", "A_train", "B_train", "A_heldout", "B_heldout") + COIN_KEYS
    return keys + (("B_scrambled",) if arm == "SC" else ())


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


class Stage2R3SNN(PureSNN):
    """PureSNN [2, 20, 2] that logs every state reset / weight restore (Gate 4 item 2)."""

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


def build_network(w1: np.ndarray, w2: np.ndarray, beta_a: float) -> Stage2R3SNN:
    net = Stage2R3SNN(
        [2, 20, 2], dt=1.0, weight_scale=0.0, tau_m=20.0, tau_syn=0.0,
        v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0,
    ).double()
    net.neurons[0] = CountingALIF(20, tau_m=20.0, v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0,
                                  dt=1.0, tau_a=TAU_A, beta_a=beta_a)
    with torch.no_grad():
        net.synapses[0].weight.copy_(torch.as_tensor(w1, dtype=torch.float64))
        net.synapses[1].weight.copy_(torch.as_tensor(w2, dtype=torch.float64))
    # No autograd: R-STDP is the only weight writer (arithmetic identical; avoids a 400k-step graph).
    for p in net.parameters():
        p.requires_grad_(False)
    amp = 1.0 / 25.0
    net.plasticities = [
        CountingRSTDP(syn, lr=0.25, tau_elig=25.0, dt=1.0, a_plus=amp, a_minus=amp,
                      tau_plus=20.0, tau_minus=20.0, w_min=lo, w_max=hi, credit="eligibility")
        for syn, (lo, hi) in zip(net.synapses, (W1_BOUNDS, W2_BOUNDS))
    ]
    return net


def init_state(net: Stage2R3SNN) -> None:
    """Sec. 5.1 start state: v_rest everywhere, zero spikes/traces/eligibility/adaptation."""
    net.reset_online_state(batch_size=1, dtype=torch.float64)


ONEHOT = (torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([[0.0, 1.0]], dtype=torch.float64))


def make_reward_fn(xt: int) -> Callable[[torch.Tensor], float]:
    """Sec. 3.1 / 4.2: reward_fn(f) = (2*x_t - 1) * (f[0, 0] - f[0, 1]); ONE global scalar."""
    sign = 1.0 if xt else -1.0
    return lambda f: sign * (float(f[0, O1]) - float(f[0, O0]))


def state_bytes(net: Stage2R3SNN) -> bytes:
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


def start_state_ok(net: Stage2R3SNN) -> bool:
    ok = all(bool((v == -70.0).all()) for v in net._v)
    ok &= all(bool((z == 0).all()) for z in net._spikes)
    ok &= bool((net.neurons[0].a == 0).all())
    for p in net.plasticities:
        for t in (p.eligibility, p.pre_trace, p.post_trace, p.last_pairing):
            ok &= bool((t == 0).all())
    return ok


# ---------------------------------------------------------------- evaluation
class CoinReader:
    """Read-counting view of one checkpoint's tie-coin vector (sec. 14 item 14c access log)."""

    def __init__(self, coin: np.ndarray):
        self._coin = coin
        self.reads = 0
        self.reads_scored = 0
        self.indices: List[int] = []

    def read(self, i: int, scored: bool) -> int:
        self.reads += 1
        self.reads_scored += int(scored)
        self.indices.append(i)
        return int(self._coin[i])


def predict(z1: int, z0: int, i: int, coin: CoinReader, scored: bool) -> int:
    """Sec. 3.1 prediction rule."""
    if z1 == 1 and z0 == 0:
        return 1
    if z1 == 0 and z0 == 1:
        return 0
    return coin.read(i, scored)


def evaluate(live: Stage2R3SNN, x: np.ndarray, coin: np.ndarray, log: Optional[SourceLog] = None,
             name: str = "", coin_name: str = "", warmup: int = EVAL_WARMUP,
             return_predictions: bool = False) -> Dict[str, object]:
    """Non-mutating held-out evaluation (sec. 5): deepcopy, sec. 5.1 start state, no reward."""
    ev = copy.deepcopy(live)
    ev.reset_log = []
    init_state(ev)
    start_ok = start_state_ok(ev)
    w_before = [s.weight.data.clone() for s in ev.synapses]
    reader = CoinReader(coin)
    correct = ones = scored = 0
    n_o1 = n_o0 = n_silent = n_both = 0
    nontie_correct = 0
    preds: List[int] = []
    for i in range(len(x)):
        xi = int(x[i])
        out = ev.current_output()
        z1, z0 = int(out[0, O1].item()), int(out[0, O0].item())
        is_scored = i >= warmup
        pred = predict(z1, z0, i, reader, is_scored)
        ev.online_step(ONEHOT[xi], reward_fn=None)
        if return_predictions:
            preds.append(pred)
        if is_scored:
            scored += 1
            correct += int(pred == xi)
            ones += pred
            if z1 != z0:
                nontie_correct += int(pred == xi)
                if z1:
                    n_o1 += 1
                else:
                    n_o0 += 1
            elif z1:
                n_both += 1
            else:
                n_silent += 1
    w_const = all(torch.equal(a, s.weight.data) for a, s in zip(w_before, ev.synapses))
    if log is not None:
        log.add("eval", name, len(x))
        log.add("eval", coin_name, reader.reads)
    nontie = n_o1 + n_o0
    res: Dict[str, object] = {
        "accuracy": correct / scored,
        "prediction_one_fraction": ones / scored,
        "O1_only_fraction": n_o1 / scored,
        "O0_only_fraction": n_o0 / scored,
        "both_silent_fraction": n_silent / scored,
        "both_fire_fraction": n_both / scored,
        "nontie_count": nontie,
        "nontie_accuracy": (nontie_correct / nontie) if nontie else None,
        "scored": scored,
        "coin_reads": reader.reads,
        "coin_reads_scored": reader.reads_scored,
        "coin_reads_scored_equal_scored_ties": reader.reads_scored == n_silent + n_both,
        "eval_copy_start_state_ok": bool(start_ok),
        "eval_copy_init_calls": ev.reset_counts()["reset_online_state"],
        "eval_weights_bitwise_constant": bool(w_const),
    }
    if return_predictions:
        res["predictions"] = preds
        res["coin_read_indices"] = list(reader.indices)
    return res


# ------------------------------------------------------------------ training
class PhaseStats:
    def __init__(self):
        self.steps = 0
        self.steps_with_reward_fn = 0
        self.out_spikes = [0, 0]
        self.state_counts = {"O1_only": 0, "O0_only": 0, "both_silent": 0, "both_fire": 0}
        self.hidden_spikes = torch.zeros(1, 20, dtype=torch.float64)
        self.pos = 0
        self.neg = 0
        self.abs_dw_w1 = 0.0
        self.abs_dw_w2_rows = [0.0, 0.0]
        self.rewards = array("d")
        self.dw_out = array("d")
        self.dw_rows = (array("d"), array("d"))
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
        r = np.frombuffer(self.rewards, dtype=float) if len(self.rewards) else np.zeros(0)
        d = np.frombuffer(self.dw_out, dtype=float) if len(self.dw_out) else np.zeros(0)
        rows = [np.frombuffer(x, dtype=float) if len(x) else np.zeros(0) for x in self.dw_rows]
        w2_abs = self.abs_dw_w2_rows[0] + self.abs_dw_w2_rows[1]
        return {
            "reward_dw_sufficient_stats": {
                "n": int(r.size), "sum_r": float(r.sum()), "sum_d": float(d.sum()),
                "sum_rr": float((r * r).sum()), "sum_dd": float((d * d).sum()), "sum_rd": float((r * d).sum()),
            },
            "steps": self.steps,
            "steps_with_reward_fn": self.steps_with_reward_fn,
            "output_spikes": self.out_spikes[0] + self.out_spikes[1],
            "output_spikes_O1": self.out_spikes[0],
            "output_spikes_O0": self.out_spikes[1],
            "output_rate_hz_O1": 1000.0 * self.out_spikes[0] / n,
            "output_rate_hz_O0": 1000.0 * self.out_spikes[1] / n,
            "hidden_rate_hz": 1000.0 * float(self.hidden_spikes.sum()) / (20.0 * n),
            "O1_only_fraction": self.state_counts["O1_only"] / n,
            "O0_only_fraction": self.state_counts["O0_only"] / n,
            "both_silent_fraction": self.state_counts["both_silent"] / n,
            "both_fire_fraction": self.state_counts["both_fire"] / n,
            "reward_events_positive": self.pos,
            "reward_events_negative": self.neg,
            "abs_weight_change_w1_mv": self.abs_dw_w1,
            "abs_weight_change_w2_mv": w2_abs,
            "abs_weight_change_w2_O1_mv": self.abs_dw_w2_rows[0],
            "abs_weight_change_w2_O0_mv": self.abs_dw_w2_rows[1],
            "total_abs_weight_change_mv": self.abs_dw_w1 + w2_abs,
            "reward_signed_dw_out_corr": pearson(r, d),
            "reward_signed_dw_O1_corr": pearson(r, rows[0]),
            "reward_signed_dw_O0_corr": pearson(r, rows[1]),
            "sfa_a_mean": a_mean,
            "sfa_a_max": a_max,
            "sfa_threshold_contribution_mean_mv": beta_a * a_mean,
            "sfa_threshold_contribution_max_mv": beta_a * a_max,
            "sfa_threshold_sampled_max_abs_mv": self.thr_samples_max_abs,
            "instrumentation_samples": self.samples,
            "nonfinite_detected": self.nonfinite,
            "out_of_bounds_detected": self.out_of_bounds,
        }


def _check(net: Stage2R3SNN, st: PhaseStats) -> None:
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


_STATE_KEY = {(1, 0): "O1_only", (0, 1): "O0_only", (0, 0): "both_silent", (1, 1): "both_fire"}


def train_phase(net: Stage2R3SNN, x: np.ndarray, plastic: bool, st: PhaseStats,
                log: Optional[SourceLog] = None, name: str = "",
                step_hook: Optional[Callable[[Stage2R3SNN, int, float], None]] = None) -> None:
    w = [s.weight.data for s in net.synapses]
    alif = net.neurons[0]
    for t in range(len(x)):
        xt = int(x[t])
        cur = net.current_output()
        z1, z0 = int(cur[0, O1].item()), int(cur[0, O0].item())
        if plastic:
            will_reward = z1 != z0
            before = [u.clone() for u in w] if will_reward else None
            out, reward = net.online_step(ONEHOT[xt], reward_fn=make_reward_fn(xt))
            st.steps_with_reward_fn += 1
            if reward != 0.0:
                assert before is not None
                d2 = w[1] - before[1]
                st.abs_dw_w1 += float((w[0] - before[0]).abs().sum())
                st.abs_dw_w2_rows[0] += float(d2[O1].abs().sum())
                st.abs_dw_w2_rows[1] += float(d2[O0].abs().sum())
                st.rewards.append(reward)
                st.dw_out.append(float(d2.sum()))
                st.dw_rows[0].append(float(d2[O1].sum()))
                st.dw_rows[1].append(float(d2[O0].sum()))
                if reward > 0:
                    st.pos += 1
                else:
                    st.neg += 1
        else:
            out, reward = net.online_step(ONEHOT[xt], reward_fn=None)
        st.out_spikes[0] += z1
        st.out_spikes[1] += z0
        st.state_counts[_STATE_KEY[(z1, z0)]] += 1
        st.hidden_spikes.add_(net._spikes[0])
        st.a_sum.add_(alif.a)
        torch.maximum(st.a_max, alif.a, out=st.a_max)
        st.steps += 1
        if st.steps % SAMPLE_EVERY == 0:
            _check(net, st)
        if step_hook is not None:
            step_hook(net, t, reward)
    _check(net, st)
    if log is not None:
        log.add("train", name, len(x))


def weight_arrays(net: Stage2R3SNN) -> Tuple[np.ndarray, np.ndarray]:
    return tuple(s.weight.data.numpy().copy() for s in net.synapses)  # type: ignore[return-value]


# --------------------------------------------------- sec. 8.1 phase-end fields
PHASE_END_SUFFIXES = (
    "w1_min", "w1_max", "w1_lower_hits", "w1_upper_hits",
    "w2_min", "w2_max", "w2_lower_hits", "w2_upper_hits",
    "w2_O1_min", "w2_O1_max", "w2_O1_lower_hits", "w2_O1_upper_hits",
    "w2_O0_min", "w2_O0_max", "w2_O0_lower_hits", "w2_O0_upper_hits",
    "weights_sha256",
)
PHASE_END_FIELDS = tuple(f"{p}_{s}" for p in ("a_end", "b_end") for s in PHASE_END_SUFFIXES)
assert len(PHASE_END_SUFFIXES) == 17 and len(PHASE_END_FIELDS) == 34


def phase_end_fields(prefix: str, w1: np.ndarray, w2: np.ndarray) -> Dict[str, object]:
    """Sec. 8.1 field set from live weights (W1 (20,2), W2 (2,20)); bound hit = float64 ==."""
    def four(tag, a, lo, hi):
        return {
            f"{prefix}_{tag}_min": float(a.min()),
            f"{prefix}_{tag}_max": float(a.max()),
            f"{prefix}_{tag}_lower_hits": int((a == lo).sum()),
            f"{prefix}_{tag}_upper_hits": int((a == hi).sum()),
        }

    out: Dict[str, object] = {}
    out.update(four("w1", w1, *W1_BOUNDS))
    out.update(four("w2", w2, *W2_BOUNDS))
    out.update(four("w2_O1", w2[O1], *W2_BOUNDS))
    out.update(four("w2_O0", w2[O0], *W2_BOUNDS))
    out[f"{prefix}_weights_sha256"] = sha256(np.asarray(w1, dtype=np.float64), np.asarray(w2, dtype=np.float64))
    return out


class SchemaError(ValueError):
    pass


def validate_row(row: Dict[str, object]) -> None:
    """Sec. 8.1 schema-completeness assertion; raises SchemaError (writer then emits no row)."""
    for p in ("a_end", "b_end"):
        for s in PHASE_END_SUFFIXES:
            k = f"{p}_{s}"
            if k not in row:
                raise SchemaError(f"missing field {k}")
            v = row[k]
            if s == "weights_sha256":
                if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
                    raise SchemaError(f"{k} is not a 64-hex string")
            elif s.endswith("_hits"):
                if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                    raise SchemaError(f"{k} is not a non-negative int")
            else:
                if not (isinstance(v, float) and math.isfinite(v)):
                    raise SchemaError(f"{k} is not a finite float")
        for side in ("lower", "upper"):
            tot = row[f"{p}_w2_{side}_hits"]
            parts = row[f"{p}_w2_O1_{side}_hits"] + row[f"{p}_w2_O0_{side}_hits"]
            if tot != parts:
                raise SchemaError(f"{p}_w2_{side}_hits {tot} != O1 + O0 {parts}")


# ------------------------------------------------------------------- one run
def run_arm_seed(arm: str, seed: int, expected_hashes: Optional[Dict[str, str]] = None,
                 phase_steps: int = PHASE_STEPS, eval_len: int = EVAL_LEN,
                 eval_warmup: int = EVAL_WARMUP,
                 phase_end_hook: Optional[Callable[[str, Stage2R3SNN], None]] = None,
                 step_hook: Optional[Callable[[Stage2R3SNN, int, float], None]] = None) -> Dict[str, object]:
    """One continuous Stage 2 r3 run (sec. 3-8) for one arm and seed."""
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
    train_phase(net, obj["A_train"], plastic_a, st_a, log, "A_train", step_hook)
    # sec. 8.1 a_end_*: live weights after step 199,999, before the first B step.
    w_shift = weight_arrays(net)
    a_end = phase_end_fields("a_end", *w_shift)
    if phase_end_hook is not None:
        phase_end_hook("A", net)
    counts_after_a = net.reset_counts()
    live_bytes_before_pre = state_bytes(net)
    evals: Dict[str, Dict[str, object]] = {}
    for ck, stream in (("A_pre", "A_heldout"), ("B_pre", "B_heldout")):
        evals[ck] = evaluate(net, obj[stream], obj[f"coin_{ck}"], log, stream, f"coin_{ck}", eval_warmup)
    pre_isolated = state_bytes(net) == live_bytes_before_pre

    st_b = PhaseStats()
    train_phase(net, b_stream, plastic_b, st_b, log, b_stream_name, step_hook)
    # sec. 8.1 b_end_*: live weights after step 399,999.
    w_final = weight_arrays(net)
    b_end = phase_end_fields("b_end", *w_final)
    if phase_end_hook is not None:
        phase_end_hook("B", net)
    live_bytes_before_post = state_bytes(net)
    for ck, stream in (("B_post", "B_heldout"), ("A_post", "A_heldout")):
        evals[ck] = evaluate(net, obj[stream], obj[f"coin_{ck}"], log, stream, f"coin_{ck}", eval_warmup)
    post_isolated = state_bytes(net) == live_bytes_before_post
    counts_final = net.reset_counts()
    n_weights_after = sum(s.weight.numel() for s in net.synapses)

    def maxdiff(a, b):
        return float(max(np.max(np.abs(a[0] - b[0])), np.max(np.abs(a[1] - b[1]))))

    train_reads = {k: v for k, v in log.counts.items() if k.startswith("train:")}
    heldout_consumed_by_training = sum(v for k, v in train_reads.items() if "heldout" in k)
    coin_reads_by_training = sum(v for k, v in train_reads.items() if "coin" in k)
    row: Dict[str, object] = {
        "spec_revision": "r3",
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "seed": seed,
        "engine": "snn/core.py PureSNN.online_step [2,20,2] + ALIFNeuron hidden (reward_mode=per_spike_next_step)",
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
        "initial_weights_sha256": sha256(*w_init),
        **a_end,
        **b_end,
        "n_trainable_weights_before": n_weights_before,
        "n_trainable_weights_after": n_weights_after,
        "live_init_counts": init_counts,
        "live_reset_counts_after_A": counts_after_a,
        "live_reset_counts_final": counts_final,
        "live_post_init_reset_calls": {k: counts_final[k] - init_counts[k] for k in counts_final},
        "live_init_start_state_ok": bool(init_start_ok),
        "access_log": dict(sorted(log.counts.items())),
        "heldout_observations_consumed_by_training": heldout_consumed_by_training,
        "coin_reads_by_training": coin_reads_by_training,
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
    validate_row(row)
    return row
