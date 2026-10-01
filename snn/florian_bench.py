"""Florian (2007) XOR bench on the repository core (``snn/core.py``).

Reward mode: ``REWARD_PER_SPIKE_NEXT_STEP``: +1 (target 1) or -1 (target 0)
for each output spike, delivered on the following 1 ms step, online, batch 1,
continuous state across patterns and epochs.

The substrate is ``PureSNN`` / ``LIFNeuron`` / ``Synapse`` / ``RSTDPPlasticity``.
This module does not import ``ladder``; the protocol constants (sizes,
encodings, bounds, gammas, time constants) are restated here from the paper
and ``docs/FLORIAN_DIFFERENCE_TABLE.md``. Random draws (initial weights,
symbol trains, pattern order, Poisson input) follow the same NumPy
``default_rng(seed)`` call order as the independent anchor, so seeds pair
one-to-one for the seed-for-seed comparison.  See docs/FLORIAN_CORE_GATE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from snn.core import REWARD_PER_SPIKE_NEXT_STEP, PureSNN, RSTDPPlasticity

PATTERNS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
U_REST_MV = -70.0
THETA_MV = -54.0
PUBLISHED_SUCCESS_PCT = {
    ("rate", "mstdp"): 99.1,
    ("rate", "mstdpet"): 98.2,
    ("temporal", "mstdp"): 89.7,
    ("temporal", "mstdpet"): 99.5,
}


@dataclass(frozen=True)
class CoreFlorianConfig:
    task: str
    rule: str
    epochs: int = 200
    pattern_ms: int = 500
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_stdp_ms: float = 20.0
    tau_elig_ms: float = 25.0
    plastic: bool = True
    reward_mode: str = REWARD_PER_SPIKE_NEXT_STEP

    def __post_init__(self):
        if self.task not in ("rate", "temporal"):
            raise ValueError("task must be rate or temporal")
        if self.rule not in ("mstdp", "mstdpet"):
            raise ValueError("rule must be mstdp or mstdpet")
        if self.reward_mode != REWARD_PER_SPIKE_NEXT_STEP:
            raise ValueError("the Florian core bench runs only the per-spike next-step reward mode")

    @property
    def sizes(self) -> Tuple[int, int, int]:
        return (60, 60, 1) if self.task == "rate" else (2, 20, 1)

    @property
    def gamma_mv(self) -> float:
        if self.task == "rate":
            return 0.1 if self.rule == "mstdp" else 0.625
        return 0.01 if self.rule == "mstdp" else 0.25

    @property
    def published_success_pct(self) -> float:
        return PUBLISHED_SUCCESS_PCT[(self.task, self.rule)]


class CoreFlorianBench:
    """Builds a ``PureSNN`` in Florian's units and runs the XOR protocol."""

    def __init__(self, config: CoreFlorianConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        n_in, n_hidden, n_out = config.sizes
        net = PureSNN(
            [n_in, n_hidden, n_out],
            dt=config.dt_ms,
            weight_scale=0.0,
            tau_m=config.tau_m_ms,
            tau_syn=0.0,  # direct voltage jump, no synaptic filter (eq. 4.1)
            v_thresh=THETA_MV,
            v_rest=U_REST_MV,
            v_reset=U_REST_MV,
        ).double()
        self.net = net

        # Initial weights, bounds, and encodings: same rng call order as the anchor.
        if config.task == "rate":
            sign = np.asarray(([1] * 15 + [-1] * 15) * 2, dtype=float)
            excit = sign > 0
            w1 = np.empty((n_hidden, n_in))
            w1[:, excit] = self.rng.uniform(0.0, 5.0, size=(n_hidden, int(excit.sum())))
            w1[:, ~excit] = self.rng.uniform(-5.0, 0.0, size=(n_hidden, int((~excit).sum())))
            w1_low = torch.as_tensor(np.broadcast_to(np.where(excit, 0.0, -5.0), w1.shape).copy())
            w1_high = torch.as_tensor(np.broadcast_to(np.where(excit, 5.0, 0.0), w1.shape).copy())
            w2 = self.rng.uniform(0.0, 5.0, size=(n_out, n_hidden))
            w2_low, w2_high = 0.0, 5.0
            self.symbol_trains = None
        else:
            w1 = self.rng.uniform(-10.0, 10.0, size=(n_hidden, n_in))
            w1_low, w1_high = -10.0, 10.0
            w2 = self.rng.uniform(0.0, 10.0, size=(n_out, n_hidden))
            w2_low, w2_high = 0.0, 10.0
            self.symbol_trains = np.zeros((2, config.pattern_ms))
            for symbol in (0, 1):
                self.symbol_trains[symbol, self.rng.choice(config.pattern_ms, size=50, replace=False)] = 1.0

        with torch.no_grad():
            net.synapses[0].weight.copy_(torch.as_tensor(w1))
            net.synapses[1].weight.copy_(torch.as_tensor(w2))

        if config.rule == "mstdpet":
            # z(t+dt) = beta z(t) + zeta(t)/tau_z with A+ = 1, A- = -1  ->  a = 1/tau_z.
            amp, credit = 1.0 / config.tau_elig_ms, "eligibility"
        else:
            # dw = gamma r(t+dt) zeta(t), A+ = 1, A- = -1, no trace.
            amp, credit = 1.0, "pairing"
        bounds = ((w1_low, w1_high), (w2_low, w2_high))
        net.plasticities = [
            RSTDPPlasticity(
                syn,
                lr=config.gamma_mv,
                tau_elig=config.tau_elig_ms,
                dt=config.dt_ms,
                a_plus=amp,
                a_minus=amp,
                tau_plus=config.tau_stdp_ms,
                tau_minus=config.tau_stdp_ms,
                w_min=low,
                w_max=high,
                credit=credit,
            )
            for syn, (low, high) in zip(net.synapses, bounds)
        ]
        net.reset_online_state(batch_size=1, dtype=torch.float64)
        self.plastic = config.plastic
        self.total_output_spikes = 0
        self.total_abs_weight_change = 0.0
        self.reward_events: List[Tuple[float, float]] = []  # (r, signed d sum(w_out))
        self.record_rewards = True
        self._scratch = torch.zeros(1, config.sizes[0], dtype=torch.float64)

    # ----------------------------------------------------------------- input
    def input_for(self, pattern: Tuple[int, int], timestep: int) -> np.ndarray:
        if self.config.task == "rate":
            spikes = np.zeros(60)
            if pattern[0]:
                spikes[:30] = self.rng.random(30) < 0.04
            if pattern[1]:
                spikes[30:] = self.rng.random(30) < 0.04
            return spikes
        assert self.symbol_trains is not None
        return np.asarray([self.symbol_trains[pattern[0], timestep], self.symbol_trains[pattern[1], timestep]])

    # ------------------------------------------------------------------ step
    def step(self, input_spikes: np.ndarray, target: int) -> int:
        x = self._scratch
        x[0].copy_(torch.from_numpy(np.asarray(input_spikes, dtype=np.float64)))
        sign = 1.0 if target else -1.0
        if self.plastic:
            # Reward is nonzero only on transitions after an output spike; snapshot only then.
            will_reward = bool(self.net.current_output()[0, 0].item())
            w = [s.weight.data for s in self.net.synapses]
            before = [t.clone() for t in w] if will_reward else None
            out, reward = self.net.online_step(x, reward_fn=lambda f: sign * float(f[0, 0]))
            if reward != 0.0:
                assert before is not None
                self.total_abs_weight_change += float(sum((a - b).abs().sum() for a, b in zip(w, before)))
                if self.record_rewards:
                    self.reward_events.append((reward, float((w[-1] - before[-1]).sum())))
        else:
            out, reward = self.net.online_step(x, reward_fn=None)
        fired = int(out[0, 0].item())
        self.total_output_spikes += fired
        return fired

    def present(self, pattern: Tuple[int, int]) -> int:
        target = int(pattern[0] != pattern[1])
        return sum(self.step(self.input_for(pattern, t), target) for t in range(self.config.pattern_ms))

    def run(self) -> Dict[str, object]:
        last_counts = {p: 0 for p in PATTERNS}
        for epoch in range(self.config.epochs):
            order = list(PATTERNS)
            self.rng.shuffle(order)
            for pattern in order:
                count = self.present(pattern)
                if epoch == self.config.epochs - 1:
                    last_counts[pattern] = count
        rates = {f"{a}{b}": 2.0 * last_counts[(a, b)] for a, b in PATTERNS}
        w1 = self.net.synapses[0].weight.data
        w2 = self.net.synapses[1].weight.data
        return {
            "success": bool(rates["11"] < rates["01"] and rates["11"] < rates["10"]),
            "last_epoch_rates_hz": rates,
            "total_output_spikes": self.total_output_spikes,
            "total_abs_weight_change_mv": self.total_abs_weight_change,
            "w1_min": float(w1.min()),
            "w1_max": float(w1.max()),
            "w2_min": float(w2.min()),
            "w2_max": float(w2.max()),
        }

    def weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return tuple(s.weight.data.detach().cpu().numpy().copy() for s in self.net.synapses)  # type: ignore[return-value]


# ------------------------------------------------------------ diagnostics
def gate_margin_hz(rates: Dict[str, float]) -> float:
    return min(rates["01"], rates["10"]) - rates["11"]


def pearson(x, y) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def present_frozen(bench: CoreFlorianBench, rng: np.random.Generator, symbol_trains=None) -> Dict[str, float]:
    """Present the four patterns once with plasticity off (anchor protocol)."""
    saved = bench.plastic, bench.rng, bench.symbol_trains
    bench.plastic, bench.rng = False, rng
    if symbol_trains is not None:
        bench.symbol_trains = symbol_trains
    try:
        order = list(PATTERNS)
        rng.shuffle(order)
        counts = {p: bench.present(p) for p in order}
    finally:
        bench.plastic, bench.rng, bench.symbol_trains = saved
    return {f"{a}{b}": 2.0 * counts[(a, b)] for a, b in PATTERNS}


def new_symbol_pair(rng: np.random.Generator, pattern_ms: int = 500) -> np.ndarray:
    trains = np.zeros((2, pattern_ms))
    for symbol in (0, 1):
        trains[symbol, rng.choice(pattern_ms, size=50, replace=False)] = 1.0
    return trains


def post_training_checks(bench: CoreFlorianBench, seed: int, n_eval: int = 10) -> Dict[str, object]:
    """Same evaluation seeds and protocol as the anchor's frozen retest/generalization."""
    out: Dict[str, object] = {}
    retest = [present_frozen(bench, np.random.default_rng([seed, 101, k])) for k in range(n_eval)]
    out["retest_rates_hz"] = retest
    out["retest_pass_count"] = int(sum(gate_margin_hz(r) > 0 for r in retest))
    out["retest_median_margin_hz"] = float(np.median([gate_margin_hz(r) for r in retest]))
    if bench.config.task == "temporal":
        gen = []
        for k in range(n_eval):
            symbols = new_symbol_pair(np.random.default_rng([seed, 202, k]), bench.config.pattern_ms)
            gen.append(present_frozen(bench, np.random.default_rng([seed, 203, k]), symbols))
        out["generalization_rates_hz"] = gen
        out["generalization_pass_count"] = int(sum(gate_margin_hz(r) > 0 for r in gen))
        out["generalization_median_margin_hz"] = float(np.median([gate_margin_hz(r) for r in gen]))
    out["n_eval"] = n_eval
    return out


def run_core_experiment(task: str, rule: str, seed: int, epochs: int = 200, plastic: bool = True,
                        n_eval: int = 10, record_output: bool = False) -> Dict[str, object]:
    torch.set_num_threads(1)
    cfg = CoreFlorianConfig(task=task, rule=rule, epochs=epochs, plastic=plastic)
    bench = CoreFlorianBench(cfg, seed)
    output_train: List[int] = []
    if record_output:
        inner = bench.step

        def recorded(inp, target):
            fired = inner(inp, target)
            output_train.append(fired)
            return fired

        bench.step = recorded  # type: ignore[method-assign]
    result = bench.run()
    if record_output:
        del bench.step
    rewards = [r for r, _ in bench.reward_events]
    deltas = [d for _, d in bench.reward_events]
    corr = pearson(rewards, deltas)
    w1, w2 = bench.weights()
    checks = post_training_checks(bench, seed, n_eval) if n_eval else {}
    row: Dict[str, object] = {
        "engine": "snn/core.py PureSNN.online_step (reward_mode=per_spike_next_step)",
        "task": task,
        "rule": rule,
        "seed": seed,
        "epochs": epochs,
        "plastic": plastic,
        "gamma_mv": cfg.gamma_mv,
        "published_success_pct": cfg.published_success_pct,
        "gate_margin_hz": gate_margin_hz(result["last_epoch_rates_hz"]),  # type: ignore[arg-type]
        "reward_events": len(rewards),
        "reward_events_positive": int(sum(r > 0 for r in rewards)),
        "reward_events_negative": int(sum(r < 0 for r in rewards)),
        "reward_signed_dw_out_corr": corr,
        **result,
        **checks,
        "_final_w1": w1,
        "_final_w2": w2,
    }
    if record_output:
        row["_output_train"] = np.asarray(output_train, dtype=np.int8)
    return row
