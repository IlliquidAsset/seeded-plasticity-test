"""Independent implementation of Florian (2007), sections 4.1-4.3.

This module intentionally does not reuse snn.core: it is a reference anchor for
paper equations 2.7-2.8, 3.6-3.12, and 4.1.  See the frozen pre-run difference
table in docs/FLORIAN_DIFFERENCE_TABLE.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

PATTERNS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class FlorianConfig:
    task: str
    rule: str
    epochs: int = 200
    pattern_ms: int = 500
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_stdp_ms: float = 20.0
    tau_elig_ms: float = 25.0

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
        values = {
            ("rate", "mstdp"): 99.1,
            ("rate", "mstdpet"): 98.2,
            ("temporal", "mstdp"): 89.7,
            ("temporal", "mstdpet"): 99.5,
        }
        return values[(self.task, self.rule)]


class FlorianNetwork:
    """Continuous-state, two-layer IF network with online R-STDP."""

    def __init__(self, config: FlorianConfig, seed: int):
        if config.task not in {"rate", "temporal"}:
            raise ValueError("task must be rate or temporal")
        if config.rule not in {"mstdp", "mstdpet"}:
            raise ValueError("rule must be mstdp or mstdpet")
        self.config = config
        self.rng = np.random.default_rng(seed)
        n_in, n_hidden, _ = config.sizes

        if config.task == "rate":
            # Half of each 30-neuron bit population is excitatory, half inhibitory.
            self.input_sign = np.asarray(([1] * 15 + [-1] * 15) * 2, dtype=float)
            self.w1 = np.empty((n_hidden, n_in), dtype=float)
            excit = self.input_sign > 0
            self.w1[:, excit] = self.rng.uniform(0.0, 5.0, size=(n_hidden, int(excit.sum())))
            self.w1[:, ~excit] = self.rng.uniform(-5.0, 0.0, size=(n_hidden, int((~excit).sum())))
            self.w1_low = np.broadcast_to(np.where(excit, 0.0, -5.0), self.w1.shape)
            self.w1_high = np.broadcast_to(np.where(excit, 5.0, 0.0), self.w1.shape)
            self.w2 = self.rng.uniform(0.0, 5.0, size=(1, n_hidden))
            self.w2_low, self.w2_high = 0.0, 5.0
            self.symbol_trains = None
        else:
            self.input_sign = None
            self.w1 = self.rng.uniform(-10.0, 10.0, size=(n_hidden, n_in))
            self.w1_low, self.w1_high = -10.0, 10.0
            self.w2 = self.rng.uniform(0.0, 10.0, size=(1, n_hidden))
            self.w2_low, self.w2_high = 0.0, 10.0
            # The two symbols are fixed for the entire experiment and reused on
            # both input channels, exactly as the paper describes.
            self.symbol_trains = np.zeros((2, config.pattern_ms), dtype=np.float64)
            for symbol in (0, 1):
                indexes = self.rng.choice(config.pattern_ms, size=50, replace=False)
                self.symbol_trains[symbol, indexes] = 1.0

        self.v_hidden = np.full(n_hidden, -70.0)
        self.v_output = np.full(1, -70.0)
        self.hidden_spikes = np.zeros(n_hidden)
        self.output_spikes = np.zeros(1)

        # P+ tracks presynaptic spikes; P- tracks postsynaptic spikes.
        self.pre1 = np.zeros(n_in)
        self.post1 = np.zeros(n_hidden)
        self.pre2 = np.zeros(n_hidden)
        self.post2 = np.zeros(1)

        self.z1 = np.zeros_like(self.w1)
        self.z2 = np.zeros_like(self.w2)
        self.z_scale = 1.0
        self.total_output_spikes = 0
        self.total_abs_weight_change = 0.0

    @property
    def trace_beta(self) -> float:
        return math.exp(-self.config.dt_ms / self.config.tau_stdp_ms)

    @property
    def elig_beta(self) -> float:
        return math.exp(-self.config.dt_ms / self.config.tau_elig_ms)

    @property
    def membrane_beta(self) -> float:
        return math.exp(-self.config.dt_ms / self.config.tau_m_ms)

    def input_for(self, pattern: Tuple[int, int], timestep: int) -> np.ndarray:
        if self.config.task == "rate":
            spikes = np.zeros(60)
            if pattern[0]:
                spikes[:30] = self.rng.random(30) < 0.04
            if pattern[1]:
                spikes[30:] = self.rng.random(30) < 0.04
            return spikes
        assert self.symbol_trains is not None
        return np.asarray(
            [self.symbol_trains[pattern[0], timestep], self.symbol_trains[pattern[1], timestep]]
        )

    def _update_traces(self, input_spikes: np.ndarray):
        beta = self.trace_beta
        self.pre1 = beta * self.pre1 + input_spikes
        self.post1 = beta * self.post1 - self.hidden_spikes
        self.pre2 = beta * self.pre2 + self.hidden_spikes
        self.post2 = beta * self.post2 - self.output_spikes

    def _xi(self, pre_trace, post_trace, pre_spikes, post_spikes):
        return np.outer(post_spikes, pre_trace) + np.outer(post_trace, pre_spikes)

    def _update_eligibility(self, input_spikes: np.ndarray):
        self.z_scale *= self.elig_beta
        divisor = self.config.tau_elig_ms * self.z_scale
        hidden_indexes = np.flatnonzero(self.hidden_spikes)
        input_indexes = np.flatnonzero(input_spikes)
        output_fired = bool(self.output_spikes[0])

        if hidden_indexes.size:
            self.z1[hidden_indexes, :] += self.pre1[None, :] / divisor
        if input_indexes.size:
            self.z1[:, input_indexes] += self.post1[:, None] / divisor
        if output_fired:
            self.z2[0, :] += self.pre2 / divisor
        if hidden_indexes.size:
            self.z2[:, hidden_indexes] += self.post2[:, None] / divisor

    def _materialize_eligibility(self):
        if self.config.rule == "mstdpet":
            self.z1 *= self.z_scale
            self.z2 *= self.z_scale
            self.z_scale = 1.0

    def _apply_reward(self, input_spikes: np.ndarray, reward: float):
        if reward == 0.0:
            return
        gamma = self.config.gamma_mv
        old1 = self.w1.copy()
        old2 = self.w2.copy()
        if self.config.rule == "mstdpet":
            delta1 = gamma * reward * (self.z_scale * self.z1)
            delta2 = gamma * reward * (self.z_scale * self.z2)
        else:
            delta1 = gamma * reward * self._xi(
                self.pre1, self.post1, input_spikes, self.hidden_spikes
            )
            delta2 = gamma * reward * self._xi(
                self.pre2, self.post2, self.hidden_spikes, self.output_spikes
            )
        self.w1 = np.clip(self.w1 + delta1, self.w1_low, self.w1_high)
        self.w2 = np.clip(self.w2 + delta2, self.w2_low, self.w2_high)
        self.total_abs_weight_change += float(np.abs(self.w1 - old1).sum() + np.abs(self.w2 - old2).sum())

    def _advance_neurons(self, input_spikes: np.ndarray):
        # Eq. 4.1: current spikes affect membrane potential on the following step.
        hidden_next_v = -70.0 + self.membrane_beta * (self.v_hidden + 70.0) + self.w1 @ input_spikes
        output_next_v = -70.0 + self.membrane_beta * (self.v_output + 70.0) + self.w2 @ self.hidden_spikes
        hidden_next = (hidden_next_v >= -54.0).astype(float)
        output_next = (output_next_v >= -54.0).astype(float)
        hidden_next_v[hidden_next.astype(bool)] = -70.0
        output_next_v[output_next.astype(bool)] = -70.0
        self.v_hidden = hidden_next_v
        self.v_output = output_next_v
        self.hidden_spikes = hidden_next
        self.output_spikes = output_next

    def step(self, input_spikes: np.ndarray, target: int) -> int:
        self._update_traces(input_spikes)
        if self.config.rule == "mstdpet":
            self._update_eligibility(input_spikes)
        # The reward emitted for this output spike is delivered on the transition
        # to the next 1 ms state, after z(t+dt) includes the current pairing.
        fired = int(self.output_spikes[0])
        reward = float((1 if target else -1) * fired)
        self._apply_reward(input_spikes, reward)
        self._advance_neurons(input_spikes)
        self.total_output_spikes += fired
        return fired

    def run(self) -> Dict[str, object]:
        last_counts = {pattern: 0 for pattern in PATTERNS}
        for epoch in range(self.config.epochs):
            order = list(PATTERNS)
            self.rng.shuffle(order)
            for pattern in order:
                count = 0
                target = int(pattern[0] != pattern[1])
                for timestep in range(self.config.pattern_ms):
                    count += self.step(self.input_for(pattern, timestep), target)
                if epoch == self.config.epochs - 1:
                    last_counts[pattern] = count
                # Avoid an underflowing lazy scale while retaining continuous state.
                self._materialize_eligibility()

        rates = {f"{a}{b}": 2.0 * last_counts[(a, b)] for a, b in PATTERNS}
        success = rates["11"] < rates["01"] and rates["11"] < rates["10"]
        return {
            "success": bool(success),
            "last_epoch_rates_hz": rates,
            "total_output_spikes": self.total_output_spikes,
            "total_abs_weight_change_mv": self.total_abs_weight_change,
            "w1_min": float(self.w1.min()),
            "w1_max": float(self.w1.max()),
            "w2_min": float(self.w2.min()),
            "w2_max": float(self.w2.max()),
        }


def run_experiment(task: str, rule: str, seed: int, epochs: int = 200) -> Dict[str, object]:
    config = FlorianConfig(task=task, rule=rule, epochs=epochs)
    result = FlorianNetwork(config, seed).run()
    return {
        "task": task,
        "rule": rule,
        "seed": seed,
        "epochs": epochs,
        "gamma_mv": config.gamma_mv,
        "published_success_pct": config.published_success_pct,
        **result,
    }
