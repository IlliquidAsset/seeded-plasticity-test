"""Executed tiny spiking systems for ladder rungs 1-3.

Every rung built on this module runs real integrate-and-fire neurons and plastic
synapses, one 1 ms step at a time.  Eligibility is produced only by the spikes
the circuit generates, and reward is scheduled only from spikes the circuit
actually emitted (or from an exogenous schedule for random-reward controls).

Discrete equations (Florian 2007):

* membrane, eq. 4.1: u(t+1) = u_r + (u(t) - u_r) exp(-dt/tau_m) + sum_j w_j f_j(t)
  and a neuron fires at t+1 when u(t+1) >= theta, after which u is reset.
* spike traces, eqs. 3.11-3.12: P+(t) = P+(t-1) exp(-dt/tau+) + A+ f_pre(t),
  P-(t) = P-(t-1) exp(-dt/tau-) + A- f_post(t).
* pairing term, eq. 3.10: zeta(t) = P+(t) f_post(t) + P-(t) f_pre(t).
* eligibility, eq. 2.8: z(t+1) = exp(-dt/tau_z) z(t) + zeta(t) / tau_z.
* weight, eq. 2.7: w(t+1) = w(t) + gamma r(t+1) z(t+1), then hard bounds.

A reward caused by a spike at step t with extra delay d is delivered at step
t + 1 + d and multiplies z(t + 1 + d).

This simulator deliberately shares no code with ``ladder.handcheck``.  The
checker recomputes weight changes from recorded spike times with explicit
pair sums, so a bug in this trace recursion cannot also hide in the checker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

U_REST_MV = -70.0
THETA_MV = -54.0
U_RESET_MV = -70.0
TAU_M_MS = 20.0


@dataclass(frozen=True)
class Rule:
    """Florian 2007 section 4.1 plasticity constants."""

    a_plus: float = 1.0
    a_minus: float = -1.0
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    tau_z_ms: float = 25.0
    gamma_mv: float = 0.25


@dataclass(frozen=True)
class SynapseSpec:
    name: str
    pre: int
    post: int
    w0_mv: float
    low_mv: float
    high_mv: float
    plastic: bool = True


@dataclass
class Circuit:
    """A feed-forward circuit of externally driven inputs and IF neurons.

    Neuron indexes 0..n_inputs-1 are input neurons whose spikes come from a
    raster.  Indexes n_inputs..n_inputs+n_if-1 are integrate-and-fire neurons.
    """

    n_inputs: int
    n_if: int
    synapses: Tuple[SynapseSpec, ...]
    names: Tuple[str, ...]

    @property
    def n(self) -> int:
        return self.n_inputs + self.n_if

    def index(self, name: str) -> int:
        return self.names.index(name)

    def synapse(self, name: str) -> SynapseSpec:
        for spec in self.synapses:
            if spec.name == name:
                return spec
        raise KeyError(name)


# A reward policy maps (step, spike vector at step, spike history so far) to the
# reward value to schedule for delivery at step + 1 + delay.
RewardPolicy = Callable[[int, np.ndarray, np.ndarray], float]


@dataclass
class RunRecord:
    spikes: np.ndarray  # (T, N) bool
    weights_final: Dict[str, float]
    weights_initial: Dict[str, float]
    weight_trace: Dict[str, np.ndarray]  # sampled at every step end
    reward_delivered: Dict[int, float]  # delivery step -> summed reward
    clip_events: Dict[str, int]
    delay_ms: int

    def spike_times(self, neuron: int) -> np.ndarray:
        return np.flatnonzero(self.spikes[:, neuron])

    def count(self, neuron: int) -> int:
        return int(self.spikes[:, neuron].sum())


class Simulator:
    """Step-by-step executor.  Methods are small so tests can mutate one."""

    def __init__(self, circuit: Circuit, rule: Rule, plastic: bool = True):
        self.circuit = circuit
        self.rule = rule
        self.plastic = plastic
        self.pre = np.asarray([s.pre for s in circuit.synapses], dtype=int)
        self.post = np.asarray([s.post for s in circuit.synapses], dtype=int)
        self.low = np.asarray([s.low_mv for s in circuit.synapses], dtype=float)
        self.high = np.asarray([s.high_mv for s in circuit.synapses], dtype=float)
        self.learn = np.asarray([s.plastic for s in circuit.synapses], dtype=bool)
        self.w = np.asarray([s.w0_mv for s in circuit.synapses], dtype=float)
        self.u = np.full(circuit.n, U_REST_MV)
        self.p_plus = np.zeros(circuit.n)
        self.p_minus = np.zeros(circuit.n)
        self.z = np.zeros(len(circuit.synapses))
        self.next_if_spikes = np.zeros(circuit.n_if, dtype=bool)
        self.pending: Dict[int, float] = {}
        self.delivered: Dict[int, float] = {}
        self.clips = np.zeros(len(circuit.synapses), dtype=int)
        self.b_plus = math.exp(-1.0 / rule.tau_plus_ms)
        self.b_minus = math.exp(-1.0 / rule.tau_minus_ms)
        self.b_z = math.exp(-1.0 / rule.tau_z_ms)
        self.b_m = math.exp(-1.0 / TAU_M_MS)

    # --- one-step pieces -------------------------------------------------
    def update_traces(self, f: np.ndarray) -> None:
        self.p_plus = self.b_plus * self.p_plus + self.rule.a_plus * f
        self.p_minus = self.b_minus * self.p_minus + self.rule.a_minus * f

    def pairing(self, f: np.ndarray) -> np.ndarray:
        return self.p_plus[self.pre] * f[self.post] + self.p_minus[self.post] * f[self.pre]

    def update_eligibility(self, zeta: np.ndarray) -> None:
        self.z = self.b_z * self.z + zeta / self.rule.tau_z_ms

    def schedule(self, t: int, value: float, delay_ms: int) -> None:
        if value != 0.0:
            when = t + 1 + delay_ms
            self.pending[when] = self.pending.get(when, 0.0) + value

    def deliver(self, t: int) -> float:
        r = self.pending.pop(t + 1, 0.0)
        if r != 0.0:
            self.delivered[t + 1] = self.delivered.get(t + 1, 0.0) + r
        return r

    def apply_reward(self, r: float) -> None:
        if not self.plastic or r == 0.0:
            return
        proposed = self.w + self.rule.gamma_mv * r * self.z * self.learn
        clipped = np.clip(proposed, self.low, self.high)
        self.clips += (clipped != proposed).astype(int)
        self.w = clipped

    def advance_neurons(self, f: np.ndarray) -> np.ndarray:
        drive = np.zeros(self.circuit.n)
        np.add.at(drive, self.post, self.w * f[self.pre])
        self.u = U_REST_MV + (self.u - U_REST_MV) * self.b_m + drive
        fired = self.u >= THETA_MV
        fired[: self.circuit.n_inputs] = False
        self.u[fired] = U_RESET_MV
        self.u[: self.circuit.n_inputs] = U_REST_MV
        return fired[self.circuit.n_inputs :]

    def step(self, t: int, f: np.ndarray, history: np.ndarray, policy, delay_ms: int) -> None:
        """One 1 ms transition.  Order: traces, eligibility, reward, membrane."""
        self.update_traces(f)
        self.update_eligibility(self.pairing(f))
        if policy is not None:
            self.schedule(t, float(policy(t, f, history)), delay_ms)
        self.apply_reward(self.deliver(t))
        self.next_if_spikes = self.advance_neurons(f)

    # --- driver -----------------------------------------------------------
    def run(
        self,
        raster: np.ndarray,
        policy: Optional[RewardPolicy] = None,
        delay_ms: int = 0,
        exogenous: Optional[Dict[int, float]] = None,
    ) -> RunRecord:
        steps = raster.shape[0]
        if raster.shape[1] != self.circuit.n_inputs:
            raise ValueError("raster width must equal n_inputs")
        spikes = np.zeros((steps, self.circuit.n), dtype=bool)
        w_init = {s.name: float(self.w[k]) for k, s in enumerate(self.circuit.synapses)}
        w_trace = np.zeros((steps, len(self.circuit.synapses)))
        for when, value in (exogenous or {}).items():
            self.pending[int(when)] = self.pending.get(int(when), 0.0) + float(value)
        for t in range(steps):
            f = np.zeros(self.circuit.n)
            f[: self.circuit.n_inputs] = raster[t]
            f[self.circuit.n_inputs :] = self.next_if_spikes
            spikes[t] = f.astype(bool)
            self.step(t, f, spikes, policy, delay_ms)
            w_trace[t] = self.w
        return RunRecord(
            spikes=spikes,
            weights_final={s.name: float(self.w[k]) for k, s in enumerate(self.circuit.synapses)},
            weights_initial=w_init,
            weight_trace={s.name: w_trace[:, k].copy() for k, s in enumerate(self.circuit.synapses)},
            reward_delivered=dict(self.delivered),
            clip_events={s.name: int(self.clips[k]) for k, s in enumerate(self.circuit.synapses)},
            delay_ms=delay_ms,
        )


def with_weights(circuit: Circuit, weights: Dict[str, float]) -> Circuit:
    """Return a copy of ``circuit`` whose initial weights are ``weights``."""
    specs = tuple(
        SynapseSpec(s.name, s.pre, s.post, float(weights.get(s.name, s.w0_mv)), s.low_mv, s.high_mv, s.plastic)
        for s in circuit.synapses
    )
    return Circuit(circuit.n_inputs, circuit.n_if, specs, circuit.names)


def poisson_raster(rng: np.random.Generator, steps: int, rates_hz: Sequence[float]) -> np.ndarray:
    p = np.asarray(rates_hz, dtype=float) / 1000.0
    return rng.random((steps, len(rates_hz))) < p


def evaluate_frozen(circuit: Circuit, weights: Dict[str, float], raster: np.ndarray, rule: Rule) -> RunRecord:
    """Drive a non-plastic copy carrying ``weights`` with ``raster``."""
    return Simulator(with_weights(circuit, weights), rule, plastic=False).run(raster)
