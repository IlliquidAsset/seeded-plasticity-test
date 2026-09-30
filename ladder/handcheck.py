"""Independent hand-expectation checker for the executed ladder.

Nothing here imports ``ladder.spiking``.  Every expectation is recomputed from
recorded spike times with explicit sums over spike pairs (not the trace
recursion used by the simulator):

* pairing term (Florian eq. 3.10-3.12 unrolled):
  zeta(t) = f_post(t) * sum_{pre spikes s <= t} A+ exp(-(t-s)/tau+)
          + f_pre(t)  * sum_{post spikes s <= t} A- exp(-(t-s)/tau-)
* eligibility (eq. 2.8 unrolled): z(s) = sum_{t < s} zeta(t)/tau_z exp(-(s-1-t)/tau_z)
* weight (eq. 2.7): w <- clip(w + gamma r(s) z(s)) for each delivery step s in order.
* reward schedule: derived here from the recorded spikes and the declarative
  policy description, not read back from the simulator's reward queue.
* membrane (eq. 4.1 unrolled since last reset):
  u(t) = u_r + sum_{s in (reset, t]} drive(s-1) exp(-(t-s)/tau_m), where
  drive(s-1) = sum_j w_j(s-1) f_j(s-1) from the recorded weight trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Constants:
    a_plus: float = 1.0
    a_minus: float = -1.0
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_z: float = 25.0
    gamma: float = 0.25
    u_rest: float = -70.0
    theta: float = -54.0
    tau_m: float = 20.0


@dataclass(frozen=True)
class PolicySpec:
    """Declarative reward policy the checker evaluates on recorded spikes.

    kind:
      ``none``            no reward.
      ``per_spike``       ``value`` for each spike of ``neuron`` (optionally only when
                          ``require_prior`` fired on the immediately preceding step).
      ``per_silent_step`` ``value`` for each step on which ``neuron`` did not fire.
      ``baseline``        r = f(t) - rbar(t-1), rbar an EMA of f with ``tau_bar_ms``.
      ``exogenous``       fixed schedule {delivery_step: value}, independent of spikes.
    """

    kind: str
    neuron: int = -1
    value: float = 1.0
    require_prior: Optional[int] = None
    tau_bar_ms: float = 1000.0
    delay_ms: int = 0
    schedule: Optional[Tuple[Tuple[int, float], ...]] = None


def reward_schedule(spikes: np.ndarray, policy: PolicySpec) -> Dict[int, float]:
    """Delivery step -> reward, computed from recorded spikes only."""
    steps = spikes.shape[0]
    out: Dict[int, float] = {}
    if policy.kind == "none":
        return out
    if policy.kind == "exogenous":
        for when, value in policy.schedule or ():
            out[int(when)] = out.get(int(when), 0.0) + float(value)
        return out
    f = spikes[:, policy.neuron].astype(bool)
    if policy.kind == "per_spike":
        times = np.flatnonzero(f)
        if policy.require_prior is not None:
            prior = spikes[:, policy.require_prior].astype(bool)
            times = [t for t in times if t >= 1 and prior[t - 1]]
        for t in times:
            when = int(t) + 1 + policy.delay_ms
            out[when] = out.get(when, 0.0) + policy.value
        return out
    if policy.kind == "per_silent_step":
        for t in np.flatnonzero(~f):
            when = int(t) + 1 + policy.delay_ms
            out[when] = out.get(when, 0.0) + policy.value
        return out
    if policy.kind == "baseline":
        b = math.exp(-1.0 / policy.tau_bar_ms)
        spike_times = np.flatnonzero(f)
        for t in range(steps):
            # rbar(t-1) by explicit sum over earlier spikes.
            earlier = spike_times[spike_times <= t - 1]
            rbar_prev = (1.0 - b) * float(np.sum(b ** (t - 1 - earlier))) if earlier.size else 0.0
            r = float(f[t]) - rbar_prev
            if r != 0.0:
                when = t + 1 + policy.delay_ms
                out[when] = out.get(when, 0.0) + r
        return out
    raise ValueError(policy.kind)


def pairing_events(pre_times: np.ndarray, post_times: np.ndarray, c: Constants) -> Dict[int, float]:
    """Nonzero zeta(t) by explicit pair sums."""
    zeta: Dict[int, float] = {}
    pre_times = np.asarray(pre_times, dtype=int)
    post_times = np.asarray(post_times, dtype=int)
    for t in post_times:
        earlier = pre_times[pre_times <= t]
        if earlier.size:
            zeta[int(t)] = zeta.get(int(t), 0.0) + c.a_plus * float(np.sum(np.exp(-(t - earlier) / c.tau_plus)))
    for t in pre_times:
        earlier = post_times[post_times <= t]
        if earlier.size:
            zeta[int(t)] = zeta.get(int(t), 0.0) + c.a_minus * float(np.sum(np.exp(-(t - earlier) / c.tau_minus)))
    return {t: v for t, v in zeta.items() if v != 0.0}


def eligibility_at(delivery_steps: Sequence[int], zeta: Dict[int, float], c: Constants) -> np.ndarray:
    """z(s) for each delivery step s by explicit weighted sums (chunked)."""
    s = np.asarray(delivery_steps, dtype=float)
    if not zeta or s.size == 0:
        return np.zeros(s.size)
    times = np.asarray(sorted(zeta), dtype=float)
    vals = np.asarray([zeta[int(t)] for t in times]) / c.tau_z
    out = np.zeros(s.size)
    for lo in range(0, s.size, 2048):
        chunk = s[lo : lo + 2048, None]
        lag = chunk - 1.0 - times[None, :]
        mask = lag >= 0.0
        weights = np.where(mask, np.exp(-np.where(mask, lag, 0.0) / c.tau_z), 0.0)
        out[lo : lo + 2048] = weights @ vals
    return out


def expected_weight_path(
    w0: float,
    low: float,
    high: float,
    pre_times: np.ndarray,
    post_times: np.ndarray,
    schedule: Dict[int, float],
    c: Constants,
    steps: int,
    rule: str = "mstdpet",
) -> Tuple[np.ndarray, float]:
    """Per-step weight after each 1 ms update, plus the unclipped sum of updates.

    A delivery at step s is applied at the end of step s-1 (eq. 2.7 maps
    w(t) -> w(t+dt) with r(t+dt) z(t+dt)); only deliveries with s <= steps occur.
    """
    when = sorted(s for s in schedule if 1 <= s <= steps)
    zeta = pairing_events(pre_times, post_times, c)
    if rule == "mstdpet":
        z = eligibility_at(when, zeta, c)
    elif rule == "mstdp":
        # Eq. 3.9: w(t+dt) = w(t) + gamma r(t+dt) zeta(t); no trace, no /tau_z.
        z = np.asarray([zeta.get(s - 1, 0.0) for s in when])
    else:
        raise ValueError(rule)
    path = np.full(steps, float(w0))
    w = float(w0)
    raw = 0.0
    change_steps: List[int] = []
    change_values: List[float] = []
    for s, zs in zip(when, z):
        delta = c.gamma * schedule[s] * zs
        raw += delta
        w = min(high, max(low, w + delta))
        change_steps.append(s - 1)
        change_values.append(w)
    if change_steps:
        idx = np.searchsorted(np.asarray(change_steps), np.arange(steps), side="right") - 1
        has = idx >= 0
        path[has] = np.asarray(change_values)[idx[has]]
    return path, raw


def expected_weight(w0, low, high, pre_times, post_times, schedule, c, steps, rule="mstdpet") -> Tuple[float, float]:
    """Return (final weight with sequential clipping, unclipped sum of updates)."""
    path, raw = expected_weight_path(w0, low, high, pre_times, post_times, schedule, c, steps, rule)
    return float(path[-1]), raw


def replay_if_spikes(
    n_inputs: int,
    names: Sequence[str],
    synapses: Sequence[Tuple[str, int, int]],
    spikes: np.ndarray,
    weight_trace: Dict[str, np.ndarray],
    c: Constants,
) -> np.ndarray:
    """Recompute IF spikes from recorded inputs and a per-step weight path.

    Callers pass the checker's own weight path (from ``expected_weight_path``),
    so the replay closes a fixed point: recorded spikes -> hand weights ->
    replayed spikes must equal recorded spikes.

    ``synapses`` is the declared (name, pre, post) connectivity.  Returns the
    replayed (T, N) spike matrix; input columns are copied from the record.
    Integrate-and-fire neurons are recomputed in topological index order, so a
    downstream neuron sees the replayed (not recorded) upstream spikes.
    """
    steps, n = spikes.shape
    replay = np.zeros_like(spikes, dtype=bool)
    replay[:, :n_inputs] = spikes[:, :n_inputs]
    b = math.exp(-1.0 / c.tau_m)
    for neuron in range(n_inputs, n):
        incoming = [(name, pre) for name, pre, post in synapses if post == neuron]
        # drive into u(t) comes from spikes at t-1 through weights after reward at t-1.
        drive = np.zeros(steps)
        for name, pre in incoming:
            w = weight_trace[name]
            f = replay[:, pre].astype(float)
            drive[1:] += w[:-1] * f[:-1]
        last_reset = 0
        for t in range(1, steps):
            # Terms older than 800 ms are below exp(-40) ~ 4e-18 of their size.
            lags = np.arange(max(last_reset + 1, t - 800), t + 1)
            u = c.u_rest + float(np.sum(drive[lags] * b ** (t - lags)))
            if u >= c.theta:
                replay[t, neuron] = True
                last_reset = t
    return replay
