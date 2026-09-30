"""Equation and reward-order check for the Florian NumPy anchor.

Records every layer's spikes and the target from a real ``FlorianNetwork`` run,
then recomputes selected final weights with ``ladder.handcheck`` (explicit pair
sums, reward schedule derived from recorded output spikes and targets).  Shares
no update code with ``ladder.florian``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ladder import handcheck as hc
from ladder.florian import FlorianConfig, FlorianNetwork


def record_run(net: FlorianNetwork) -> Dict[str, np.ndarray]:
    """Run ``net`` while observing f(t) of every layer from outside ``step``.

    Wrapping the bound method (rather than adding a hook inside the anchor)
    leaves the anchor's code path unchanged and also observes subclasses that
    override ``step``.
    """
    rows: Dict[str, List[np.ndarray]] = {"inp": [], "hid": [], "out": [], "target": []}
    inner = net.step

    def observed_step(input_spikes, target):
        rows["inp"].append(np.asarray(input_spikes).astype(bool))
        rows["hid"].append(net.hidden_spikes.astype(bool))
        rows["out"].append(net.output_spikes.astype(bool))
        rows["target"].append(target)
        return inner(input_spikes, target)

    w1_0, w2_0 = net.w1.copy(), net.w2.copy()
    net.step = observed_step  # type: ignore[method-assign]
    try:
        net.run()
    finally:
        del net.step
    return {
        "inp": np.asarray(rows["inp"]),
        "hid": np.asarray(rows["hid"]),
        "out": np.asarray(rows["out"])[:, 0],
        "target": np.asarray(rows["target"]),
        "w1_0": w1_0,
        "w2_0": w2_0,
    }


def reward_from_record(rec: Dict[str, np.ndarray]) -> Dict[int, float]:
    """Paper 4.2: +1 (target 1) or -1 (target 0) per output spike, next step."""
    schedule: Dict[int, float] = {}
    for t in np.flatnonzero(rec["out"]):
        schedule[int(t) + 1] = 1.0 if rec["target"][t] else -1.0
    return schedule


def check_network(
    net: FlorianNetwork,
    n_w1_samples: int = 40,
    sample_seed: int = 0,
) -> Dict[str, object]:
    cfg = net.config
    rec = record_run(net)
    steps = rec["out"].shape[0]
    const = hc.Constants(gamma=cfg.gamma_mv, tau_z=cfg.tau_elig_ms, tau_plus=cfg.tau_stdp_ms, tau_minus=cfg.tau_stdp_ms)
    schedule = reward_from_record(rec)
    errors = []
    checked = 0
    out_times = np.flatnonzero(rec["out"])
    w2_low, w2_high = net.w2_low, net.w2_high
    for j in range(net.w2.shape[1]):
        expected, _ = hc.expected_weight(
            float(rec["w2_0"][0, j]), w2_low, w2_high, np.flatnonzero(rec["hid"][:, j]), out_times, schedule, const, steps, cfg.rule
        )
        errors.append(abs(expected - float(net.w2[0, j])))
        checked += 1
    rng = np.random.default_rng(sample_seed)
    low1 = np.broadcast_to(net.w1_low, net.w1.shape)
    high1 = np.broadcast_to(net.w1_high, net.w1.shape)
    for _ in range(n_w1_samples):
        h = int(rng.integers(net.w1.shape[0]))
        i = int(rng.integers(net.w1.shape[1]))
        expected, _ = hc.expected_weight(
            float(rec["w1_0"][h, i]),
            float(low1[h, i]),
            float(high1[h, i]),
            np.flatnonzero(rec["inp"][:, i]),
            np.flatnonzero(rec["hid"][:, h]),
            schedule,
            const,
            steps,
            cfg.rule,
        )
        errors.append(abs(expected - float(net.w1[h, i])))
        checked += 1
    total_change = float(np.abs(net.w2 - rec["w2_0"]).sum() + np.abs(net.w1 - rec["w1_0"]).sum())
    scale = max(1.0, float(np.max(np.abs(net.w1))), float(np.max(np.abs(net.w2))))
    max_err = float(max(errors))
    return {
        "task": cfg.task,
        "rule": cfg.rule,
        "epochs": cfg.epochs,
        "steps": steps,
        "output_spikes": int(rec["out"].sum()),
        "reward_events": len(schedule),
        "synapses_checked": checked,
        "total_abs_weight_change_mv": total_change,
        "max_abs_error_mv": max_err,
        "pass": max_err < 1e-8 * scale and total_change > 0.0,
    }


def run_equation_checks(epochs: int = 2, seeds=(0, 1)) -> List[Dict[str, object]]:
    rows = []
    for task in ("rate", "temporal"):
        for rule in ("mstdp", "mstdpet"):
            for seed in seeds:
                net = FlorianNetwork(FlorianConfig(task, rule, epochs=epochs), seed)
                row = check_network(net)
                row["seed"] = seed
                rows.append(row)
    return rows
