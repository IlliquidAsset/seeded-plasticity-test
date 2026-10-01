"""Seed-for-seed comparison between the core Florian bench and the independent anchor.

This is the only place where the core path and ``ladder.florian`` meet, and it
only observes them. The core engine never imports this module.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ladder.florian import FlorianConfig, FlorianNetwork
from snn.florian_bench import CoreFlorianBench, CoreFlorianConfig


def _record(bench) -> List[np.ndarray]:
    rows: List[np.ndarray] = []
    inner = bench.step

    def observed(inp, target):
        fired = inner(inp, target)
        rows.append(fired)
        return fired

    bench.step = observed
    return rows


def run_anchor_recorded(task: str, rule: str, seed: int, epochs: int, plastic: bool = True):
    net = FlorianNetwork(FlorianConfig(task, rule, epochs=epochs, plastic=plastic), seed)
    rows = _record(net)
    result = net.run()
    del net.step
    return net, result, np.asarray(rows, dtype=np.int8)


def run_core_recorded(task: str, rule: str, seed: int, epochs: int, plastic: bool = True):
    bench = CoreFlorianBench(CoreFlorianConfig(task, rule, epochs=epochs, plastic=plastic), seed)
    rows = _record(bench)
    result = bench.run()
    del bench.step
    return bench, result, np.asarray(rows, dtype=np.int8)


def compare_trains(core_train: np.ndarray, anchor_train: np.ndarray) -> Dict[str, object]:
    n = min(core_train.size, anchor_train.size)
    diff = np.flatnonzero(core_train[:n] != anchor_train[:n])
    return {
        "output_train_identical": bool(core_train.size == anchor_train.size and diff.size == 0),
        "first_divergent_step": int(diff[0]) if diff.size else None,
        "n_divergent_steps": int(diff.size),
        "core_output_spikes": int(core_train.sum()),
        "anchor_output_spikes": int(anchor_train.sum()),
    }


def parity(task: str, rule: str, seed: int, epochs: int, plastic: bool = True) -> Dict[str, object]:
    anchor, a_res, a_train = run_anchor_recorded(task, rule, seed, epochs, plastic)
    bench, c_res, c_train = run_core_recorded(task, rule, seed, epochs, plastic)
    w1, w2 = bench.weights()
    return {
        "task": task,
        "rule": rule,
        "seed": seed,
        "epochs": epochs,
        "plastic": plastic,
        **compare_trains(c_train, a_train),
        "max_abs_w1_diff_mv": float(np.max(np.abs(w1 - anchor.w1))),
        "max_abs_w2_diff_mv": float(np.max(np.abs(w2 - anchor.w2))),
        "core_success": c_res["success"],
        "anchor_success": a_res["success"],
        "core_rates_hz": c_res["last_epoch_rates_hz"],
        "anchor_rates_hz": a_res["last_epoch_rates_hz"],
    }
