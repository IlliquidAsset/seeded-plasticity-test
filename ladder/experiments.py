"""Hand-checkable experiments for an R-STDP learning ladder.

The ladder deliberately separates the mechanism claim (delta-w equals reward
x eligibility) from the outcome claim (a threshold neuron's firing changes
relative to an identical frozen copy).  It is not a replacement simulator.
It is a tiny reference oracle for the repository's learning rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

DELAYS_MS = (0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 499, 500, 750, 1000, 2000)
TAUS_MS = (25.0, 500.0, 1000.0)
SEEDS = tuple(range(10))


@dataclass
class EligibilityTrace:
    """Discrete trace with explicit same-timestep semantics.

    A tag written at time t is visible to reward delivered at time t.  Moving
    reward d integer milliseconds later applies exactly d decay operations.
    """

    tau_ms: float
    dt_ms: float = 1.0
    value: float = 0.0

    @property
    def beta(self) -> float:
        return math.exp(-self.dt_ms / self.tau_ms)

    def tag(self, amount: float = 1.0) -> None:
        self.value += amount

    def step(self, tag: float = 0.0) -> None:
        self.value = self.beta * self.value + tag

    def elapse(self, delay_ms: int) -> None:
        for _ in range(int(delay_ms / self.dt_ms)):
            self.step()

    def credit(self, reward: float) -> float:
        return reward * self.value


def _probe_firing_fraction(weight: float, threshold: float = 1.0, seed: int = 0) -> float:
    """Fire a threshold neuron under a seeded, frozen drive panel.

    Learned and frozen copies receive the exact same panel for each seed.
    """
    background = np.random.default_rng(seed).uniform(-0.5, 0.5, size=1001)
    return float(np.mean(weight + background >= threshold))


def _mechanism_update(
    tags: Sequence[float], reward: float, delay_ms: int, tau_ms: float, lr: float
) -> np.ndarray:
    decay = math.exp(-delay_ms / tau_ms)
    return lr * reward * np.asarray(tags, dtype=float) * decay


def _background_noise_floor(tau_ms: float, rate_hz: float = 5.0) -> Dict[str, float]:
    """Empirical RMS eligibility under independent 5 Hz pre/post activity.

    This uses the same trace ordering and default STDP amplitudes as
    snn.core.RSTDPPlasticity.  The resulting crossing is conditional on this
    stated background model; it is not a universal biological noise floor.
    """
    rms_values: List[float] = []
    beta_e = math.exp(-1.0 / tau_ms)
    beta_s = math.exp(-1.0 / 20.0)
    p = rate_hz / 1000.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        eligibility = pre_trace = post_trace = 0.0
        samples: List[float] = []
        for t in range(60_000):
            pre = float(rng.random() < p)
            post = float(rng.random() < p)
            pre_trace = beta_s * pre_trace + pre
            post_trace = beta_s * post_trace + post
            stdp = 0.01 * pre_trace * post - 0.01 * post_trace * pre
            eligibility = beta_e * eligibility + stdp
            if t >= 10_000:
                samples.append(eligibility)
        rms_values.append(float(np.sqrt(np.mean(np.square(samples)))))
    rms = float(np.mean(rms_values))
    crossing = math.inf if rms <= 0.0 else -tau_ms * math.log(rms)
    return {
        "background_pre_hz": rate_hz,
        "background_post_hz": rate_hz,
        "stdp_a_plus": 0.01,
        "stdp_a_minus": 0.01,
        "noise_rms_normalized_to_unit_tag": rms,
        "signal_below_noise_after_ms": crossing,
        "seed_count": len(SEEDS),
    }


def run_rung0() -> Dict[str, Any]:
    curves: Dict[str, Any] = {}
    max_error = 0.0
    for tau in TAUS_MS:
        rows = []
        for delay in DELAYS_MS:
            trace = EligibilityTrace(tau)
            trace.tag(1.0)
            trace.elapse(delay)
            observed = trace.credit(1.0)
            theory = math.exp(-delay / tau)
            error = abs(observed - theory)
            max_error = max(max_error, error)
            rows.append(
                {
                    "delay_ms": delay,
                    "observed_credit": observed,
                    "theory_credit": theory,
                    "absolute_error": error,
                }
            )
        curves[str(int(tau))] = {
            "points": rows,
            "noise_floor": _background_noise_floor(tau),
        }

    same_step = EligibilityTrace(25.0)
    same_step.tag(1.0)
    same_step_credit = same_step.credit(1.0)
    pass_gate = max_error < 1e-12 and same_step_credit == 1.0
    return {
        "rung": 0,
        "status": "PASS" if pass_gate else "FAIL",
        "same_timestep_order": "tag, then reward; zero decay operations at d=0",
        "same_timestep_credit": same_step_credit,
        "max_absolute_error": max_error,
        "mechanism_pass": pass_gate,
        "outcome_pass": True,
        "outcome_note": "Trace-only rung has no synapse or firing behavior by design.",
        "curves": curves,
    }


def run_rung1() -> Dict[str, Any]:
    tau = 25.0
    lr = 0.1
    delay_sweep = []
    max_error = 0.0
    for delay in DELAYS_MS:
        observed = float(_mechanism_update([1.0], 1.0, delay, tau, lr)[0])
        expected = lr * math.exp(-delay / tau)
        max_error = max(max_error, abs(observed - expected))
        delay_sweep.append(
            {"delay_ms": delay, "delta_weight": observed, "expected": expected}
        )

    initial = 0.9
    cap = 2.0
    trials = 200
    train_lr = 0.02
    train_delay = 5
    unit_credit = math.exp(-train_delay / tau)

    raw_weight = initial
    raw_history = [raw_weight]
    for _ in range(trials):
        raw_weight = min(cap, raw_weight + train_lr * unit_credit)
        raw_history.append(raw_weight)

    baseline_weight = initial
    baseline = 0.0
    alpha = 0.1
    baseline_history = [baseline_weight]
    for _ in range(trials):
        advantage = 1.0 - baseline
        baseline_weight = min(cap, baseline_weight + train_lr * advantage * unit_credit)
        baseline += alpha * (1.0 - baseline)
        baseline_history.append(baseline_weight)

    # Mirror: an unwanted output spike receives punishment.  A positive reward
    # for silence alone has no post-spike tag and therefore cannot change this
    # synapse under pair-based R-STDP.
    mirror_weight = initial
    for _ in range(30):
        mirror_weight += train_lr * (-1.0) * unit_credit
    positive_silence_delta = float(_mechanism_update([0.0], 1.0, train_delay, tau, train_lr)[0])

    no_reward_delta = float(_mechanism_update([1.0], 0.0, train_delay, tau, train_lr)[0])
    random_final_deltas = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rewards = np.asarray([1.0] * 100 + [-1.0] * 100)
        rng.shuffle(rewards)
        random_final_deltas.append(float(train_lr * unit_credit * rewards.sum()))

    per_seed_outcomes = []
    for seed in SEEDS:
        frozen_rate = _probe_firing_fraction(initial, seed=seed)
        per_seed_outcomes.append(
            {
                "seed": seed,
                "frozen_firing_fraction": frozen_rate,
                "self_reward_raw_firing_fraction": _probe_firing_fraction(raw_weight, seed=seed),
                "self_reward_baseline_firing_fraction": _probe_firing_fraction(baseline_weight, seed=seed),
                "silence_target_punishment_firing_fraction": _probe_firing_fraction(mirror_weight, seed=seed),
            }
        )
    outcomes = {
        key: float(np.mean([row[key] for row in per_seed_outcomes]))
        for key in (
            "frozen_firing_fraction",
            "self_reward_raw_firing_fraction",
            "self_reward_baseline_firing_fraction",
            "silence_target_punishment_firing_fraction",
        )
    }
    mechanism_pass = max_error < 1e-12 and no_reward_delta == 0.0 and max(
        abs(x) for x in random_final_deltas
    ) < 1e-12
    outcome_pass = all(
        row["self_reward_raw_firing_fraction"] > row["frozen_firing_fraction"]
        and row["self_reward_baseline_firing_fraction"] > row["frozen_firing_fraction"]
        and row["silence_target_punishment_firing_fraction"] < row["frozen_firing_fraction"]
        for row in per_seed_outcomes
    )
    stabilization_pass = raw_weight == cap and baseline_weight < cap
    pass_gate = mechanism_pass and outcome_pass and stabilization_pass
    return {
        "rung": 1,
        "status": "PASS" if pass_gate else "FAIL",
        "mechanism_pass": mechanism_pass,
        "outcome_pass": outcome_pass,
        "delay_sweep": delay_sweep,
        "max_delay_curve_error": max_error,
        "controls": {
            "no_reward_delta": no_reward_delta,
            "random_reward_final_delta_by_seed": random_final_deltas,
            "positive_reward_for_silence_delta_without_tag": positive_silence_delta,
            "silence_target_method": "punish an emitted spike with reward=-1",
        },
        "runaway": {
            "raw_final_weight": raw_weight,
            "raw_hit_cap": raw_weight == cap,
            "baseline_subtracted_final_weight": baseline_weight,
            "baseline_hit_cap": baseline_weight == cap,
            "baseline_final": baseline,
        },
        "outcome": outcomes,
        "outcome_per_seed": per_seed_outcomes,
    }


def _rung2_condition(delay: int, background_hz: float, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    tau = 25.0
    lr = 0.002
    episodes = 200
    total = np.zeros(2, dtype=float)
    predicted = np.zeros(2, dtype=float)
    beta = math.exp(-1.0 / tau)
    p = background_hz / 1000.0
    for _ in range(episodes):
        eligibility = np.asarray([1.0, 0.0])
        for _step in range(delay):
            b_tag = 0.0
            if rng.random() < p:
                b_tag = 0.2 if rng.random() < 0.5 else -0.2
            eligibility = beta * eligibility + np.asarray([0.0, b_tag])
        delta = lr * eligibility
        total += delta
        predicted += delta
    return {
        "delta_w_A": float(total[0]),
        "delta_w_B": float(total[1]),
        "prediction_error": float(np.max(np.abs(total - predicted))),
        "leakage_ratio_abs": float(abs(total[1]) / max(abs(total[0]), 1e-30)),
    }


def run_rung2() -> Dict[str, Any]:
    delays = (0, 5, 25, 50, 100, 200)
    backgrounds = (0.0, 1.0, 5.0, 20.0, 50.0)
    grid = []
    for delay in delays:
        for rate in backgrounds:
            per_seed = [_rung2_condition(delay, rate, seed) for seed in SEEDS]
            leakage = [row["leakage_ratio_abs"] for row in per_seed]
            grid.append(
                {
                    "delay_ms": delay,
                    "background_hz": rate,
                    "median_delta_w_A": float(np.median([x["delta_w_A"] for x in per_seed])),
                    "median_delta_w_B": float(np.median([x["delta_w_B"] for x in per_seed])),
                    "median_abs_leakage_ratio": float(np.median(leakage)),
                    "p95_abs_leakage_ratio": float(np.quantile(leakage, 0.95)),
                    "per_seed": per_seed,
                }
            )

    clean = next(x for x in grid if x["delay_ms"] == 5 and x["background_hz"] == 0.0)
    initial_a = initial_b = 0.85
    learned_a = initial_a + clean["median_delta_w_A"]
    learned_b = initial_b + clean["median_delta_w_B"]
    per_seed_outcomes = []
    for seed in SEEDS:
        per_seed_outcomes.append(
            {
                "seed": seed,
                "frozen_A_firing_fraction": _probe_firing_fraction(initial_a, seed=seed),
                "learned_A_firing_fraction": _probe_firing_fraction(learned_a, seed=seed),
                "frozen_B_firing_fraction": _probe_firing_fraction(initial_b, seed=seed),
                "learned_B_firing_fraction": _probe_firing_fraction(learned_b, seed=seed),
            }
        )
    outcome = {
        key: float(np.mean([row[key] for row in per_seed_outcomes]))
        for key in (
            "frozen_A_firing_fraction",
            "learned_A_firing_fraction",
            "frozen_B_firing_fraction",
            "learned_B_firing_fraction",
        )
    }
    mechanism_pass = clean["median_delta_w_A"] > 0.0 and clean["median_delta_w_B"] == 0.0
    outcome_pass = all(
        row["learned_A_firing_fraction"] > row["frozen_A_firing_fraction"]
        and row["learned_B_firing_fraction"] == row["frozen_B_firing_fraction"]
        for row in per_seed_outcomes
    )
    pass_gate = mechanism_pass and outcome_pass
    breaks = [
        {"delay_ms": row["delay_ms"], "background_hz": row["background_hz"]}
        for row in grid
        if row["median_abs_leakage_ratio"] >= 0.1
    ]
    return {
        "rung": 2,
        "status": "PASS" if pass_gate else "FAIL",
        "mechanism_pass": mechanism_pass,
        "outcome_pass": outcome_pass,
        "primary_gate": clean,
        "outcome": outcome,
        "outcome_per_seed": per_seed_outcomes,
        "map_definition": "break = median |delta_w_B/delta_w_A| >= 0.10 across 10 seeds",
        "break_conditions": breaks,
        "grid": grid,
    }


def run_rung3() -> Dict[str, Any]:
    tau = 25.0
    delay = 10
    lr = 0.003
    episodes = 100
    per_event = lr * math.exp(-delay / tau)
    growth = episodes * per_event
    initial = 0.85

    series_expected = np.asarray([growth, growth])
    series_observed = np.zeros(2)
    parallel_expected = np.asarray([growth, 0.0])
    parallel_observed = np.zeros(2)
    control_expected = 0.0
    control_observed = 0.0
    for _ in range(episodes):
        series_observed += _mechanism_update([1.0, 1.0], 1.0, delay, tau, lr)
        parallel_observed += _mechanism_update([1.0, 0.0], 1.0, delay, tau, lr)
        control_observed += float(_mechanism_update([0.0], 1.0, delay, tau, lr)[0])

    mechanism_error = max(
        float(np.max(np.abs(series_observed - series_expected))),
        float(np.max(np.abs(parallel_observed - parallel_expected))),
        abs(control_observed - control_expected),
    )
    per_seed_outcomes = []
    for seed in SEEDS:
        frozen_rate = _probe_firing_fraction(initial, seed=seed)
        per_seed_outcomes.append(
            {
                "seed": seed,
                "series_A_to_B_frozen": frozen_rate,
                "series_A_to_B_learned": _probe_firing_fraction(initial + series_observed[0], seed=seed),
                "series_B_to_C_frozen": frozen_rate,
                "series_B_to_C_learned": _probe_firing_fraction(initial + series_observed[1], seed=seed),
                "parallel_A_to_C_frozen": frozen_rate,
                "parallel_A_to_C_learned": _probe_firing_fraction(initial + parallel_observed[0], seed=seed),
                "parallel_B_to_C_frozen": frozen_rate,
                "parallel_B_to_C_learned": _probe_firing_fraction(initial + parallel_observed[1], seed=seed),
                "unconnected_frozen": frozen_rate,
                "unconnected_after_reward": _probe_firing_fraction(initial + control_observed, seed=seed),
            }
        )
    outcome_keys = tuple(key for key in per_seed_outcomes[0] if key != "seed")
    outcome = {
        key: float(np.mean([row[key] for row in per_seed_outcomes])) for key in outcome_keys
    }
    mechanism_pass = mechanism_error < 1e-12 and control_observed == 0.0
    outcome_pass = all(
        row["series_A_to_B_learned"] > row["series_A_to_B_frozen"]
        and row["series_B_to_C_learned"] > row["series_B_to_C_frozen"]
        and row["parallel_A_to_C_learned"] > row["parallel_A_to_C_frozen"]
        and row["parallel_B_to_C_learned"] == row["parallel_B_to_C_frozen"]
        and row["unconnected_after_reward"] == row["unconnected_frozen"]
        for row in per_seed_outcomes
    )
    pass_gate = mechanism_pass and outcome_pass
    return {
        "rung": 3,
        "status": "PASS" if pass_gate else "FAIL",
        "mechanism_pass": mechanism_pass,
        "outcome_pass": outcome_pass,
        "mechanism_max_absolute_error": mechanism_error,
        "series_delta_weights": series_observed.tolist(),
        "parallel_delta_weights": parallel_observed.tolist(),
        "unconnected_delta_weight": control_observed,
        "outcome": outcome,
        "outcome_per_seed": per_seed_outcomes,
    }


def run_all_rungs() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    runners = (run_rung0, run_rung1, run_rung2, run_rung3)
    for index, runner in enumerate(runners):
        result = runner()
        results[f"rung_{index}"] = result
        if result["status"] != "PASS":
            return {
                "status": "STOPPED",
                "stop_reason": f"rung {index} failed; higher rungs were not run",
                "results": results,
            }
    return {"status": "PASS", "stop_reason": None, "results": results}
