"""R-STDP learning ladder.

Rung 0 (trace mechanics) is unchanged from commit eb972c7 and was independently
passed by Nora in t_a0a9775c.  The synthetic rungs 1-3 of eb972c7 (preassigned
eligibility vectors plus a separate threshold probe) were rejected in that
review and are removed.  Rungs 1-3 now execute real spiking circuits; see
``ladder.executed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ladder.executed import run_rung1, run_rung2, run_rung3

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
