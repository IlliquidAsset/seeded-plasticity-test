"""Executed ladder rungs 1-3 (supersedes the synthetic rungs of commit eb972c7).

Each rung instantiates real integrate-and-fire neurons and plastic synapses
(``ladder.spiking``), drives them with generated Poisson input, schedules reward
only from spikes the circuit actually emitted, and then:

1. Mechanism: recomputes every final weight independently from the recorded
   spike times with explicit pair sums (``ladder.handcheck``) and replays every
   IF spike from recorded inputs + weight trajectory.  Predicted and observed
   are produced by different code paths.
2. Outcome: drives the learned weights and an identical frozen copy (initial
   weights) with the same held-out input panel and compares output spikes at
   the motif output.

Controls per rung: no reward, time-shuffled reward (same count of +1 events at
random steps), sign-randomized reward (same count, random +/-1 at random
steps), paired frozen copy, and (rung 3) disconnected synapses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ladder import handcheck as hc
from ladder.spiking import Circuit, Rule, RunRecord, Simulator, SynapseSpec, evaluate_frozen, poisson_raster

SEEDS = tuple(range(10))
TRAIN_MS = 20_000
R1_TRAIN_MS = 60_000
TEST_MS = 10_000
TOL_MV = 1e-9
# A weight below SILENT_W_MV cannot drive a neuron to threshold even if its
# input fired every 1 ms: max depolarization = w / (1 - exp(-1/20)) < 16 mV.
SILENT_W_MV = 0.5
RULE = Rule(gamma_mv=0.2)
CONST = hc.Constants(gamma=RULE.gamma_mv)


# ---------------------------------------------------------------------------
# Reward policies.  Simulator-side callables are written independently of the
# checker's declarative evaluation in ``handcheck.reward_schedule``.
# ---------------------------------------------------------------------------

def sim_policy(spec: hc.PolicySpec) -> Optional[Callable[[int, np.ndarray, np.ndarray], float]]:
    if spec.kind in ("none", "exogenous"):
        return None
    if spec.kind == "per_spike":
        n, v, prior = spec.neuron, spec.value, spec.require_prior

        def per_spike(t, f, history):
            if not f[n]:
                return 0.0
            if prior is not None and (t == 0 or not history[t - 1, prior]):
                return 0.0
            return v

        return per_spike
    if spec.kind == "per_silent_step":
        n, v = spec.neuron, spec.value
        return lambda t, f, history: 0.0 if f[n] else v
    if spec.kind == "baseline":
        n = spec.neuron
        b = math.exp(-1.0 / spec.tau_bar_ms)
        state = {"rbar": 0.0}

        def baseline(t, f, history):
            r = float(f[n]) - state["rbar"]
            state["rbar"] = b * state["rbar"] + (1.0 - b) * float(f[n])
            return r

        return baseline
    raise ValueError(spec.kind)


def run_arm(circuit: Circuit, raster: np.ndarray, spec: hc.PolicySpec, rule: Rule = RULE, sim_cls=Simulator) -> RunRecord:
    exo: Optional[Dict[int, float]] = None
    if spec.kind == "exogenous" and spec.schedule:
        # Sum duplicates: two random events may land on the same step.
        exo = {}
        for when, value in spec.schedule:
            exo[int(when)] = exo.get(int(when), 0.0) + float(value)
    return sim_cls(circuit, rule).run(raster, sim_policy(spec), delay_ms=spec.delay_ms, exogenous=exo)


def random_schedule(rng: np.random.Generator, count: int, steps: int, signed: bool) -> Tuple[Tuple[int, float], ...]:
    when = rng.integers(1, steps, size=count)
    values = rng.choice([-1.0, 1.0], size=count) if signed else np.ones(count)
    return tuple((int(w), float(v)) for w, v in zip(when, values))


# ---------------------------------------------------------------------------
# Independent mechanism check
# ---------------------------------------------------------------------------

def check_mechanism(circuit: Circuit, rec: RunRecord, spec: hc.PolicySpec, const: hc.Constants = CONST) -> Dict[str, Any]:
    steps = rec.spikes.shape[0]
    schedule = hc.reward_schedule(rec.spikes, spec)
    per_syn = {}
    hand_paths: Dict[str, np.ndarray] = {}
    worst = 0.0
    for s in circuit.synapses:
        if s.plastic:
            path, raw = hc.expected_weight_path(
                s.w0_mv, s.low_mv, s.high_mv, rec.spike_times(s.pre), rec.spike_times(s.post), schedule, const, steps
            )
        else:
            path, raw = np.full(steps, s.w0_mv), 0.0
        hand_paths[s.name] = path
        expected = float(path[-1])
        observed = rec.weights_final[s.name]
        err = abs(observed - expected)
        worst = max(worst, err, float(np.max(np.abs(path - rec.weight_trace[s.name]))))
        per_syn[s.name] = {
            "w0_mv": s.w0_mv,
            "observed_final_mv": observed,
            "hand_expected_final_mv": expected,
            "observed_delta_mv": observed - s.w0_mv,
            "hand_unclipped_delta_mv": raw,
            "abs_error_mv": err,
            "clip_events": rec.clip_events[s.name],
        }
    replay = hc.replay_if_spikes(
        circuit.n_inputs,
        circuit.names,
        [(s.name, s.pre, s.post) for s in circuit.synapses],
        rec.spikes,
        hand_paths,
        const,
    )
    mismatches = int((replay != rec.spikes).sum())
    reward_mismatch = _schedule_mismatch(schedule, rec.reward_delivered, steps)
    return {
        "synapses": per_syn,
        "max_abs_weight_error_mv": worst,
        "spike_replay_mismatches": mismatches,
        "reward_schedule_mismatch": reward_mismatch,
        "reward_events": int(sum(1 for s in schedule if s <= steps)),
        "reward_sum": float(sum(v for s, v in schedule.items() if s <= steps)),
        "pass": worst < TOL_MV and mismatches == 0 and reward_mismatch < 1e-9,
    }


def _schedule_mismatch(expected: Dict[int, float], delivered: Dict[int, float], steps: int) -> float:
    keys = {k for k in expected if k <= steps} | set(delivered)
    return max((abs(expected.get(k, 0.0) - delivered.get(k, 0.0)) for k in keys), default=0.0)


def output_counts(circuit: Circuit, learned: Dict[str, float], raster: np.ndarray, neuron: int) -> Tuple[int, int]:
    frozen = {s.name: s.w0_mv for s in circuit.synapses}
    a = evaluate_frozen(circuit, learned, raster, RULE).count(neuron)
    b = evaluate_frozen(circuit, frozen, raster, RULE).count(neuron)
    return a, b


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(list(key))


# ---------------------------------------------------------------------------
# Rung 1: one input -> one output, one plastic synapse
# ---------------------------------------------------------------------------

R1_INPUT_HZ = 40.0
R1_W0 = 9.0
R1_BOUNDS = (0.0, 20.0)
R1_DELAYS = (0, 1, 5, 10, 25, 50, 100)


def r1_circuit(w0: float = R1_W0) -> Circuit:
    return Circuit(1, 1, (SynapseSpec("in->out", 0, 1, w0, *R1_BOUNDS),), ("in", "out"))


def r1_single_pairing(delay_ms: int, rule: Rule = RULE, sim_cls=Simulator) -> Dict[str, float]:
    """Deterministic: one input spike at t=10 fires the output at t=11.

    Closed form (hand): zeta(11) = A+ exp(-1/tau+); z(12) = zeta/tau_z;
    reward for the spike at 11 arrives at 12 + d, so
    delta_w = gamma * exp(-1/tau+) / tau_z * exp(-d/tau_z).
    """
    circuit = r1_circuit(w0=17.0)
    raster = np.zeros((200 + delay_ms, 1), dtype=bool)
    raster[10, 0] = True
    rec = sim_cls(circuit, rule).run(raster, lambda t, f, h: float(f[1]), delay_ms=delay_ms)
    hand = RULE.gamma_mv * math.exp(-1.0 / 20.0) / 25.0 * math.exp(-delay_ms / 25.0)
    observed = rec.weights_final["in->out"] - 17.0
    return {
        "delay_ms": delay_ms,
        "output_spike_times": rec.spike_times(1).tolist(),
        "observed_delta_mv": observed,
        "hand_closed_form_delta_mv": hand,
        "ratio_to_zero_delay": observed / (RULE.gamma_mv * math.exp(-1.0 / 20.0) / 25.0),
        "exp_minus_d_over_tau": math.exp(-delay_ms / 25.0),
        "abs_error_mv": abs(observed - hand),
    }


def _r1_seed(seed: int) -> Dict[str, Any]:
    circuit = r1_circuit()
    out = circuit.index("out")
    train = poisson_raster(_rng(1, seed, 0), TRAIN_MS, [R1_INPUT_HZ])
    test = poisson_raster(_rng(1, seed, 1), TEST_MS, [R1_INPUT_HZ])
    frozen_train = evaluate_frozen(circuit, {"in->out": R1_W0}, train, RULE)
    arms: Dict[str, hc.PolicySpec] = {
        "self_reward": hc.PolicySpec("per_spike", neuron=out, value=1.0),
        "self_reward_baseline": hc.PolicySpec("baseline", neuron=out, tau_bar_ms=1000.0),
        "punish_emitted_spikes": hc.PolicySpec("per_spike", neuron=out, value=-1.0),
        "reward_every_silent_step": hc.PolicySpec("per_silent_step", neuron=out, value=1.0 / 25.0),
        "no_reward": hc.PolicySpec("none"),
    }
    results: Dict[str, Any] = {}
    records: Dict[str, RunRecord] = {}
    for name, spec in arms.items():
        rec = run_arm(circuit, train, spec)
        records[name] = rec
        results[name] = {"spec": spec.kind, "mechanism": check_mechanism(circuit, rec, spec)}
    n_events = results["self_reward"]["mechanism"]["reward_events"]
    for name, signed in (("time_shuffled_reward", False), ("sign_randomized_reward", True)):
        spec = hc.PolicySpec("exogenous", schedule=random_schedule(_rng(1, seed, 2 + signed), n_events, TRAIN_MS, signed))
        rec = run_arm(circuit, train, spec)
        records[name] = rec
        results[name] = {"spec": spec.kind, "mechanism": check_mechanism(circuit, rec, spec)}
    for name, rec in records.items():
        learned, frozen = output_counts(circuit, rec.weights_final, test, out)
        results[name]["outcome"] = {
            "final_weight_mv": rec.weights_final["in->out"],
            "heldout_learned_output_spikes": learned,
            "heldout_frozen_output_spikes": frozen,
            "heldout_delta_spikes": learned - frozen,
            "train_output_spikes": rec.count(out),
            "train_frozen_copy_output_spikes": frozen_train.count(out),
            "weight_hit_upper_bound": rec.weights_final["in->out"] >= R1_BOUNDS[1],
            "weight_at_5s_10s_15s_20s_mv": [float(rec.weight_trace["in->out"][k - 1]) for k in (5000, 10000, 15000, 20000)],
        }

    # Silence mirror, fully silent operationalization: a SILENT_W_MV synapse
    # provably cannot fire the output, so there is never a postsynaptic tag.
    silent_circuit = r1_circuit(w0=SILENT_W_MV)
    spec = hc.PolicySpec("per_silent_step", neuron=out, value=1.0 / 25.0)
    rec = run_arm(silent_circuit, train, spec)
    results["reward_silence_output_never_fires"] = {
        "spec": spec.kind,
        "mechanism": check_mechanism(silent_circuit, rec, spec),
        "outcome": {
            "train_output_spikes": rec.count(out),
            "final_weight_mv": rec.weights_final["in->out"],
            "delta_weight_mv": rec.weights_final["in->out"] - SILENT_W_MV,
        },
    }

    # Runaway study: raw self-reward vs expected-reward baseline over 60 s.
    long_train = poisson_raster(_rng(1, seed, 5), R1_TRAIN_MS, [R1_INPUT_HZ])
    runaway = {}
    for name, spec in (
        ("raw", hc.PolicySpec("per_spike", neuron=out, value=1.0)),
        ("baseline", hc.PolicySpec("baseline", neuron=out, tau_bar_ms=1000.0)),
    ):
        rec = run_arm(circuit, long_train, spec)
        mech = check_mechanism(circuit, rec, spec)
        w = rec.weight_trace["in->out"]
        rate = [1000.0 * rec.spikes[k : k + 10_000, out].mean() for k in range(0, R1_TRAIN_MS, 10_000)]
        runaway[name] = {
            "mechanism_pass": mech["pass"],
            "max_abs_weight_error_mv": mech["max_abs_weight_error_mv"],
            "weight_every_10s_mv": [float(w[k - 1]) for k in range(10_000, R1_TRAIN_MS + 1, 10_000)],
            "output_rate_per_10s_hz": rate,
            "hit_upper_bound": bool(np.any(w >= R1_BOUNDS[1])),
            "first_step_at_bound": int(np.argmax(w >= R1_BOUNDS[1])) if np.any(w >= R1_BOUNDS[1]) else None,
            "clip_events": rec.clip_events["in->out"],
        }

    # Delay sweep on the stochastic system (self-reward).
    delay_rows = []
    for d in R1_DELAYS:
        spec = hc.PolicySpec("per_spike", neuron=out, value=1.0, delay_ms=d)
        # Short 5 s window keeps the weight far from the bound, so the
        # comparison isolates delay rather than saturation.
        rec = run_arm(circuit, train[:5000], spec)
        mech = check_mechanism(circuit, rec, spec)
        delay_rows.append(
            {
                "delay_ms": d,
                "observed_delta_mv": rec.weights_final["in->out"] - R1_W0,
                "reward_events": mech["reward_events"],
                "mechanism_pass": mech["pass"],
                "max_abs_weight_error_mv": mech["max_abs_weight_error_mv"],
            }
        )
    return {"seed": seed, "arms": results, "delay_sweep_5s": delay_rows, "runaway_60s": runaway}


def run_rung1(seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    closed = [r1_single_pairing(d) for d in R1_DELAYS]
    per_seed = [_r1_seed(s) for s in seeds]
    mech_ok = (
        all(arm["mechanism"]["pass"] for row in per_seed for arm in row["arms"].values())
        and all(r["mechanism_pass"] for row in per_seed for r in row["delay_sweep_5s"])
        and all(v["mechanism_pass"] for row in per_seed for v in row["runaway_60s"].values())
    )
    closed_ok = all(r["abs_error_mv"] < 1e-12 and r["output_spike_times"] == [11] for r in closed)
    no_reward_exact = all(
        row["arms"]["no_reward"]["outcome"]["final_weight_mv"] == R1_W0 for row in per_seed
    )
    silence_zero = all(
        row["arms"]["reward_silence_output_never_fires"]["outcome"]["delta_weight_mv"] == 0.0
        and row["arms"]["reward_silence_output_never_fires"]["outcome"]["train_output_spikes"] == 0
        for row in per_seed
    )
    o = lambda row, arm: row["arms"][arm]["outcome"]
    self_up = all(o(r, "self_reward")["heldout_delta_spikes"] > 0 for r in per_seed)
    punish_down = all(o(r, "punish_emitted_spikes")["heldout_delta_spikes"] < 0 for r in per_seed)
    beats_shuffled = all(
        o(r, "self_reward")["heldout_delta_spikes"] > o(r, "time_shuffled_reward")["heldout_delta_spikes"] for r in per_seed
    )
    beats_signed = all(
        o(r, "self_reward")["heldout_delta_spikes"] > o(r, "sign_randomized_reward")["heldout_delta_spikes"] for r in per_seed
    )
    mechanism_pass = mech_ok and closed_ok and no_reward_exact and silence_zero
    outcome_pass = self_up and punish_down and beats_shuffled and beats_signed
    raw = [r["runaway_60s"]["raw"] for r in per_seed]
    base = [r["runaway_60s"]["baseline"] for r in per_seed]
    runaway = {
        "raw_self_reward_hit_bound_seeds": sum(x["hit_upper_bound"] for x in raw),
        "baseline_subtracted_hit_bound_seeds": sum(x["hit_upper_bound"] for x in base),
        "raw_first_ms_at_bound": [x["first_step_at_bound"] for x in raw],
        "baseline_first_ms_at_bound": [x["first_step_at_bound"] for x in base],
        "raw_weight_every_10s_mv_median": np.median([x["weight_every_10s_mv"] for x in raw], axis=0).tolist(),
        "baseline_weight_every_10s_mv_median": np.median([x["weight_every_10s_mv"] for x in base], axis=0).tolist(),
        "raw_output_rate_per_10s_hz_median": np.median([x["output_rate_per_10s_hz"] for x in raw], axis=0).tolist(),
        "baseline_output_rate_per_10s_hz_median": np.median([x["output_rate_per_10s_hz"] for x in base], axis=0).tolist(),
        "predicted_failure_mode_reproduced": all(x["hit_upper_bound"] for x in raw),
        "baseline_prevents_saturation_all_seeds": not any(x["hit_upper_bound"] for x in base),
        "note": (
            "Reported, not gated. With pure self-reward, r(t)=f_out(t) is perfectly correlated with the "
            "post-spike tag it multiplies, so subtracting a running mean removes the average-reward drive "
            "but not the spike-contingent drive. Rough estimate (inferred, ignoring post-before-pre pairings "
            "and overlapping tags): each output spike nets credit proportional to 1 - rbar*tau_z, which "
            "only reaches zero near an output rate of 1/tau_z = 40 Hz."
        ),
    }
    return {
        "rung": 1,
        "status": "PASS" if mechanism_pass and outcome_pass else "FAIL",
        "mechanism_pass": mechanism_pass,
        "outcome_pass": outcome_pass,
        "gates": {
            "all_arms_hand_checked_and_replayed": mech_ok,
            "single_pairing_closed_form": closed_ok,
            "no_reward_weight_unchanged_exactly": no_reward_exact,
            "positive_reward_on_silence_with_silent_output_is_zero": silence_zero,
            "self_reward_heldout_firing_above_frozen_all_seeds": self_up,
            "punishment_heldout_firing_below_frozen_all_seeds": punish_down,
            "self_reward_beats_time_shuffled_all_seeds": beats_shuffled,
            "self_reward_beats_sign_randomized_all_seeds": beats_signed,
        },
        "single_pairing_closed_form": closed,
        "runaway_and_baseline": runaway,
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Rung 2: inputs A and B -> one output, two plastic synapses
# ---------------------------------------------------------------------------

R2_A_HZ = 40.0
R2_B_HZ = 20.0
R2_W0 = 9.0
R2_BOUNDS = (0.0, 20.0)
R2_DELAYS = (0, 5, 25, 50, 100)
R2_B_RATES = (0.0, 5.0, 20.0, 50.0)
R2_MAP_SEEDS = tuple(range(5))


def r2_circuit() -> Circuit:
    return Circuit(
        2,
        1,
        (SynapseSpec("A->out", 0, 2, R2_W0, *R2_BOUNDS), SynapseSpec("B->out", 1, 2, R2_W0, *R2_BOUNDS)),
        ("A", "B", "out"),
    )


def _a_caused(spikes: np.ndarray, a: int, out: int) -> int:
    f_out = spikes[1:, out]
    return int(np.sum(f_out & spikes[:-1, a]))


def _b_only_caused(spikes: np.ndarray, a: int, b: int, out: int) -> int:
    return int(np.sum(spikes[1:, out] & spikes[:-1, b] & ~spikes[:-1, a]))


def _r2_seed(seed: int, b_hz: float = R2_B_HZ, delay_ms: int = 0, with_controls: bool = True) -> Dict[str, Any]:
    circuit = r2_circuit()
    a, b, out = 0, 1, 2
    train = poisson_raster(_rng(2, seed, int(b_hz), 0), TRAIN_MS, [R2_A_HZ, b_hz])
    main = hc.PolicySpec("per_spike", neuron=out, value=1.0, require_prior=a, delay_ms=delay_ms)
    rec = run_arm(circuit, train, main)
    mech = check_mechanism(circuit, rec, main)
    row: Dict[str, Any] = {
        "seed": seed,
        "b_rate_hz": b_hz,
        "delay_ms": delay_ms,
        "rewarded_a_caused_events": mech["reward_events"],
        "train_a_caused_output_spikes": _a_caused(rec.spikes, a, out),
        "train_b_only_output_spikes": _b_only_caused(rec.spikes, a, b, out),
        "delta_w_A_mv": rec.weights_final["A->out"] - R2_W0,
        "delta_w_B_mv": rec.weights_final["B->out"] - R2_W0,
        "mechanism": mech,
    }
    row["leakage_ratio"] = row["delta_w_B_mv"] / row["delta_w_A_mv"] if row["delta_w_A_mv"] else float("nan")
    if not with_controls:
        return row
    test = poisson_raster(_rng(2, seed, int(b_hz), 1), TEST_MS, [R2_A_HZ, b_hz])
    test_b_only = poisson_raster(_rng(2, seed, int(b_hz), 2), TEST_MS, [0.0, max(b_hz, 20.0)])
    frozen_w = {"A->out": R2_W0, "B->out": R2_W0}

    def outcome(weights):
        learned = evaluate_frozen(circuit, weights, test, RULE)
        frozen = evaluate_frozen(circuit, frozen_w, test, RULE)
        lb = evaluate_frozen(circuit, weights, test_b_only, RULE).count(out)
        fb = evaluate_frozen(circuit, frozen_w, test_b_only, RULE).count(out)
        return {
            "heldout_a_caused_learned": _a_caused(learned.spikes, a, out),
            "heldout_a_caused_frozen": _a_caused(frozen.spikes, a, out),
            "heldout_b_only_caused_learned": _b_only_caused(learned.spikes, a, b, out),
            "heldout_b_only_caused_frozen": _b_only_caused(frozen.spikes, a, b, out),
            "heldout_total_learned": learned.count(out),
            "heldout_total_frozen": frozen.count(out),
            "b_only_panel_learned": lb,
            "b_only_panel_frozen": fb,
        }

    row["outcome"] = outcome(rec.weights_final)
    controls = {}
    for name, spec in (
        ("no_reward", hc.PolicySpec("none")),
        ("time_shuffled_reward", hc.PolicySpec("exogenous", schedule=random_schedule(_rng(2, seed, 7), mech["reward_events"], TRAIN_MS, False))),
        ("sign_randomized_reward", hc.PolicySpec("exogenous", schedule=random_schedule(_rng(2, seed, 8), mech["reward_events"], TRAIN_MS, True))),
    ):
        crec = run_arm(circuit, train, spec)
        cm = check_mechanism(circuit, crec, spec)
        controls[name] = {
            "delta_w_A_mv": crec.weights_final["A->out"] - R2_W0,
            "delta_w_B_mv": crec.weights_final["B->out"] - R2_W0,
            "mechanism_pass": cm["pass"],
            "max_abs_weight_error_mv": cm["max_abs_weight_error_mv"],
            "outcome": outcome(crec.weights_final),
        }
    row["controls"] = controls
    return row


def run_rung2(seeds: Sequence[int] = SEEDS, map_seeds: Sequence[int] = R2_MAP_SEEDS) -> Dict[str, Any]:
    primary = [_r2_seed(s) for s in seeds]
    grid = []
    for d in R2_DELAYS:
        for rate in R2_B_RATES:
            rows = [_r2_seed(s, b_hz=rate, delay_ms=d, with_controls=False) for s in map_seeds]
            grid.append(
                {
                    "delay_ms": d,
                    "b_rate_hz": rate,
                    "median_delta_w_A_mv": float(np.median([r["delta_w_A_mv"] for r in rows])),
                    "median_delta_w_B_mv": float(np.median([r["delta_w_B_mv"] for r in rows])),
                    "median_leakage_ratio": float(np.median([r["leakage_ratio"] for r in rows])),
                    "mechanism_pass_all": all(r["mechanism"]["pass"] for r in rows),
                    "max_abs_weight_error_mv": max(r["mechanism"]["max_abs_weight_error_mv"] for r in rows),
                    "b_zero_implies_delta_w_B_zero": (rate != 0.0) or all(r["delta_w_B_mv"] == 0.0 for r in rows),
                }
            )
    mech = all(r["mechanism"]["pass"] for r in primary) and all(
        c["mechanism_pass"] for r in primary for c in r["controls"].values()
    ) and all(g["mechanism_pass_all"] and g["b_zero_implies_delta_w_B_zero"] for g in grid)
    no_reward_exact = all(
        r["controls"]["no_reward"]["delta_w_A_mv"] == 0.0 and r["controls"]["no_reward"]["delta_w_B_mv"] == 0.0 for r in primary
    )
    a_grows = all(r["delta_w_A_mv"] > 0.0 for r in primary)
    a_beats_b = all(r["delta_w_A_mv"] > r["delta_w_B_mv"] for r in primary)
    a_behaviour = all(r["outcome"]["heldout_a_caused_learned"] > r["outcome"]["heldout_a_caused_frozen"] for r in primary)
    beats_shuffled = all(
        (r["outcome"]["heldout_a_caused_learned"] - r["outcome"]["heldout_a_caused_frozen"])
        > (r["controls"]["time_shuffled_reward"]["outcome"]["heldout_a_caused_learned"] - r["outcome"]["heldout_a_caused_frozen"])
        for r in primary
    )
    beats_signed = all(
        (r["outcome"]["heldout_a_caused_learned"] - r["outcome"]["heldout_a_caused_frozen"])
        > (r["controls"]["sign_randomized_reward"]["outcome"]["heldout_a_caused_learned"] - r["outcome"]["heldout_a_caused_frozen"])
        for r in primary
    )
    mechanism_pass = mech and no_reward_exact and a_grows and a_beats_b
    outcome_pass = a_behaviour and beats_shuffled and beats_signed
    breaks = [
        {"delay_ms": g["delay_ms"], "b_rate_hz": g["b_rate_hz"], "median_leakage_ratio": g["median_leakage_ratio"]}
        for g in grid
        if g["b_rate_hz"] > 0 and not (g["median_leakage_ratio"] < 0.5)
    ]
    return {
        "rung": 2,
        "status": "PASS" if mechanism_pass and outcome_pass else "FAIL",
        "mechanism_pass": mechanism_pass,
        "outcome_pass": outcome_pass,
        "gates": {
            "all_arms_hand_checked_and_replayed": mech,
            "no_reward_weights_unchanged_exactly": no_reward_exact,
            "delta_w_A_positive_all_seeds": a_grows,
            "delta_w_A_exceeds_delta_w_B_all_seeds": a_beats_b,
            "heldout_A_caused_firing_above_frozen_all_seeds": a_behaviour,
            "A_caused_gain_beats_time_shuffled_all_seeds": beats_shuffled,
            "A_caused_gain_beats_sign_randomized_all_seeds": beats_signed,
        },
        "primary_condition": {"a_hz": R2_A_HZ, "b_hz": R2_B_HZ, "delay_ms": 0, "w0_mv": R2_W0},
        "primary": primary,
        "leakage_map_definition": "credit breakdown = median(delta_w_B / delta_w_A) >= 0.5 over 5 seeds (B earns at least half of A's credit)",
        "leakage_breakdown_conditions": breaks,
        "leakage_map": grid,
    }


# ---------------------------------------------------------------------------
# Rung 3: series, parallel, and disconnected motifs
# ---------------------------------------------------------------------------

R3_BOUNDS = (0.0, 20.0)


def r3_series() -> Circuit:
    # A (input) -> B (IF) -> C (IF); reward on C spikes.
    return Circuit(
        1,
        2,
        (SynapseSpec("A->B", 0, 1, 9.0, *R3_BOUNDS), SynapseSpec("B->C", 1, 2, 14.0, *R3_BOUNDS)),
        ("A", "B", "C"),
    )


def r3_parallel() -> Circuit:
    # A (input) -> C, B (input) -> C; reward on C spikes.
    return Circuit(
        2,
        1,
        (SynapseSpec("A->C", 0, 2, 9.0, *R3_BOUNDS), SynapseSpec("B->C", 1, 2, 9.0, *R3_BOUNDS)),
        ("A", "B", "C"),
    )


def r3_disconnected() -> Circuit:
    # A (input) -> C is the rewarded motif.  D (input) -> E (IF, active) and
    # D -> F (IF, subthreshold) have no path to C.  Reward is on C spikes.
    return Circuit(
        2,
        3,
        (
            SynapseSpec("A->C", 0, 2, 9.0, *R3_BOUNDS),
            SynapseSpec("D->E_active", 1, 3, 17.0, *R3_BOUNDS),
            SynapseSpec("D->F_silent", 1, 4, SILENT_W_MV, *R3_BOUNDS),
        ),
        ("A", "D", "C", "E", "F"),
    )


def _motif_arms(circuit: Circuit, train: np.ndarray, reward_neuron: int, seed: int, key: int):
    main = hc.PolicySpec("per_spike", neuron=reward_neuron, value=1.0)
    rec = run_arm(circuit, train, main)
    mech = check_mechanism(circuit, rec, main)
    out = {"reward_on_output": (rec, mech)}
    for name, spec in (
        ("no_reward", hc.PolicySpec("none")),
        ("time_shuffled_reward", hc.PolicySpec("exogenous", schedule=random_schedule(_rng(3, seed, key, 1), mech["reward_events"], TRAIN_MS, False))),
        ("sign_randomized_reward", hc.PolicySpec("exogenous", schedule=random_schedule(_rng(3, seed, key, 2), mech["reward_events"], TRAIN_MS, True))),
    ):
        crec = run_arm(circuit, train, spec)
        out[name] = (crec, check_mechanism(circuit, crec, spec))
    return out


def _summarize_arm(circuit, rec, mech, panels: Dict[str, np.ndarray], out_neuron: int):
    frozen = {s.name: s.w0_mv for s in circuit.synapses}
    res = {
        "delta_w_mv": {k: rec.weights_final[k] - rec.weights_initial[k] for k in rec.weights_final},
        "mechanism_pass": mech["pass"],
        "max_abs_weight_error_mv": mech["max_abs_weight_error_mv"],
        "spike_replay_mismatches": mech["spike_replay_mismatches"],
        "reward_events": mech["reward_events"],
        "panels": {},
    }
    for pname, raster in panels.items():
        learned = evaluate_frozen(circuit, rec.weights_final, raster, RULE)
        fz = evaluate_frozen(circuit, frozen, raster, RULE)
        res["panels"][pname] = {
            "output_learned": learned.count(out_neuron),
            "output_frozen": fz.count(out_neuron),
            "per_neuron_learned": {n: learned.count(i) for i, n in enumerate(circuit.names)},
            "per_neuron_frozen": {n: fz.count(i) for i, n in enumerate(circuit.names)},
        }
    return res


def _r3_seed(seed: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {"seed": seed}

    series = r3_series()
    train = poisson_raster(_rng(3, seed, 1, 0), TRAIN_MS, [40.0])
    panels = {"A_40Hz": poisson_raster(_rng(3, seed, 1, 9), TEST_MS, [40.0])}
    row["series"] = {k: _summarize_arm(series, r, m, panels, 2) for k, (r, m) in _motif_arms(series, train, 2, seed, 1).items()}

    par = r3_parallel()
    train = poisson_raster(_rng(3, seed, 2, 0), TRAIN_MS, [40.0, 40.0])
    panels = {
        "A_and_B_40Hz": poisson_raster(_rng(3, seed, 2, 9), TEST_MS, [40.0, 40.0]),
        "A_only_80Hz": poisson_raster(_rng(3, seed, 2, 10), TEST_MS, [80.0, 0.0]),
        "B_only_80Hz": poisson_raster(_rng(3, seed, 2, 11), TEST_MS, [0.0, 80.0]),
    }
    row["parallel_both_active"] = {k: _summarize_arm(par, r, m, panels, 2) for k, (r, m) in _motif_arms(par, train, 2, seed, 2).items()}

    # Parallel with branch B silent during training: B is wired but never eligible.
    train_b_silent = poisson_raster(_rng(3, seed, 3, 0), TRAIN_MS, [40.0, 0.0])
    row["parallel_B_silent_in_training"] = {
        k: _summarize_arm(par, r, m, panels, 2) for k, (r, m) in _motif_arms(par, train_b_silent, 2, seed, 3).items()
    }

    disc = r3_disconnected()
    train = poisson_raster(_rng(3, seed, 4, 0), TRAIN_MS, [40.0, 40.0])
    panels = {"A_and_D_40Hz": poisson_raster(_rng(3, seed, 4, 9), TEST_MS, [40.0, 40.0])}
    row["disconnected"] = {k: _summarize_arm(disc, r, m, panels, 2) for k, (r, m) in _motif_arms(disc, train, 2, seed, 4).items()}
    return row


def _gain(arm, panel):
    p = arm["panels"][panel]
    return p["output_learned"] - p["output_frozen"]


def run_rung3(seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    per_seed = [_r3_seed(s) for s in seeds]
    motifs = ("series", "parallel_both_active", "parallel_B_silent_in_training", "disconnected")
    mech = all(arm["mechanism_pass"] for r in per_seed for m in motifs for arm in r[m].values())
    no_reward_exact = all(all(v == 0.0 for v in r[m]["no_reward"]["delta_w_mv"].values()) for r in per_seed for m in motifs)

    def all_seeds(fn):
        return all(fn(r) for r in per_seed)

    series_links_grow = all_seeds(lambda r: all(v > 0 for v in r["series"]["reward_on_output"]["delta_w_mv"].values()))
    series_output_up = all_seeds(lambda r: _gain(r["series"]["reward_on_output"], "A_40Hz") > 0)
    series_beats_random = all_seeds(
        lambda r: _gain(r["series"]["reward_on_output"], "A_40Hz")
        > max(_gain(r["series"]["time_shuffled_reward"], "A_40Hz"), _gain(r["series"]["sign_randomized_reward"], "A_40Hz"))
    )
    par_both_grow = all_seeds(lambda r: all(v > 0 for v in r["parallel_both_active"]["reward_on_output"]["delta_w_mv"].values()))
    par_output_up = all_seeds(lambda r: _gain(r["parallel_both_active"]["reward_on_output"], "A_and_B_40Hz") > 0)
    par_beats_random = all_seeds(
        lambda r: _gain(r["parallel_both_active"]["reward_on_output"], "A_and_B_40Hz")
        > max(
            _gain(r["parallel_both_active"]["time_shuffled_reward"], "A_and_B_40Hz"),
            _gain(r["parallel_both_active"]["sign_randomized_reward"], "A_and_B_40Hz"),
        )
    )
    silent_branch_exact = all_seeds(
        lambda r: all(arm["delta_w_mv"]["B->C"] == 0.0 for arm in r["parallel_B_silent_in_training"].values())
        and _gain(r["parallel_B_silent_in_training"]["reward_on_output"], "B_only_80Hz") == 0
    )
    silent_branch_active_side_up = all_seeds(
        lambda r: r["parallel_B_silent_in_training"]["reward_on_output"]["delta_w_mv"]["A->C"] > 0
        and _gain(r["parallel_B_silent_in_training"]["reward_on_output"], "A_only_80Hz") > 0
    )
    disc_silent_exact = all_seeds(
        lambda r: all(arm["delta_w_mv"]["D->F_silent"] == 0.0 for arm in r["disconnected"].values())
    )
    # Output C has no path from D/E/F, so its held-out firing must match the
    # A-only learning effect regardless of what the disconnected synapses did.
    def _c_with_only_a_learned(r):
        weights = {"A->C": 9.0 + r["disconnected"]["reward_on_output"]["delta_w_mv"]["A->C"], "D->E_active": 17.0, "D->F_silent": SILENT_W_MV}
        panel = poisson_raster(_rng(3, r["seed"], 4, 9), TEST_MS, [40.0, 40.0])
        return evaluate_frozen(r3_disconnected(), weights, panel, RULE).count(2)

    disc_output_isolated = all_seeds(
        lambda r: r["disconnected"]["reward_on_output"]["panels"]["A_and_D_40Hz"]["per_neuron_learned"]["C"]
        == _c_with_only_a_learned(r)
    )
    mechanism_pass = mech and no_reward_exact and series_links_grow and par_both_grow and silent_branch_exact and disc_silent_exact
    outcome_pass = (
        series_output_up and series_beats_random and par_output_up and par_beats_random
        and silent_branch_active_side_up and disc_output_isolated
    )
    active_disc = {
        "reward_on_output_delta_w_D_E_mv": [r["disconnected"]["reward_on_output"]["delta_w_mv"]["D->E_active"] for r in per_seed],
        "time_shuffled_delta_w_D_E_mv": [r["disconnected"]["time_shuffled_reward"]["delta_w_mv"]["D->E_active"] for r in per_seed],
        "sign_randomized_delta_w_D_E_mv": [r["disconnected"]["sign_randomized_reward"]["delta_w_mv"]["D->E_active"] for r in per_seed],
        "interpretation": "An active synapse with no path to the rewarded output still has eligibility; a global reward changes it about as much as time-shuffled reward does. Eligibility, not wiring, gates credit.",
    }
    return {
        "rung": 3,
        "status": "PASS" if mechanism_pass and outcome_pass else "FAIL",
        "mechanism_pass": mechanism_pass,
        "outcome_pass": outcome_pass,
        "gates": {
            "all_arms_hand_checked_and_replayed": mech,
            "no_reward_weights_unchanged_exactly": no_reward_exact,
            "series_both_links_grow_all_seeds": series_links_grow,
            "series_output_C_above_frozen_all_seeds": series_output_up,
            "series_gain_beats_both_random_controls_all_seeds": series_beats_random,
            "parallel_both_branches_grow_all_seeds": par_both_grow,
            "parallel_output_C_above_frozen_all_seeds": par_output_up,
            "parallel_gain_beats_both_random_controls_all_seeds": par_beats_random,
            "parallel_silent_branch_unchanged_exactly_and_B_only_output_identical": silent_branch_exact,
            "parallel_silent_branch_active_side_grows_and_A_only_output_up": silent_branch_active_side_up,
            "disconnected_subthreshold_synapse_unchanged_exactly": disc_silent_exact,
            "disconnected_changes_do_not_reach_output_C": disc_output_isolated,
        },
        "disconnected_but_active_synapse": active_disc,
        "per_seed": per_seed,
    }
