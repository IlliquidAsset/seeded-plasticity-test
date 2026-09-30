#!/usr/bin/env python3
"""
Track 3 four-arm wave-gated plasticity experiment.

Runs the frozen Track 3 design on the post-A-F bench:
  - binary_classification (easy, weight_scale=2.0)
  - temporal_xor (paper protocol, 500 ms / 50 spikes)
  - temporal_sequence (dead-network diagnosis config)

Mandatory arms: wave-coherent, phase-scrambled,
eligibility-timescale-only, and frozen.  Three matched seeds per arm.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snn.core import PureSNN
from snn.instrumentation import InstrumentedRSTDP
from snn.tasks import (
    BinaryClassificationTask,
    TemporalSequenceTask,
    TemporalXORPaperTask,
)
from snn.wave_plasticity import WaveGatedPlasticity


REPO_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_DIR / "results_track3"
RESULTS_DIR.mkdir(exist_ok=True)

ARMS = [
    "wave-coherent",
    "phase-scrambled",
    "eligibility-timescale-only",
    "frozen",
]
SEEDS = [42, 123, 256]
N_TRIALS = 300
BATCH_SIZE = 16
EVAL_EVERY = 25
LR = 0.005
WEIGHT_SCALE = 2.0
TAU_SYN = 10.0
TAU_M = 20.0
TAU_ELIG_FAST = 20.0
TAU_ELIG_ONLY = 1000.0
TAU_ELIG_SLOW = 500.0
A_PLUS = 0.01
A_MINUS = 0.01
TAU_PLUS = 20.0
TAU_MINUS = 20.0
FREQUENCY = 8.0

TASKS = {
    "binary_classification": {
        "cls": BinaryClassificationTask,
        "kwargs": {"n_input": 16, "timesteps": 100},
        "layers": [16, 2],
        "description": "rate-based easy config, weight_scale=2.0",
    },
    "temporal_xor": {
        "cls": TemporalXORPaperTask,
        "kwargs": {"timesteps": 500, "dt": 1.0},
        "layers": [2, 20, 2],
        "description": "paper protocol: 500 ms, 50 uniform spikes per active channel",
    },
    "temporal_sequence": {
        "cls": TemporalSequenceTask,
        "kwargs": {
            "n_channels": 8,
            "timesteps": 50,
            "signal_window_fraction": 0.16,
        },
        "layers": [8, 12, 8],
        "description": "dead-network diagnosis config, weight_scale=2.0",
    },
}


def _commit_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
    ).strip()


def _attach_plasticity(net, arm, seed):
    """Attach exactly the rule declared for one Track 3 arm."""
    if arm == "frozen":
        net.plasticities = []
        return

    net.plasticities = []
    for syn in net.synapses:
        common = dict(
            lr=LR,
            dt=net.dt,
            a_plus=A_PLUS,
            a_minus=A_MINUS,
            tau_plus=TAU_PLUS,
            tau_minus=TAU_MINUS,
            record=True,
        )
        if arm == "eligibility-timescale-only":
            p = InstrumentedRSTDP(syn, tau_elig=TAU_ELIG_ONLY, **common)
        elif arm == "wave-coherent":
            p = WaveGatedPlasticity(
                syn,
                tau_elig=TAU_ELIG_FAST,
                tau_elig_slow=TAU_ELIG_SLOW,
                frequency=FREQUENCY,
                phi_0=0.0,
                scramble_phase=False,
                **common,
            )
        elif arm == "phase-scrambled":
            p = WaveGatedPlasticity(
                syn,
                tau_elig=TAU_ELIG_FAST,
                tau_elig_slow=TAU_ELIG_SLOW,
                frequency=FREQUENCY,
                phi_0=0.0,
                scramble_phase=True,
                phase_seed=seed + 10_000,
                **common,
            )
        else:
            raise ValueError(f"unknown arm: {arm}")
        net.plasticities.append(p)


def _aggregate_layer_metrics(plasticities, method):
    reports = [getattr(p, method)() for p in plasticities if hasattr(p, method)]
    if not reports:
        return {}
    keys = set().union(*(r.keys() for r in reports))
    out = {}
    for key in keys:
        vals = [r[key] for r in reports if isinstance(r.get(key), (int, float))]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def run_one(task_name, arm, seed):
    """Run one task/arm/seed replicate and return serializable metrics."""
    cfg = TASKS[task_name]
    torch.manual_seed(seed)
    task = cfg["cls"](**cfg["kwargs"])
    task.set_seed(seed + 1)

    net = PureSNN(
        cfg["layers"],
        dt=1.0,
        weight_scale=WEIGHT_SCALE,
        tau_m=TAU_M,
        tau_syn=TAU_SYN,
    )
    frozen = net.get_frozen_weights()
    _attach_plasticity(net, arm, seed)

    started = time.time()
    accuracy_trace = []
    weight_norm_trace = []
    max_abs_weight = max(syn.weight.data.abs().max().item() for syn in net.synapses)
    nan_detected = False
    weight_limit_exceeded = False

    for trial in range(N_TRIALS):
        inputs, targets = task.generate_batch(BATCH_SIZE)
        output_spikes = net(inputs)
        reward = task.compute_reward(output_spikes, targets)

        if arm == "frozen":
            net.restore_frozen_weights(frozen)
        else:
            for p in net.plasticities:
                p.apply_reward(reward)

        trial_max = max(syn.weight.data.abs().max().item() for syn in net.synapses)
        max_abs_weight = max(max_abs_weight, trial_max)
        has_nan = any(torch.isnan(syn.weight.data).any().item() for syn in net.synapses)
        if has_nan:
            nan_detected = True
            break
        if trial_max > 5.0:
            weight_limit_exceeded = True
            break

        if trial % EVAL_EVERY == 0 or trial == N_TRIALS - 1:
            with torch.no_grad():
                decisions = task.decode_output(output_spikes)
                accuracy = (decisions == targets).float().mean().item()
            accuracy_trace.append([trial, accuracy])
            weight_norm_trace.append([trial, net.weight_norm()])

    liveness = _aggregate_layer_metrics(net.plasticities, "get_liveness_metrics")
    wave_metrics = _aggregate_layer_metrics(net.plasticities, "get_wave_metrics")

    return {
        "task": task_name,
        "arm": arm,
        "seed": seed,
        "final_accuracy": accuracy_trace[-1][1] if accuracy_trace else None,
        "accuracy_trace": accuracy_trace,
        "weight_norm_trace": weight_norm_trace,
        "liveness": liveness,
        "wave_metrics": wave_metrics,
        "max_abs_weight": max_abs_weight,
        "nan_detected": nan_detected,
        "weight_limit_exceeded": weight_limit_exceeded,
        "wall_time_seconds": time.time() - started,
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _standard_error(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def evaluate_criteria(results):
    """Evaluate only the frozen PASS / NULL / STOP criteria."""
    by_task = {}
    for task_name in TASKS:
        by_task[task_name] = {}
        for arm in ARMS:
            rows = [r for r in results if r["task"] == task_name and r["arm"] == arm]
            rows.sort(key=lambda r: SEEDS.index(r["seed"]))
            by_task[task_name][arm] = rows

    any_nan = any(r["nan_detected"] for r in results)
    any_weight_limit = any(r["weight_limit_exceeded"] for r in results)

    task_evaluations = {}
    pass_tasks = []
    null_tasks = []
    guardrail_failures = []
    degenerate_coherent_tasks = []

    for task_name, arms in by_task.items():
        A = [r["final_accuracy"] for r in arms["wave-coherent"]]
        B = [r["final_accuracy"] for r in arms["phase-scrambled"]]
        C = [r["final_accuracy"] for r in arms["eligibility-timescale-only"]]
        D = [r["final_accuracy"] for r in arms["frozen"]]

        a_gt_c_hits = sum(a > c for a, c in zip(A, C))
        c_gt_d_margin_hits = sum((c - d) >= 0.05 for c, d in zip(C, D))
        a_gt_b_hits = sum(a > b for a, b in zip(A, B))
        se = max(_standard_error(A), _standard_error(C))
        approx_threshold = max(0.05, se)
        a_approx_c_hits = sum(abs(a - c) <= approx_threshold for a, c in zip(A, C))

        a_corrs = [
            r.get("liveness", {}).get("reward_weight_corr", 0.0)
            for r in arms["wave-coherent"]
        ]
        mean_a_corr = _mean(a_corrs)
        zero_fracs = [
            r.get("wave_metrics", {}).get("fraction_updates_gate_zero", 0.0)
            for r in arms["wave-coherent"]
        ]
        mean_zero_fraction = _mean(zero_fracs)

        guardrail = a_gt_b_hits >= 2
        pass_candidate = (
            a_gt_c_hits >= 2
            and c_gt_d_margin_hits >= 2
            and mean_a_corr > 0.1
        )
        null_candidate = a_approx_c_hits >= 2 and c_gt_d_margin_hits >= 2

        # "Close to 1" is operationalized conservatively as >= 0.95 solely
        # for the mandatory degenerate-gate STOP check.
        degenerate = mean_zero_fraction >= 0.95

        if pass_candidate and guardrail and not degenerate:
            pass_tasks.append(task_name)
        if null_candidate and guardrail and not degenerate:
            null_tasks.append(task_name)
        if (pass_candidate or null_candidate) and not guardrail:
            guardrail_failures.append(task_name)
        if degenerate:
            degenerate_coherent_tasks.append(task_name)

        task_evaluations[task_name] = {
            "wave_coherent": A,
            "phase_scrambled": B,
            "eligibility_timescale_only": C,
            "frozen": D,
            "means": {
                "A": _mean(A),
                "B": _mean(B),
                "C": _mean(C),
                "D": _mean(D),
            },
            "A_gt_C_hits": a_gt_c_hits,
            "C_minus_D_ge_0_05_hits": c_gt_d_margin_hits,
            "A_approx_C_hits": a_approx_c_hits,
            "A_approx_C_threshold": approx_threshold,
            "A_gt_B_guardrail_hits": a_gt_b_hits,
            "A_reward_weight_corr_per_seed": a_corrs,
            "A_reward_weight_corr_mean": mean_a_corr,
            "A_degenerate_gate_fraction_per_seed": zero_fracs,
            "A_degenerate_gate_fraction_mean": mean_zero_fraction,
            "pass_candidate": pass_candidate,
            "null_candidate": null_candidate,
            "guardrail_pass": guardrail,
            "degenerate_gate_stop": degenerate,
        }

    # Explicit STOP conditions have precedence.  A degenerate gate on any task
    # invalidates that task, but does not erase a valid PASS/NULL on another task.
    if any_nan or any_weight_limit:
        outcome = "STOP"
        reason = "NaN detected" if any_nan else "synaptic weight exceeded |w| > 5.0"
    elif pass_tasks:
        outcome = "PASS"
        reason = f"PASS criteria satisfied on: {', '.join(pass_tasks)}"
    elif null_tasks:
        outcome = "NULL"
        reason = f"NULL criteria satisfied on: {', '.join(null_tasks)}"
    else:
        outcome = "STOP"
        if guardrail_failures:
            reason = "phase-scrambled guardrail failed"
        elif degenerate_coherent_tasks:
            reason = "wave-coherent gate degenerated to zero on tested task(s)"
        else:
            reason = "neither wave-coherent nor eligibility-timescale-only rescued any task"

    return {
        "outcome": outcome,
        "reason": reason,
        "pass_tasks": pass_tasks,
        "null_tasks": null_tasks,
        "guardrail_failures": guardrail_failures,
        "degenerate_coherent_tasks": degenerate_coherent_tasks,
        "any_nan": any_nan,
        "any_weight_limit_exceeded": any_weight_limit,
        "tasks": task_evaluations,
    }


def run_track3(selected_tasks=None, selected_arms=None, selected_seeds=None):
    tasks = selected_tasks or list(TASKS)
    arms = selected_arms or ARMS
    seeds = selected_seeds or SEEDS
    started = time.time()
    command = shlex.join([sys.executable, *sys.argv])
    commit_sha = _commit_sha()

    results = []
    for task_name in tasks:
        for arm in arms:
            for seed in seeds:
                print(f"\n--- {task_name} | {arm} | seed {seed} ---", flush=True)
                result = run_one(task_name, arm, seed)
                results.append(result)
                print(
                    f"accuracy={result['final_accuracy']:.4f} "
                    f"max|w|={result['max_abs_weight']:.4f} "
                    f"wall={result['wall_time_seconds']:.1f}s",
                    flush=True,
                )
                if result["nan_detected"] or result["weight_limit_exceeded"]:
                    print("STOP condition reached; ending run.", flush=True)
                    break
            if results and (results[-1]["nan_detected"] or results[-1]["weight_limit_exceeded"]):
                break
        if results and (results[-1]["nan_detected"] or results[-1]["weight_limit_exceeded"]):
            break

    is_full = set(tasks) == set(TASKS) and set(arms) == set(ARMS) and set(seeds) == set(SEEDS)
    criteria = evaluate_criteria(results) if is_full and len(results) == 36 else None

    summary = {
        "experiment": "Track 3 wave-gated plasticity",
        "parameters": {
            "tasks": {
                k: {
                    "class": TASKS[k]["cls"].__name__,
                    "kwargs": TASKS[k]["kwargs"],
                    "layers": TASKS[k]["layers"],
                    "description": TASKS[k]["description"],
                }
                for k in tasks
            },
            "arms": arms,
            "seeds": seeds,
            "n_trials": N_TRIALS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_scale": WEIGHT_SCALE,
            "tau_syn_ms": TAU_SYN,
            "tau_m_ms": TAU_M,
            "tau_elig_by_arm_ms": {
                "wave-coherent_fast": TAU_ELIG_FAST,
                "phase-scrambled_fast": TAU_ELIG_FAST,
                "eligibility-timescale-only": TAU_ELIG_ONLY,
                "frozen": None,
            },
            "tau_elig_slow_ms": TAU_ELIG_SLOW,
            "a_plus": A_PLUS,
            "a_minus": A_MINUS,
            "tau_plus_ms": TAU_PLUS,
            "tau_minus_ms": TAU_MINUS,
            "wave_frequency_hz": FREQUENCY,
        },
        "criterion_evaluation": criteria,
        "results": results,
        "commit_sha_at_run": commit_sha,
        "exact_command": command,
        "wall_time_seconds": time.time() - started,
    }

    out_name = "summary.json" if is_full else "partial_summary.json"
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    if criteria:
        print(f"TRACK 3 OUTCOME: {criteria['outcome']} — {criteria['reason']}")
    else:
        print("PARTIAL RUN COMPLETE (no criterion evaluation)")
    print(f"Results: {out_path}")
    print(f"Wall time: {summary['wall_time_seconds']:.1f}s")
    print("=" * 72)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), action="append")
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    args = parser.parse_args()
    run_track3(args.task, args.arm, args.seed)


if __name__ == "__main__":
    main()
