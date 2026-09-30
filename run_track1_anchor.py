#!/usr/bin/env python3
"""
Track 1: source-matched Florian anchor run.

Uses the paper-faithful TemporalXORPaperTask (Revision A), the corrected
per-sample apply_reward (Revision B), and Florian 2007's actual eligibility
time constant (tau_elig = 25 ms).

Runs 3 seeds with plasticity ON vs frozen control, 200 epochs each, and
persists a JSON summary under results_track1_anchor/.
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snn.core import PureSNN
from snn.instrumentation import InstrumentedRSTDP
from snn.tasks import TemporalXORPaperTask


REPO_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_DIR / "results_track1_anchor"
RESULTS_DIR.mkdir(exist_ok=True)


# Florian 2007, section 4.1 / 4.3 (temporally coded XOR):
# tau_+ = tau_- = 20 ms, tau_z (eligibility trace) = 25 ms.
FLORIAN_TAU_ELIG = 25.0

# Layers: 2 input, 20 hidden, 2 output.  Florian used 1 output neuron; the
# repo's paper task has 2 outputs, so we keep the 2-output argmax decoder.
FLORIAN_LAYERS = [2, 20, 2]

SEEDS = [42, 123, 256]
N_TRIALS = 200
BATCH_SIZE = 16
LR = 0.005
WEIGHT_SCALE = 2.0
TAU_SYN = 10.0
TAU_M = 20.0
A_PLUS = 0.02
A_MINUS = 0.015
TAU_PLUS = 20.0
TAU_MINUS = 20.0


def _get_commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
    except Exception:
        return None


def _compute_pattern_rates(net, task, pattern, n_samples=8):
    """
    Compute the average firing rate of each output neuron for a fixed input
    pattern.  `pattern` is one of (0,0), (0,1), (1,0), (1,1).
    Returns dict with per-neuron rates and total output rate.
    """
    a_val, b_val = pattern
    with torch.no_grad():
        # Build a batch where both bits are fixed.
        input_spikes = torch.zeros(n_samples, 2, task.timesteps)
        if a_val == 1:
            for _ in range(n_samples):
                idx = torch.randperm(task.timesteps, generator=task.rng)[:50]
                input_spikes[_, 0, idx] = 1.0
        if b_val == 1:
            for _ in range(n_samples):
                idx = torch.randperm(task.timesteps, generator=task.rng)[:50]
                input_spikes[_, 1, idx] = 1.0

        output_spikes = net(input_spikes)
        # output_spikes: (n_samples, n_output, timesteps)
        per_neuron = output_spikes.sum(dim=-1).mean(dim=0) / (task.timesteps * 1e-3)
        total = output_spikes.sum().item() / (n_samples * task.timesteps * 1e-3)
    return {
        "per_neuron": per_neuron.tolist(),
        "total_hz": total,
    }


def _run_seed(seed, plasticity_on=True):
    torch.manual_seed(seed)
    task = TemporalXORPaperTask(timesteps=500, dt=1.0)
    task.set_seed(seed + 1)

    net = PureSNN(
        FLORIAN_LAYERS,
        dt=1.0,
        weight_scale=WEIGHT_SCALE,
        tau_m=TAU_M,
        tau_syn=TAU_SYN,
    )

    frozen = net.get_frozen_weights()

    if plasticity_on:
        net.plasticities = []
        for syn in net.synapses:
            p = InstrumentedRSTDP(
                syn,
                lr=LR,
                tau_elig=FLORIAN_TAU_ELIG,
                a_plus=A_PLUS,
                a_minus=A_MINUS,
                tau_plus=TAU_PLUS,
                tau_minus=TAU_MINUS,
                record=True,
            )
            net.plasticities.append(p)

    accuracies = []
    for trial in range(N_TRIALS):
        inputs, targets = task.generate_batch(BATCH_SIZE)
        output_spikes = net(inputs)
        reward = task.compute_reward(output_spikes, targets)

        if plasticity_on:
            for p in net.plasticities:
                p.apply_reward(reward)
        else:
            net.restore_frozen_weights(frozen)

        if trial % 25 == 0 or trial == N_TRIALS - 1:
            with torch.no_grad():
                decisions = task.decode_output(output_spikes)
                accuracy = (decisions == targets).float().mean().item()
                accuracies.append((trial, accuracy))

    # Final pattern-rate probe for the gate: rate({1,1}) vs rate({0,1}).
    # We report the firing rate of output neuron 1 (class-1 neuron).
    rates_11 = _compute_pattern_rates(net, task, (1, 1))
    rates_01 = _compute_pattern_rates(net, task, (0, 1))

    # Liveness metrics from the first plastic layer.
    liveness = {}
    if plasticity_on and net.plasticities:
        liveness = net.plasticities[0].get_liveness_metrics()

    return {
        "seed": seed,
        "plasticity_on": plasticity_on,
        "final_accuracy": accuracies[-1][1] if accuracies else 0.0,
        "accuracy_trace": accuracies,
        "rate_11": rates_11,
        "rate_01": rates_01,
        "liveness": liveness,
    }


def run_track1_anchor():
    start_wall = time.time()
    commit_sha = _get_commit_sha()

    per_seed = []
    frozen_per_seed = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} | plasticity ON ---")
        on_result = _run_seed(seed, plasticity_on=True)
        print(f"  final_accuracy={on_result['final_accuracy']:.4f}")
        print(f"  rate(1,1) neuron1={on_result['rate_11']['per_neuron'][1]:.4f} Hz")
        print(f"  rate(0,1) neuron1={on_result['rate_01']['per_neuron'][1]:.4f} Hz")
        print(f"  elig@reward_pct_of_peak={on_result['liveness'].get('eligibility_at_reward_pct_of_peak', 0.0):.6f}")
        print(f"  reward_weight_corr={on_result['liveness'].get('reward_weight_corr', 0.0):.6f}")
        per_seed.append(on_result)

        print(f"\n--- Seed {seed} | frozen control ---")
        off_result = _run_seed(seed, plasticity_on=False)
        print(f"  final_accuracy={off_result['final_accuracy']:.4f}")
        frozen_per_seed.append(off_result)

    # Gate: >=2/3 seeds show rate({1,1}) < rate({0,1}).
    gate_hits = 0
    for r in per_seed:
        if r["rate_11"]["per_neuron"][1] < r["rate_01"]["per_neuron"][1]:
            gate_hits += 1

    outcome = "PASS" if gate_hits >= 2 else "NULL"

    summary = {
        "outcome": outcome,
        "gate_criterion": ">=2/3 seeds show rate({1,1}) < rate({0,1})",
        "gate_hits": gate_hits,
        "total_seeds": len(SEEDS),
        "parameters": {
            "task": "TemporalXORPaperTask",
            "layers": FLORIAN_LAYERS,
            "tau_elig_ms": FLORIAN_TAU_ELIG,
            "tau_plus_ms": TAU_PLUS,
            "tau_minus_ms": TAU_MINUS,
            "a_plus": A_PLUS,
            "a_minus": A_MINUS,
            "lr": LR,
            "n_trials": N_TRIALS,
            "batch_size": BATCH_SIZE,
            "weight_scale": WEIGHT_SCALE,
            "tau_syn_ms": TAU_SYN,
            "tau_m_ms": TAU_M,
        },
        "commit_sha": commit_sha,
        "command": "python3 run_track1_anchor.py",
        "wall_time_seconds": time.time() - start_wall,
        "per_seed": per_seed,
        "frozen_per_seed": frozen_per_seed,
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Track 1 outcome: {outcome} ({gate_hits}/{len(SEEDS)} seeds passed gate)")
    print(f"Summary written to: {summary_path}")
    print(f"Wall time: {summary['wall_time_seconds']:.1f}s")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_track1_anchor()
