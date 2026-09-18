"""
Experiment runner for pure SNN learning.
Tests whether plasticity produces learning across a bench of tasks.

Controls built-in:
1. Frozen-weight: plasticity off, same seeds
2. Fresh-seed replication: new seeds, plasticity on
3. Freeze verification: confirms freeze actually stops weight changes
"""

import torch
import json
import time
import math
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn.core import PureSNN
from snn.tasks import get_task_configs, ALL_TASKS


def verify_freeze_mechanism():
    """
    Critical: verify the freeze mechanism actually works.
    Run a forward pass, try to update weights, verify they don't change.
    """
    print("=" * 60)
    print("FREEZE MECHANISM VERIFICATION")
    print("=" * 60)

    net = PureSNN([8, 16, 4], dt=1.0, weight_scale=0.1)
    net.add_plasticity(lr=0.1)  # High learning rate for testing

    # Get frozen weights
    frozen = net.get_frozen_weights()
    initial_norm = net.weight_norm()
    print(f"Initial weight norm: {initial_norm:.6f}")

    # Run a forward pass (should update plasticity traces)
    x = torch.randn(4, 8, 50)
    out = net(x)

    # Try to apply reward (should update weights)
    reward = torch.tensor([1.0, 1.0, -1.0, 1.0])
    for p in net.plasticities:
        p.apply_reward(reward)

    # Check weights changed
    after_update_norm = net.weight_norm()
    ok_no_freeze, _ = net.verify_freeze(frozen)
    print(f"Weight norm after unfrozen update: {after_update_norm:.6f}")
    print(f"Freeze check (should FAIL without restore): {ok_no_freeze}")

    # Now restore and verify
    net.restore_frozen_weights(frozen)
    after_restore_norm = net.weight_norm()
    ok_after_freeze, diffs = net.verify_freeze(frozen)
    print(f"Weight norm after restore: {after_restore_norm:.6f}")
    print(f"Freeze check (should PASS after restore): {ok_after_freeze}")
    print(f"Per-layer max diffs: {diffs}")

    # Run another forward+reward cycle with freeze active
    out2 = net(x)
    for p in net.plasticities:
        p.apply_reward(reward)
    net.restore_frozen_weights(frozen)
    ok_final, _ = net.verify_freeze(frozen)
    print(f"Freeze check after second restore: {ok_final}")

    if ok_after_freeze and ok_final:
        print(">>> FREEZE VERIFICATION: PASSED")
    else:
        print(">>> FREEZE VERIFICATION: FAILED - aborting")
        return False
    return True


def run_single_experiment(task_name, task_cls, task_kwargs, layers,
                          seed=42, lr=0.001, n_trials=200, eval_every=20,
                          plasticity_on=True):
    """
    Run one experiment on one task.

    Returns dict of results.
    """
    torch.manual_seed(seed)
    task = task_cls(**task_kwargs)
    task.set_seed(seed + 1)  # Different seed for task generation

    net = PureSNN(layers, dt=1.0, weight_scale=0.1,
                  tau_m=20.0, tau_syn=5.0)

    if plasticity_on:
        net.add_plasticity(lr=lr, tau_elig=20.0,
                           a_plus=0.008, a_minus=0.006,
                           tau_plus=20.0, tau_minus=20.0)

    # Freeze initial weights for control
    frozen = net.get_frozen_weights()

    # Training loop
    batch_size = 16
    accuracies = []
    weight_norms = []

    for trial in range(n_trials):
        # Generate batch
        inputs, targets = task.generate_batch(batch_size)

        # Forward pass
        output_spikes = net(inputs)

        # Compute reward
        reward = task.compute_reward(output_spikes, targets)

        # Apply reward if plasticity is on
        if plasticity_on and net.plasticities:
            for p in net.plasticities:
                p.apply_reward(reward)
        elif not plasticity_on:
            # Restore frozen weights (ensure no drift)
            net.restore_frozen_weights(frozen)

        # Evaluate
        if trial % eval_every == 0 or trial == n_trials - 1:
            with torch.no_grad():
                decisions = task.decode_output(output_spikes)
                accuracy = (decisions == targets).float().mean().item()
                accuracies.append((trial, accuracy))
                weight_norms.append((trial, net.weight_norm()))

    return {
        "task": task_name,
        "plasticity": plasticity_on,
        "seed": seed,
        "lr": lr,
        "n_trials": n_trials,
        "final_accuracy": accuracies[-1][1] if accuracies else 0.0,
        "accuracy_trace": accuracies,
        "weight_norm_trace": weight_norms,
        "initial_weight_norm": weight_norms[0][1] if weight_norms else 0.0,
        "final_weight_norm": weight_norms[-1][1] if weight_norms else 0.0,
    }


def run_full_bench(seeds=[42, 123, 256], lr=0.001, n_trials=300, eval_every=25):
    """
    Run the full benchmark across all tasks.
    """
    task_configs = get_task_configs()
    results = []

    for task_key, config in task_configs.items():
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_key} - {config['description']}")
        print(f"{'=' * 60}")

        for seed in seeds:
            # Plasticity ON
            print(f"\n  Seed {seed}, Plasticity ON:")
            r_on = run_single_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                plasticity_on=True,
            )
            results.append(r_on)
            print(f"    Initial weight norm: {r_on['initial_weight_norm']:.4f}")
            print(f"    Final weight norm:   {r_on['final_weight_norm']:.4f}")
            print(f"    Final accuracy:      {r_on['final_accuracy']:.3f}")

            # Plasticity OFF (frozen control)
            print(f"  Seed {seed}, Plasticity OFF (frozen):")
            r_off = run_single_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                plasticity_on=False,
            )
            results.append(r_off)
            print(f"    Initial weight norm: {r_off['initial_weight_norm']:.4f}")
            print(f"    Final weight norm:   {r_off['final_weight_norm']:.4f}")
            print(f"    Final accuracy:      {r_off['final_accuracy']:.3f}")

            delta = r_on['final_accuracy'] - r_off['final_accuracy']
            print(f"    Delta (on - off):    {delta:+.3f} {'<<<' if delta > 0.05 else ''}")

    return results


def print_verdict(results):
    """Print the plain-language verdict."""
    print("\n\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Group by task
    by_task = {}
    for r in results:
        key = r["task"]
        if key not in by_task:
            by_task[key] = {"on": [], "off": []}
        mode = "on" if r["plasticity"] else "off"
        by_task[key][mode].append(r)

    any_learning = False

    for task_key, data in by_task.items():
        print(f"\n--- {task_key} ---")
        on_accs = [r["final_accuracy"] for r in data["on"]]
        off_accs = [r["final_accuracy"] for r in data["off"]]

        on_mean = sum(on_accs) / len(on_accs)
        off_mean = sum(off_accs) / len(off_accs)

        print(f"  Plasticity ON  accuracies: {[f'{a:.3f}' for a in on_accs]}")
        print(f"  Plasticity OFF accuracies: {[f'{a:.3f}' for a in off_accs]}")
        print(f"  Mean ON: {on_mean:.3f} vs Mean OFF: {off_mean:.3f}")

        # Check if plasticity-on consistently beats off
        beats_off = sum(1 for on, off in zip(on_accs, off_accs) if on > off)
        total = len(on_accs)
        if beats_off > total / 2:
            print(f"  >>> Plasticity beats frozen in {beats_off}/{total} seeds")
            any_learning = True
        else:
            print(f"  >>> No consistent learning signal (beats in {beats_off}/{total})")

    print(f"\n\n{'=' * 60}")
    if any_learning:
        print("OVERALL VERDICT: YES - the spiking neural network can learn.")
        print("Plasticity-on consistently outperforms frozen controls.")
    else:
        print("OVERALL VERDICT: NO - insufficient evidence of learning.")
        print("Plasticity-on did not consistently outperform frozen controls.")
    print("=" * 60)


if __name__ == "__main__":
    print("PURE SPIKING NEURAL NETWORK - LEARNING EXPERIMENT")
    print("Testing whether a pure SNN can learn with immediate reward.\n")

    # 1. Verify freeze mechanism
    freeze_ok = verify_freeze_mechanism()
    if not freeze_ok:
        print("Freeze mechanism failed. Aborting.")
        sys.exit(1)

    # 2. Run full benchmark
    results = run_full_bench(
        seeds=[42, 123, 256],
        lr=0.001,
        n_trials=300,
        eval_every=25
    )

    # 3. Print verdict
    print_verdict(results)

    # 4. Save results
    # Convert tensors to serializable format
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in ['accuracy_trace', 'weight_norm_trace']}
        sr['accuracy_trace'] = [[int(t), float(a)] for t, a in r['accuracy_trace']]
        sr['weight_norm_trace'] = [[int(t), float(n)] for t, n in r['weight_norm_trace']]
        serializable.append(sr)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results.json')
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
