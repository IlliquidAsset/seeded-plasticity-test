"""
Structural plasticity experiment.
Tests whether a spiking neural network can FIND a route, not just strengthen a fixed one.

Four arms (compare all):
1. Weight plasticity only (R-STDP, fixed structure, fully connected)
2. Structural plasticity only (pruning/sprouting, no R-STDP weight updates)
3. Both together (R-STDP + structural plasticity)
4. Frozen control — no plasticity of either kind

Critical design: starting routes are seeded BADLY on purpose.
The network must walk its way from a bad route to a good one.

Signal: track which connections survive over time. Report whether the
surviving path earns reward and beats the seeded route.
"""

import torch
import json
import time
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn.core import PureSNN
from snn.tasks import get_task_configs, get_hard_task_configs
from snn.structural import StructuralPlasticity


def verify_freeze_mechanism():
    """
    Verify the freeze mechanism works for all 4 arms.
    Same as base experiment but checks mask freeze too.
    """
    print("=" * 60)
    print("FREEZE MECHANISM VERIFICATION (Structural)")
    print("=" * 60)

    # Test with mask active
    net = PureSNN([8, 16, 4], dt=1.0, weight_scale=0.1)
    net.add_plasticity(lr=0.1)

    # Enable mask
    for syn in net.synapses:
        syn.use_mask = True

    frozen_weights = net.get_frozen_weights()
    frozen_masks = [syn.connection_mask.data.clone() for syn in net.synapses]

    init_norm = net.weight_norm()
    print(f"Initial weight norm: {init_norm:.6f}")

    # Run forward + apply reward
    x = torch.randn(4, 8, 50)
    out = net(x)
    reward = torch.tensor([1.0, 1.0, -1.0, 1.0])
    for p in net.plasticities:
        p.apply_reward(reward)

    # Verify weights changed (no freeze yet)
    ok_no_freeze, _ = net.verify_freeze(frozen_weights)
    after_norm = net.weight_norm()
    print(f"Weight norm after update: {after_norm:.6f}")
    print(f"Freeze check (should FAIL): {ok_no_freeze}")

    # Restore and verify
    net.restore_frozen_weights(frozen_weights)
    for syn, fm in zip(net.synapses, frozen_masks):
        syn.connection_mask.data.copy_(fm)
    ok_restored, _ = net.verify_freeze(frozen_weights)
    print(f"Freeze check after restore (should PASS): {ok_restored}")

    # Verify mask preserved through freeze cycle
    mask_ok = all(
        torch.equal(syn.connection_mask.data, fm)
        for syn, fm in zip(net.synapses, frozen_masks)
    )
    print(f"Mask freeze check (should PASS): {mask_ok}")

    if ok_restored and mask_ok:
        print(">>> FREEZE VERIFICATION: PASSED")
        return True
    else:
        print(">>> FREEZE VERIFICATION: FAILED")
        return False


def run_struct_experiment(
    task_name, task_cls, task_kwargs, layers,
    seed=42, lr=0.001, n_trials=300, eval_every=20,
    weight_plasticity=True, structural_plasticity=True,
    struct_kwargs=None,
):
    """
    Run one experiment on one task with configurable plasticity types.

    Args:
        weight_plasticity: R-STDP weight updates active
        structural_plasticity: pruning/sprouting active
        struct_kwargs: overrides for StructuralPlasticity constructor

    Returns dict of results including connectivity traces.
    """
    if struct_kwargs is None:
        struct_kwargs = {}

    torch.manual_seed(seed)
    task = task_cls(**task_kwargs)
    task.set_seed(seed + 1)

    net = PureSNN(layers, dt=1.0, weight_scale=0.1,
                  tau_m=20.0, tau_syn=5.0)

    # Attach plasticity rules if weight plasticity is on
    if weight_plasticity:
        net.add_plasticity(lr=lr, tau_elig=20.0,
                           a_plus=0.008, a_minus=0.006,
                           tau_plus=20.0, tau_minus=20.0)

    # Attach structural plasticity per layer
    struct_plasticities = []
    if structural_plasticity:
        for i, syn in enumerate(net.synapses):
            pre_n = layers[i]
            post_n = layers[i + 1]
            sp = StructuralPlasticity(
                syn, pre_n, post_n,
                **struct_kwargs
            )
            # Seed with deliberately BAD routes
            sp.seed_initial_mask(n_routes=3, seed=seed + i * 7)
            # Override intervals based on n_trials so pruning/sprouting fires
            # at least 3 times during the run
            sp.prune_interval = max(1, n_trials // 6)
            sp.sprout_interval = max(1, n_trials // 6)
            struct_plasticities.append(sp)
    else:
        # Even without structural plasticity, enable mask for consistent comparison
        # but set it to fully connected (so no pruning/sprouting happens)
        for syn in net.synapses:
            syn.use_mask = True
            syn.connection_mask.data.fill_(1.0)

    # Freeze initial state for control and weight-movement reporting
    frozen_weights = net.get_frozen_weights()
    frozen_masks = [syn.connection_mask.data.clone() for syn in net.synapses]
    initial_weight_norm = net.weight_norm()

    # Training loop
    batch_size = 16
    accuracies = []
    weight_norms = []
    connectivity_snapshots = []  # (trial_num, list_of_mask_similarities_to_initial)
    final_masks = None

    for trial in range(n_trials):
        inputs, targets = task.generate_batch(batch_size)

        # Forward pass with return_all if structural plasticity needs per-layer activity
        need_all_layers = structural_plasticity and struct_plasticities
        if need_all_layers:
            output_spikes, all_layer = net(inputs, return_all=True)
        else:
            output_spikes = net(inputs)

        reward = task.compute_reward(output_spikes, targets)

        # --- Apply plasticity ---

        # 1. Weight plasticity (R-STDP)
        if weight_plasticity and net.plasticities:
            for p in net.plasticities:
                p.apply_reward(reward)

        # 2. Record activity for structural plasticity
        if structural_plasticity and struct_plasticities and need_all_layers:
            # all_layer[i] is (batch, n_neurons, timesteps) for layer i output spikes
            for i, sp in enumerate(struct_plasticities):
                if i == 0:
                    pre_spikes = inputs
                else:
                    pre_spikes = all_layer[i - 1]
                post_spikes = all_layer[i]

                # Sum over timesteps for activity
                pre_total = pre_spikes.sum(dim=-1)  # (batch, pre_n)
                post_total = post_spikes.sum(dim=-1)  # (batch, post_n)

                sp.record_activity(pre_total, post_total)

        # 3. Record reward for structural plasticity
        if structural_plasticity and struct_plasticities:
            for sp in struct_plasticities:
                sp.record_reward(reward)

        # 4. Apply structural updates (pruning/sprouting)
        if structural_plasticity and struct_plasticities:
            for sp in struct_plasticities:
                sp.step(trial)

        # --- Freeze control ---
        if not weight_plasticity and not structural_plasticity:
            # Full freeze: restore weights AND masks
            net.restore_frozen_weights(frozen_weights)
            for syn, fm in zip(net.synapses, frozen_masks):
                syn.connection_mask.data.copy_(fm)
        elif not weight_plasticity:
            # Only structural plasticity: freeze weights but allow mask changes
            net.restore_frozen_weights(frozen_weights)

        # Evaluate
        if trial % eval_every == 0 or trial == n_trials - 1:
            with torch.no_grad():
                decisions = task.decode_output(output_spikes)
                accuracy = (decisions == targets).float().mean().item()
                accuracies.append((trial, accuracy))
                weight_norms.append((trial, net.weight_norm()))

            # Connectivity snapshot
            if structural_plasticity and struct_plasticities:
                sims = []
                for sp, fm in zip(struct_plasticities, frozen_masks):
                    sim = sp.get_mask_similarity(fm)
                    sims.append(sim)
                connectivity_snapshots.append((trial, sims))
                # Also record raw stats
                for idx, sp in enumerate(struct_plasticities):
                    stats = sp.get_connectivity_stats()
                    connectivity_snapshots.append((f"stats_{idx}_{trial}", stats))

    # Final connectivity analysis
    final_connectivity = {}
    if structural_plasticity and struct_plasticities:
        final_masks = [sp.synapse.connection_mask.data.clone() for sp in struct_plasticities]
        for idx, sp in enumerate(struct_plasticities):
            stats = sp.get_connectivity_stats()
            sim_to_initial = sp.get_mask_similarity(frozen_masks[idx])
            final_connectivity[f"layer_{idx}"] = {
                "stats": stats,
                "jaccard_to_initial": sim_to_initial,
                "n_snapshots": len(sp.snapshots),
            }

    # Did structural plasticity change the connectivity?
    connectivity_changed = False
    if structural_plasticity and struct_plasticities:
        for idx, sp in enumerate(struct_plasticities):
            sim = sp.get_mask_similarity(frozen_masks[idx])
            if sim < 0.95:  # Less than 95% overlap = meaningful change
                connectivity_changed = True
                break

    # Did the surviving connections earn reward? (proxy: accuracy at end)
    final_accuracy = accuracies[-1][1] if accuracies else 0.0
    initial_accuracy = accuracies[0][1] if accuracies else 0.0

    final_weight_norm = net.weight_norm()
    signed_weight_norm_change = final_weight_norm - initial_weight_norm
    absolute_weight_norm_change = abs(signed_weight_norm_change)
    weight_norm_change_pct = (
        absolute_weight_norm_change / initial_weight_norm * 100.0
        if initial_weight_norm else 0.0
    )
    signed_weight_norm_change_pct = (
        signed_weight_norm_change / initial_weight_norm * 100.0
        if initial_weight_norm else 0.0
    )
    weights_nan = any(
        not torch.isfinite(syn.weight.data).all().item() for syn in net.synapses
    )
    max_abs_weight = max(
        syn.weight.data.abs().max().item() for syn in net.synapses
    )

    return {
        "task": task_name,
        "weight_plasticity": weight_plasticity,
        "structural_plasticity": structural_plasticity,
        "seed": seed,
        "lr": lr,
        "n_trials": n_trials,
        "initial_weight_norm": initial_weight_norm,
        "final_weight_norm": final_weight_norm,
        "signed_weight_norm_change": signed_weight_norm_change,
        "absolute_weight_norm_change": absolute_weight_norm_change,
        "signed_weight_norm_change_pct": signed_weight_norm_change_pct,
        "weight_norm_change_pct": weight_norm_change_pct,
        "weights_moved_over_1pct": weight_norm_change_pct > 1.0,
        "weights_nan": weights_nan,
        "weights_exploded": max_abs_weight > 5.0,
        "max_abs_weight": max_abs_weight,
        "final_accuracy": final_accuracy,
        "initial_accuracy": initial_accuracy,
        "accuracy_trace": accuracies,
        "weight_norm_trace": weight_norms,
        "connectivity_snapshots": connectivity_snapshots,
        "final_connectivity": final_connectivity,
        "connectivity_changed": connectivity_changed,
        "found_better_route": final_accuracy > initial_accuracy + 0.1,
    }


def verify_frozen_floor(
    task_configs, seeds=[42, 123, 256], n_trials=100, details_out=None
):
    """Verify frozen controls stay no more than 10 points above chance."""
    failures = {}
    means = {}

    print("\n" + "=" * 76)
    print("FROZEN-AT-FLOOR PRECHECK")
    print("=" * 76)
    print(f"{'Task':30s} {'Frozen mean':>12s} {'Chance':>10s} {'Limit':>10s} {'Status':>9s}")
    print("-" * 76)

    for task_name, config in task_configs.items():
        seed_means = []
        for seed in seeds:
            result = run_struct_experiment(
                task_name,
                config["cls"],
                config["kwargs"],
                config["layers"],
                seed=seed,
                n_trials=n_trials,
                eval_every=1,
                weight_plasticity=False,
                structural_plasticity=False,
            )
            trace = result["accuracy_trace"]
            seed_means.append(sum(acc for _, acc in trace) / len(trace))

        mean_accuracy = sum(seed_means) / len(seed_means)
        chance = 1.0 / config["layers"][-1]
        limit = chance + 0.10
        means[task_name] = mean_accuracy
        status = "PASS" if mean_accuracy <= limit else "FAIL"
        if status == "FAIL":
            failures[task_name] = mean_accuracy
        print(
            f"{task_name:30s} {mean_accuracy:12.3f} {chance:10.3f} "
            f"{limit:10.3f} {status:>9s}"
        )

    print("=" * 76)
    if failures:
        details = ", ".join(f"{name}={mean:.3f}" for name, mean in failures.items())
        print(f"FROZEN FLOOR GATE: FAILED — harden these tasks: {details}")
    else:
        print("FROZEN FLOOR GATE: PASSED")
    if details_out is not None:
        details_out.update(means)
    return not failures, failures


def calibrate_lr(
    task_configs=None,
    lr_sweep=[0.001, 0.01, 0.05, 0.1],
    task_name="binary_classification",
    seed=42,
    n_trials=100,
):
    """Sweep weight-only learning rates and choose the highest stable value."""
    if task_configs is None:
        task_configs = get_hard_task_configs()
    if task_name not in task_configs:
        raise KeyError(f"Unknown calibration task: {task_name}")

    config = task_configs[task_name]
    sweep_results = []
    print("\n" + "=" * 88)
    print(f"WEIGHT-PLASTICITY LR CALIBRATION ({task_name}, seed={seed})")
    print("=" * 88)
    print(
        f"{'LR':>8s} {'Initial norm':>14s} {'Final norm':>14s} "
        f"{'Signed move':>13s} {'Max |w|':>10s} {'Stable':>9s}"
    )
    print("-" * 88)

    for lr in lr_sweep:
        result = run_struct_experiment(
            task_name,
            config["cls"],
            config["kwargs"],
            config["layers"],
            seed=seed,
            lr=lr,
            n_trials=n_trials,
            eval_every=max(1, n_trials // 4),
            weight_plasticity=True,
            structural_plasticity=False,
        )
        stable = not result["weights_nan"] and not result["weights_exploded"]
        calibration = {
            "lr": lr,
            "initial_weight_norm": result["initial_weight_norm"],
            "final_weight_norm": result["final_weight_norm"],
            "signed_weight_norm_change_pct": result["signed_weight_norm_change_pct"],
            "weight_norm_change_pct": result["weight_norm_change_pct"],
            "max_abs_weight": result["max_abs_weight"],
            "weights_nan": result["weights_nan"],
            "weights_exploded": result["weights_exploded"],
            "stable": stable,
        }
        sweep_results.append(calibration)
        print(
            f"{lr:8.3g} {result['initial_weight_norm']:14.6f} "
            f"{result['final_weight_norm']:14.6f} "
            f"{result['signed_weight_norm_change_pct']:12.3f}% "
            f"{result['max_abs_weight']:10.4f} "
            f"{('YES' if stable else 'NO'):>9s}"
        )

    stable_lrs = [item["lr"] for item in sweep_results if item["stable"]]
    if not stable_lrs:
        raise RuntimeError("LR calibration failed: every candidate was unstable")
    chosen_lr = max(stable_lrs)
    print("-" * 88)
    print(
        f"Chosen LR: {chosen_lr:g} — highest candidate with finite weights "
        "and max |weight| <= 5.0"
    )
    print("=" * 88)
    return chosen_lr, sweep_results


def run_struct_bench(
    seeds=[42, 123, 256],
    lr=0.001,
    n_trials=300,
    eval_every=25,
    tasks_to_run=None,
    task_configs=None,
    result_callback=None,
):
    """
    Run the full 4-arm structural plasticity benchmark.

    Args:
        tasks_to_run: list of task keys to include (None = all)
    """
    if task_configs is None:
        task_configs = get_task_configs()
    results = []

    def record(result):
        results.append(result)
        if result_callback is not None:
            result_callback(result, len(results))

    for task_key, config in task_configs.items():
        if tasks_to_run is not None and task_key not in tasks_to_run:
            continue

        print(f"\n{'=' * 60}")
        print(f"TASK: {task_key} - {config['description']}")
        print(f"{'=' * 60}")

        for seed in seeds:
            # ARM 1: Weight plasticity only
            print(f"\n  Seed {seed}, ARM 1: Weight plasticity only:")
            r1 = run_struct_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                weight_plasticity=True, structural_plasticity=False,
            )
            record(r1)
            print(f"    Final accuracy: {r1['final_accuracy']:.3f}")

            # ARM 2: Structural plasticity only
            print(f"  Seed {seed}, ARM 2: Structural plasticity only:")
            r2 = run_struct_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                weight_plasticity=False, structural_plasticity=True,
            )
            record(r2)
            print(f"    Final accuracy: {r2['final_accuracy']:.3f}")
            if r2.get("final_connectivity"):
                for lidx, ldata in r2["final_connectivity"].items():
                    print(f"      {lidx}: {ldata['stats']}")
                    print(f"      Jaccard to initial: {ldata['jaccard_to_initial']:.3f}")

            # ARM 3: Both together
            print(f"  Seed {seed}, ARM 3: Both together:")
            r3 = run_struct_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                weight_plasticity=True, structural_plasticity=True,
            )
            record(r3)
            print(f"    Final accuracy: {r3['final_accuracy']:.3f}")
            if r3.get("final_connectivity"):
                for lidx, ldata in r3["final_connectivity"].items():
                    print(f"      {lidx}: {ldata['stats']}")
                    print(f"      Jaccard to initial: {ldata['jaccard_to_initial']:.3f}")

            # ARM 4: Frozen control
            print(f"  Seed {seed}, ARM 4: Frozen control:")
            r4 = run_struct_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                weight_plasticity=False, structural_plasticity=False,
            )
            record(r4)
            print(f"    Final accuracy: {r4['final_accuracy']:.3f}")
            print(f"    Freeze check - no change expected: acc={r4['final_accuracy']:.3f}")

    return results


def summarize_struct_results(results, real_margin=0.10):
    """Aggregate arm accuracies, weight movement, and frozen comparisons."""
    arm_names = {
        (True, False): "weight_only",
        (False, True): "structural_only",
        (True, True): "both",
        (False, False): "frozen",
    }
    grouped = {}
    for result in results:
        task = grouped.setdefault(result["task"], {})
        arm = arm_names[(result["weight_plasticity"], result["structural_plasticity"])]
        task.setdefault(arm, []).append(result)

    task_summary = {}
    deltas = []
    for task_name, arms in grouped.items():
        task_summary[task_name] = {}
        for arm_name, runs in arms.items():
            accuracies = [run["final_accuracy"] for run in runs]
            task_summary[task_name][arm_name] = {
                "accuracies": accuracies,
                "mean_accuracy": sum(accuracies) / len(accuracies),
            }
        frozen = task_summary[task_name].get("frozen")
        if frozen:
            frozen_mean = frozen["mean_accuracy"]
            for arm_name in ("weight_only", "structural_only", "both"):
                if arm_name in task_summary[task_name]:
                    delta = task_summary[task_name][arm_name]["mean_accuracy"] - frozen_mean
                    task_summary[task_name][arm_name]["delta_vs_frozen"] = delta
                    deltas.append(delta)

    weight_movement = {}
    for arm_name in ("weight_only", "both"):
        runs = [
            run for arms in grouped.values() for run in arms.get(arm_name, [])
        ]
        if not runs:
            continue
        mean_abs = sum(run["absolute_weight_norm_change"] for run in runs) / len(runs)
        mean_pct = sum(run["weight_norm_change_pct"] for run in runs) / len(runs)
        weight_movement[arm_name] = {
            "mean_absolute_norm_change": mean_abs,
            "mean_percentage_change": mean_pct,
            "passes_1pct": mean_pct > 1.0,
        }

    if any(delta >= real_margin for delta in deltas):
        verdict = "YES"
    elif deltas and all(delta <= 0 for delta in deltas):
        verdict = "NO"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "real_margin": real_margin,
        "tasks": task_summary,
        "weight_movement": weight_movement,
        "verdict": verdict,
    }


def print_struct_verdict(results):
    """Print the plain-language verdict for structural plasticity experiment."""
    print("\n\n" + "=" * 60)
    print("STRUCTURAL PLASTICITY VERDICT")
    print("=" * 60)

    # Group by task
    by_task = {}
    for r in results:
        key = r["task"]
        if key not in by_task:
            by_task[key] = {"arm1": [], "arm2": [], "arm3": [], "arm4": []}
        if r["weight_plasticity"] and not r["structural_plasticity"]:
            by_task[key]["arm1"].append(r)
        elif not r["weight_plasticity"] and r["structural_plasticity"]:
            by_task[key]["arm2"].append(r)
        elif r["weight_plasticity"] and r["structural_plasticity"]:
            by_task[key]["arm3"].append(r)
        else:
            by_task[key]["arm4"].append(r)

    arm_labels = {
        "arm1": "Weight plasticity only",
        "arm2": "Structural plasticity only",
        "arm3": "Both together",
        "arm4": "Frozen control",
    }

    for task_key, data in by_task.items():
        print(f"\n--- {task_key} ---")
        for arm_key, label in arm_labels.items():
            runs = data[arm_key]
            if not runs:
                continue
            accs = [r["final_accuracy"] for r in runs]
            mean_acc = sum(accs) / len(accs)
            found_route = sum(1 for r in runs if r.get("found_better_route", False))
            conn_changed = sum(1 for r in runs if r.get("connectivity_changed", False))
            print(f"  {label}:")
            print(f"    Accuracies: {[f'{a:.3f}' for a in accs]}")
            print(f"    Mean accuracy: {mean_acc:.3f}")
            if arm_key in ("arm2", "arm3"):
                print(f"    Found better route: {found_route}/{len(runs)} seeds")
                print(f"    Connectivity changed: {conn_changed}/{len(runs)} seeds")
                # Check if surviving connections earned reward
                for r in runs:
                    if r.get("final_connectivity"):
                        better = "YES" if r.get("found_better_route") else "NO"
                        changed = "YES" if r.get("connectivity_changed") else "NO"
                        print(f"      Seed {r['seed']}: better_route={better}, connectivity_changed={changed}")
                        for lidx, ldata in r["final_connectivity"].items():
                            js = ldata.get("jaccard_to_initial", 1.0)
                            dens = ldata.get("stats", {}).get("density", 0)
                            print(f"        {lidx}: Jaccard={js:.3f}, density={dens:.4f}")

    summary = summarize_struct_results(results)
    print(f"\n\n{'=' * 60}")
    print("WEIGHT MOVEMENT")
    print("=" * 60)
    for arm_name, movement in summary["weight_movement"].items():
        status = "PASS" if movement["passes_1pct"] else "FAIL"
        print(
            f"{arm_name}: mean absolute norm change="
            f"{movement['mean_absolute_norm_change']:.6f}, "
            f"mean percentage change={movement['mean_percentage_change']:.3f}% "
            f"({status}: {'>' if movement['passes_1pct'] else '<='} 1%)"
        )

    print(f"\n{'=' * 60}")
    print("PLASTICITY VS FROZEN (10-POINT REAL MARGIN)")
    print("=" * 60)
    for task_name, arms in summary["tasks"].items():
        frozen_mean = arms.get("frozen", {}).get("mean_accuracy", float("nan"))
        print(f"{task_name}: frozen={frozen_mean:.3f}")
        for arm_name in ("weight_only", "structural_only", "both"):
            if arm_name in arms:
                print(
                    f"  {arm_name}: mean={arms[arm_name]['mean_accuracy']:.3f}, "
                    f"delta={arms[arm_name].get('delta_vs_frozen', float('nan')):+.3f}"
                )

    print(f"\n{'=' * 60}")
    print("OVERALL VERDICT")
    print("=" * 60)
    print(f"{summary['verdict']} - does any plasticity arm beat frozen by >= 0.10?")
    print("=" * 60)


def save_struct_results(results, out_path=None):
    """Save results to JSON."""
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'struct_results.json')

    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()
              if k not in ['accuracy_trace', 'weight_norm_trace', 'connectivity_snapshots']}
        sr['accuracy_trace'] = [[int(t), float(a)] for t, a in r['accuracy_trace']]
        sr['weight_norm_trace'] = [[int(t), float(n)] for t, n in r['weight_norm_trace']]
        # Connectivity snapshots are complex; save compact version
        sr['final_connectivity'] = r.get('final_connectivity', {})
        serializable.append(sr)

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    print("STRUCTURAL PLASTICITY EXPERIMENT")
    print("Can the network find its own route, not just strengthen a fixed one?")
    print("Four arms, 5 tasks, 3 seeds each.\n")

    # 1. Verify freeze mechanism
    freeze_ok = verify_freeze_mechanism()
    if not freeze_ok:
        print("Freeze mechanism failed. Aborting.")
        sys.exit(1)

    # 2. Run benchmark
    results = run_struct_bench(
        seeds=[42, 123, 256],
        lr=0.001,
        n_trials=300,
        eval_every=25,
    )

    # 3. Print verdict
    print_struct_verdict(results)

    # 4. Save results
    save_struct_results(results)
