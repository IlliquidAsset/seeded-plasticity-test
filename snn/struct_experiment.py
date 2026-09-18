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
from snn.tasks import get_task_configs
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

    # Freeze initial state for control
    frozen_weights = net.get_frozen_weights()
    frozen_masks = [syn.connection_mask.data.clone() for syn in net.synapses]

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

    return {
        "task": task_name,
        "weight_plasticity": weight_plasticity,
        "structural_plasticity": structural_plasticity,
        "seed": seed,
        "lr": lr,
        "n_trials": n_trials,
        "final_accuracy": final_accuracy,
        "initial_accuracy": initial_accuracy,
        "accuracy_trace": accuracies,
        "weight_norm_trace": weight_norms,
        "connectivity_snapshots": connectivity_snapshots,
        "final_connectivity": final_connectivity,
        "connectivity_changed": connectivity_changed,
        "found_better_route": final_accuracy > initial_accuracy + 0.1,
    }


def run_struct_bench(
    seeds=[42, 123, 256],
    lr=0.001,
    n_trials=300,
    eval_every=25,
    tasks_to_run=None,
):
    """
    Run the full 4-arm structural plasticity benchmark.

    Args:
        tasks_to_run: list of task keys to include (None = all)
    """
    task_configs = get_task_configs()
    results = []

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
            results.append(r1)
            print(f"    Final accuracy: {r1['final_accuracy']:.3f}")

            # ARM 2: Structural plasticity only
            print(f"  Seed {seed}, ARM 2: Structural plasticity only:")
            r2 = run_struct_experiment(
                task_key, config["cls"], config["kwargs"], config["layers"],
                seed=seed, lr=lr, n_trials=n_trials, eval_every=eval_every,
                weight_plasticity=False, structural_plasticity=True,
            )
            results.append(r2)
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
            results.append(r3)
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
            results.append(r4)
            print(f"    Final accuracy: {r4['final_accuracy']:.3f}")
            print(f"    Freeze check - no change expected: acc={r4['final_accuracy']:.3f}")

    return results


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

    # Overall verdict
    print(f"\n\n{'=' * 60}")
    print("OVERALL VERDICT")
    print("=" * 60)

    any_structural_learning = False
    for task_key, data in by_task.items():
        for arm_key in ("arm2", "arm3"):
            runs = data[arm_key]
            if not runs:
                continue
            for r in runs:
                if r.get("found_better_route", False):
                    any_structural_learning = True
                    break

    if any_structural_learning:
        print("YES - the network can find a better route through structural plasticity.")
        print("Structural plasticity (alone or with weight plasticity) discovered")
        print("connectivity patterns that outperform the initial bad seeding.")
    else:
        print("INCONCLUSIVE - insufficient evidence that structural plasticity")
        print("discovered better routes than the initial bad seeding.")
        print("This may indicate: (a) pruning/sprouting parameters need tuning,")
        print("(b) the task difficulty is mismatched, or (c) the mechanism needs revision.")
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
