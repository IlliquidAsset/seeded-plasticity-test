#!/usr/bin/env python3
"""
Lightning.ai entry point for pure SNN learning experiment.
Runs on the free tier (4 CPU cores).
"""
import sys
import os
import json
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snn.experiment import verify_freeze_mechanism, run_full_bench, print_verdict


def main():
    print("=" * 60)
    print("PURE SPIKING NEURAL NETWORK - LEARNING EXPERIMENT")
    print("Lightning.ai Free Tier (4 CPU cores)")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Verify freeze mechanism
    print("\n--- Phase 1: Freeze Verification ---")
    freeze_ok = verify_freeze_mechanism()
    if not freeze_ok:
        print("CRITICAL: Freeze mechanism failed. Aborting.")
        sys.exit(1)
    print("Freeze mechanism: VERIFIED")

    # Phase 2: Run benchmark
    print("\n--- Phase 2: Full Benchmark ---")
    results = run_full_bench(
        seeds=[42, 123, 256],
        lr=0.005,
        n_trials=300,
        eval_every=25,
    )

    # Phase 3: Print verdict
    print("\n--- Phase 3: Verdict ---")
    print_verdict(results)

    # Phase 4: Save results
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k not in ['accuracy_trace', 'weight_norm_trace']}
        sr['accuracy_trace'] = [[int(t), float(a)] for t, a in r['accuracy_trace']]
        sr['weight_norm_trace'] = [[int(t), float(n)] for t, n in r['weight_norm_trace']]
        serializable.append(sr)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    # Determine overall verdict
    by_task = {}
    for r in results:
        key = r["task"]
        if key not in by_task:
            by_task[key] = {"on": [], "off": []}
        mode = "on" if r["plasticity"] else "off"
        by_task[key][mode].append(r["final_accuracy"])

    print("\nSummary Table:")
    print(f"{'Task':<30} {'Seeds':>5} {'Plasticity ON':>14} {'Plasticity OFF':>15} {'Delta':>8}")
    print("-" * 72)
    for task_key, data in by_task.items():
        on_accs = data["on"]
        off_accs = data["off"]
        on_mean = sum(on_accs) / len(on_accs)
        off_mean = sum(off_accs) / len(off_accs)
        delta = on_mean - off_mean
        n_seeds = len(on_accs)
        print(f"{task_key:<30} {n_seeds:>5} {on_mean:>13.3f} {off_mean:>14.3f} {delta:>+7.3f}")
    print("-" * 72)


if __name__ == "__main__":
    main()
