#!/usr/bin/env python3
"""
Lightning.ai entry point for the structural plasticity experiment.
Same compute as the main bench (free tier, 4 CPU cores).

Four arms:
1. Weight plasticity only
2. Structural plasticity only
3. Both together
4. Frozen control

Runs on Lightning.ai free tier via SDK.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snn.struct_experiment import (
    verify_freeze_mechanism,
    run_struct_bench,
    print_struct_verdict,
    save_struct_results,
)


def main():
    print("=" * 60)
    print("STRUCTURAL PLASTICITY EXPERIMENT")
    print("Can the network find its own route?")
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

    # Phase 2: Run 4-arm benchmark
    print("\n--- Phase 2: 4-Arm Structural Plasticity Benchmark ---")
    results = run_struct_bench(
        seeds=[42, 123, 256],
        lr=0.001,
        n_trials=300,
        eval_every=25,
    )

    # Phase 3: Print verdict
    print("\n--- Phase 3: Verdict ---")
    print_struct_verdict(results)

    # Phase 4: Save results
    print("\n--- Phase 4: Saving Results ---")
    save_struct_results(results)

    # Print final summary table
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    by_task = {}
    for r in results:
        key = r["task"]
        if key not in by_task:
            by_task[key] = {"arm1": [], "arm2": [], "arm3": [], "arm4": []}
        if r["weight_plasticity"] and not r["structural_plasticity"]:
            by_task[key]["arm1"].append(r["final_accuracy"])
        elif not r["weight_plasticity"] and r["structural_plasticity"]:
            by_task[key]["arm2"].append(r["final_accuracy"])
        elif r["weight_plasticity"] and r["structural_plasticity"]:
            by_task[key]["arm3"].append(r["final_accuracy"])
        else:
            by_task[key]["arm4"].append(r["final_accuracy"])

    print(f"\n{'Task':<30} {'Arm1(W)':>8} {'Arm2(S)':>8} {'Arm3(W+S)':>10} {'Arm4(F)':>8}")
    print("-" * 64)
    for task_key, data in by_task.items():
        def mean_or_dash(vals):
            if vals:
                return f"{sum(vals)/len(vals):.3f}"
            return "-"
        print(f"{task_key:<30} {mean_or_dash(data['arm1']):>8} {mean_or_dash(data['arm2']):>8} "
              f"{mean_or_dash(data['arm3']):>10} {mean_or_dash(data['arm4']):>8}")


if __name__ == "__main__":
    main()
