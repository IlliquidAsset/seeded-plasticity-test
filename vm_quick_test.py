#!/usr/bin/env python3
"""Quick test script to run on Lightning VM."""
import sys
sys.path.insert(0, '/tmp/snn')

from snn.experiment import verify_freeze_mechanism, run_single_experiment
from snn.tasks import BinaryClassificationTask

print("=== Quick Test Start ===")
freeze_ok = verify_freeze_mechanism()
print(f"Freeze OK: {freeze_ok}")

r = run_single_experiment(
    "test", BinaryClassificationTask, {"n_input": 16, "timesteps": 100}, [16, 2],
    seed=42, lr=0.001, n_trials=50, eval_every=10, plasticity_on=True
)
print(f"ON: final_acc={r['final_accuracy']:.3f}, init_norm={r['initial_weight_norm']:.4f}, final_norm={r['final_weight_norm']:.4f}")

r2 = run_single_experiment(
    "test", BinaryClassificationTask, {"n_input": 16, "timesteps": 100}, [16, 2],
    seed=42, lr=0.001, n_trials=50, eval_every=10, plasticity_on=False
)
print(f"OFF: final_acc={r2['final_accuracy']:.3f}")

print("=== Quick Test Done ===")
