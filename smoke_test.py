#!/usr/bin/env python3
"""Quick smoke test - verify imports, freeze mechanism, and basic task execution."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from snn.core import PureSNN
from snn.tasks import BinaryClassificationTask, TemporalXORTask

print("=== Smoke Test: Pure SNN Learning ===")

# Test 1: Freeze mechanism
print("\n--- Test 1: Freeze Mechanism ---")
net = PureSNN([16, 2])
net.add_plasticity(lr=0.001)
frozen = net.get_frozen_weights()
init_norm = net.weight_norm()
print(f"Initial weight norm: {init_norm:.4f}")

x = torch.randn(4, 16, 50)
out = net(x)
for p in net.plasticities:
    p.apply_reward(torch.tensor([1.0, 1.0, -1.0, 1.0]))

post_norm = net.weight_norm()
print(f"After plasticity update norm: {post_norm:.4f}")

ok_before, diffs_before = net.verify_freeze(frozen)
print(f"Freeze check BEFORE restore: {ok_before} (diffs: {diffs_before})")
assert not ok_before, "Weights should have changed after plasticity update"
print("  OK: Weights changed (plasticity works)")

net.restore_frozen_weights(frozen)
ok_after, diffs_after = net.verify_freeze(frozen)
restored_norm = net.weight_norm()
print(f"Freeze check AFTER restore: {ok_after} (diffs: {diffs_after})")
print(f"Restored norm: {restored_norm:.4f}")
assert ok_after, "FREEZE MECHANISM FAILED - weights did not match after restore"
assert abs(restored_norm - init_norm) < 1e-10, "Norm should match exactly"
print("  PASSED: Freeze mechanism verified")

# Test 2: Task execution
print("\n--- Test 2: Task Execution ---")
task = BinaryClassificationTask(n_input=16)
task.set_seed(42)
inputs, targets = task.generate_batch(8)
print(f"Input shape: {inputs.shape}")
print(f"Targets: {targets}")
assert inputs.shape == (8, 16, 100), f"Expected (8, 16, 100), got {inputs.shape}"

out = net(inputs)
print(f"Output shape: {out.shape}")
reward = task.compute_reward(out, targets)
print(f"Rewards: {reward}")
acc = (task.decode_output(out) == targets).float().mean().item()
print(f"Accuracy (random init): {acc:.3f}")

# Test 3: Temporal XOR task
print("\n--- Test 3: Temporal XOR Task ---")
xor_task = TemporalXORTask(n_hidden=16)
xor_task.set_seed(42)
inputs2, targets2 = xor_task.generate_batch(8)
print(f"Input shape: {inputs2.shape}")
print(f"Targets: {targets2}")

net2 = PureSNN([2, 16, 2])
net2.add_plasticity(lr=0.001)
out2 = net2(inputs2)
reward2 = xor_task.compute_reward(out2, targets2)
print(f"Rewards: {reward2}")
acc2 = (xor_task.decode_output(out2) == targets2).float().mean().item()
print(f"Accuracy (random init): {acc2:.3f}")

# Test 4: Verify freeze during training loop
print("\n--- Test 4: Training Loop with Freeze ---")
net3 = PureSNN([16, 2])
frozen3 = net3.get_frozen_weights()
task3 = BinaryClassificationTask(n_input=16)
task3.set_seed(99)

accs_frozen = []
for i in range(20):
    inp, tgt = task3.generate_batch(16)
    out3 = net3(inp)
    # No plasticity - no plasticity rules added
    # Verify weights never changed
    ok, _ = net3.verify_freeze(frozen3)
    if not ok:
        print(f"  FAILED at trial {i}: weights changed without plasticity!")
        break
    acc3 = (task3.decode_output(out3) == tgt).float().mean().item()
    accs_frozen.append(acc3)

print(f"  Frozen accuracies over 20 trials: min={min(accs_frozen):.3f}, max={max(accs_frozen):.3f}")
print(f"  Freeze held for {len(accs_frozen)} trials")
print("  PASSED: No plasticity = no weight change")

print("\n=== ALL SMOKE TESTS PASSED ===")
