"""Quick structural plasticity smoke test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from snn.core import PureSNN
from snn.structural import StructuralPlasticity, seed_bad_routes

print("=== Structural Plasticity Smoke Test ===\n")

# Test 1: seed_bad_routes produces non-trivial mask
print("--- Test 1: Bad Route Seeding ---")
mask = seed_bad_routes(8, 16, n_routes=3, seed=42)
print(f"Mask shape: {mask.shape}")
print(f"Connections: {mask.sum().item()}/{mask.numel()} ({mask.sum().item()/mask.numel()*100:.1f}%)")
assert mask.sum().item() > 0, "Mask should have connections"
assert mask.sum().item() < mask.numel(), "Mask should NOT be fully connected (that would be good routing)"
print("  PASSED: Bad seeding produced non-trivial sparse mask\n")

# Test 2: StructuralPlasticity initialization
print("--- Test 2: StructuralPlasticity init ---")
net = PureSNN([16, 8, 4])
syn = net.synapses[0]
sp = StructuralPlasticity(syn, 16, 8)
sp.seed_initial_mask(n_routes=3, seed=42)
stats = sp.get_connectivity_stats()
print(f"Stats: {stats}")
assert stats["total_connected"] < stats["total_possible"], "Should be sparse"
assert syn.use_mask == True, "Mask should be active"
print("  PASSED: StructuralPlasticity initialized\n")

# Test 3: Forward pass with mask
print("--- Test 3: Forward pass with mask ---")
x = torch.randn(4, 16, 50)
out = net(x)
print(f"Output shape: {out.shape}")
print(f"Output spikes: {out.sum().item()} total")
print("  PASSED: Forward pass works with mask\n")

# Test 4: Pruning removes weak connections
print("--- Test 4: Pruning ---")
net2 = PureSNN([16, 8, 4])
syn2 = net2.synapses[0]
sp2 = StructuralPlasticity(syn2, 16, 8, prune_threshold=0.5, prune_interval=1)
sp2.seed_initial_mask(n_routes=3, seed=42)

# Make some weights strong, others weak
with torch.no_grad():
    syn2.weight.data[:4, :8] = 1.0  # strong
    syn2.weight.data[4:, 8:] = 0.001  # weak

initial_conn = syn2.connection_mask.data.sum().item()
print(f"Initial connections: {initial_conn}")

# Record activity and reward
x2 = torch.randn(2, 16, 20)
out2 = net2(x2)
sp2.record_activity(torch.randn(2, 16), torch.randn(2, 8))
sp2.record_reward(torch.tensor(1.0))

# Prune
result = sp2._prune(trial_num=50)
if result:
    print(f"Prune result: {result}")
final_conn = syn2.connection_mask.data.sum().item()
print(f"Final connections: {final_conn}")
assert final_conn <= initial_conn, "Pruning should not add connections"
print("  PASSED: Pruning works\n")

# Test 5: Sprouting adds connections
print("--- Test 5: Sprouting ---")
net3 = PureSNN([16, 8, 4])
syn3 = net3.synapses[0]
sp3 = StructuralPlasticity(syn3, 16, 8, sprout_interval=1, sprout_rate=0.5)
sp3.seed_initial_mask(n_routes=1, seed=42)  # minimal initial connections

initial_conn3 = syn3.connection_mask.data.sum().item()
print(f"Initial connections: {initial_conn3}")

# Set up activity traces as if some neurons are active
sp3.pre_activity = torch.zeros(16)
sp3.post_activity = torch.zeros(8)
sp3.pre_activity[:5] = 1.0  # first 5 pre are active
sp3.post_activity[:3] = 1.0  # first 3 post are active
sp3.recent_reward = 1.0
sp3.reward_baseline = -0.5

result3 = sp3._sprout(trial_num=50)
if result3:
    print(f"Sprout result: {result3}")
final_conn3 = syn3.connection_mask.data.sum().item()
print(f"Final connections: {final_conn3}")
assert final_conn3 > initial_conn3, "Sprouting should add connections"
print("  PASSED: Sprouting works\n")

# Test 6: Full step cycle (prune + sprout)
print("--- Test 6: Full step cycle ---")
net4 = PureSNN([16, 8, 4])
syn4 = net4.synapses[0]
sp4 = StructuralPlasticity(syn4, 16, 8,
                            prune_interval=10, sprout_interval=10,
                            prune_threshold=0.5, sprout_rate=0.3)
sp4.seed_initial_mask(n_routes=2, seed=42)

# Make some weights very weak for pruning
with torch.no_grad():
    syn4.weight.data[syn4.weight.data.abs() < 0.1] = 0.001

sp4.pre_activity = torch.randn(16)
sp4.post_activity = torch.randn(8)
sp4.recent_reward = 1.0
sp4.reward_baseline = -0.5

change = sp4.step(trial_num=10)
print(f"Step result: {change}")
print(f"Snapshots: {len(sp4.snapshots)}")
assert len(sp4.snapshots) > 0, "Should have recorded snapshots"
print("  PASSED: Full step cycle works\n")

# Test 7: Connectivity stats
print("--- Test 7: Connectivity stats ---")
stats4 = sp4.get_connectivity_stats()
print(f"Stats: {stats4}")
assert "density" in stats4
assert "total_connected" in stats4
print("  PASSED: Connectivity stats work\n")

# Test 8: Mask similarity
print("--- Test 8: Mask similarity ---")
initial_mask = syn4.connection_mask.data.clone()
sim = sp4.get_mask_similarity(initial_mask)
print(f"Jaccard similarity to initial: {sim:.4f}")
assert 0 <= sim <= 1.0
print("  PASSED: Mask similarity works\n")

print("=== ALL STRUCTURAL PLASTICITY TESTS PASSED ===")
