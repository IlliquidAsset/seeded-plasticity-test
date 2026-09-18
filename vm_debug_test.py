#!/usr/bin/env python3
"""Debug: check if plasticity traces and weight updates are working."""
import sys
sys.path.insert(0, '/tmp/snn')

import torch
from snn.core import PureSNN

print("=== Debug: Plasticity Traces ===")

# Create a tiny network
net = PureSNN([2, 2, 1], dt=1.0, weight_scale=0.1)
net.add_plasticity(lr=0.1, tau_elig=20.0, a_plus=0.01, a_minus=0.01)

# Print initial weights
for i, syn in enumerate(net.synapses):
    print(f"Layer {i} initial weights: {syn.weight.data}")

# Create simple input that should cause spiking
# High-rate Poisson for 50 timesteps
x = torch.zeros(1, 2, 50)
# Make first input neuron fire every few timesteps
for t in range(0, 50, 5):
    x[0, 0, t] = 1.0
# Make second input fire at different times
for t in range(2, 50, 5):
    x[0, 1, t] = 1.0

print(f"\nInput spikes: {x.sum()} total spikes")

# Forward pass
out = net(x)
print(f"Output spikes: {out.sum()} total spikes")

# Check eligibility traces
for i, p in enumerate(net.plasticities):
    print(f"\nLayer {i} eligibility trace:")
    print(f"  Shape: {p.eligibility.shape}")
    print(f"  Mean: {p.eligibility.mean().item():.6f}")
    print(f"  Max: {p.eligibility.max().item():.6f}")
    print(f"  Min: {p.eligibility.min().item():.6f}")
    print(f"  Non-zero: {(p.eligibility.abs() > 1e-10).sum().item()}")

# Apply reward
reward = torch.tensor([1.0])
print(f"\nApplying reward: {reward}")
for p in net.plasticities:
    delta = 0.1 * reward.mean() * p.eligibility.mean(dim=0)
    print(f"  Layer delta_w mean: {delta.mean().item():.6f}, max: {delta.max().item():.6f}")
    p.apply_reward(reward)

# Check weights after update
for i, syn in enumerate(net.synapses):
    print(f"Layer {i} after update weights: {syn.weight.data}")

print("\n=== Debug Complete ===")
