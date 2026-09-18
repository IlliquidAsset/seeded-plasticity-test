"""
Pure SNN Learning Experiment
============================
Can a spiking neural network learn with immediate reward?

Built entirely from spiking primitives - no transformers, no ANN-to-SNN conversion,
no seeded weights.

Design:
- LIF neurons with current-based synapses
- Reward-modulated STDP (R-STDP) plasticity
- Immediate reward at end of each trial

Controls:
1. Frozen-weight: plasticity disabled, weights restored after each trial
2. Fresh-seed replication: 3 different random seeds per task
3. Freeze mechanism verified before experiment runs

Tasks (varying difficulty):
- binary_classification: Rate-based binary decision (easy)
- frequency_discrimination: 3-class rate discrimination (easy-medium)
- temporal_xor: Non-linear temporal XOR (medium)
- temporal_sequence: Signal-in-noise temporal integration (medium-hard)
- associative_memory: Pattern association (hard)

Author: Amanda Bigoletits for Kendrick Kirk
Project: AIB Research - Pure SNN Learning
"""

__version__ = "0.1.0"
