"""
Pure spiking neural network core.
LIF neurons, current-based synapses, R-STDP learning.
Timestep-by-timestep forward pass.

No transformers. No ANN-to-SNN conversion. No seeded weights.
Built from scratch.
"""

import torch
import torch.nn as nn
import math


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.
    Processes one timestep at a time.
    """
    def __init__(self, n_neurons, tau_m=20.0, v_thresh=-55.0, v_rest=-70.0, v_reset=-75.0, dt=1.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.dt = dt
        self.beta = math.exp(-dt / tau_m) if tau_m > 0 else 0.0

    def forward(self, I_syn, v=None):
        """
        One timestep.
        I_syn: (batch, n_neurons)
        v: (batch, n_neurons) or None for reset

        Returns: spikes (batch, n_neurons), new v (batch, n_neurons)
        """
        if v is None:
            v = torch.full_like(I_syn, self.v_rest)

        # Leaky integrate
        v = self.v_rest + self.beta * (v - self.v_rest) + I_syn
        # Fire
        fired = (v >= self.v_thresh).float()
        # Soft reset
        v = torch.where(fired > 0, torch.full_like(v, self.v_reset), v)

        return fired, v


class Synapse(nn.Module):
    """
    Synaptic connection between two populations.
    Current-based with exponential decay, one timestep at a time.
    """
    def __init__(self, pre_n, post_n, weight_scale=0.1, tau_syn=5.0, dt=1.0):
        super().__init__()
        self.pre_n = pre_n
        self.post_n = post_n
        self.tau_syn = tau_syn
        self.dt = dt
        self.beta_syn = math.exp(-dt / tau_syn) if tau_syn > 0 else 0.0

        # Initialize weights uniformly
        self.weight = nn.Parameter(
            torch.randn(post_n, pre_n) * weight_scale / math.sqrt(pre_n)
        )

    def forward(self, pre_spikes, I_syn=None):
        """
        One timestep.
        pre_spikes: (batch, pre_n)
        I_syn: (batch, post_n) or None

        Returns: I_syn_new (batch, post_n)
        """
        if I_syn is None:
            I_syn = torch.zeros(pre_spikes.shape[0], self.post_n, device=pre_spikes.device)

        # Decay
        I_syn = self.beta_syn * I_syn
        # Add current from spikes
        I_syn = I_syn + torch.mm(pre_spikes, self.weight.t())

        return I_syn


class RSTDPPlasticity:
    """
    Reward-modulated STDP learning rule.
    Tracks eligibility trace per synapse, modulated by global reward.

    Δw_ij = η * R * e_ij
    e_ij = eligibility trace (low-pass filtered STDP signal)
    """
    def __init__(self, synapse, lr=0.001, tau_elig=20.0, dt=1.0,
                 a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0):
        self.synapse = synapse
        self.lr = lr
        self.tau_elig = tau_elig
        self.dt = dt
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.beta_elig = math.exp(-dt / tau_elig) if tau_elig > 0 else 0.0
        self.beta_plus = math.exp(-dt / tau_plus) if tau_plus > 0 else 0.0
        self.beta_minus = math.exp(-dt / tau_minus) if tau_minus > 0 else 0.0

    def reset(self, batch_size, device):
        """Reset traces for a new trial."""
        post_n, pre_n = self.synapse.weight.shape
        self.eligibility = torch.zeros(batch_size, post_n, pre_n, device=device)
        self.pre_trace = torch.zeros(batch_size, pre_n, device=device)
        self.post_trace = torch.zeros(batch_size, post_n, device=device)

    def step(self, pre_spikes, post_spikes):
        """
        Update traces for one timestep.

        pre_spikes: (batch, pre_n)
        post_spikes: (batch, post_n)
        """
        # Decay and update spike traces
        self.pre_trace = self.beta_plus * self.pre_trace + pre_spikes
        self.post_trace = self.beta_minus * self.post_trace + post_spikes

        # STDP update to eligibility trace:
        # pre before post -> potentiation (pre_trace * post_spike)
        # post before pre -> depression (post_trace * pre_spike)
        pre_t_3d = self.pre_trace.unsqueeze(1)    # (batch, 1, pre_n)
        post_s_3d = post_spikes.unsqueeze(2)      # (batch, post_n, 1)
        post_t_3d = self.post_trace.unsqueeze(2)  # (batch, post_n, 1)
        pre_s_3d = pre_spikes.unsqueeze(1)         # (batch, 1, pre_n)

        pot = self.a_plus * pre_t_3d * post_s_3d
        dep = -self.a_minus * post_t_3d * pre_s_3d

        stdp_delta = pot + dep  # (batch, post_n, pre_n)

        # Decay and accumulate eligibility
        self.eligibility = self.beta_elig * self.eligibility + stdp_delta

    def apply_reward(self, reward):
        """
        Apply reward to update weights.

        reward: scalar tensor or (batch,) tensor
        """
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        # Average over batch
        delta_w = self.lr * reward.mean() * self.eligibility.mean(dim=0)
        with torch.no_grad():
            self.synapse.weight.data.add_(delta_w)
            # Keep weights bounded
            self.synapse.weight.data.clamp_(-2.0, 2.0)


class PureSNN(nn.Module):
    """
    Pure spiking neural network.
    Timestep-by-timestep forward pass through all layers.
    """
    def __init__(self, layer_sizes, dt=1.0, weight_scale=0.1,
                 tau_m=20.0, tau_syn=5.0):
        """
        layer_sizes: [input_size, hidden1, ..., output_size]
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.dt = dt
        self.n_layers = len(layer_sizes) - 1

        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()

        for i in range(self.n_layers):
            pre_n = layer_sizes[i]
            post_n = layer_sizes[i + 1]
            self.synapses.append(
                Synapse(pre_n, post_n, weight_scale=weight_scale, tau_syn=tau_syn, dt=dt)
            )
            self.neurons.append(
                LIFNeuron(post_n, tau_m=tau_m, dt=dt)
            )

        # Plasticity rules (attached after construction)
        self.plasticities = []

    def add_plasticity(self, lr=0.001, tau_elig=20.0,
                       a_plus=0.01, a_minus=0.01,
                       tau_plus=20.0, tau_minus=20.0):
        """Attach R-STDP plasticity to all synapses."""
        self.plasticities = []
        for syn in self.synapses:
            p = RSTDPPlasticity(
                syn, lr=lr, tau_elig=tau_elig, dt=self.dt,
                a_plus=a_plus, a_minus=a_minus,
                tau_plus=tau_plus, tau_minus=tau_minus
            )
            self.plasticities.append(p)

    def forward(self, input_spikes, return_all=False):
        """
        input_spikes: (batch, input_size, timesteps)

        Returns: output_spikes (batch, output_size, timesteps)
        """
        batch_size = input_spikes.shape[0]
        timesteps = input_spikes.shape[-1]
        device = input_spikes.device

        # Reset plasticity traces
        for p in self.plasticities:
            p.reset(batch_size, device)

        # Layer states
        syn_current = [None] * self.n_layers
        v = [None] * self.n_layers

        # Store all layer outputs if needed
        if return_all:
            all_layer_outputs = [[] for _ in range(self.n_layers)]

        output_spikes = torch.zeros(batch_size, self.layer_sizes[-1], timesteps, device=device)

        # Timestep-by-timestep
        for t in range(timesteps):
            # Input to first layer
            layer_in = input_spikes[..., t]  # (batch, input_size)

            for i in range(self.n_layers):
                # Synapse
                syn_current[i] = self.synapses[i](layer_in, syn_current[i])
                # Neuron
                spikes, v[i] = self.neurons[i](syn_current[i], v[i])

                # Plasticity step
                if self.plasticities:
                    self.plasticities[i].step(layer_in, spikes)

                # Store for return
                if return_all:
                    all_layer_outputs[i].append(spikes)

                # Next layer input
                layer_in = spikes

            # Store output
            output_spikes[..., t] = layer_in

        if return_all:
            # Convert lists to tensors
            all_tensors = []
            for layer_out in all_layer_outputs:
                all_tensors.append(torch.stack(layer_out, dim=-1))
            return output_spikes, all_tensors

        return output_spikes

    def get_frozen_weights(self):
        """Deep copy of all weights for freeze verification."""
        return [syn.weight.data.clone() for syn in self.synapses]

    def restore_frozen_weights(self, frozen):
        """Restore weights from frozen copy."""
        for syn, fw in zip(self.synapses, frozen):
            syn.weight.data.copy_(fw)

    def verify_freeze(self, frozen):
        """Verify freeze is active. Returns (ok, per_layer_diffs)."""
        diffs = []
        for i, (syn, fw) in enumerate(zip(self.synapses, frozen)):
            diff = (syn.weight.data - fw).abs().max().item()
            diffs.append(diff)
        return max(diffs) < 1e-10, diffs

    def weight_norm(self):
        """L2 norm of all weights."""
        return math.sqrt(sum(syn.weight.data.norm().item() ** 2 for syn in self.synapses))
