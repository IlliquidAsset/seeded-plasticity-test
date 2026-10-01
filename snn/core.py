"""
Pure spiking neural network core.
LIF neurons, current-based synapses, R-STDP learning.
Timestep-by-timestep forward pass.

No transformers. No ANN-to-SNN conversion. No seeded weights.
Built from scratch.

Reward schedules (two clearly separated modes):

* ``REWARD_END_OF_TRIAL`` (legacy, Track 1-3): ``PureSNN.forward`` runs a
  whole trial with traces reset at its start; the caller then delivers ONE
  scalar reward per sample via ``RSTDPPlasticity.apply_reward``, which
  multiplies whatever eligibility is left at the end of the trial.
* ``REWARD_PER_SPIKE_NEXT_STEP`` (Florian 2007, sec. 4.2): continuous online
  state via ``PureSNN.reset_online_state`` + ``PureSNN.online_step``; a
  reward computed from the output spikes of step t is applied on the
  transition t -> t+dt, after the eligibility update that includes step t's
  pairings and before the membranes integrate step t's spikes.
"""

import torch
import torch.nn as nn
import math

REWARD_END_OF_TRIAL = "end_of_trial"
REWARD_PER_SPIKE_NEXT_STEP = "per_spike_next_step"
REWARD_MODES = (REWARD_END_OF_TRIAL, REWARD_PER_SPIKE_NEXT_STEP)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.
    Processes one timestep at a time.
    """
    def __init__(self, n_neurons, tau_m=20.0, v_thresh=1.0, v_rest=0.0, v_reset=0.0, dt=1.0):
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
        I_syn: (batch, n_neurons) - postsynaptic current
        v: (batch, n_neurons) or None for reset

        Returns: spikes (batch, n_neurons), new v (batch, n_neurons)
        """
        if v is None:
            v = torch.full_like(I_syn, self.v_rest)

        # Leaky integrate: v decays toward rest, adds current
        v = self.v_rest + self.beta * (v - self.v_rest) + I_syn
        # Fire if above threshold
        fired = (v >= self.v_thresh).to(v.dtype)
        # Soft reset
        v = torch.where(fired > 0, torch.full_like(v, self.v_reset), v)

        return fired, v


class Synapse(nn.Module):
    """
    Synaptic connection between two populations.
    Current-based with exponential decay, one timestep at a time.

    Supports optional connection mask for structural plasticity.
    When use_mask=True, effective weight = weight * connection_mask.
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

        # Connection mask for structural plasticity (1.0 = connected)
        # Initially all ones = fully connected (classic dense mode)
        self.register_buffer('connection_mask', torch.ones(post_n, pre_n))
        self.use_mask = False  # toggle to activate masking

    def get_effective_weight(self):
        """Return weight * mask when masking is active, else plain weight."""
        if self.use_mask:
            return self.weight * self.connection_mask
        return self.weight

    def forward(self, pre_spikes, I_syn=None):
        """
        One timestep.
        pre_spikes: (batch, pre_n)
        I_syn: (batch, post_n) or None

        Returns: I_syn_new (batch, post_n)
        """
        if I_syn is None:
            I_syn = torch.zeros(pre_spikes.shape[0], self.post_n, device=pre_spikes.device,
                                dtype=self.weight.dtype)

        # Decay
        I_syn = self.beta_syn * I_syn
        # Add current from spikes, using effective weight (masked if active)
        effective_w = self.get_effective_weight()
        I_syn = I_syn + torch.mm(pre_spikes.to(effective_w.dtype), effective_w.t())

        return I_syn


class RSTDPPlasticity:
    """
    Reward-modulated STDP learning rule.
    Tracks eligibility trace per synapse, modulated by global reward.

    Δw_ij = η * R * e_ij
    e_ij = eligibility trace (low-pass filtered STDP signal)

    ``credit`` selects what the reward multiplies:
      ``"eligibility"`` (default, Florian MSTDPET): the decaying trace e_ij.
      ``"pairing"`` (Florian MSTDP, eq. 3.9): the instantaneous STDP pairing
      term of the most recent ``step`` (no trace).
    ``w_min`` / ``w_max`` are floats or tensors broadcastable to the weight
    (sign-specific bounds); the defaults keep the legacy clamp [-2, 2].
    """
    CREDIT_MODES = ("eligibility", "pairing")

    def __init__(self, synapse, lr=0.001, tau_elig=1000.0, dt=1.0,
                 a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0,
                 w_min=-2.0, w_max=2.0, credit="eligibility"):
        if credit not in self.CREDIT_MODES:
            raise ValueError(f"credit must be one of {self.CREDIT_MODES}")
        self.synapse = synapse
        self.lr = lr
        self.tau_elig = tau_elig
        self.dt = dt
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        self.credit = credit
        self.beta_elig = math.exp(-dt / tau_elig) if tau_elig > 0 else 0.0
        self.beta_plus = math.exp(-dt / tau_plus) if tau_plus > 0 else 0.0
        self.beta_minus = math.exp(-dt / tau_minus) if tau_minus > 0 else 0.0

    def reset(self, batch_size, device, dtype=None):
        """Reset traces for a new trial (dtype defaults to the weight dtype)."""
        post_n, pre_n = self.synapse.weight.shape
        dtype = dtype or self.synapse.weight.dtype
        self.eligibility = torch.zeros(batch_size, post_n, pre_n, device=device, dtype=dtype)
        self.pre_trace = torch.zeros(batch_size, pre_n, device=device, dtype=dtype)
        self.post_trace = torch.zeros(batch_size, post_n, device=device, dtype=dtype)
        self.last_pairing = torch.zeros(batch_size, post_n, pre_n, device=device, dtype=dtype)

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
        self.last_pairing = stdp_delta

        # Decay and accumulate eligibility
        self.eligibility = self.beta_elig * self.eligibility + stdp_delta

    def credit_signal(self):
        """The quantity the reward multiplies (see ``credit``)."""
        return self.eligibility if self.credit == "eligibility" else self.last_pairing

    def apply_reward(self, reward):
        """
        Apply reward to update weights.

        reward: scalar tensor or (batch,) tensor
        """
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        # Per-sample reward x eligibility, then average over the batch.
        # reward: (batch,) -> (batch, 1, 1); eligibility: (batch, post, pre)
        r_expanded = reward.reshape(-1, 1, 1)
        delta_w = self.lr * (r_expanded * self.credit_signal()).mean(dim=0)
        with torch.no_grad():
            self.synapse.weight.data.add_(delta_w)
            # Keep weights bounded
            if torch.is_tensor(self.w_min) or torch.is_tensor(self.w_max):
                self.synapse.weight.data.copy_(
                    torch.clamp(self.synapse.weight.data, min=self.w_min, max=self.w_max)
                )
            else:
                self.synapse.weight.data.clamp_(self.w_min, self.w_max)


class PureSNN(nn.Module):
    """
    Pure spiking neural network.
    Timestep-by-timestep forward pass through all layers.
    """
    def __init__(self, layer_sizes, dt=1.0, weight_scale=0.1,
                 tau_m=20.0, tau_syn=5.0, v_thresh=1.0, v_rest=0.0, v_reset=0.0):
        """
        layer_sizes: [input_size, hidden1, ..., output_size]
        tau_syn=0 gives a direct voltage jump of w on the integrating step
        (no synaptic filter), as in Florian 2007 eq. 4.1.
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
                LIFNeuron(post_n, tau_m=tau_m, v_thresh=v_thresh,
                          v_rest=v_rest, v_reset=v_reset, dt=dt)
            )

        # Plasticity rules (attached after construction)
        self.plasticities = []

    def add_plasticity(self, lr=0.001, tau_elig=1000.0,
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

    # ------------------------------------------------------------------
    # Reward mode REWARD_PER_SPIKE_NEXT_STEP: continuous online simulation.
    # ------------------------------------------------------------------
    def reset_online_state(self, batch_size=1, device=None, dtype=None):
        """Start a continuous online simulation (membranes at rest, no spikes).

        Unlike ``forward``, state then persists across ``online_step`` calls
        and across patterns; nothing is reset between trials.
        """
        w0 = self.synapses[0].weight
        device = device or w0.device
        dtype = dtype or w0.dtype
        self._v = [torch.full((batch_size, n.n_neurons), n.v_rest, device=device, dtype=dtype)
                   for n in self.neurons]
        self._spikes = [torch.zeros(batch_size, n.n_neurons, device=device, dtype=dtype)
                        for n in self.neurons]
        for p in self.plasticities:
            p.reset(batch_size, device, dtype)

    def current_output(self):
        """Output-layer spikes f_out(t) of the current step (before advancing)."""
        return self._spikes[-1]

    def online_step(self, input_spikes, reward_fn=None):
        """One dt transition t -> t+dt in Florian's discrete order (eq. 2.7-2.8, 4.1).

        input_spikes: (batch, input_size) spikes at step t.
        reward_fn: optional callable(output_spikes_t) -> float; its value is
            the reward delivered on this transition (per-spike, next step).

        Order within the transition:
          1. traces and eligibility take in all pairings at step t;
          2. reward r(t+dt) = reward_fn(f_out(t)) multiplies the updated credit;
          3. each layer integrates its presynaptic spikes of step t through
             the just-updated weights (one discrete step per layer) and emits
             f(t+dt).
        Returns (output_spikes_t, reward_delivered).
        """
        pres = [input_spikes] + self._spikes[:-1]
        for i, p in enumerate(self.plasticities):
            p.step(pres[i], self._spikes[i])
        out_t = self._spikes[-1]
        reward = 0.0
        if reward_fn is not None:
            reward = float(reward_fn(out_t))
            if reward != 0.0 and self.plasticities:
                r = torch.tensor(reward, dtype=out_t.dtype, device=out_t.device)
                for p in self.plasticities:
                    p.apply_reward(r)
        new_spikes = []
        for i in range(self.n_layers):
            current = self.synapses[i](pres[i], None)
            spikes, self._v[i] = self.neurons[i](current, self._v[i])
            new_spikes.append(spikes)
        self._spikes = new_spikes
        return out_t, reward

    # ------------------------------------------------------------------
    # Reward mode REWARD_END_OF_TRIAL (legacy): one trial, traces reset,
    # reward applied by the caller after the trial.
    # ------------------------------------------------------------------
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
