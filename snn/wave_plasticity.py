"""
Wave-gated eligibility plasticity.

Implements WaveGatedPlasticity, a subclass of RSTDPPlasticity that gates a
slow eligibility trace by a rectified cosine of an 8 Hz theta phase at
reward time.  The fast eligibility trace is left untouched so the wave rule
remains a pure additive gating mechanism.
"""

import math
import random
import torch

from snn.core import RSTDPPlasticity


class PhaseOscillator:
    """
    Pure phase reference.  No current injection; only a phase phi(t) that is
    read at reward time.
    """

    def __init__(self, frequency=8.0, phi_0=0.0):
        self.frequency = frequency
        self.phi_0 = phi_0
        self.t = 0.0

    def reset(self, phi_0=None):
        """Reset time to zero and optionally set a new initial phase."""
        if phi_0 is not None:
            self.phi_0 = phi_0
        self.t = 0.0

    def step(self, dt=1.0):
        """Advance time by one simulation step (dt in ms)."""
        self.t += dt

    def phase_at_reward(self):
        """
        Return the phase at the current reward time in radians.
        Time is in ms, frequency is in Hz, so t is converted to seconds.
        """
        t_seconds = self.t / 1000.0
        return 2.0 * math.pi * self.frequency * t_seconds + self.phi_0


class WaveGatedPlasticity(RSTDPPlasticity):
    """
    Reward-modulated STDP with a theta-wave gate on a slow eligibility trace.

    Inherits RSTDPPlasticity without modifying the base class.  Maintains:
      - e_fast: the inherited eligibility trace (tau = tau_elig, default 20 ms)
      - e_slow: a second eligibility trace with tau = tau_elig_slow (500 ms)

    At reward time the slow trace is gated by max(0, cos(phi_reward)) and the
    update is applied per-sample then averaged over the batch.
    """

    def __init__(
        self,
        synapse,
        lr=0.001,
        tau_elig=20.0,
        dt=1.0,
        a_plus=0.01,
        a_minus=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
        tau_elig_slow=500.0,
        frequency=8.0,
        phi_0=0.0,
        scramble_phase=False,
        phase_seed=None,
        record=True,
    ):
        super().__init__(
            synapse,
            lr=lr,
            tau_elig=tau_elig,
            dt=dt,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
        )
        self.tau_elig_slow = tau_elig_slow
        self.beta_elig_slow = math.exp(-dt / tau_elig_slow) if tau_elig_slow > 0 else 0.0
        self.frequency = frequency
        self.phi_0 = phi_0
        self.scramble_phase = scramble_phase
        # Isolated RNG prevents phase-control draws from perturbing task/input
        # randomness.  Identical phase_seed values synchronize all layers.
        self._phase_rng = random.Random(phase_seed)
        self.oscillator = PhaseOscillator(frequency=frequency, phi_0=phi_0)
        self.record = record
        self.reset_recording()

    def reset_recording(self):
        """Clear instrumentation buffers."""
        self._records = []
        self._phase_history = []
        self._gate_history = []
        self._degenerate_gate_count = 0
        self._total_update_count = 0
        self._trial_peak_abs_elig = 0.0

    def reset(self, batch_size, device):
        """Reset fast and slow traces and the oscillator for a new trial."""
        super().reset(batch_size, device)
        post_n, pre_n = self.synapse.weight.shape
        self.e_slow = torch.zeros(batch_size, post_n, pre_n, device=device)

        # Per-trial random initial phase if requested.
        if self.scramble_phase:
            phi_0 = self._phase_rng.random() * 2.0 * math.pi
        else:
            phi_0 = self.phi_0
        self.oscillator.reset(phi_0=phi_0)
        self._trial_peak_abs_elig = 0.0

    def step(self, pre_spikes, post_spikes):
        """
        Update both fast and slow eligibility traces for one timestep.

        Fast trace is updated by the base class.  Slow trace sees the same
        STDP pairing but with a longer time constant.
        """
        # Fast trace (inherited eligibility buffer)
        super().step(pre_spikes, post_spikes)

        # Recompute the same STDP delta using the updated spike traces.
        # These are the same causal terms used by the base class.
        pre_t_3d = self.pre_trace.unsqueeze(1)    # (batch, 1, pre)
        post_s_3d = post_spikes.unsqueeze(2)      # (batch, post, 1)
        post_t_3d = self.post_trace.unsqueeze(2)  # (batch, post, 1)
        pre_s_3d = pre_spikes.unsqueeze(1)         # (batch, 1, pre)

        pot = self.a_plus * pre_t_3d * post_s_3d
        dep = -self.a_minus * post_t_3d * pre_s_3d
        stdp_delta = pot + dep  # (batch, post, pre)

        # Slow trace with independent decay constant
        self.e_slow = self.beta_elig_slow * self.e_slow + stdp_delta
        self.oscillator.step(self.dt)

        # Track peak slow eligibility for instrumentation.
        with torch.no_grad():
            peak = self.e_slow.abs().max().item()
            if peak > self._trial_peak_abs_elig:
                self._trial_peak_abs_elig = peak

    def apply_reward(self, reward):
        """
        Apply reward gated by the rectified cosine of the phase at reward time.

        reward: scalar tensor or (batch,) tensor
        """
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

        phi_reward = self.oscillator.phase_at_reward()
        cosine = math.cos(phi_reward)
        # Numerical cos(pi / 2) is a tiny positive value rather than exact zero.
        # The frozen spec defines that boundary as gate == 0.
        gate = 0.0 if cosine <= 1e-15 else cosine

        # Per-sample product, then mean over batch (Revision B semantics).
        r_expanded = reward.reshape(-1, 1, 1)  # (batch, 1, 1)
        scaled = r_expanded * gate * self.e_slow
        delta_w = self.lr * scaled.mean(dim=0)  # (post, pre)

        with torch.no_grad():
            self.synapse.weight.data.add_(delta_w)
            self.synapse.weight.data.clamp_(-2.0, 2.0)

        if self.record:
            self._total_update_count += 1
            if gate == 0.0:
                self._degenerate_gate_count += 1
            self._phase_history.append(phi_reward)
            self._gate_history.append(gate)
            mean_abs_delta = delta_w.abs().mean().item()
            mean_abs_elig = self.e_slow.abs().mean().item()
            max_abs_elig = max(self.e_slow.abs().max().item(), self._trial_peak_abs_elig, 1e-12)
            pct_of_peak = 100.0 * mean_abs_elig / max_abs_elig

            self._records.append({
                "mean_abs_eligibility": mean_abs_elig,
                "max_abs_eligibility": max_abs_elig,
                "eligibility_at_reward_pct_of_peak": pct_of_peak,
                "gate": gate,
                "phase_at_reward": phi_reward,
                "mean_reward": reward.mean().item(),
                "mean_abs_weight_delta": mean_abs_delta,
            })

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def get_wave_metrics(self):
        """Return a dict of wave-gate diagnostics."""
        if self._total_update_count == 0:
            return {
                "fraction_updates_gate_zero": 0.0,
                "n_updates": 0,
                "mean_gate": 0.0,
                "std_gate": 0.0,
            }

        gates = self._gate_history
        n_zero = sum(1.0 for g in gates if g == 0.0)
        mean_gate = sum(gates) / len(gates)
        if len(gates) > 1:
            var = sum((g - mean_gate) ** 2 for g in gates) / len(gates)
            std_gate = math.sqrt(var)
        else:
            std_gate = 0.0

        return {
            "fraction_updates_gate_zero": n_zero / self._total_update_count,
            "n_updates": self._total_update_count,
            "mean_gate": mean_gate,
            "std_gate": std_gate,
        }

    def get_liveness_metrics(self):
        """Return eligibility/reward diagnostics (Track 2 instrumentation)."""
        if len(self._records) < 2:
            return {
                "n_trials_recorded": len(self._records),
                "mean_abs_eligibility": 0.0,
                "eligibility_at_reward_pct_of_peak": 0.0,
                "reward_weight_corr": 0.0,
            }

        mean_abs_eligs = [r["mean_abs_eligibility"] for r in self._records]
        pcts = [r["eligibility_at_reward_pct_of_peak"] for r in self._records]
        rewards = [r["mean_reward"] for r in self._records]
        deltas = [r["mean_abs_weight_delta"] for r in self._records]

        def pearson(x, y):
            n = len(x)
            if n < 2:
                return 0.0
            mx = sum(x) / n
            my = sum(y) / n
            num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
            sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
            sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
            if sx == 0 or sy == 0:
                return 0.0
            return num / (sx * sy)

        return {
            "n_trials_recorded": len(self._records),
            "mean_abs_eligibility": sum(mean_abs_eligs) / len(mean_abs_eligs),
            "eligibility_at_reward_pct_of_peak": sum(pcts) / len(pcts),
            "reward_weight_corr": pearson(rewards, deltas),
        }
