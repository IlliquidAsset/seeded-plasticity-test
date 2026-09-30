"""
Instrumentation for R-STDP bench runs.

Tracks eligibility liveness at reward time and computes reward-weight correlation.
This module intentionally does not change the learning rule; it only records and
reports diagnostics.
"""

import math
import torch

from snn.core import RSTDPPlasticity


class InstrumentedRSTDP(RSTDPPlasticity):
    """RSTDP with eligibility liveness and reward-weight diagnostics."""

    def __init__(self, *args, record=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.record = record
        self.reset_recording()

    def reset_recording(self):
        """Clear all recorded diagnostics."""
        self._records = []
        self._weight_before = None

    def reset(self, batch_size, device):
        super().reset(batch_size, device)
        self._trial_peak_abs_elig = 0.0

    def step(self, pre_spikes, post_spikes):
        super().step(pre_spikes, post_spikes)
        with torch.no_grad():
            peak = self.eligibility.abs().max().item()
            if peak > self._trial_peak_abs_elig:
                self._trial_peak_abs_elig = peak

    def apply_reward(self, reward):
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

        mean_abs_elig = self.eligibility.abs().mean().item()
        max_abs_elig = max(
            self.eligibility.abs().max().item(), self._trial_peak_abs_elig, 1e-12
        )
        pct_of_peak = 100.0 * mean_abs_elig / max_abs_elig

        if self.record:
            weight_before = self.synapse.weight.data.detach().clone()
            super().apply_reward(reward)
            weight_after = self.synapse.weight.data.detach()
            mean_abs_delta = (weight_after - weight_before).abs().mean().item()
            self._records.append({
                "mean_abs_eligibility": mean_abs_elig,
                "max_abs_eligibility": max_abs_elig,
                "eligibility_at_reward_pct_of_peak": pct_of_peak,
                "mean_reward": reward.mean().item(),
                "mean_abs_weight_delta": mean_abs_delta,
            })
        else:
            super().apply_reward(reward)

    def get_liveness_metrics(self):
        """Return a dict of eligibility/reward diagnostics."""
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
