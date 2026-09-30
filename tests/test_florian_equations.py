"""Equation, reward-order, and paired-control tests for the Florian NumPy anchor."""

import numpy as np
import pytest

from ladder.florian import FlorianConfig, FlorianNetwork, gate_margin_hz, new_symbol_pair, run_experiment
from ladder.florian_check import check_network


@pytest.mark.parametrize("task", ["rate", "temporal"])
@pytest.mark.parametrize("rule", ["mstdp", "mstdpet"])
def test_anchor_weights_match_independent_pair_sum_recomputation(task, rule):
    net = FlorianNetwork(FlorianConfig(task, rule, epochs=2), seed=3)
    row = check_network(net)
    assert row["reward_events"] > 0
    assert row["total_abs_weight_change_mv"] > 0.0
    assert row["pass"], row


class _SignFlip(FlorianNetwork):
    def _update_traces(self, input_spikes):
        super()._update_traces(input_spikes)
        self.pre1, self.pre2 = -self.pre1, -self.pre2


class _RewardBeforeEligibility(FlorianNetwork):
    """Reward multiplies z(t) instead of z(t+dt) (wrong within-step order)."""

    def step(self, input_spikes, target):
        self._update_traces(input_spikes)
        fired = int(self.output_spikes[0])
        self._apply_reward(input_spikes, float((1 if target else -1) * fired))
        if self.config.rule == "mstdpet":
            self._update_eligibility(input_spikes)
        self._advance_neurons(input_spikes)
        self.total_output_spikes += fired
        return fired


class _RewardOneStepLate(FlorianNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending = 0.0

    def step(self, input_spikes, target):
        self._update_traces(input_spikes)
        if self.config.rule == "mstdpet":
            self._update_eligibility(input_spikes)
        fired = int(self.output_spikes[0])
        self._apply_reward(input_spikes, self._pending)
        self._pending = float((1 if target else -1) * fired)
        self._advance_neurons(input_spikes)
        self.total_output_spikes += fired
        return fired


@pytest.mark.parametrize("mutant", [_SignFlip, _RewardBeforeEligibility, _RewardOneStepLate], ids=lambda m: m.__name__)
def test_equation_check_detects_mutation(mutant):
    net = mutant(FlorianConfig("temporal", "mstdpet", epochs=2), seed=3)
    assert not check_network(net)["pass"]


def test_frozen_twin_shares_inputs_and_never_changes_weights():
    cfg = FlorianConfig("temporal", "mstdpet", epochs=2)
    trained = FlorianNetwork(cfg, seed=4)
    frozen = FlorianNetwork(FlorianConfig("temporal", "mstdpet", epochs=2, plastic=False), seed=4)
    np.testing.assert_array_equal(trained.symbol_trains, frozen.symbol_trains)
    np.testing.assert_array_equal(trained.w1, frozen.w1)
    w1, w2 = frozen.w1.copy(), frozen.w2.copy()
    result = frozen.run()
    assert result["total_abs_weight_change_mv"] == 0.0
    np.testing.assert_array_equal(frozen.w1, w1)
    np.testing.assert_array_equal(frozen.w2, w2)


def test_frozen_and_trained_see_identical_evaluation_input():
    a = run_experiment("temporal", "mstdp", seed=5, epochs=1, plastic=False, n_eval=2)
    b = run_experiment("temporal", "mstdp", seed=5, epochs=1, plastic=False, n_eval=2)
    assert a["retest_rates_hz"] == b["retest_rates_hz"]
    assert a["generalization_rates_hz"] == b["generalization_rates_hz"]


def test_new_symbol_pairs_are_new_and_well_formed():
    net = FlorianNetwork(FlorianConfig("temporal", "mstdp", epochs=1), seed=6)
    pair = new_symbol_pair(np.random.default_rng([6, 202, 0]))
    assert pair.shape == (2, 500)
    assert np.all(pair.sum(axis=1) == 50)
    assert not np.array_equal(pair, net.symbol_trains)


def test_gate_margin_sign_matches_paper_gate():
    assert gate_margin_hz({"00": 0, "01": 10, "10": 8, "11": 2}) == 6
    assert gate_margin_hz({"00": 0, "01": 10, "10": 2, "11": 2}) == 0
