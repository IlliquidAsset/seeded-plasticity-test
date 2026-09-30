"""Discriminating tests for the executed ladder (rungs 1-3).

The first block proves the simulator and the independent checker agree.  The
mutation block proves the checker is not a tautology: each intentional sign,
delay, reward-order, or connectivity bug in the simulator is detected.
"""

import math

import numpy as np
import pytest

from ladder import executed as ex
from ladder import handcheck as hc
from ladder.spiking import Simulator, poisson_raster


# --------------------------------------------------------------------------
# Intentional simulator mutations
# --------------------------------------------------------------------------

class SignFlipSTDP(Simulator):
    """A+ has the wrong sign: pre-before-post depresses."""

    def update_traces(self, f):
        self.p_plus = self.b_plus * self.p_plus - self.rule.a_plus * f
        self.p_minus = self.b_minus * self.p_minus + self.rule.a_minus * f


class LateReward(Simulator):
    """Reward lands one extra step late (off-by-one delay)."""

    def schedule(self, t, value, delay_ms):
        super().schedule(t, value, delay_ms + 1)


class RewardBeforeEligibility(Simulator):
    """Reward multiplies z(t) before the current pairing enters z (wrong order)."""

    def step(self, t, f, history, policy, delay_ms):
        self.update_traces(f)
        if policy is not None:
            self.schedule(t, float(policy(t, f, history)), delay_ms)
        self.apply_reward(self.deliver(t))
        self.update_eligibility(self.pairing(f))
        self.next_if_spikes = self.advance_neurons(f)


class CrossedWires(Simulator):
    """Membrane drive is routed through the other synapse's presynaptic neuron."""

    def advance_neurons(self, f):
        saved = self.pre.copy()
        self.pre = saved[::-1].copy()
        try:
            return super().advance_neurons(f)
        finally:
            self.pre = saved


class PlasticityLeaksToAll(Simulator):
    """Reward changes every synapse by the mean eligibility (no synapse specificity)."""

    def apply_reward(self, r):
        if not self.plastic or r == 0.0:
            return
        self.w = np.clip(self.w + self.rule.gamma_mv * r * self.z.mean(), self.low, self.high)


MUTANTS = [SignFlipSTDP, LateReward, RewardBeforeEligibility, CrossedWires, PlasticityLeaksToAll]


def _r2_short(sim_cls, seed=0, steps=5000):
    circuit = ex.r2_circuit()
    raster = poisson_raster(np.random.default_rng([77, seed]), steps, [ex.R2_A_HZ, ex.R2_B_HZ])
    spec = hc.PolicySpec("per_spike", neuron=2, value=1.0, require_prior=0)
    rec = ex.run_arm(circuit, raster, spec, sim_cls=sim_cls)
    return circuit, rec, spec


def test_correct_simulator_passes_independent_check():
    circuit, rec, spec = _r2_short(Simulator)
    mech = ex.check_mechanism(circuit, rec, spec)
    assert mech["reward_events"] > 20
    assert mech["pass"], mech
    assert mech["max_abs_weight_error_mv"] < 1e-9
    assert mech["spike_replay_mismatches"] == 0


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.__name__)
def test_checker_detects_intentional_mutation(mutant):
    circuit, rec, spec = _r2_short(mutant)
    mech = ex.check_mechanism(circuit, rec, spec)
    assert not mech["pass"], f"{mutant.__name__} escaped detection: {mech['max_abs_weight_error_mv']}"


@pytest.mark.parametrize("mutant", [SignFlipSTDP, LateReward, RewardBeforeEligibility])
def test_single_pairing_closed_form_detects_mutation(mutant):
    good = ex.r1_single_pairing(5)
    bad = ex.r1_single_pairing(5, sim_cls=mutant)
    assert good["abs_error_mv"] < 1e-12
    assert bad["abs_error_mv"] > 1e-4


def test_checker_rejects_wrong_reward_policy():
    """A reward schedule derived from the wrong neuron must not match."""
    circuit, rec, spec = _r2_short(Simulator)
    wrong = hc.PolicySpec("per_spike", neuron=2, value=1.0, require_prior=1)
    assert not ex.check_mechanism(circuit, rec, wrong)["pass"]


# --------------------------------------------------------------------------
# Closed-form facts computed by hand from recorded event times
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delay", ex.R1_DELAYS)
def test_single_pairing_matches_hand_formula(delay):
    row = ex.r1_single_pairing(delay)
    # Input at t=10 drives the output across threshold at t=11 (17 mV > 16 mV).
    assert row["output_spike_times"] == [11]
    hand = 0.2 * math.exp(-1 / 20) / 25 * math.exp(-delay / 25)
    assert row["observed_delta_mv"] == pytest.approx(hand, abs=1e-14)


def test_pair_sum_checker_matches_manual_arithmetic():
    c = hc.Constants(gamma=1.0)
    zeta = hc.pairing_events(np.array([3]), np.array([5, 9]), c)
    # post at 5 after pre at 3: +exp(-2/20); post at 9: +exp(-6/20).
    assert zeta[5] == pytest.approx(math.exp(-2 / 20))
    assert zeta[9] == pytest.approx(math.exp(-6 / 20))
    z = hc.eligibility_at([10], zeta, c)[0]
    manual = math.exp(-2 / 20) / 25 * math.exp(-4 / 25) + math.exp(-6 / 20) / 25
    assert z == pytest.approx(manual, rel=1e-12)


def test_silent_output_with_positive_reward_changes_nothing():
    circuit = ex.r1_circuit(w0=ex.SILENT_W_MV)
    raster = poisson_raster(np.random.default_rng(3), 10_000, [200.0])
    spec = hc.PolicySpec("per_silent_step", neuron=1, value=1.0)
    rec = ex.run_arm(circuit, raster, spec)
    assert rec.count(1) == 0
    assert rec.weights_final["in->out"] == ex.SILENT_W_MV
    assert ex.check_mechanism(circuit, rec, spec)["pass"]


def test_no_reward_is_exactly_frozen():
    circuit, rec, _ = _r2_short(Simulator)
    raster = rec.spikes[:, :2]
    none = ex.run_arm(circuit, raster, hc.PolicySpec("none"))
    frozen = ex.evaluate_frozen(circuit, {"A->out": ex.R2_W0, "B->out": ex.R2_W0}, raster, ex.RULE)
    assert none.weights_final == {"A->out": ex.R2_W0, "B->out": ex.R2_W0}
    np.testing.assert_array_equal(none.spikes, frozen.spikes)


def test_disconnected_output_ignores_disconnected_weights():
    circuit = ex.r3_disconnected()
    panel = poisson_raster(np.random.default_rng(5), 5000, [40.0, 40.0])
    a = ex.evaluate_frozen(circuit, {"A->C": 12.0, "D->E_active": 17.0, "D->F_silent": 0.5}, panel, ex.RULE)
    b = ex.evaluate_frozen(circuit, {"A->C": 12.0, "D->E_active": 3.0, "D->F_silent": 19.0}, panel, ex.RULE)
    np.testing.assert_array_equal(a.spikes[:, 2], b.spikes[:, 2])
    assert not np.array_equal(a.spikes[:, 3], b.spikes[:, 3])
