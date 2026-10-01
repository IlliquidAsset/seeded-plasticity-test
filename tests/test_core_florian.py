"""Core-substrate Florian reward mode: parity, ordering, controls, legacy preservation."""

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import snn.core as core
from snn.core import REWARD_END_OF_TRIAL, REWARD_MODES, REWARD_PER_SPIKE_NEXT_STEP, PureSNN, RSTDPPlasticity, Synapse
from snn.florian_bench import CoreFlorianBench, CoreFlorianConfig, run_core_experiment
from snn.florian_parity import parity

ROOT = Path(__file__).resolve().parents[1]


def test_both_reward_modes_are_named_and_distinct():
    assert REWARD_MODES == (REWARD_END_OF_TRIAL, REWARD_PER_SPIKE_NEXT_STEP)
    with pytest.raises(ValueError):
        CoreFlorianConfig("rate", "mstdp", reward_mode=REWARD_END_OF_TRIAL)


def _legacy_run(mod):
    torch.manual_seed(7)
    net = mod.PureSNN([8, 16, 4], dt=1.0, weight_scale=2.0, tau_m=20.0, tau_syn=10.0)
    net.add_plasticity(lr=0.05, tau_elig=25.0, a_plus=0.02, a_minus=0.015)
    outs = []
    for _ in range(5):
        x = (torch.rand(3, 8, 100) < 0.1).float()
        outs.append(net(x))
        for p in net.plasticities:
            p.apply_reward(torch.tensor([1.0, -1.0, 0.5]))
    return torch.cat([s.weight.data.flatten() for s in net.synapses]), torch.cat([o.flatten() for o in outs])


def test_end_of_trial_mode_bit_identical_to_pre_change_core():
    """The legacy path must be untouched: compare against snn/core.py at 490cb1c."""
    try:
        src = subprocess.check_output(["git", "show", "490cb1c:snn/core.py"], cwd=ROOT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git history unavailable")
    spec = importlib.util.spec_from_loader("core_490cb1c", loader=None)
    old = importlib.util.module_from_spec(spec)
    exec(src, old.__dict__)
    w_old, o_old = _legacy_run(old)
    w_new, o_new = _legacy_run(core)
    assert w_new.dtype == torch.float32
    assert torch.equal(o_old, o_new)
    assert torch.equal(w_old, w_new)


def test_legacy_defaults_unchanged():
    p = RSTDPPlasticity(Synapse(2, 2))
    assert (p.w_min, p.w_max, p.credit) == (-2.0, 2.0, "eligibility")
    net = PureSNN([2, 3, 1])
    assert net.neurons[0].v_thresh == 1.0 and net.neurons[0].v_rest == 0.0


def test_pairing_credit_uses_instantaneous_term_only():
    syn = Synapse(1, 1, weight_scale=0.0, tau_syn=0.0)
    p = RSTDPPlasticity(syn, lr=1.0, tau_elig=25.0, a_plus=1.0, a_minus=1.0, credit="pairing", w_min=-9, w_max=9)
    p.reset(1, syn.weight.device)
    one, zero = torch.ones(1, 1), torch.zeros(1, 1)
    p.step(one, zero)
    p.step(zero, one)  # pre 1 ms before post: zeta = exp(-1/20)
    p.apply_reward(torch.tensor(1.0))
    assert syn.weight.item() == pytest.approx(np.exp(-1 / 20), rel=1e-6)
    p.step(zero, zero)  # no pairing now: pairing credit is zero even though eligibility is not
    p.apply_reward(torch.tensor(1.0))
    assert syn.weight.item() == pytest.approx(np.exp(-1 / 20), rel=1e-6)
    assert p.eligibility.abs().item() > 0


def test_tensor_bounds_are_sign_specific():
    syn = Synapse(2, 1, weight_scale=0.0)
    p = RSTDPPlasticity(syn, lr=100.0, w_min=torch.tensor([[0.0, -5.0]]), w_max=torch.tensor([[5.0, 0.0]]))
    p.reset(1, syn.weight.device)
    p.eligibility.fill_(1.0)
    p.apply_reward(torch.tensor(1.0))
    assert syn.weight.tolist() == [[5.0, 0.0]]
    p.apply_reward(torch.tensor(-1.0))
    assert syn.weight.tolist() == [[0.0, -5.0]]


def test_online_reward_lands_on_next_step_after_eligibility_update():
    """Output spike at step t -> reward on the t->t+1 transition, using z(t+dt)."""
    net = PureSNN([1, 1], tau_syn=0.0, v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0).double()
    with torch.no_grad():
        net.synapses[0].weight.fill_(20.0)  # one input spike -> one output spike next step
    net.plasticities = [RSTDPPlasticity(net.synapses[0], lr=1.0, tau_elig=25.0, a_plus=1 / 25, a_minus=1 / 25,
                                        w_min=0.0, w_max=100.0)]
    net.reset_online_state(1, dtype=torch.float64)
    one = torch.ones(1, 1, dtype=torch.float64)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    seen = []
    out, r = net.online_step(one, reward_fn=lambda f: float(f[0, 0]))  # t=0: pre spike
    seen.append((float(out), r))
    out, r = net.online_step(zero, reward_fn=lambda f: float(f[0, 0]))  # t=1: output spike, reward 0 so far
    seen.append((float(out), r))
    w_before = net.synapses[0].weight.item()
    out, r = net.online_step(zero, reward_fn=lambda f: float(f[0, 0]))  # t=1->2: reward delivered
    seen.append((float(out), r))
    # Output spike appears at step 1 (one discrete step after the input spike);
    # its reward is delivered on the 1 -> 2 transition, never earlier or later.
    assert seen == [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    # Pairing at step 1 (pre trace exp(-1/20), post spike) enters z BEFORE the
    # reward multiplies it: w = 20 + 1 * exp(-1/20)/25.  Reward-before-update
    # would give 20 exactly; a one-step-late reward would give exp(-1/20)*exp(-1/25)/25.
    expected = 20.0 + np.exp(-1 / 20) / 25.0
    assert w_before == pytest.approx(expected, rel=1e-12)
    assert net.synapses[0].weight.item() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("task", ["rate", "temporal"])
@pytest.mark.parametrize("rule", ["mstdp", "mstdpet"])
def test_core_matches_independent_anchor_exactly(task, rule):
    """G0(b,c): spike-for-spike identical output and weights within 1e-9 mV over 2 epochs."""
    row = parity(task, rule, seed=3, epochs=2)
    assert row["core_output_spikes"] > 0
    assert row["output_train_identical"], row
    assert row["max_abs_w1_diff_mv"] < 1e-9 and row["max_abs_w2_diff_mv"] < 1e-9, row


def test_core_frozen_twin_never_changes_weights_and_matches_anchor_frozen():
    row = parity("temporal", "mstdpet", seed=4, epochs=2, plastic=False)
    assert row["output_train_identical"]
    bench = CoreFlorianBench(CoreFlorianConfig("temporal", "mstdpet", epochs=2, plastic=False), seed=4)
    w1, w2 = bench.weights()
    res = bench.run()
    assert res["total_abs_weight_change_mv"] == 0.0
    np.testing.assert_array_equal(bench.weights()[0], w1)
    np.testing.assert_array_equal(bench.weights()[1], w2)
    assert bench.reward_events == []


def test_reward_sign_and_signed_dw_correlation_recorded():
    row = run_core_experiment("temporal", "mstdpet", seed=2, epochs=2, n_eval=0)
    assert row["reward_events"] == row["reward_events_positive"] + row["reward_events_negative"]
    assert row["reward_events"] > 0
    assert row["reward_signed_dw_out_corr"] is not None


def test_core_bench_does_not_import_ladder():
    src = (ROOT / "snn" / "florian_bench.py").read_text() + (ROOT / "snn" / "core.py").read_text()
    assert "ladder" not in src.replace("does not import ``ladder``", "")
