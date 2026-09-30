"""
Unit tests for WaveGatedPlasticity.

Covers the frozen spec requirements:
  - gate function (rectified cosine, sign/zero behaviour)
  - slow-trace decay (tau = 500 ms)
  - phase-scramble draws per trial
  - per-sample update semantics
  - degenerate-gate counter
  - shakedown: wave does not disturb base STDP traces
"""

import math
import torch

from snn.core import PureSNN, RSTDPPlasticity, Synapse
from snn.instrumentation import InstrumentedRSTDP
from snn.wave_plasticity import WaveGatedPlasticity, PhaseOscillator
from run_wave_replicate import (
    TAU_ELIG_FAST,
    TAU_ELIG_ONLY,
    _attach_plasticity,
)


# ---------------------------------------------------------------------------
# PhaseOscillator
# ---------------------------------------------------------------------------

def test_phase_oscillator_phase_at_reward():
    """Gate phase increases linearly with time and respects phi_0."""
    osc = PhaseOscillator(frequency=8.0, phi_0=0.0)
    # After 100 ms, phase should be 2*pi*8*0.1 = 1.6*pi
    for _ in range(100):
        osc.step(1.0)
    phi = osc.phase_at_reward()
    assert math.isclose(phi, 1.6 * math.pi, abs_tol=1e-6)

    osc.reset(phi_0=math.pi / 2)
    assert osc.t == 0.0
    assert osc.phase_at_reward() == math.pi / 2


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def test_rectified_cosine_gate():
    """Gate is max(0, cos(phi)); zero for negative/zero cosine."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, phi_0=0.0, record=False)

    # Inject a non-zero slow eligibility trace.
    p.reset(batch_size=2, device=syn.weight.device)
    p.e_slow = torch.ones(2, 3, 2)

    # phi = 0 -> cos(0) = 1 -> gate = 1 -> weights change
    weight_before = syn.weight.data.clone()
    p.apply_reward(torch.tensor([1.0, 1.0]))
    assert not torch.allclose(syn.weight.data, weight_before, atol=1e-8)

    # phi = pi -> cos(pi) = -1 -> gate = 0 -> no change
    p.reset(batch_size=2, device=syn.weight.device)
    p.e_slow = torch.ones(2, 3, 2)
    # Manually set oscillator time so phase = pi at reward.
    p.oscillator.reset(phi_0=0.0)
    p.oscillator.t = 1000.0 * (math.pi / (2.0 * math.pi * 8.0))  # ms
    weight_before = syn.weight.data.clone()
    p.apply_reward(torch.tensor([1.0, 1.0]))
    assert torch.allclose(syn.weight.data, weight_before, atol=1e-8)

    # phi = pi/2 -> cos = 0 -> gate = 0 -> no change
    p.reset(batch_size=2, device=syn.weight.device)
    p.e_slow = torch.ones(2, 3, 2)
    p.oscillator.reset(phi_0=math.pi / 2)
    weight_before = syn.weight.data.clone()
    p.apply_reward(torch.tensor([1.0, 1.0]))
    assert torch.allclose(syn.weight.data, weight_before, atol=1e-8)


def test_phase_scramble_per_trial():
    """scramble_phase=True draws a new phi_0 on every reset."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, scramble_phase=True, phase_seed=1234, record=False)

    phases = []
    for _ in range(20):
        p.reset(batch_size=2, device=syn.weight.device)
        phases.append(p.oscillator.phi_0)

    assert len(set(phases)) > 1, "expected a distribution of initial phases"
    # All in [0, 2*pi)
    for phi in phases:
        assert 0.0 <= phi < 2.0 * math.pi

    # A second layer with the same phase seed receives the same per-trial
    # sequence, so the network has one coherent scrambled reference.
    p2 = WaveGatedPlasticity(syn, scramble_phase=True, phase_seed=1234, record=False)
    phases2 = []
    for _ in range(20):
        p2.reset(batch_size=2, device=syn.weight.device)
        phases2.append(p2.oscillator.phi_0)
    assert phases2 == phases


# ---------------------------------------------------------------------------
# Slow trace
# ---------------------------------------------------------------------------

def test_slow_trace_decay_constant():
    """Slow trace has the correct decay constant for tau=500 ms."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, tau_elig_slow=500.0, record=False)
    expected_beta = math.exp(-1.0 / 500.0)
    assert math.isclose(p.beta_elig_slow, expected_beta, rel_tol=1e-6)


def test_default_fast_trace_uses_frozen_20_ms_tau():
    """The inherited fast trace defaults to the frozen 20 ms constant."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, record=False)
    assert math.isclose(p.beta_elig, math.exp(-1.0 / 20.0), rel_tol=1e-6)


def test_runner_uses_distinct_fast_and_timescale_only_taus():
    """Wave arms use 20 ms fast traces; the no-wave control uses 1000 ms."""
    wave_net = PureSNN([2, 3], weight_scale=0.1)
    _attach_plasticity(wave_net, "wave-coherent", seed=42)
    wave_rule = wave_net.plasticities[0]
    assert isinstance(wave_rule, WaveGatedPlasticity)
    assert math.isclose(
        wave_rule.beta_elig, math.exp(-1.0 / TAU_ELIG_FAST), rel_tol=1e-6
    )

    scrambled_net = PureSNN([2, 3], weight_scale=0.1)
    _attach_plasticity(scrambled_net, "phase-scrambled", seed=42)
    scrambled_rule = scrambled_net.plasticities[0]
    assert isinstance(scrambled_rule, WaveGatedPlasticity)
    assert math.isclose(
        scrambled_rule.beta_elig, math.exp(-1.0 / TAU_ELIG_FAST), rel_tol=1e-6
    )

    control_net = PureSNN([2, 3], weight_scale=0.1)
    _attach_plasticity(control_net, "eligibility-timescale-only", seed=42)
    control_rule = control_net.plasticities[0]
    assert isinstance(control_rule, InstrumentedRSTDP)
    assert math.isclose(
        control_rule.beta_elig, math.exp(-1.0 / TAU_ELIG_ONLY), rel_tol=1e-6
    )


def test_slow_trace_is_slower_than_fast():
    """After the same stimulus, the slow trace decays more slowly."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, tau_elig=20.0, tau_elig_slow=500.0, record=False)

    p.reset(batch_size=1, device=syn.weight.device)
    pre = torch.zeros(1, 2)
    post = torch.zeros(1, 3)
    # Pre-before-post pairing creates a non-zero potentiation signal.
    pre[0, 0] = 1.0
    p.step(pre, post)
    pre.zero_()
    post[0, 0] = 1.0
    p.step(pre, post)
    # Then decay once with no new spikes.
    post.zero_()
    p.step(pre, post)

    fast = p.eligibility[0, 0, 0].item()
    slow = p.e_slow[0, 0, 0].item()
    assert slow > fast, "slow trace should retain more eligibility than fast trace after one step"


# ---------------------------------------------------------------------------
# Per-sample update semantics
# ---------------------------------------------------------------------------

def test_per_sample_update_semantics():
    """Update equals eta * mean(R * gate * e_slow) with per-sample product."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, lr=1.0, phi_0=0.0, record=False)
    p.reset(batch_size=2, device=syn.weight.device)

    p.e_slow = torch.tensor([
        [[1.0, 0.0], [0.5, -0.5], [0.0, 0.0]],
        [[0.0, 1.0], [-0.5, 0.5], [0.0, 0.0]],
    ], device=syn.weight.device)
    p.oscillator.reset(phi_0=0.0)  # gate = 1

    reward = torch.tensor([1.0, -1.0])
    expected_delta = (reward.reshape(-1, 1, 1) * p.e_slow).mean(dim=0)

    weight_before = syn.weight.data.clone()
    p.apply_reward(reward)
    actual_delta = syn.weight.data - weight_before

    assert torch.allclose(actual_delta, expected_delta, atol=1e-6)


# ---------------------------------------------------------------------------
# Degenerate gate counter
# ---------------------------------------------------------------------------

def test_degenerate_gate_counter():
    """fraction_updates_gate_zero tracks updates with gate == 0."""
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = WaveGatedPlasticity(syn, phi_0=math.pi / 2, record=True)
    p.reset(batch_size=2, device=syn.weight.device)
    p.e_slow = torch.ones(2, 3, 2)

    for _ in range(5):
        p.apply_reward(torch.tensor([1.0, 1.0]))

    metrics = p.get_wave_metrics()
    assert metrics["n_updates"] == 5
    assert metrics["fraction_updates_gate_zero"] == 1.0


# ---------------------------------------------------------------------------
# Shakedown: wave does not disturb base STDP traces
# ---------------------------------------------------------------------------

def test_fast_trace_matches_base_class():
    """Wave class fast trace equals base class eligibility on same spikes."""
    torch.manual_seed(42)
    syn1 = Synapse(4, 3, weight_scale=0.5, tau_syn=5.0)
    syn2 = Synapse(4, 3, weight_scale=0.5, tau_syn=5.0)
    # Copy weights so synapses are identical.
    syn2.weight.data.copy_(syn1.weight.data)

    base = RSTDPPlasticity(syn1, lr=0.001, tau_elig=20.0, dt=1.0)
    wave = WaveGatedPlasticity(syn2, lr=0.001, tau_elig=20.0, tau_elig_slow=500.0, dt=1.0, record=False)

    base.reset(batch_size=2, device=syn1.weight.device)
    wave.reset(batch_size=2, device=syn2.weight.device)

    # Generate synthetic spike trains.
    torch.manual_seed(123)
    for _ in range(150):
        pre = (torch.rand(2, 4) < 0.3).float()
        post = (torch.rand(2, 3) < 0.3).float()
        base.step(pre, post)
        wave.step(pre, post)

    diff = (base.eligibility - wave.eligibility).abs().max().item()
    assert diff < 1e-5, f"fast trace diverged from base class by {diff}"


def test_slow_trace_matches_base_with_same_tau():
    """Wave slow trace equals base eligibility when tau_elig == tau_elig_slow."""
    torch.manual_seed(7)
    syn1 = Synapse(4, 3, weight_scale=0.5, tau_syn=5.0)
    syn2 = Synapse(4, 3, weight_scale=0.5, tau_syn=5.0)
    syn2.weight.data.copy_(syn1.weight.data)

    base = RSTDPPlasticity(syn1, lr=0.001, tau_elig=500.0, dt=1.0)
    wave = WaveGatedPlasticity(syn2, lr=0.001, tau_elig=20.0, tau_elig_slow=500.0, dt=1.0, record=False)

    base.reset(batch_size=2, device=syn1.weight.device)
    wave.reset(batch_size=2, device=syn2.weight.device)

    torch.manual_seed(456)
    for _ in range(150):
        pre = (torch.rand(2, 4) < 0.3).float()
        post = (torch.rand(2, 3) < 0.3).float()
        base.step(pre, post)
        wave.step(pre, post)

    diff = (base.eligibility - wave.e_slow).abs().max().item()
    assert diff < 1e-5, f"slow trace diverged from base eligibility by {diff}"
