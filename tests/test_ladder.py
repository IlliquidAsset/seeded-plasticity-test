import math

import pytest
import torch

from ladder.experiments import (
    DELAYS_MS,
    TAUS_MS,
    EligibilityTrace,
    run_all_rungs,
    run_rung0,
    run_rung1,
    run_rung2,
    run_rung3,
)
from snn.core import RSTDPPlasticity, Synapse


@pytest.mark.parametrize("tau", TAUS_MS)
@pytest.mark.parametrize("delay", DELAYS_MS)
def test_rung0_reference_matches_closed_form(tau, delay):
    trace = EligibilityTrace(tau)
    trace.tag(1.0)
    trace.elapse(delay)
    assert trace.credit(1.0) == pytest.approx(math.exp(-delay / tau), abs=1e-13)


@pytest.mark.parametrize("tau", TAUS_MS)
@pytest.mark.parametrize("delay", DELAYS_MS)
def test_repository_core_trace_matches_closed_form(tau, delay):
    synapse = Synapse(1, 1, weight_scale=0.0)
    rule = RSTDPPlasticity(synapse, lr=1.0, tau_elig=tau)
    rule.reset(batch_size=1, device=synapse.weight.device)
    rule.eligibility.fill_(1.0)
    zeros = torch.zeros(1, 1)
    for _ in range(delay):
        rule.step(zeros, zeros)
    assert rule.eligibility.item() == pytest.approx(math.exp(-delay / tau), rel=3e-5, abs=1e-7)


def test_same_timestep_reward_sees_fresh_tag_before_decay():
    trace = EligibilityTrace(25.0)
    trace.tag()
    assert trace.credit(1.0) == 1.0


def test_rung1_controls_and_behavior():
    result = run_rung1()
    assert result["status"] == "PASS"
    assert result["controls"]["no_reward_delta"] == 0.0
    assert result["controls"]["positive_reward_for_silence_delta_without_tag"] == 0.0
    assert result["runaway"]["raw_hit_cap"] is True
    assert result["runaway"]["baseline_hit_cap"] is False


def test_rung2_assigns_clean_credit_only_to_a():
    result = run_rung2()
    assert result["status"] == "PASS"
    assert result["primary_gate"]["median_delta_w_A"] > 0.0
    assert result["primary_gate"]["median_delta_w_B"] == 0.0


def test_rung3_unconnected_control_never_changes():
    result = run_rung3()
    assert result["status"] == "PASS"
    assert result["unconnected_delta_weight"] == 0.0
    assert result["outcome"]["unconnected_after_reward"] == result["outcome"]["unconnected_frozen"]


def test_sequential_runner_reaches_rung3_only_after_all_pass():
    result = run_all_rungs()
    assert result["status"] == "PASS"
    assert list(result["results"]) == ["rung_0", "rung_1", "rung_2", "rung_3"]
    assert run_rung0()["max_absolute_error"] < 1e-12
