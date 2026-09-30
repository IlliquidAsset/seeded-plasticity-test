import math

import pytest
import torch

from ladder.experiments import DELAYS_MS, TAUS_MS, EligibilityTrace
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
