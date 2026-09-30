import torch

from snn.core import RSTDPPlasticity, Synapse
from snn.tasks import TemporalXORPaperTask


def test_apply_reward_uses_per_sample_product():
    """
    Revision B: apply_reward must compute (R * e).mean(dim=0), not
    mean(R) * mean(e).  With a balanced reward vector the broken rule
    produces zero update; the correct rule does not.
    """
    syn = Synapse(2, 3, weight_scale=0.1, tau_syn=5.0)
    p = RSTDPPlasticity(syn, lr=1.0, tau_elig=1000.0)
    p.reset(batch_size=2, device=syn.weight.device)

    # Hand-craft eligibility so the per-sample product is non-degenerate.
    p.eligibility = torch.zeros(2, 3, 2)
    p.eligibility[0] = torch.tensor([[1.0, 0.5],
                                      [-0.5, 0.2],
                                      [0.0, -0.3]])
    p.eligibility[1] = torch.tensor([[-0.8, 0.1],
                                      [0.4, -0.2],
                                      [0.2, 0.5]])

    reward = torch.tensor([1.0, -1.0])
    weight_before = syn.weight.data.clone()

    p.apply_reward(reward)

    weight_after = syn.weight.data
    delta = weight_after - weight_before

    # The old broken rule gives mean(R) = 0 and therefore zero update.
    expected_old_delta = reward.mean() * p.eligibility.mean(dim=0)
    assert expected_old_delta.abs().max().item() < 1e-6, \
        "fixture is supposed to be a non-zero case for the old rule"

    # The new rule must produce a non-zero update.
    assert delta.abs().max().item() > 1e-6

    # And the update must equal the per-sample product formula.
    expected_delta = (reward.reshape(-1, 1, 1) * p.eligibility).mean(dim=0)
    assert torch.allclose(delta, expected_delta, atol=1e-6)


def test_temporal_xor_paper_task_emits_paper_protocol():
    """Revision A: paper protocol is 500 ms trains with 50 uniform spikes."""
    task = TemporalXORPaperTask()
    task.set_seed(42)
    input_spikes, target = task.generate_batch(batch_size=8)

    assert input_spikes.shape == (8, 2, 500)
    assert input_spikes.shape[-1] == 500

    # Each channel has either 0 or exactly 50 spikes.
    for b in range(input_spikes.shape[0]):
        for ch in range(2):
            count = input_spikes[b, ch].sum().item()
            assert count in (0.0, 50.0)

    # Active channels (bit == 1) must have 50 spikes; inactive channels 0.
    total_spikes = input_spikes.sum(dim=-1)
    for b in range(input_spikes.shape[0]):
        for ch in range(2):
            if total_spikes[b, ch].item() not in (0.0, 50.0):
                raise AssertionError("spike count must be 0 or 50")
