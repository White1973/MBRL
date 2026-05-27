"""Tests for GAE computation, especially mask/padding handling."""
from __future__ import annotations

import torch

from ..gae import compute_gae, trajectory_to_ppo_batch
from ..trajectory import Trajectory


def test_gae_no_mask_no_done():
    """Basic GAE with no masking, no termination — pure imagined rollout case."""
    B, H = 2, 4
    rewards = torch.ones(B, H)
    values = torch.zeros(B, H + 1)
    dones = torch.zeros(B, H)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (B, H)
    assert ret.shape == (B, H)
    # With zero values, advantages == discounted returns
    assert torch.allclose(adv, ret)
    # Last step should have advantage == reward (no future)
    # Actually with GAE: adv[3] = delta[3] = r[3] + gamma*V[4] - V[3] = 1
    assert torch.allclose(adv[:, -1], torch.ones(B))


def test_gae_with_done():
    """GAE should cut bootstrap at done steps."""
    B, H = 1, 3
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5, 10.0]])  # big bootstrap
    dones = torch.tensor([[0.0, 1.0, 0.0]])  # done at step 1

    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

    # Step 2 (after reset): delta = r2 + gamma*V3 - V2 = 1 + 0.99*10 - 0.5 = 10.4
    # Step 1 (done): delta = r1 + 0 - V1 = 1 - 0.5 = 0.5, no carry from step 2
    # Step 0: delta = r0 + gamma*V1 - V0 = 1 + 0.99*0.5 - 0.5 = 0.995
    #         next_adv from step 1 = 0.5
    #         adv0 = 0.995 + 0.99*0.95*0.5 = 0.995 + 0.47025 = 1.46525
    assert abs(adv[0, 1].item() - 0.5) < 1e-4, f"step 1 adv: {adv[0, 1].item()}"
    assert abs(adv[0, 0].item() - 1.46525) < 1e-3, f"step 0 adv: {adv[0, 0].item()}"


def test_gae_padding_no_leak():
    """Padding steps should not leak into valid steps' advantages."""
    B, H = 1, 4
    rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    values = torch.tensor([[0.0, 0.0, 0.0, 99.0, 99.0]])  # garbage in padding
    dones = torch.zeros(B, H)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # only first 2 steps valid

    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, mask=mask)

    # Padding steps should be zero
    assert adv[0, 2].item() == 0.0
    assert adv[0, 3].item() == 0.0
    assert ret[0, 2].item() == 0.0
    assert ret[0, 3].item() == 0.0

    # Now verify valid steps are NOT polluted by the garbage values in padding.
    # If padding leaked, the advantages would be wildly wrong.
    # Step 1 (last valid): delta = r1 + gamma*V2 - V1 = 1 + 0 - 0 = 1
    #   next_advantage from step 2 is gated to 0, so adv1 = 1.0
    assert abs(adv[0, 1].item() - 1.0) < 1e-4
    # Step 0: delta = r0 + gamma*V1 - V0 = 1 + 0 - 0 = 1
    #   next_advantage from step 1 = 1.0
    #   adv0 = 1 + 0.99*0.95*1.0 = 1 + 0.9405 = 1.9405
    assert abs(adv[0, 0].item() - 1.9405) < 1e-3


def test_trajectory_to_ppo_batch_strips_padding():
    """trajectory_to_ppo_batch should strip padding, not pass it to PPO."""
    B, H = 2, 3
    state_dim = 4
    traj = Trajectory(
        states=torch.randn(B, H, state_dim),
        actions=torch.randint(0, 4, (B, H)),
        rewards=torch.ones(B, H),
        dones=torch.zeros(B, H),
        log_probs=torch.randn(B, H),
        values=torch.randn(B, H + 1),
        mask=torch.tensor([
            [1.0, 1.0, 0.0],  # 2 valid steps
            [1.0, 1.0, 1.0],  # 3 valid steps
        ]),
    )
    batch = trajectory_to_ppo_batch(traj, gamma=0.99, gae_lambda=0.95)
    assert batch.states.shape[0] == 5  # 2 + 3 = 5 valid steps
    assert batch.actions.shape[0] == 5


def test_trajectory_to_ppo_batch_no_mask():
    """Without mask, all B*H steps are included."""
    B, H = 2, 3
    traj = Trajectory(
        states=torch.randn(B, H, 4),
        actions=torch.randint(0, 4, (B, H)),
        rewards=torch.ones(B, H),
        dones=torch.zeros(B, H),
        log_probs=torch.randn(B, H),
        values=torch.randn(B, H + 1),
    )
    batch = trajectory_to_ppo_batch(traj)
    assert batch.states.shape[0] == B * H
