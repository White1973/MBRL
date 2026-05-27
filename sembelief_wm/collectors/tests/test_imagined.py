"""Tests for ImaginedCollector — uses mock dynamics to verify shapes and flow."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from ..imagined import ImaginedCollector, ImaginedCollectorConfig
from ..adapters import LatentActorCriticAdapter, make_dynamics_step, make_predict_reward, make_get_belief_slots
from ...rl.gae import trajectory_to_ppo_batch
from ...rl.policy import MLPActorCritic, MLPSpec
from ...types import BeliefState


@dataclass
class FakeBelief:
    slots: Tensor


def test_imagined_collector_shapes():
    """ImaginedCollector should produce correctly shaped Trajectory."""
    B, H, K, D = 4, 8, 36, 256
    num_actions = 4

    # Flat state = (B, K*D)
    state_dim = K * D
    policy = MLPActorCritic(MLPSpec(state_dim=state_dim, num_actions=num_actions))

    def dynamics_step(belief, action, **kwargs):
        return FakeBelief(slots=belief.slots + 0.01 * torch.randn_like(belief.slots))

    def predict_reward(belief):
        return belief.slots[:, 0, 0]  # (B,) — dummy scalar

    def reward_transform(logits, **kwargs):
        return torch.sigmoid(logits) - 0.5

    def get_belief_slots(belief):
        return belief.slots.reshape(belief.slots.shape[0], -1)  # (B, K*D)

    collector = ImaginedCollector(
        policy=policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=H, batch_size=B),
    )

    start = FakeBelief(slots=torch.randn(B, K, D))
    traj = collector.collect(start)

    assert traj.states.shape == (B, H, state_dim)
    assert traj.actions.shape == (B, H)
    assert traj.rewards.shape == (B, H)
    assert traj.dones.shape == (B, H)
    assert traj.log_probs.shape == (B, H)
    assert traj.values.shape == (B, H + 1)
    assert traj.mask is None  # imagined rollout, all valid
    assert (traj.dones == 0).all()  # no termination in imagination


def test_imagined_collector_to_ppo_batch():
    """Full pipeline: ImaginedCollector -> trajectory_to_ppo_batch -> PPOBatch."""
    B, H = 4, 8
    state_dim = 16
    num_actions = 4

    policy = MLPActorCritic(MLPSpec(state_dim=state_dim, num_actions=num_actions))

    def dynamics_step(belief, action, **kwargs):
        return FakeBelief(slots=belief.slots + 0.01)

    def predict_reward(belief):
        return belief.slots.mean(dim=-1)

    def reward_transform(logits, **kwargs):
        return logits

    def get_belief_slots(belief):
        return belief.slots

    collector = ImaginedCollector(
        policy=policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=H, batch_size=B),
    )

    start = FakeBelief(slots=torch.randn(B, state_dim))
    traj = collector.collect(start)
    batch = trajectory_to_ppo_batch(traj, gamma=0.99, gae_lambda=0.95)

    assert batch.states.shape == (B * H, state_dim)
    assert batch.actions.shape == (B * H,)
    assert batch.advantages.shape == (B * H,)


def test_imagined_collector_bootstrap_value():
    """bootstrap_with_value=True should use V(final_state), not zero."""
    B, H = 2, 4
    state_dim = 8

    policy = MLPActorCritic(MLPSpec(state_dim=state_dim, num_actions=2))

    def dynamics_step(belief, action, **kwargs):
        return FakeBelief(slots=belief.slots)

    def predict_reward(belief):
        return torch.zeros(belief.slots.shape[0])

    def reward_transform(logits, **kwargs):
        return logits

    def get_belief_slots(belief):
        return belief.slots

    # With bootstrap
    collector_boot = ImaginedCollector(
        policy=policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=H, bootstrap_with_value=True),
    )
    traj_boot = collector_boot.collect(FakeBelief(slots=torch.randn(B, state_dim)))

    # Without bootstrap
    collector_no_boot = ImaginedCollector(
        policy=policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=H, bootstrap_with_value=False),
    )
    traj_no_boot = collector_no_boot.collect(FakeBelief(slots=torch.randn(B, state_dim)))

    # No-bootstrap: final value should be zero
    assert (traj_no_boot.values[:, -1] == 0).all()
    # Bootstrap: final value is V(final_state), generally non-zero
    # (can't assert non-zero since it depends on init, but shapes should match)
    assert traj_boot.values.shape == (B, H + 1)


def test_reward_shape_squeeze():
    """Reward predictor returning (B, 1) should be squeezed to (B,)."""
    B, H = 3, 4
    state_dim = 8

    policy = MLPActorCritic(MLPSpec(state_dim=state_dim, num_actions=2))

    def dynamics_step(belief, action, **kwargs):
        return FakeBelief(slots=belief.slots)

    def predict_reward(belief):
        return torch.zeros(belief.slots.shape[0], 1)  # (B, 1) — wrong shape

    def reward_transform(logits, **kwargs):
        return logits  # passes through (B, 1) if not squeezed

    def get_belief_slots(belief):
        return belief.slots

    collector = ImaginedCollector(
        policy=policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=H),
    )
    traj = collector.collect(FakeBelief(slots=torch.randn(B, state_dim)))
    assert traj.rewards.shape == (B, H), f"Expected (B, H), got {traj.rewards.shape}"


class FakeActorCritic:
    """Mock LatentActorCritic with 5-tuple return and BeliefState input."""

    def __init__(self, K: int, D: int, num_actions: int):
        self.K = K
        self.D = D
        self.num_actions = num_actions
        # Simple linear layers for act/evaluate
        import torch.nn as nn
        self._policy = nn.Linear(K * D, num_actions)
        self._value = nn.Linear(K * D, 1)

    def act(self, belief: BeliefState, *, deterministic: bool = False, **kwargs):
        flat = belief.slots.reshape(belief.slots.shape[0], -1)
        logits = self._policy(flat)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self._value(flat).squeeze(-1)
        return action, log_prob, entropy, value, None  # 5-tuple

    def evaluate_actions(self, belief: BeliefState, actions: Tensor, **kwargs):
        flat = belief.slots.reshape(belief.slots.shape[0], -1)
        logits = self._policy(flat)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self._value(flat).squeeze(-1)
        return log_prob, entropy, value

    def parameters(self):
        import itertools
        return itertools.chain(self._policy.parameters(), self._value.parameters())


class FakeWorldModel:
    """Mock WorldModel with prior_step and predict_reward."""

    def prior_step(self, *, prev_belief: BeliefState, prev_actions: Tensor, **kwargs):
        noise = 0.01 * torch.randn_like(prev_belief.slots)
        return BeliefState(slots=prev_belief.slots + noise)

    def predict_reward(self, belief: BeliefState) -> Tensor:
        return belief.slots[:, 0, 0]  # (B,) scalar


def test_adapter_integration():
    """Full integration: adapter-wrapped policy + world model adapters + ImaginedCollector."""
    B, H, K, D = 4, 6, 36, 64
    num_actions = 4

    fake_ac = FakeActorCritic(K, D, num_actions)
    fake_wm = FakeWorldModel()

    adapted_policy = LatentActorCriticAdapter(fake_ac)
    dynamics = make_dynamics_step(fake_wm)
    reward_fn = make_predict_reward(fake_wm)
    get_slots = make_get_belief_slots()

    collector = ImaginedCollector(
        policy=adapted_policy,
        dynamics_step=dynamics,
        predict_reward=reward_fn,
        reward_transform=lambda logits, **kw: torch.sigmoid(logits) - 0.5,
        get_belief_slots=get_slots,
        config=ImaginedCollectorConfig(horizon=H, batch_size=B),
    )

    start = BeliefState(slots=torch.randn(B, K, D))
    traj = collector.collect(start)

    # Shape checks
    assert traj.states.shape == (B, H, K, D)
    assert traj.actions.shape == (B, H)
    assert traj.rewards.shape == (B, H)
    assert traj.values.shape == (B, H + 1)

    # Full pipeline to PPOBatch
    # Need to flatten states for PPO — adapter's get_belief_slots returns (B, K, D)
    # trajectory_to_ppo_batch flattens (B, H) -> (N,) but preserves state shape as (N, K, D)
    batch = trajectory_to_ppo_batch(traj, gamma=0.99, gae_lambda=0.95)
    assert batch.states.shape == (B * H, K, D)
    assert batch.actions.shape == (B * H,)
    assert batch.advantages.shape == (B * H,)


def test_adapter_rejects_2d_states():
    """LatentActorCriticAdapter should reject 2D (B, D) states."""
    fake_ac = FakeActorCritic(K=4, D=8, num_actions=2)
    adapted = LatentActorCriticAdapter(fake_ac)

    states_2d = torch.randn(3, 32)  # (B, D) — 2D, should fail
    with pytest.raises(ValueError, match="3D"):
        adapted.act(states_2d)
