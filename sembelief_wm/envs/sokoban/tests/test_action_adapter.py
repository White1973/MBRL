"""Tests for Sokoban action adapter and LLM policy integration."""
import pytest
import torch
import torch.nn as nn

from sembelief_wm.rl.action_adapter import ActionAdapter
from sembelief_wm.envs.sokoban.action_adapter import SokobanActionAdapter
from sembelief_wm.rl.llm_policy import LLMActorCritic, LLMPolicyConfig  # no BeliefProjector
from sembelief_wm.rl.ppo import PPOUpdater, PPOConfig
from sembelief_wm.rl.gae import trajectory_to_ppo_batch
from sembelief_wm.rl.trajectory import Trajectory


# -- Fixtures --

@pytest.fixture
def adapter():
    return SokobanActionAdapter(hidden_dim=64)


class MockBackbone(nn.Module):
    """A real nn.Module backbone with trainable params — verifies gradient flow."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)  # (B, K, D) → (B, K, D)


@pytest.fixture
def policy(adapter):
    cfg = LLMPolicyConfig(hidden_dim=64, num_slots=4)
    return LLMActorCritic(MockBackbone(64), adapter, cfg)


# -- ActionAdapter protocol --

def test_sokoban_implements_protocol(adapter):
    assert isinstance(adapter, ActionAdapter)


def test_num_actions(adapter):
    assert adapter.num_actions == 4


def test_format_action(adapter):
    assert adapter.format_action(0) == "up"
    assert adapter.format_action(1) == "down"
    assert adapter.format_action(2) == "left"
    assert adapter.format_action(3) == "right"


def test_decode_text_exact(adapter):
    assert adapter.decode_text("up") == 0
    assert adapter.decode_text("Down") == 1
    assert adapter.decode_text("move left") == 2
    assert adapter.decode_text("push right") == 3


def test_decode_text_regex(adapter):
    assert adapter.decode_text("I should go up now") == 0
    assert adapter.decode_text("the best move is down") == 1


def test_decode_text_none(adapter):
    assert adapter.decode_text("") is None
    assert adapter.decode_text("foobar") is None


def test_model_env_action_roundtrip():
    for model_a in range(4):
        env_a = SokobanActionAdapter.model_to_env_action(model_a)
        assert SokobanActionAdapter.env_to_model_action(env_a) == model_a


# -- LLMActorCritic --

def test_act_shapes(policy):
    states = torch.randn(3, 4 * 64)
    actions, lp, ent, val = policy.act(states)
    assert actions.shape == (3,)
    assert lp.shape == (3,)
    assert ent.shape == (3,)
    assert val.shape == (3,)


def test_act_3d_input(policy):
    states = torch.randn(3, 4, 64)
    actions, lp, ent, val = policy.act(states)
    assert actions.shape == (3,)


def test_evaluate_actions_shapes(policy):
    states = torch.randn(3, 4 * 64)
    actions = torch.randint(0, 4, (3,))
    lp, ent, val = policy.evaluate_actions(states, actions)
    assert lp.shape == (3,)
    assert ent.shape == (3,)
    assert val.shape == (3,)


def test_deterministic_act(policy):
    states = torch.randn(2, 4 * 64)
    a1, _, _, _ = policy.act(states, deterministic=True)
    a2, _, _, _ = policy.act(states, deterministic=True)
    assert torch.equal(a1, a2)


# -- End-to-end: LLM policy → ImaginedCollector → PPO --

def test_llm_policy_ppo_integration(policy):
    """Full pipeline: LLM policy → trajectory → GAE → PPO update."""
    B, H, K, D = 4, 8, 4, 64
    state_dim = K * D

    # Simulate a trajectory from ImaginedCollector
    states = torch.randn(B, H, state_dim)
    actions = torch.randint(0, 4, (B, H))
    rewards = torch.randn(B, H)
    dones = torch.zeros(B, H)
    log_probs = torch.randn(B, H)
    values = torch.randn(B, H + 1)

    traj = Trajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
    )

    batch = trajectory_to_ppo_batch(traj, gamma=0.99, gae_lambda=0.95)
    assert batch.states.shape[0] == B * H

    ppo_cfg = PPOConfig(epochs=2, minibatch_size=8)
    updater = PPOUpdater(ppo_cfg, policy)
    metrics = updater.update(batch)

    assert metrics.policy_loss != 0.0 or metrics.value_loss != 0.0


def test_backbone_receives_gradients(policy):
    """Verify that gradients flow through the backbone during PPO update."""
    # Zero out backbone grads
    for p in policy.backbone.parameters():
        if p.grad is not None:
            p.grad.zero_()

    states = torch.randn(4, 4 * 64)
    actions = torch.randint(0, 4, (4,))
    lp, ent, val = policy.evaluate_actions(states, actions)
    loss = -lp.mean() + val.mean()
    loss.backward()

    # Backbone params must have nonzero gradients
    backbone_grads = [p.grad for p in policy.backbone.parameters() if p.grad is not None]
    assert len(backbone_grads) > 0, "No gradients reached backbone"
    assert any(g.abs().sum() > 0 for g in backbone_grads), "Backbone grads are all zero"


def test_no_belief_projector():
    """Verify LLMActorCritic has no BeliefProjector — belief slots go directly to backbone."""
    adapter = SokobanActionAdapter(hidden_dim=64)
    cfg = LLMPolicyConfig(hidden_dim=64, num_slots=4)
    pol = LLMActorCritic(MockBackbone(64), adapter, cfg)

    assert not hasattr(pol, 'belief_projector'), "BeliefProjector should not exist"
    states = torch.randn(3, 4 * 64)
    actions, lp, ent, val = pol.act(states)
    assert actions.shape == (3,)
