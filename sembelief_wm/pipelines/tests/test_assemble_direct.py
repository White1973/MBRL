"""Tests that directly call assemble_pipeline() and assemble_llm_pipeline().

Unlike test_assemble.py which manually replicates assembly logic, these tests
exercise the REAL assembly functions end-to-end, catching interface drift.
"""
import torch
import torch.nn as nn
from torch import Tensor

from sembelief_wm.config import Config, Phase2Config, PPOConfig as LegacyPPOConfig
from sembelief_wm.types import BeliefState, SequenceBatch
from sembelief_wm.envs.sokoban.action_adapter import SokobanActionAdapter
from sembelief_wm.pipelines.assemble import assemble_pipeline, assemble_llm_pipeline


# ---------------------------------------------------------------------------
# Small config — overrides hidden_dim so tensors are tiny
# ---------------------------------------------------------------------------

K, D, T = 4, 32, 6  # slots, hidden dim, seq len


def _small_config() -> Config:
    """Config with small hidden_dim for fast tests."""
    return Config(
        hidden_dim=D,
        phase2=Phase2Config(
            world_model_mode="frozen_wm",
            ppo=LegacyPPOConfig(
                total_updates=2,
                rollout_batch_size=4,
                rollout_horizon=3,
                epochs_per_update=2,
                minibatch_size=4,
                actor_lr=1e-3,
                eval_every=0,
                checkpoint_every=0,
                collect_every=0,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Fake WorldModel — duck-types the real WorldModel interface
# ---------------------------------------------------------------------------

class FakeQwenBackbone(nn.Module):
    """Backbone with LoRA-named params so trainable_parameters() finds them."""

    def __init__(self):
        super().__init__()
        self.lora_a = nn.Linear(D, 4, bias=False)
        self.lora_b = nn.Linear(4, D, bias=False)
        self.base = nn.Linear(D, D)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x))


class FakeWorldModel(nn.Module):
    """Minimal nn.Module that satisfies what assemble_* functions need."""

    def __init__(self):
        super().__init__()
        self.transition = nn.Linear(D, D)
        self.reward_head = nn.Linear(D, 1)
        self.backbone = FakeQwenBackbone()
        self._initial_belief = nn.Parameter(torch.zeros(1, K, D), requires_grad=False)

    def prior_step(self, *, prev_belief, prev_actions, **kwargs):
        new_slots = self.transition(prev_belief.slots)
        return BeliefState(slots=new_slots)

    def predict_reward(self, belief):
        pooled = belief.slots.mean(dim=1)
        return self.reward_head(pooled).squeeze(-1)

    def get_initial_belief(self, batch_size, *, device=None, dtype=None):
        slots = self._initial_belief.expand(batch_size, -1, -1)
        if device is not None:
            slots = slots.to(device=device)
        if dtype is not None:
            slots = slots.to(dtype=dtype)
        return BeliefState(slots=slots.clone())

    def posterior_step(self, *, prev_belief, prev_actions, observation_tokens, env_ids=None):
        # Simple posterior: combine prev belief with observation
        new_slots = prev_belief.slots + observation_tokens.mean(dim=-2, keepdim=True).expand_as(prev_belief.slots) * 0.01
        return BeliefState(slots=new_slots)


# ---------------------------------------------------------------------------
# Fake OfflineDataSource — returns SequenceBatch without needing real data
# ---------------------------------------------------------------------------

class FakeOfflineDataSource:
    """Duck-types OfflineDataSource.sample_batch(batch_size) -> SequenceBatch."""

    def sample_batch(self, batch_size: int) -> SequenceBatch:
        return SequenceBatch(
            obs_tokens=torch.randn(batch_size, T, K, D),
            actions=torch.randint(0, 4, (batch_size, T)),
            rewards=torch.zeros(batch_size, T),
            episode_lengths=torch.full((batch_size,), T, dtype=torch.long),
            env_ids=torch.zeros(batch_size, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Fake LatentActorCritic — 5-tuple act return
# ---------------------------------------------------------------------------

class FakeLatentActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(K * D, 4)
        self.value = nn.Linear(K * D, 1)

    def act(self, belief, *, deterministic=False, **kwargs):
        flat = belief.slots.reshape(belief.slots.shape[0], -1)
        logits = self.policy(flat)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        value = self.value(flat).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value, None

    def evaluate_actions(self, belief, actions, **kwargs):
        flat = belief.slots.reshape(belief.slots.shape[0], -1)
        logits = self.policy(flat)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value(flat).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value


# ---------------------------------------------------------------------------
# Tests — directly calling assemble_pipeline / assemble_llm_pipeline
# ---------------------------------------------------------------------------

def test_assemble_pipeline_direct():
    """Call assemble_pipeline() directly and run one update."""
    config = _small_config()
    wm = FakeWorldModel()
    ac = FakeLatentActorCritic()
    data_source = FakeOfflineDataSource()

    pipeline = assemble_pipeline(
        config=config,
        world_model=wm,
        actor_critic=ac,
        data_source=data_source,
        device=torch.device("cpu"),
    )

    pipeline.train()  # Should complete 2 updates without crashing


def test_assemble_llm_pipeline_direct_shared():
    """Call assemble_llm_pipeline() with shared backbone and run one update."""
    config = _small_config()
    wm = FakeWorldModel()
    action_adapter = SokobanActionAdapter(hidden_dim=D)
    data_source = FakeOfflineDataSource()

    pipeline, llm_policy = assemble_llm_pipeline(
        config=config,
        world_model=wm,
        action_adapter=action_adapter,
        data_source=data_source,
        device=torch.device("cpu"),
        shared_backbone=True,
    )

    pipeline.train()  # Should complete 2 updates without crashing


def test_assemble_llm_pipeline_shared_backbone_updated():
    """PPO updates through assemble_llm_pipeline actually move shared LoRA weights."""
    config = _small_config()
    wm = FakeWorldModel()
    action_adapter = SokobanActionAdapter(hidden_dim=D)
    data_source = FakeOfflineDataSource()

    lora_before = wm.backbone.lora_a.weight.data.clone()

    pipeline, llm_policy = assemble_llm_pipeline(
        config=config,
        world_model=wm,
        action_adapter=action_adapter,
        data_source=data_source,
        device=torch.device("cpu"),
        shared_backbone=True,
    )

    pipeline.train()

    lora_after = wm.backbone.lora_a.weight.data
    assert not torch.equal(lora_before, lora_after), (
        "Shared LoRA weights were not updated — assemble_llm_pipeline wiring is broken"
    )
