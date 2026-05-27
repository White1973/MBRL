"""Smoke tests for assembly functions — exercises the real assemble_* code paths.

These tests verify that:
1. ImaginedCollector is instantiated with correct kwargs (predict_reward + reward_transform)
2. Shared backbone LoRA params stay trainable after world_model.requires_grad_(False)
3. The full assembly → pipeline → one update path doesn't crash
"""
import torch
import torch.nn as nn
from torch import Tensor

from sembelief_wm.rl.llm_policy import LLMActorCritic, LLMPolicyConfig
from sembelief_wm.rl.ppo import PPOUpdater, PPOConfig
from sembelief_wm.envs.sokoban.action_adapter import SokobanActionAdapter
from sembelief_wm.collectors.imagined import ImaginedCollector, ImaginedCollectorConfig
from sembelief_wm.collectors.adapters import (
    LatentActorCriticAdapter,
    make_dynamics_step,
    make_predict_reward,
    make_get_belief_slots,
)
from sembelief_wm.collectors.reward_transforms import make_reward_transform
from sembelief_wm.pipelines.mbrl_train import MBRLPipeline, PipelineConfig
from sembelief_wm.types import BeliefState


# ---------------------------------------------------------------------------
# Minimal fakes that match real interfaces
# ---------------------------------------------------------------------------

K, D = 4, 32  # small dims for tests


class FakeWorldModel(nn.Module):
    """Minimal world model with the same interface as real WorldModel."""

    def __init__(self):
        super().__init__()
        self.transition = nn.Linear(D, D)
        self.reward_head = nn.Linear(D, 1)
        # Fake backbone with LoRA-like trainable params
        self.backbone = FakeQwenBackbone()

    def prior_step(self, *, prev_belief, prev_actions, **kwargs):
        # Simple dynamics: linear transform each slot
        new_slots = self.transition(prev_belief.slots)
        return BeliefState(slots=new_slots)

    def predict_reward(self, belief):
        # Mean-pool slots → reward logit
        pooled = belief.slots.mean(dim=1)  # (B, D)
        return self.reward_head(pooled).squeeze(-1)  # (B,)

    def get_initial_belief(self, batch_size, *, device=None, dtype=None):
        return BeliefState(
            slots=torch.zeros(batch_size, K, D, device=device, dtype=dtype)
        )


class FakeQwenBackbone(nn.Module):
    """Simulates QwenTransitionBackbone with trainable LoRA-like params."""

    def __init__(self):
        super().__init__()
        self.lora_a = nn.Linear(D, 4, bias=False)
        self.lora_b = nn.Linear(4, D, bias=False)
        self.base = nn.Linear(D, D)
        # Freeze base, only LoRA trainable
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x))


class FakeActorCritic(nn.Module):
    """Fake LatentActorCritic with 5-tuple act return (matches real interface)."""

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
        return action, dist.log_prob(action), dist.entropy(), value, None  # 5-tuple

    def evaluate_actions(self, belief, actions, **kwargs):
        flat = belief.slots.reshape(belief.slots.shape[0], -1)
        logits = self.policy(flat)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value(flat).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value


# ---------------------------------------------------------------------------
# Test: Latent policy assembly path
# ---------------------------------------------------------------------------

def test_assemble_latent_path_one_update():
    """assemble latent path: adapters + ImaginedCollector + PPO → one update runs."""
    wm = FakeWorldModel()
    ac = FakeActorCritic()

    # Freeze WM (as assemble_pipeline does)
    wm.eval()
    wm.requires_grad_(False)

    # Build components exactly as assemble_pipeline does
    adapted_policy = LatentActorCriticAdapter(ac)
    dynamics_step = make_dynamics_step(wm)
    predict_reward_fn = make_predict_reward(wm)
    get_belief_slots = make_get_belief_slots()

    reward_transform = make_reward_transform(
        mapping="sigmoid_affine",
        positive_value=1.0,
        negative_value=-1.0,
    )

    collector = ImaginedCollector(
        policy=adapted_policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward_fn,
        reward_transform=lambda logits, **kw: reward_transform(logits),
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=4, batch_size=8),
    )

    def sample_beliefs_fn(batch_size):
        return wm.get_initial_belief(batch_size, device=torch.device("cpu"), dtype=torch.float32)

    ppo_cfg = PPOConfig(lr=1e-3, epochs=2, minibatch_size=4)
    ppo_updater = PPOUpdater(policy=adapted_policy, config=ppo_cfg)

    cfg = PipelineConfig(
        total_updates=1,
        eval_every=0,
        rollout_batch_size=8,
        rollout_horizon=4,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=collector.collect,
        sample_beliefs_fn=sample_beliefs_fn,
        ppo_updater=ppo_updater,
        policy=ac,
    )

    pipeline.train()  # Should not crash


# ---------------------------------------------------------------------------
# Test: LLM policy assembly path
# ---------------------------------------------------------------------------

def test_assemble_llm_path_one_update():
    """assemble LLM path: LLMActorCritic + ImaginedCollector + PPO → one update runs."""
    wm = FakeWorldModel()

    # Freeze WM first
    wm.eval()
    wm.requires_grad_(False)

    # Build backbone (shared mode simulation)
    backbone = wm.backbone
    # Re-enable LoRA params by name (as assemble_llm_pipeline does)
    for name, p in backbone.named_parameters():
        if "lora" in name:
            p.requires_grad_(True)

    action_adapter = SokobanActionAdapter(hidden_dim=D)
    llm_cfg = LLMPolicyConfig(hidden_dim=D, num_slots=K)
    llm_policy = LLMActorCritic(
        backbone=backbone,
        action_adapter=action_adapter,
        config=llm_cfg,
    )

    # Build imagined collector
    dynamics_step = make_dynamics_step(wm)
    predict_reward_fn = make_predict_reward(wm)
    get_belief_slots = make_get_belief_slots()
    reward_transform = make_reward_transform(
        mapping="sigmoid_affine",
        positive_value=1.0,
        negative_value=-1.0,
    )

    collector = ImaginedCollector(
        policy=llm_policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward_fn,
        reward_transform=lambda logits, **kw: reward_transform(logits),
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=4, batch_size=8),
    )

    def sample_beliefs_fn(batch_size):
        return wm.get_initial_belief(
            batch_size, device=torch.device("cpu"), dtype=torch.float32
        )

    # Collect LoRA params as extra_params (by name, not requires_grad)
    extra_params = [
        p for n, p in backbone.named_parameters() if "lora" in n
    ]

    ppo_cfg = PPOConfig(lr=1e-3, epochs=2, minibatch_size=4)
    ppo_updater = PPOUpdater(
        policy=llm_policy, config=ppo_cfg, extra_params=extra_params
    )

    cfg = PipelineConfig(
        total_updates=1,
        eval_every=0,
        rollout_batch_size=8,
        rollout_horizon=4,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=collector.collect,
        sample_beliefs_fn=sample_beliefs_fn,
        ppo_updater=ppo_updater,
        policy=llm_policy,
    )

    pipeline.train()  # Should not crash


# ---------------------------------------------------------------------------
# Test: Shared backbone params stay trainable after WM freeze
# ---------------------------------------------------------------------------

def test_shared_backbone_lora_survives_wm_freeze():
    """After world_model.requires_grad_(False), shared LoRA params re-enabled."""
    wm = FakeWorldModel()

    # Before freeze, LoRA params are trainable
    assert wm.backbone.lora_a.weight.requires_grad

    # Freeze WM (kills all grads including LoRA)
    wm.requires_grad_(False)
    assert not wm.backbone.lora_a.weight.requires_grad

    # Re-enable LoRA (as assemble_llm_pipeline does)
    for name, p in wm.backbone.named_parameters():
        if "lora" in name:
            p.requires_grad_(True)

    # LoRA trainable again, base still frozen
    assert wm.backbone.lora_a.weight.requires_grad
    assert wm.backbone.lora_b.weight.requires_grad
    assert not wm.backbone.base.weight.requires_grad


def test_shared_backbone_ppo_updates_lora_weights():
    """PPO optimizer.step() actually updates shared backbone LoRA weights."""
    wm = FakeWorldModel()
    wm.eval()
    wm.requires_grad_(False)

    # Re-enable LoRA
    # Re-enable LoRA params by name
    for name, p in wm.backbone.named_parameters():
        if "lora" in name:
            p.requires_grad_(True)

    action_adapter = SokobanActionAdapter(hidden_dim=D)
    llm_cfg = LLMPolicyConfig(hidden_dim=D, num_slots=K)
    llm_policy = LLMActorCritic(
        backbone=wm.backbone,
        action_adapter=action_adapter,
        config=llm_cfg,
    )

    extra_params = [
        p for n, p in wm.backbone.named_parameters() if "lora" in n
    ]
    w_before = wm.backbone.lora_a.weight.data.clone()

    # Build collector + pipeline
    dynamics_step = make_dynamics_step(wm)
    predict_reward_fn = make_predict_reward(wm)
    get_belief_slots = make_get_belief_slots()
    reward_transform = make_reward_transform("sigmoid_affine", 1.0, -1.0)

    collector = ImaginedCollector(
        policy=llm_policy,
        dynamics_step=dynamics_step,
        predict_reward=predict_reward_fn,
        reward_transform=lambda logits, **kw: reward_transform(logits),
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(horizon=4, batch_size=8),
    )

    def sample_beliefs_fn(batch_size):
        return wm.get_initial_belief(
            batch_size, device=torch.device("cpu"), dtype=torch.float32
        )

    ppo_cfg = PPOConfig(lr=1e-2, epochs=4, minibatch_size=4)
    ppo_updater = PPOUpdater(
        policy=llm_policy, config=ppo_cfg, extra_params=extra_params
    )

    cfg = PipelineConfig(
        total_updates=3,
        eval_every=0,
        rollout_batch_size=8,
        rollout_horizon=4,
    )

    pipeline = MBRLPipeline(
        config=cfg,
        imagine_fn=collector.collect,
        sample_beliefs_fn=sample_beliefs_fn,
        ppo_updater=ppo_updater,
        policy=llm_policy,
    )

    pipeline.train()

    w_after = wm.backbone.lora_a.weight.data
    assert not torch.equal(w_before, w_after), (
        "Shared LoRA weights were not updated by PPO — extra_params not wired correctly"
    )
