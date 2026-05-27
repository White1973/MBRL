#!/usr/bin/env python3
"""CPU smoke test for the Phase 2 latent RL stack.

This script exercises three Phase 2 paths with tiny mock components:

1. Latent MLP PPO end-to-end on top of a frozen mock world model
2. VLM freeform policy with token-level PPO
3. VLM freeform policy with hybrid scoring + lookahead planner

The freeform route uses a tiny fake tokenizer and causal LM so the code path
can be validated without downloading Qwen weights.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sembelief_wm.agent.phase2 import (
    LatentActorCritic,
    Phase2Trainer,
)
from sembelief_wm.config import (
    ActorCriticConfig,
    AntiCollapseConfig,
    BackboneConfig,
    BeliefConfig,
    Config,
    EncoderConfig,
    EnvironmentConfig,
    FreeformPolicyConfig,
    LookaheadConfig,
    Phase2Config,
    PPOConfig,
    TrainingConfig,
    WandbConfig,
)
from sembelief_wm.model import TransitionBackbone, WorldModel
from sembelief_wm.types import SequenceBatch


class MockBackbone(TransitionBackbone):
    """Small trainable backbone for smoke tests."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = tokens.mean(dim=1, keepdim=True)
        return self.token_proj(tokens) + self.context_proj(context)


class DummyDataSource:
    """Deterministic random data source for Phase 2 smoke tests."""

    def __init__(self, config: Config, *, seq_len: int = 6) -> None:
        self.config = config
        self.seq_len = seq_len
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(123)

    def sample_batch(self, batch_size: int) -> SequenceBatch:
        obs_tokens = torch.randn(
            batch_size,
            self.seq_len,
            self.config.encoder.compressed_tokens,
            self.config.hidden_dim,
            generator=self.generator,
        )
        actions = torch.randint(
            low=0,
            high=self.config.env.num_actions,
            size=(batch_size, self.seq_len),
            generator=self.generator,
        )
        rewards = torch.where(
            torch.rand(batch_size, self.seq_len, generator=self.generator) < 0.1,
            torch.full((batch_size, self.seq_len), 10.9),
            torch.full((batch_size, self.seq_len), -0.1),
        )
        episode_lengths = torch.full((batch_size,), self.seq_len, dtype=torch.long)
        env_ids = torch.zeros(batch_size, dtype=torch.long)
        return SequenceBatch(
            obs_tokens=obs_tokens,
            actions=actions,
            rewards=rewards,
            episode_lengths=episode_lengths,
            env_ids=env_ids,
        )


class CaptureLogger:
    def __init__(self) -> None:
        self.history: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        self.history.append((step, metrics))


def make_test_config(
    *,
    policy_family: str = "mlp",
    freeform_update_mode: str = "token_ppo",
    use_lookahead: bool = False,
    lookahead_width: int = 4,
) -> Config:
    return Config(
        hidden_dim=32,
        belief=BeliefConfig(num_slots=4),
        encoder=EncoderConfig(
            vjepa2_raw_tokens=8,
            vjepa2_raw_dim=16,
            compressed_tokens=4,
        ),
        backbone=BackboneConfig(
            model_name="fake-qwen",
            attention_mode="bidirectional",
            action_conditioning_mode="embedded",
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=["q_proj"],
        ),
        anti_collapse=AntiCollapseConfig(
            use_sigreg=False,
            use_ema_target=False,
            use_ema_variance=False,
        ),
        training=TrainingConfig(
            total_steps=2,
            episodes_per_step=2,
            dtype="fp32",
        ),
        phase2=Phase2Config(
            actor_critic=ActorCriticConfig(
                policy_family=policy_family, readout="mean_pool", hidden_dim=32, hidden_layers=1
            ),
            freeform_policy=FreeformPolicyConfig(
                update_mode=freeform_update_mode,
                model_name="fake-qwen",
                projector_tokens=2,
                projector_heads=4,
                max_new_tokens=6,
                do_sample=False,
                lora_rank=4,
                lora_alpha=8,
                lora_dropout=0.0,
                lora_target_modules=["q_proj"],
            ),
            ppo=PPOConfig(
                total_updates=2,
                rollout_batch_size=3,
                rollout_horizon=3,
                epochs_per_update=1,
                minibatch_size=4,
                actor_lr=1e-3,
                critic_lr=1e-3,
                checkpoint_every=1,
                reward_mapping="raw_sigmoid",
                kl_coef=0.01,
            ),
            lookahead=LookaheadConfig(
                enabled=use_lookahead,
                expansion_depth=1,
                expansion_width=lookahead_width,
                rollout_horizon=2,
                discount=0.99,
                use_value_bootstrap=True,
            ),
        ),
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
            env_ids=["sokoban"],
        ),
        wandb=WandbConfig(enabled=False),
    )


class FakeTokenizer:
    """Tiny ASCII tokenizer used by the freeform smoke path."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_name: str, trust_remote_code: bool = True) -> "FakeTokenizer":
        return cls()

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert padding
        assert return_tensors == "pt"
        encoded = [self._encode(text) for text in texts]
        max_len = max(len(ids) for ids in encoded) if encoded else 0
        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        texts: list[str] = []
        for row in token_ids.tolist():
            chars: list[str] = []
            for token in row:
                if token == self.pad_token_id:
                    continue
                if token == self.eos_token_id:
                    if not skip_special_tokens:
                        chars.append(self.eos_token)
                    continue
                chars.append(chr(max(32, token - 2)))
            texts.append("".join(chars))
        return texts

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [min(257, ord(ch) + 2) for ch in text]


class FakePeftModule:
    class LoraConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    @staticmethod
    def get_peft_model(model: nn.Module, lora_config: object) -> nn.Module:
        return model


class FakePolicyModel(nn.Module):
    """Small causal LM stub compatible with the freeform policy code path."""

    def __init__(self, hidden_dim: int, vocab_size: int = 260) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, past_key_values, use_cache, return_dict
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("FakePolicyModel requires input_ids or inputs_embeds.")
            inputs_embeds = self.embedding(input_ids)
        hidden, _ = self.rnn(inputs_embeds)
        logits = self.output(hidden)
        return SimpleNamespace(logits=logits, past_key_values=None)


class FakeTransformersModule:
    AutoTokenizer = FakeTokenizer


def _install_fake_freeform_backend() -> None:
    from sembelief_wm.agent import phase2 as phase2_module

    phase2_module._import_policy_dependencies = lambda: (FakeTransformersModule, FakePeftModule)
    phase2_module._load_policy_model = (
        lambda *, transformers, model_name, model_kwargs: FakePolicyModel(
            hidden_dim=32,
            vocab_size=260,
        )
    )


def _assert_metrics_finite(history: list[tuple[int, dict[str, float]]], label: str) -> None:
    if not history:
        raise AssertionError(f"{label}: expected at least one logged update.")
    for step, metrics in history:
        for name, value in metrics.items():
            if not torch.isfinite(torch.tensor(value)):
                raise AssertionError(f"{label}: non-finite metric at step {step}: {name}={value}")


def run_mlp_smoke() -> None:
    print("[1/3] MLP PPO smoke")
    config = make_test_config(policy_family="mlp")
    world_model = WorldModel(config, MockBackbone(config.hidden_dim))
    actor_critic = LatentActorCritic(config, device=torch.device("cpu"))
    logger = CaptureLogger()
    trainer = Phase2Trainer(
        config=config,
        world_model=world_model,
        actor_critic=actor_critic,
        data_source=DummyDataSource(config),
        device=torch.device("cpu"),
        logger=logger,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(checkpoint_dir=tmpdir)
        checkpoint = Path(tmpdir) / "latest.pt"
        if not checkpoint.exists():
            raise AssertionError("MLP smoke expected a PPO checkpoint.")
    _assert_metrics_finite(logger.history, "mlp")
    print(f"  Logged {len(logger.history)} updates; checkpoint save/load path exercised.")


def run_freeform_token_ppo_smoke() -> None:
    print("[2/3] Freeform token-PPO smoke")
    _install_fake_freeform_backend()
    config = make_test_config(policy_family="vlm_freeform", freeform_update_mode="token_ppo")
    world_model = WorldModel(config, MockBackbone(config.hidden_dim))
    actor_critic = LatentActorCritic(config, device=torch.device("cpu"))
    logger = CaptureLogger()
    trainer = Phase2Trainer(
        config=config,
        world_model=world_model,
        actor_critic=actor_critic,
        data_source=DummyDataSource(config),
        device=torch.device("cpu"),
        logger=logger,
    )
    rollout = trainer.collect_rollout()
    if rollout.response_token_ids is None or rollout.response_mask is None or rollout.old_token_log_probs is None:
        raise AssertionError("Freeform token-PPO smoke expected token-level rollout statistics.")
    metrics = trainer.update_actor_critic(rollout)
    if "phase2/kl_divergence" not in metrics:
        raise AssertionError("Freeform token-PPO smoke expected KL metric.")
    print(
        "  token_ppo rollout ok:"
        f" parse_success={float(rollout.parse_success.float().mean().item()) if rollout.parse_success is not None else 0.0:.3f},"
        f" kl={metrics['phase2/kl_divergence']:.4f}"
    )


def run_freeform_hybrid_lookahead_smoke() -> None:
    print("[3/3] Freeform hybrid-scoring + lookahead smoke")
    _install_fake_freeform_backend()
    config = make_test_config(
        policy_family="vlm_freeform",
        freeform_update_mode="hybrid_scoring",
        use_lookahead=True,
        lookahead_width=2,
    )
    world_model = WorldModel(config, MockBackbone(config.hidden_dim))
    actor_critic = LatentActorCritic(config, device=torch.device("cpu"))
    logger = CaptureLogger()
    trainer = Phase2Trainer(
        config=config,
        world_model=world_model,
        actor_critic=actor_critic,
        data_source=DummyDataSource(config),
        device=torch.device("cpu"),
        logger=logger,
    )
    rollout = trainer.collect_rollout()
    if rollout.old_log_probs is None:
        raise AssertionError("Hybrid-scoring smoke expected scalar action log-probs.")
    metrics = trainer.update_actor_critic(rollout)
    if trainer.planner is None or not trainer.planner.enabled:
        raise AssertionError("Hybrid-scoring smoke expected lookahead planner to be enabled.")
    print(
        "  hybrid_scoring + lookahead ok:"
        f" reward_mean={float(rollout.rewards.mean().item()):.4f},"
        f" clipfrac={metrics['phase2/clipfrac']:.4f}"
    )


def main() -> None:
    torch.manual_seed(0)
    run_mlp_smoke()
    run_freeform_token_ppo_smoke()
    run_freeform_hybrid_lookahead_smoke()
    print("Smoke test passed: Phase 2 MLP PPO, freeform token_ppo, and freeform hybrid lookahead.")


if __name__ == "__main__":
    main()
