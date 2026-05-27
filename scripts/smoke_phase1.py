"""End-to-end smoke test for the minimal Phase 1 training stack.

This script intentionally uses tiny dimensions and mock dependencies so the
full training path can be exercised on CPU without the real Qwen/V-JEPA
wrappers. It verifies:
- Phase1Trainer runs across all curriculum stages
- SIGReg buffer flushes on curriculum switches
- checkpoints save/load correctly
- gradients and parameter updates are non-degenerate
"""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch import Tensor

from sembelief_wm import (
    Phase1Trainer,
    SequenceBatch,
    TransitionBackbone,
    VisualTokenPreprocessor,
    WorldModel,
    duplicate_single_frame_clip,
    load_observation_tokens,
    save_observation_tokens,
)
from sembelief_wm.config import (
    BeliefConfig,
    Config,
    CurriculumConfig,
    EncoderConfig,
    EnvironmentConfig,
    RewardConfig,
    SIGRegConfig,
    TrainingConfig,
)


class MockBackbone(TransitionBackbone):
    """Shape-preserving backbone that mixes token-local and global context."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        context = tokens.mean(dim=1, keepdim=True)
        return self.token_proj(tokens) + self.context_proj(context)


class TrackingWorldModel(WorldModel):
    """World model variant that counts SIGReg-buffer flushes."""

    def __init__(self, config: Config, backbone: TransitionBackbone) -> None:
        super().__init__(config, backbone)
        self.flush_count = 0

    def flush_sigreg_buffer(self) -> None:
        self.flush_count += 1
        super().flush_sigreg_buffer()


class DummyDataSource:
    """Deterministic random data source for smoke testing."""

    def __init__(self, config: Config, *, seq_len: int = 8) -> None:
        self.config = config
        self.seq_len = seq_len
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(1234)

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
        reward_events = torch.rand(
            batch_size,
            self.seq_len,
            generator=self.generator,
        ) < 0.1
        if batch_size > 0:
            reward_events[0, 0] = True
        rewards = torch.where(
            reward_events,
            torch.full((batch_size, self.seq_len), 10.9),
            torch.full((batch_size, self.seq_len), -0.1),
        )

        episode_lengths = torch.full((batch_size,), self.seq_len, dtype=torch.long)
        if batch_size > 1 and self.seq_len > 1:
            episode_lengths[-1] = self.seq_len - 1

        return SequenceBatch(
            obs_tokens=obs_tokens,
            actions=actions,
            rewards=rewards,
            episode_lengths=episode_lengths,
        )


class CaptureLogger:
    """Simple in-memory logger for smoke-test assertions."""

    def __init__(self) -> None:
        self.history: list[tuple[int, dict[str, float]]] = []

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        self.history.append((step, metrics))


def small_test_config() -> Config:
    """Construct a tiny config that still exercises all major code paths."""

    return Config(
        hidden_dim=32,
        belief=BeliefConfig(num_slots=4, gate_bias_init=-2.0),
        encoder=EncoderConfig(
            vjepa2_raw_tokens=8,
            vjepa2_raw_dim=12,
            compressed_tokens=4,
        ),
        reward=RewardConfig(readout="mean_pool", supervision_source="posterior"),
        sigreg=SIGRegConfig(
            lambda_ep=0.05,
            lambda_var=0.005,
            num_projections=16,
            integration_range=5.0,
            num_quadrature=5,
            buffer_size=32,
            flush_on_curriculum_switch=True,
        ),
        curriculum=CurriculumConfig(
            horizons=[1, 2, 4],
            switch_steps=[0, 1, 2],
            horizon_decay=0.9,
        ),
        training=TrainingConfig(
            total_steps=3,
            episodes_per_step=2,
            lr=1e-3,
            grad_clip=1.0,
            checkpoint_every=1,
            lambda_reward=0.1,
            dtype="fp32",
        ),
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
        ),
    )


def _assert_finite_metrics(logger: CaptureLogger) -> None:
    for step, metrics in logger.history:
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise AssertionError(f"Non-finite metric at step {step}: {name}={value}")


def _assert_parameters_changed(
    before: dict[str, Tensor],
    model: nn.Module,
) -> None:
    changed = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not torch.allclose(before[name], param.detach()):
            changed = True
            break
    if not changed:
        raise AssertionError("Smoke test expected at least one trainable parameter to update.")


def _assert_nonzero_gradients(model: nn.Module) -> None:
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0:
            return
    raise AssertionError("Smoke test expected at least one non-zero finite gradient.")


def _assert_visual_preprocessing(config: Config, checkpoint_dir: Path) -> None:
    frames = torch.randn(2, 3, 8, 8)
    btchw = duplicate_single_frame_clip(frames, layout="btchw")
    bcthw = duplicate_single_frame_clip(frames, layout="bcthw")
    if tuple(btchw.shape) != (2, 2, 3, 8, 8):
        raise AssertionError(f"Unexpected btchw pseudo-clip shape: {tuple(btchw.shape)}")
    if tuple(bcthw.shape) != (2, 3, 2, 8, 8):
        raise AssertionError(f"Unexpected bcthw pseudo-clip shape: {tuple(bcthw.shape)}")

    preprocessor = VisualTokenPreprocessor(config)
    raw_tokens = torch.randn(
        2,
        config.encoder.vjepa2_raw_tokens,
        config.encoder.vjepa2_raw_dim,
    )
    obs_tokens = preprocessor(raw_tokens)
    expected = (2, config.encoder.compressed_tokens, config.hidden_dim)
    if tuple(obs_tokens.shape) != expected:
        raise AssertionError(
            f"Unexpected visual token shape: {tuple(obs_tokens.shape)} expected {expected}."
        )

    token_path = checkpoint_dir / "obs_tokens.pt"
    save_observation_tokens(obs_tokens, token_path)
    loaded = load_observation_tokens(token_path)
    if not torch.allclose(obs_tokens.cpu(), loaded):
        raise AssertionError("Saved and loaded observation tokens differ.")


def main() -> None:
    torch.manual_seed(0)
    config = small_test_config()

    world_model = TrackingWorldModel(config, MockBackbone(config.hidden_dim))
    data_source = DummyDataSource(config, seq_len=8)
    logger = CaptureLogger()
    trainer = Phase1Trainer(
        config=config,
        world_model=world_model,
        data_source=data_source,
        device="cpu",
        logger=logger,
    )

    initial_params = {
        name: param.detach().clone()
        for name, param in trainer.world_model.named_parameters()
        if param.requires_grad
    }

    with TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        _assert_visual_preprocessing(config, checkpoint_dir)
        trainer.train(checkpoint_dir=checkpoint_dir)

        checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
        if len(checkpoints) != config.training.total_steps:
            raise AssertionError(
                "Expected one checkpoint per step in the smoke test, got "
                f"{len(checkpoints)} for total_steps={config.training.total_steps}."
            )

        resumed = Phase1Trainer(
            config=config,
            world_model=TrackingWorldModel(config, MockBackbone(config.hidden_dim)),
            data_source=data_source,
            device="cpu",
        )
        resume_step = resumed.load_checkpoint(checkpoints[-1])
        if resume_step != config.training.total_steps:
            raise AssertionError(
                f"Expected resume step {config.training.total_steps}, got {resume_step}."
            )

    if world_model.flush_count != 2:
        raise AssertionError(f"Expected 2 curriculum flushes, got {world_model.flush_count}.")

    if len(logger.history) != config.training.total_steps:
        raise AssertionError(
            f"Expected {config.training.total_steps} logged steps, got {len(logger.history)}."
        )

    observed_horizons = [int(metrics["curriculum/horizon"]) for _, metrics in logger.history]
    if observed_horizons != [1, 2, 4]:
        raise AssertionError(f"Unexpected horizon schedule: {observed_horizons}")

    _assert_finite_metrics(logger)
    _assert_parameters_changed(initial_params, trainer.world_model)
    _assert_nonzero_gradients(trainer.world_model)

    print("Smoke test passed: Phase 1 trainer, curriculum, SIGReg flush, and checkpoints.")


if __name__ == "__main__":
    main()
