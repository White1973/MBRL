"""Phase 1 training entry point for SemBelief-WM.

Usage:
    # With random obs tokens (no V-JEPA 2, no real Sokoban):
    python scripts/train_phase1.py --mode mock --total-steps 100

    # With precomputed episode tokens:
    python scripts/train_phase1.py --mode offline --data-dir data/episodes

    # With real Qwen backbone:
    python scripts/train_phase1.py --mode offline --data-dir data/episodes --backbone qwen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sembelief_wm.config import (
    AntiCollapseConfig,
    BackboneConfig,
    BeliefConfig,
    Config,
    CurriculumConfig,
    EMAConfig,
    EncoderConfig,
    EnvironmentConfig,
    RewardConfig,
    SIGRegConfig,
    TrainingConfig,
    WandbConfig,
)
from sembelief_wm.data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from sembelief_wm.model import TransitionBackbone, WorldModel
from sembelief_wm.train import Phase1Trainer


class MockBackbone(TransitionBackbone):
    """Identity backbone for testing without Qwen."""
    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return tokens


class PrintLogger:
    """Simple stdout logger."""
    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
        print(f"[step {step:>6d}] {' | '.join(parts)}")


def small_config(total_steps: int = 100) -> Config:
    """Small config for mock/testing runs."""
    return Config(
        hidden_dim=64,
        belief=BeliefConfig(num_slots=4),
        encoder=EncoderConfig(compressed_tokens=4, vjepa2_raw_dim=64),
        backbone=BackboneConfig(model_name="mock"),
        reward=RewardConfig(),
        sigreg=SIGRegConfig(num_projections=32, buffer_size=32),
        curriculum=CurriculumConfig(
            horizons=[1, 2, 4],
            switch_steps=[0, max(1, total_steps // 3), max(2, 2 * total_steps // 3)],
        ),
        training=TrainingConfig(
            total_steps=total_steps,
            episodes_per_step=2,
            warmup_steps=10,
            weight_decay=0.01,
            checkpoint_every=max(1, total_steps // 2),
        ),
        wandb=WandbConfig(enabled=False),
        env=EnvironmentConfig(),
    )


def sokoban_config(
    total_steps: int = 10000,
    backbone: str = "mock",
    backbone_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    wandb_project: str = "mbrl-vlm",
    wandb_run_name: str | None = None,
    wandb_enabled: bool = True,
    attention_mode: str = "bidirectional",
    action_conditioning_mode: str = "embedded",
    anti_collapse: str = "sigreg",
    use_sigreg: bool | None = None,
    use_ema_target: bool | None = None,
    use_ema_variance: bool | None = None,
    curriculum_horizons: list[int] | None = None,
    curriculum_switch_steps: list[int] | None = None,
    encoder_type: str = "vjepa2",
    belief_slots: int = 36,
    reward_readout: str = "mean_pool",
    reward_supervision: str = "posterior",
    sigreg_lambda_ep: float = 0.05,
    sigreg_lambda_var: float = 0.005,
    sigreg_lambda_cov: float = 0.001,
    seed: int = 1,
) -> Config:
    """Config for Sokoban single-env training with real V-JEPA 2 tokens."""
    if curriculum_horizons is None:
        curriculum_horizons = [1, 2, 4]
    if curriculum_switch_steps is None:
        curriculum_switch_steps = [0, max(1, total_steps // 3), max(2, 2 * total_steps // 3)]

    if anti_collapse == "sigreg":
        anti_collapse_cfg = AntiCollapseConfig(
            use_sigreg=True if use_sigreg is None else use_sigreg,
            use_ema_target=False if use_ema_target is None else use_ema_target,
            use_ema_variance=False if use_ema_variance is None else use_ema_variance,
        )
    elif anti_collapse == "ema_teacher":
        anti_collapse_cfg = AntiCollapseConfig(
            use_sigreg=False if use_sigreg is None else use_sigreg,
            use_ema_target=True if use_ema_target is None else use_ema_target,
            use_ema_variance=False if use_ema_variance is None else use_ema_variance,
        )
    elif anti_collapse == "ema_teacher_var":
        anti_collapse_cfg = AntiCollapseConfig(
            use_sigreg=False if use_sigreg is None else use_sigreg,
            use_ema_target=True if use_ema_target is None else use_ema_target,
            use_ema_variance=True if use_ema_variance is None else use_ema_variance,
        )
    else:
        raise ValueError(f"Unsupported anti_collapse preset: {anti_collapse}")

    return Config(
        hidden_dim=3584,
        belief=BeliefConfig(num_slots=belief_slots),
        encoder=EncoderConfig(encoder_type=encoder_type, compressed_tokens=36, vjepa2_raw_dim=1408),
        backbone=BackboneConfig(
            model_name=backbone_model_name,
            attention_mode=attention_mode,
            action_conditioning_mode=action_conditioning_mode,
        ) if backbone != "mock" else BackboneConfig(model_name="mock"),
        reward=RewardConfig(readout=reward_readout, supervision_source=reward_supervision),
        sigreg=SIGRegConfig(
            lambda_ep=sigreg_lambda_ep,
            lambda_var=sigreg_lambda_var,
            lambda_cov=sigreg_lambda_cov,
        ),
        ema=EMAConfig(),
        anti_collapse=anti_collapse_cfg,
        curriculum=CurriculumConfig(
            horizons=curriculum_horizons,
            switch_steps=curriculum_switch_steps,
        ),
        training=TrainingConfig(
            total_steps=total_steps,
            episodes_per_step=4,
            warmup_steps=500,
            weight_decay=0.01,
            base_lr_factor=0.01,
            checkpoint_every=5000,
        ),
        wandb=WandbConfig(
            enabled=wandb_enabled,
            project=wandb_project,
            run_name=wandb_run_name,
        ),
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
            env_ids=["sokoban"],
        ),
    )


def build_backbone(config: Config, device: torch.device) -> TransitionBackbone:
    """Build backbone from config."""
    if config.backbone.model_name == "mock":
        if config.backbone.action_conditioning_mode == "text":
            raise ValueError("text action conditioning requires the Qwen backbone, not mock.")
        return MockBackbone()
    from sembelief_wm.model import QwenTransitionBackbone
    return QwenTransitionBackbone.from_config(config, device_map={"": device})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 world-model training")
    parser.add_argument(
        "--mode", choices=["mock", "offline"], default="mock",
        help="mock = random data + identity backbone; offline = precomputed tokens",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Episode tokens directory")
    parser.add_argument("--backbone", choices=["mock", "qwen"], default="qwen")
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--wandb-project", type=str, default="mbrl-vlm", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--val-every", type=int, default=100, help="Run validation every N steps")
    parser.add_argument("--attention-mode", choices=["bidirectional", "causal"], default="bidirectional")
    parser.add_argument("--action-conditioning-mode", choices=["embedded", "text"], default="embedded")
    parser.add_argument("--anti-collapse", choices=["sigreg", "ema_teacher", "ema_teacher_var"], default="sigreg")
    parser.add_argument("--use-sigreg", dest="use_sigreg", action="store_true", help="Enable SIGReg regularization")
    parser.add_argument("--no-sigreg", dest="use_sigreg", action="store_false", help="Disable SIGReg regularization")
    parser.add_argument("--use-ema-target", dest="use_ema_target", action="store_true", help="Use EMA posterior as the dynamics target")
    parser.add_argument("--no-ema-target", dest="use_ema_target", action="store_false", help="Use online posterior as the dynamics target")
    parser.add_argument("--ema-variance-reg", dest="use_ema_variance", action="store_true", help="Add EMA variance regularization on top of the chosen target")
    parser.add_argument("--no-ema-variance-reg", dest="use_ema_variance", action="store_false", help="Disable EMA variance regularization")
    parser.add_argument("--no-curriculum", action="store_true", help="Fixed horizon=4 from step 0")
    parser.add_argument("--encoder-type", choices=["vjepa2", "qwen"], default="vjepa2", help="Visual encoder type")
    parser.add_argument("--belief-slots", type=int, default=36, help="Number of latent belief slots")
    parser.add_argument("--reward-readout", choices=["mean_pool", "attention_pool", "learned_query"], default="mean_pool")
    parser.add_argument("--reward-supervision", choices=["posterior", "prior", "both"], default="posterior")
    parser.add_argument("--sigreg-lambda-ep", type=float, default=0.05, help="SIGReg ep regularization weight")
    parser.add_argument("--sigreg-lambda-var", type=float, default=0.005, help="SIGReg variance regularization weight")
    parser.add_argument("--sigreg-lambda-cov", type=float, default=0.001, help="SIGReg covariance regularization weight")
    parser.add_argument("--lambda-ep", dest="sigreg_lambda_ep", type=float, help="Alias for --sigreg-lambda-ep")
    parser.add_argument("--lambda-var", dest="sigreg_lambda_var", type=float, help="Alias for --sigreg-lambda-var")
    parser.add_argument("--lambda-cov", dest="sigreg_lambda_cov", type=float, help="Alias for --sigreg-lambda-cov")
    parser.add_argument("--train-episodes", type=int, default=None, help="Limit the number of training episodes")
    parser.add_argument("--val-episodes", type=int, default=None, help="Use an explicit validation set size")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.set_defaults(use_sigreg=None, use_ema_target=None, use_ema_variance=None)
    args = parser.parse_args()

    if args.mode == "offline" and args.backbone == "mock":
        raise ValueError("offline mode requires --backbone qwen; mock backbone is only supported for --mode mock")

    device = torch.device(args.device)

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "mock":
        config = small_config(args.total_steps)
        dataset = TokenizedEpisodeDataset.from_random(
            num_episodes=20,
            seq_len=8,
            config=config,
        )
    else:
        curriculum_horizons = None
        curriculum_switch_steps = None
        if args.no_curriculum:
            curriculum_horizons = [4]
            curriculum_switch_steps = [0]
        config = sokoban_config(
            args.total_steps,
            backbone=args.backbone,
            backbone_model_name=BackboneConfig().model_name,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_enabled=not args.no_wandb,
            attention_mode=args.attention_mode,
            action_conditioning_mode=args.action_conditioning_mode,
            anti_collapse=args.anti_collapse,
            use_sigreg=args.use_sigreg,
            use_ema_target=args.use_ema_target,
            use_ema_variance=args.use_ema_variance,
            curriculum_horizons=curriculum_horizons,
            curriculum_switch_steps=curriculum_switch_steps,
            encoder_type=args.encoder_type,
            belief_slots=args.belief_slots,
            reward_readout=args.reward_readout,
            reward_supervision=args.reward_supervision,
            sigreg_lambda_ep=args.sigreg_lambda_ep,
            sigreg_lambda_var=args.sigreg_lambda_var,
            sigreg_lambda_cov=args.sigreg_lambda_cov,
            seed=args.seed,
        )
        if args.data_dir is None:
            parser.error("--data-dir is required for offline mode")
        dataset = TokenizedEpisodeDataset.from_directory(args.data_dir)

    backbone = build_backbone(config, device)
    world_model = WorldModel(config, backbone)

    # Train/val split
    val_source = None
    num_val = None
    if args.val_episodes is not None:
        if args.val_episodes <= 0:
            parser.error("--val-episodes must be positive")
        num_val = args.val_episodes
    elif args.val_split > 0 and len(dataset) > 10:
        num_val = max(1, int(len(dataset) * args.val_split))

    if num_val is not None:
        if num_val >= len(dataset):
            parser.error("--val-episodes/--val-split leaves no training data")
        num_train = len(dataset) - num_val
        if args.train_episodes is not None:
            if args.train_episodes <= 0:
                parser.error("--train-episodes must be positive")
            num_train = min(num_train, args.train_episodes)
        if num_train <= 0:
            parser.error("Training split is empty after applying episode limits")
        train_dataset = TokenizedEpisodeDataset(dataset.episodes[:num_train])
        val_start = num_train
        val_stop = num_train + num_val
        if val_stop > len(dataset):
            parser.error("Requested train/val episode counts exceed dataset size")
        val_dataset = TokenizedEpisodeDataset(dataset.episodes[val_start:val_stop])
        data_source = OfflineDataSource(train_dataset, config)
        val_source = OfflineDataSource(val_dataset, config)
        print(f"  Train: {num_train} episodes, Val: {num_val} episodes")
    else:
        if args.train_episodes is not None:
            if args.train_episodes <= 0:
                parser.error("--train-episodes must be positive")
            dataset = TokenizedEpisodeDataset(dataset.episodes[:args.train_episodes])
        data_source = OfflineDataSource(dataset, config)

    logger = PrintLogger()

    trainer = Phase1Trainer(
        config=config,
        world_model=world_model,
        data_source=data_source,
        device=device,
        logger=logger,
    )

    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
        print(f"Resumed from step {start_step}")

    print(f"Training Phase 1: {config.training.total_steps} steps, device={device}")
    print(f"  Backbone: {config.backbone.model_name}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Belief slots: {config.belief.num_slots}")
    print(f"  Curriculum: {list(zip(config.curriculum.switch_steps, config.curriculum.horizons))}")
    print(
        "  Anti-collapse:"
        f" sigreg={config.anti_collapse.use_sigreg},"
        f" ema_target={config.anti_collapse.use_ema_target},"
        f" ema_variance={config.anti_collapse.use_ema_variance}"
    )

    trainer.train(
        start_step=start_step,
        checkpoint_dir=args.checkpoint_dir,
        val_source=val_source,
        val_every=args.val_every,
    )
    trainer.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
