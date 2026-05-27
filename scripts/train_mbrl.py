"""End-to-end model-based RL training entry point.

Three-stage pipeline:
  Stage 1: WM warm-up (offline supervised)  — reuse train_phase1.py or load checkpoint
  Stage 2: Policy warm-up (latent BC)       — placeholder, to be implemented
  Stage 3: Alternating online RL            — imagined PPO + periodic WM refresh + real eval

Usage:
    # Minimal smoke test (mock backbone, random data):
    python scripts/train_mbrl.py --mode mock --total-updates 2

    # Full run with existing WM checkpoint:
    python scripts/train_mbrl.py --mode full \
        --wm-checkpoint checkpoints/phase1_5k/latest.pt \
        --data-dir data/sokoban_10k/tokenized \
        --world-model-mode frozen_wm \
        --total-updates 200

    # Alternating WM + online collection:
    python scripts/train_mbrl.py --mode full \
        --wm-checkpoint checkpoints/phase1_5k/latest.pt \
        --data-dir data/sokoban_10k/tokenized \
        --world-model-mode alternating_wm \
        --collect-every 20 --collect-episodes 4 \
        --total-updates 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sembelief_wm.config import (
    BackboneConfig,
    BeliefConfig,
    Config,
    CurriculumConfig,
    EMAConfig,
    EncoderConfig,
    EnvironmentConfig,
    Phase2Config,
    PPOConfig as LegacyPPOConfig,
    RewardConfig,
    SIGRegConfig,
    TrainingConfig,
    WandbConfig,
    WorldModelRefreshConfig,
)
from sembelief_wm.data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from sembelief_wm.model import TransitionBackbone, WorldModel
from sembelief_wm.model.policy_backbone import QwenPolicyBackbone

from sembelief_wm.rl.llm_policy import LLMActorCritic, LLMPolicyConfig
from sembelief_wm.rl.ppo import PPOConfig

from sembelief_wm.envs.sokoban.action_adapter import SokobanActionAdapter

from sembelief_wm.pipelines.assemble import assemble_llm_pipeline
from sembelief_wm.pipelines.mbrl_train import PipelineConfig, Logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PrintLogger:
    """Minimal stdout logger that also forwards to wandb if available."""
    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if not k.startswith("_")]
        print(f"[update {step:>5d}] {' | '.join(parts)}")
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)


class MockBackbone(TransitionBackbone):
    """Identity backbone for testing without Qwen."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Named "lora" so QwenPolicyBackbone.trainable_parameters() can find them
        self.lora_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.lora_proj(tokens)


def build_backbone(config: Config, device: torch.device) -> TransitionBackbone:
    if config.backbone.model_name == "mock":
        return MockBackbone(hidden_dim=config.hidden_dim)
    from sembelief_wm.model import QwenTransitionBackbone
    return QwenTransitionBackbone.from_config(config, device_map={"": device})


def load_wm_checkpoint(
    world_model: WorldModel,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """Load a Phase 1 WM checkpoint. Returns the training step."""
    print(f"Loading WM checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model", ckpt)

    # Handle partial/missing keys gracefully
    missing, unexpected = world_model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (may be expected for new heads)")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")

    step = ckpt.get("step", 0)
    print(f"  Loaded at step {step}")
    return step


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def mock_config(total_updates: int = 2) -> Config:
    """Tiny config for smoke tests without Qwen."""
    return Config(
        hidden_dim=64,
        belief=BeliefConfig(num_slots=4),
        encoder=EncoderConfig(compressed_tokens=4, vjepa2_raw_dim=64),
        backbone=BackboneConfig(model_name="mock"),
        reward=RewardConfig(),
        sigreg=SIGRegConfig(num_projections=32, buffer_size=32),
        curriculum=CurriculumConfig(horizons=[1], switch_steps=[0]),
        training=TrainingConfig(total_steps=1, checkpoint_every=1),
        wandb=WandbConfig(enabled=False),
        env=EnvironmentConfig(),
        phase2=Phase2Config(
            world_model_mode="frozen_wm",
            ppo=LegacyPPOConfig(
                total_updates=total_updates,
                rollout_batch_size=4,
                rollout_horizon=4,
                epochs_per_update=2,
                minibatch_size=8,
                eval_every=1,
                eval_episodes=2,
                checkpoint_every=total_updates,
            ),
        ),
    )


def full_config(args: argparse.Namespace) -> Config:
    """Full config for real runs with Qwen backbone."""
    return Config(
        hidden_dim=3584,
        belief=BeliefConfig(num_slots=args.belief_slots),
        encoder=EncoderConfig(
            encoder_type=args.encoder_type,
            compressed_tokens=args.belief_slots,
            vjepa2_raw_dim=1408,
        ),
        backbone=BackboneConfig(
            model_name=args.backbone_model,
            attention_mode="bidirectional",
            action_conditioning_mode="text",
            attn_implementation=args.attn_implementation,
        ),
        reward=RewardConfig(readout="mean_pool", supervision_source="posterior"),
        sigreg=SIGRegConfig(),
        ema=EMAConfig(),
        curriculum=CurriculumConfig(horizons=[1, 2, 4, 8], switch_steps=[0, 1, 2, 3]),
        training=TrainingConfig(total_steps=1, checkpoint_every=1),
        wandb=WandbConfig(
            enabled=not args.no_wandb,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            group=args.wandb_group,
            job_type="mbrl_online",
        ),
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
            env_ids=["sokoban"],
        ),
        phase2=Phase2Config(
            world_model_mode=args.world_model_mode,
            ppo=LegacyPPOConfig(
                total_updates=args.total_updates,
                rollout_batch_size=args.rollout_batch_size,
                rollout_horizon=args.rollout_horizon,
                epochs_per_update=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                actor_lr=args.lr,
                critic_lr=args.lr,
                entropy_coef=args.entropy_coef,
                normalize_advantages=args.normalize_advantages,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                eval_max_steps=args.eval_max_steps,
                checkpoint_every=args.checkpoint_every,
                collect_every=args.collect_every,
                collect_episodes=args.collect_episodes,
                collect_max_steps=args.collect_max_steps,
                reward_mapping=args.reward_mapping,
            ),
            wm_refresh=WorldModelRefreshConfig(
                refresh_every=args.wm_refresh_every,
                updates_per_refresh=args.wm_refresh_updates,
                batch_size=args.wm_refresh_batch_size,
                lr=args.wm_refresh_lr,
                horizon=args.wm_refresh_horizon,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end model-based RL training (LLM policy + world model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["mock", "full"], default="mock",
        help="mock = tiny model + random data; full = real Qwen + real data",
    )

    # Data & checkpoints
    parser.add_argument("--data-dir", type=str, default=None, help="Tokenized episodes directory")
    parser.add_argument("--wm-checkpoint", type=str, default=None, help="Phase 1 WM checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Pipeline checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mbrl")

    # World model mode
    parser.add_argument(
        "--world-model-mode", choices=["frozen_wm", "alternating_wm"], default="frozen_wm",
    )
    parser.add_argument("--shared-backbone", action="store_true", default=True,
                        help="Share Qwen backbone between WM and policy")
    parser.add_argument("--independent-backbone", dest="shared_backbone", action="store_false")

    # PPO
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--normalize-advantages", action="store_true", default=True)
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false")
    parser.add_argument("--reward-mapping", choices=["sigmoid_affine", "raw_sigmoid", "clipped_logit"],
                        default="raw_sigmoid")

    # Eval
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-max-steps", type=int, default=25)

    # Online collection
    parser.add_argument("--collect-every", type=int, default=0, help="0 = no online collection")
    parser.add_argument("--collect-episodes", type=int, default=4)
    parser.add_argument("--collect-max-steps", type=int, default=0)

    # WM refresh (alternating mode)
    parser.add_argument("--wm-refresh-every", type=int, default=1)
    parser.add_argument("--wm-refresh-updates", type=int, default=4)
    parser.add_argument("--wm-refresh-batch-size", type=int, default=4)
    parser.add_argument("--wm-refresh-lr", type=float, default=1e-4)
    parser.add_argument("--wm-refresh-horizon", type=int, default=8)

    # Model
    parser.add_argument("--backbone-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--encoder-type", choices=["vjepa2", "qwen"], default="vjepa2")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                        help="Attention implementation: sdpa, flash_attention_2, eager")
    parser.add_argument("--belief-slots", type=int, default=36)
    parser.add_argument("--checkpoint-every", type=int, default=50)

    # Logging
    parser.add_argument("--wandb-project", type=str, default="mbrl-vlm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # ── Seed ──
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # ── Config ──
    if args.mode == "mock":
        config = mock_config(args.total_updates)
        # Override wandb from CLI flags (mock_config defaults to disabled)
        if not args.no_wandb:
            config.wandb = WandbConfig(
                enabled=True,
                project=args.wandb_project,
                run_name=args.wandb_run_name,
                group=args.wandb_group,
                job_type="mbrl_mock",
            )
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required for --mode full")
        config = full_config(args)

    # ── Data ──
    if args.mode == "mock":
        dataset = TokenizedEpisodeDataset.from_random(
            num_episodes=20, seq_len=8, config=config,
        )
    else:
        dataset = TokenizedEpisodeDataset.from_directory(args.data_dir)
    data_source = OfflineDataSource(dataset, config)

    print(f"Dataset: {len(dataset)} episodes")

    # ── World Model ──
    backbone = build_backbone(config, device)
    world_model = WorldModel(config, backbone)
    world_model.to(device)

    if args.wm_checkpoint:
        load_wm_checkpoint(world_model, args.wm_checkpoint, device)
    elif args.mode == "full":
        print("Warning: No WM checkpoint provided. Using random initialization.")

    # ── Action Adapter ──
    action_adapter = SokobanActionAdapter(
        hidden_dim=config.hidden_dim,
    )

    # ── WM Refresher (alternating mode) ──
    wm_refresher = None
    if config.phase2.world_model_mode == "alternating_wm":
        from sembelief_wm.train import Phase1Trainer
        from sembelief_wm.trainers.wm_refresher import WorldModelRefresher

        # Create a Phase1Trainer just for its train_one_step method
        phase1_trainer = Phase1Trainer(
            config=config,
            world_model=world_model,
            data_source=data_source,
            device=device,
        )
        wm_refresher = WorldModelRefresher(
            trainer=phase1_trainer,
            steps_per_refresh=config.phase2.wm_refresh.updates_per_refresh,
        )

    # ── Logger ──
    wandb_run = None
    if config.wandb.enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb.project,
                name=config.wandb.run_name,
                group=config.wandb.group,
                job_type=config.wandb.job_type,
                config={
                    "mode": args.mode,
                    "world_model_mode": config.phase2.world_model_mode,
                    "shared_backbone": args.shared_backbone,
                    "rollout_batch_size": config.phase2.ppo.rollout_batch_size,
                    "rollout_horizon": config.phase2.ppo.rollout_horizon,
                    "ppo_epochs": config.phase2.ppo.epochs_per_update,
                    "minibatch_size": config.phase2.ppo.minibatch_size,
                    "lr": config.phase2.ppo.actor_lr,
                    "entropy_coef": config.phase2.ppo.entropy_coef,
                    "normalize_advantages": config.phase2.ppo.normalize_advantages,
                    "eval_episodes": config.phase2.ppo.eval_episodes,
                },
            )
        except ImportError:
            print("Warning: wandb not installed, logging to stdout only")

    logger = PrintLogger(wandb_run=wandb_run)

    # ── Assemble Pipeline ──
    print("\n=== Assembling MBRL Pipeline ===")
    print(f"  Mode: {config.phase2.world_model_mode}")
    print(f"  Backbone: {'shared' if args.shared_backbone else 'independent'}")
    print(f"  Rollout: batch={config.phase2.ppo.rollout_batch_size}, horizon={config.phase2.ppo.rollout_horizon}")
    print(f"  PPO: epochs={config.phase2.ppo.epochs_per_update}, lr={config.phase2.ppo.actor_lr}, entropy={config.phase2.ppo.entropy_coef}")
    print(f"  Eval: every={config.phase2.ppo.eval_every}, episodes={config.phase2.ppo.eval_episodes}")
    if config.phase2.world_model_mode == "alternating_wm":
        wm_r = config.phase2.wm_refresh
        print(f"  WM refresh: every={wm_r.refresh_every}, steps={wm_r.updates_per_refresh}, lr={wm_r.lr}")
    if config.phase2.ppo.collect_every > 0:
        print(f"  Online collect: every={config.phase2.ppo.collect_every}, episodes={config.phase2.ppo.collect_episodes}")
    print()

    pipeline, llm_policy = assemble_llm_pipeline(
        config=config,
        world_model=world_model,
        action_adapter=action_adapter,
        data_source=data_source,
        device=device,
        shared_backbone=args.shared_backbone,
        logger=logger,
        wm_refresher=wm_refresher,
    )

    # ── Resume ──
    start_update = 0
    if args.resume:
        start_update = pipeline.load_checkpoint(args.resume)
        print(f"Resumed from update {start_update}")

    # ── Train ──
    print(f"=== Starting training: {config.phase2.ppo.total_updates} updates ===\n")
    t0 = time.time()

    pipeline.train(
        checkpoint_dir=args.checkpoint_dir,
        start_update=start_update,
    )

    elapsed = time.time() - t0
    print(f"\n=== Training complete in {elapsed:.1f}s ===")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
