"""Phase 2 latent PPO training entry point for SemBelief-WM.

Usage:
    python scripts/train_phase2.py \
        --data-dir data/episodes \
        --wm-checkpoint checkpoints/latest.pt \
        --checkpoint-dir checkpoints/phase2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sembelief_wm.config import (  # noqa: E402
    ActorCriticConfig,
    AntiCollapseConfig,
    BackboneConfig,
    BeliefConfig,
    Config,
    CurriculumConfig,
    EMAConfig,
    EncoderConfig,
    EnvironmentConfig,
    FreeformPolicyConfig,
    LookaheadConfig,
    Phase2Config,
    PPOConfig,
    RewardConfig,
    SIGRegConfig,
    TrainingConfig,
    WandbConfig,
)
from sembelief_wm.data.datasource import OfflineDataSource, TokenizedEpisodeDataset  # noqa: E402
from sembelief_wm.model import TransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.agent import LatentActorCritic, Phase2Trainer  # noqa: E402


class MockBackbone(TransitionBackbone):
    """Identity backbone for smoke tests."""

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return tokens


class PrintLogger:
    """Simple stdout logger for PPO updates."""

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
        print(f"[update {step:>5d}] {' | '.join(parts)}")


class WandbLogger:
    """Minimal W&B logger for PPO updates."""

    def __init__(self, project: str, run_name: str | None) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(project=project, name=run_name)

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        self._wandb.log(metrics, step=step)


def phase2_sokoban_config(
    *,
    backbone_model_name: str,
    attention_mode: str,
    action_conditioning_mode: str,
    anti_collapse: str,
    policy_family: str,
    freeform_update_mode: str,
    total_updates: int,
    rollout_batch_size: int,
    rollout_horizon: int,
    epochs_per_update: int,
    minibatch_size: int,
    actor_lr: float,
    critic_lr: float,
    reward_mapping: str,
    kl_coef: float,
    lookahead_enabled: bool,
    lookahead_expansion_depth: int,
    lookahead_expansion_width: int,
    lookahead_rollout_horizon: int,
    eval_every: int,
    eval_episodes: int,
    eval_max_steps: int,
    wandb_project: str,
    wandb_run_name: str | None,
    wandb_enabled: bool,
) -> Config:
    if anti_collapse == "sigreg":
        anti_collapse_cfg = AntiCollapseConfig(
            use_sigreg=True,
            use_ema_target=False,
            use_ema_variance=False,
        )
    elif anti_collapse == "ema_teacher":
        anti_collapse_cfg = AntiCollapseConfig(
            use_sigreg=False,
            use_ema_target=True,
            use_ema_variance=False,
        )
    elif anti_collapse == "ema_teacher_var":
        anti_collapse_cfg = AntiCollapseConfig(
            use_sigreg=False,
            use_ema_target=True,
            use_ema_variance=True,
        )
    else:
        raise ValueError(f"Unsupported anti_collapse preset: {anti_collapse}")

    return Config(
        hidden_dim=3584,
        belief=BeliefConfig(num_slots=36),
        encoder=EncoderConfig(compressed_tokens=36, vjepa2_raw_dim=1408),
        backbone=BackboneConfig(
            model_name=backbone_model_name,
            attention_mode=attention_mode,
            action_conditioning_mode=action_conditioning_mode,
        ),
        reward=RewardConfig(),
        sigreg=SIGRegConfig(),
        ema=EMAConfig(),
        anti_collapse=anti_collapse_cfg,
        curriculum=CurriculumConfig(),
        training=TrainingConfig(),
        phase2=Phase2Config(
            actor_critic=ActorCriticConfig(policy_family=policy_family),
            freeform_policy=FreeformPolicyConfig(update_mode=freeform_update_mode),
            ppo=PPOConfig(
                total_updates=total_updates,
                rollout_batch_size=rollout_batch_size,
                rollout_horizon=rollout_horizon,
                epochs_per_update=epochs_per_update,
                minibatch_size=minibatch_size,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                reward_mapping=reward_mapping,
                kl_coef=kl_coef,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
            ),
            lookahead=LookaheadConfig(
                enabled=lookahead_enabled,
                expansion_depth=lookahead_expansion_depth,
                expansion_width=lookahead_expansion_width,
                rollout_horizon=lookahead_rollout_horizon,
            ),
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


def build_backbone(config: Config, device: torch.device, backbone_kind: str) -> TransitionBackbone:
    if backbone_kind == "mock":
        if config.backbone.action_conditioning_mode == "text":
            raise ValueError("text action conditioning requires the Qwen backbone, not mock.")
        return MockBackbone()
    from sembelief_wm.model import QwenTransitionBackbone
    return QwenTransitionBackbone.from_config(config, device_map={"": device})


def load_world_model_checkpoint(world_model: WorldModel, checkpoint_path: str | Path, device: torch.device) -> int:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Handle key path mismatches between different Qwen/PEFT versions.
    # Common differences:
    #   - Extra "language_model." in path (Qwen2.5-VL wraps LM layers)
    #   - Extra ".model" from PEFT wrapping level
    model_keys = set(world_model.state_dict().keys())
    remapped_state: dict[str, torch.Tensor] = {}
    for k, v in model_state.items():
        if k in model_keys:
            remapped_state[k] = v
            continue
        # Try common key remappings
        candidates = [
            k.replace(".language_model.", "."),
            k.replace("base_model.model.model.", "base_model.model."),
            k.replace(".language_model.", ".").replace("base_model.model.model.", "base_model.model."),
        ]
        matched = False
        for alt in candidates:
            if alt in model_keys:
                remapped_state[alt] = v
                matched = True
                break
        if not matched:
            remapped_state[k] = v  # keep as-is, will show up as unexpected

    missing, unexpected = world_model.load_state_dict(remapped_state, strict=False)
    loaded = len(model_keys) - len(missing)
    print(f"  Phase 1 checkpoint: {loaded}/{len(model_keys)} keys loaded, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        # Check if any important trainable keys are still missing
        important_missing = [k for k in missing if any(x in k for x in ["lora", "belief_update", "reward_head"])]
        if important_missing:
            print(f"  WARNING: {len(important_missing)} important trainable keys still missing!")
            for k in important_missing[:5]:
                print(f"    {k}")
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 latent PPO training")
    parser.add_argument("--data-dir", type=str, required=True, help="Tokenized episode directory")
    parser.add_argument("--wm-checkpoint", type=str, required=True, help="Phase 1 world model checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--resume", type=str, default=None, help="Phase 2 PPO checkpoint to resume")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", choices=["mock", "qwen"], default="qwen")
    parser.add_argument("--backbone-model-name", type=str, default=BackboneConfig().model_name)
    parser.add_argument("--attention-mode", choices=["bidirectional", "causal"], default="bidirectional")
    parser.add_argument("--action-conditioning-mode", choices=["embedded", "text"], default="embedded")
    parser.add_argument("--anti-collapse", choices=["sigreg", "ema_teacher", "ema_teacher_var"], default="sigreg")
    parser.add_argument("--policy-family", choices=["mlp", "vlm_freeform"], default="mlp")
    parser.add_argument("--freeform-update-mode", choices=["token_ppo", "hybrid_scoring"], default="token_ppo")
    parser.add_argument("--total-updates", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--reward-mapping", choices=["sigmoid_affine", "raw_sigmoid", "clipped_logit"], default="sigmoid_affine")
    parser.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty coefficient against old policy (0 = disabled)")
    parser.add_argument("--lookahead", action="store_true", help="Enable lookahead planning for action selection")
    parser.add_argument("--lookahead-depth", type=int, default=1, help="Expansion depth for lookahead")
    parser.add_argument("--lookahead-width", type=int, default=4, help="Expansion width per node (num_actions = exhaustive)")
    parser.add_argument("--lookahead-horizon", type=int, default=4, help="Policy rollout steps after expansion")
    parser.add_argument("--wandb-project", type=str, default="mbrl-vlm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50, help="Run real-env eval every N updates (0=disabled)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per eval round")
    parser.add_argument("--eval-max-steps", type=int, default=25, help="Max steps per eval episode")
    parser.add_argument("--use-vjepa-eval", action="store_true", help="Load V-JEPA 2 for accurate eval tokenization")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.policy_family == "vlm_freeform" and args.backbone == "mock":
        raise ValueError("vlm_freeform policy requires the real Qwen backbone, not mock.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = phase2_sokoban_config(
        backbone_model_name=args.backbone_model_name,
        attention_mode=args.attention_mode,
        action_conditioning_mode=args.action_conditioning_mode,
        anti_collapse=args.anti_collapse,
        policy_family=args.policy_family,
        freeform_update_mode=args.freeform_update_mode,
        total_updates=args.total_updates,
        rollout_batch_size=args.rollout_batch_size,
        rollout_horizon=args.rollout_horizon,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        reward_mapping=args.reward_mapping,
        kl_coef=args.kl_coef,
        lookahead_enabled=args.lookahead,
        lookahead_expansion_depth=args.lookahead_depth,
        lookahead_expansion_width=args.lookahead_width,
        lookahead_rollout_horizon=args.lookahead_horizon,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_enabled=not args.no_wandb,
    )

    dataset = TokenizedEpisodeDataset.from_directory(args.data_dir)
    data_source = OfflineDataSource(dataset, config)

    backbone = build_backbone(config, device, args.backbone)
    world_model = WorldModel(config, backbone)
    phase1_step = load_world_model_checkpoint(world_model, args.wm_checkpoint, device)

    actor_critic = LatentActorCritic(config, device=device)
    logger = PrintLogger()
    if config.wandb.enabled:
        try:
            logger = WandbLogger(config.wandb.project, config.wandb.run_name)
        except Exception as exc:
            print(f"W&B disabled for this run: {exc}")

    trainer = Phase2Trainer(
        config=config,
        world_model=world_model,
        actor_critic=actor_critic,
        data_source=data_source,
        device=device,
        logger=logger,
    )

    start_update = 0
    if args.resume:
        start_update = trainer.load_checkpoint(args.resume)
        print(f"Resumed Phase 2 PPO checkpoint at update {start_update}")

    print(f"Loaded Phase 1 checkpoint from step {phase1_step}")
    print(f"Training Phase 2: {config.phase2.ppo.total_updates} PPO updates on {device}")
    print(f"  Backbone: {config.backbone.model_name}")
    print(f"  Action conditioning: {config.backbone.action_conditioning_mode}")
    print(f"  Policy family: {config.phase2.actor_critic.policy_family}")
    if config.phase2.actor_critic.policy_family == "vlm_freeform":
        print(f"  Freeform update mode: {config.phase2.freeform_policy.update_mode}")
    print(f"  Rollout batch size: {config.phase2.ppo.rollout_batch_size}")
    print(f"  Rollout horizon: {config.phase2.ppo.rollout_horizon}")
    print(f"  Reward mapping: {config.phase2.ppo.reward_mapping}")
    if config.phase2.ppo.kl_coef > 0:
        print(f"  KL penalty coef: {config.phase2.ppo.kl_coef}")
    if config.phase2.lookahead.enabled:
        la = config.phase2.lookahead
        print(f"  Lookahead: depth={la.expansion_depth}, width={la.expansion_width}, horizon={la.rollout_horizon}")

    # Optional: load V-JEPA 2 tokenizer for accurate real-env eval
    eval_tokenizer = None
    if args.use_vjepa_eval and config.phase2.ppo.eval_every > 0:
        from sembelief_wm.data.tokenizers.image import ImageTokenizer
        print("Loading V-JEPA 2 tokenizer for real-env evaluation...")
        eval_tokenizer = ImageTokenizer(config, device=str(device))
        print("  V-JEPA 2 tokenizer loaded.")

    if config.phase2.ppo.eval_every > 0:
        print(f"  Real-env eval: every {config.phase2.ppo.eval_every} updates, {config.phase2.ppo.eval_episodes} episodes")
        if eval_tokenizer is None:
            print("  WARNING: Using fallback tokenizer for eval (no V-JEPA 2). Eval quality may be degraded.")

    trainer.train(
        checkpoint_dir=args.checkpoint_dir,
        start_update=start_update,
        tokenizer=eval_tokenizer,
    )
    print("Phase 2 training complete.")


if __name__ == "__main__":
    main()
