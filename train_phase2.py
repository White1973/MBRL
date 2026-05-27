"""Phase 2 latent PPO training entry point for SemBelief-WM.

Usage:
    python scripts/train_phase2.py \
        --data-dir data/episodes \
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
    WorldModelRefreshConfig,
)
from sembelief_wm.data.datasource import OfflineDataSource, TokenizedEpisodeDataset  # noqa: E402
from sembelief_wm.model import TransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.agent import LatentActorCritic, OnlineReplayBuffer, Phase2Trainer  # noqa: E402


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

    def __init__(
        self,
        project: str,
        run_name: str | None,
        group: str | None = None,
        job_type: str = "phase2",
        config: dict | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(
            project=project,
            name=run_name,
            group=group,
            job_type=job_type,
            config=config,
        )

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        self._wandb.log(metrics, step=step)


def env_settings(env: str) -> dict[str, object]:
    if env == "sokoban":
        return {
            "env_ids": ["sokoban"],
            "num_actions": 4,
            "null_action_id": 4,
        }
    if env == "habitat_objectnav":
        return {
            "env_ids": ["habitat_objectnav"],
            "num_actions": 6,
            "null_action_id": 6,
        }
    raise ValueError(f"Unsupported environment: {env}")


def phase2_single_env_config(
    *,
    env: str,
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
    normalize_advantages: bool,
    reward_mapping: str,
    kl_coef: float,
    world_model_mode: str,
    warmup_ratio: float,
    wm_refresh_every: int,
    wm_refresh_updates: int,
    wm_refresh_data_mode: str,
    online_data_ratio: float,
    wm_refresh_batch_size: int,
    wm_refresh_lr: float,
    wm_refresh_horizon: int,
    wm_refresh_warmup_steps: int,
    lookahead_enabled: bool,
    lookahead_expansion_depth: int,
    lookahead_expansion_width: int,
    lookahead_rollout_horizon: int,
    eval_every: int,
    eval_episodes: int,
    eval_max_steps: int,
    collect_every: int,
    collect_episodes: int,
    collect_max_steps: int,
    wandb_project: str,
    wandb_run_name: str | None,
    wandb_group: str | None,
    wandb_job_type: str,
    wandb_enabled: bool,
) -> Config:
    settings = env_settings(env)
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
            world_model_mode=world_model_mode,
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
                normalize_advantages=normalize_advantages,
                reward_mapping=reward_mapping,
                kl_coef=kl_coef,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                collect_every=collect_every,
                collect_episodes=collect_episodes,
                collect_max_steps=collect_max_steps,
            ),
            wm_refresh=WorldModelRefreshConfig(
                warmup_ratio=warmup_ratio,
                refresh_every=wm_refresh_every,
                updates_per_refresh=wm_refresh_updates,
                data_mix_mode=wm_refresh_data_mode,
                online_data_ratio=online_data_ratio,
                batch_size=wm_refresh_batch_size,
                lr=wm_refresh_lr,
                warmup_steps=wm_refresh_warmup_steps,
                horizon=wm_refresh_horizon,
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
            group=wandb_group,
            job_type=wandb_job_type,
        ),
        env=EnvironmentConfig(
            num_actions=int(settings["num_actions"]),
            null_action_id=int(settings["null_action_id"]),
            env_ids=list(settings["env_ids"]),
        ),
    )


def resolve_wm_refresh_online_ratio(
    mode: str,
    ratio_override: float | None,
) -> tuple[str, float]:
    if ratio_override is not None:
        return "custom", float(ratio_override)
    if mode == "offline_only":
        return mode, 0.0
    if mode == "mixed":
        return mode, 0.5
    if mode == "online_only":
        return mode, 1.0
    raise ValueError(f"Unknown --wm-refresh-data-mode: {mode}")


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


def build_real_env_tokenizer(config: Config, device: torch.device, *, use_vjepa_eval: bool) -> object | None:
    env_id = config.env.env_ids[0]
    if env_id == "habitat_objectnav":
        from sembelief_wm.data.tokenizers.habitat_objectnav import HabitatObjectNavTokenizer

        return HabitatObjectNavTokenizer(
            output_dim=config.hidden_dim,
            rgb_tokens=16,
            goal_tokens=4,
            include_depth=True,
        )
    if use_vjepa_eval:
        from sembelief_wm.data.tokenizers.image import ImageTokenizer

        return ImageTokenizer(config, device=str(device))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 latent PPO training")
    parser.add_argument("--data-dir", type=str, required=True, help="Tokenized episode directory")
    parser.add_argument(
        "--wm-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional world model checkpoint. Required for frozen_wm; alternating_wm "
            "can omit it and bootstrap the WM with --warmup-ratio."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--resume", type=str, default=None, help="Phase 2 PPO checkpoint to resume")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", choices=["mock", "qwen"], default="qwen")
    parser.add_argument("--env", choices=["sokoban", "habitat_objectnav"], default="sokoban")
    parser.add_argument("--backbone-model-name", type=str, default=BackboneConfig().model_name)
    parser.add_argument("--attention-mode", choices=["bidirectional", "causal"], default="bidirectional")
    parser.add_argument("--action-conditioning-mode", choices=["embedded", "text", "continuous"], default="embedded")
    parser.add_argument("--anti-collapse", choices=["sigreg", "ema_teacher", "ema_teacher_var"], default="sigreg")
    parser.add_argument(
        "--policy-family",
        choices=["mlp", "vlm_freeform", "vlm_action_head"],
        default="vlm_action_head",
    )
    parser.add_argument("--freeform-update-mode", choices=["token_ppo", "hybrid_scoring"], default="token_ppo")
    parser.add_argument(
        "--world-model-mode",
        choices=["frozen_wm", "alternating_wm"],
        default=Phase2Config().world_model_mode,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Offline WM warm-up steps as a fraction of configured WM training.total_steps.",
    )
    parser.add_argument("--total-updates", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--epochs-per-update", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=PPOConfig().normalize_advantages,
        help="Whether to normalize PPO advantages before actor updates.",
    )
    parser.add_argument("--wm-refresh-every", type=int, default=1, help="Run supervised WM refresh every N PPO updates (0=disabled)")
    parser.add_argument("--wm-refresh-updates", type=int, default=4, help="Supervised WM steps per refresh event")
    parser.add_argument(
        "--wm-refresh-data-mode",
        choices=["offline_only", "mixed", "online_only"],
        default=WorldModelRefreshConfig().data_mix_mode,
        help="Ablation switch for WM refresh batch composition.",
    )
    parser.add_argument(
        "--online-data-ratio",
        type=float,
        default=None,
        help="Optional override for online replay fraction in each WM refresh batch; overrides --wm-refresh-data-mode.",
    )
    parser.add_argument("--online-data-dir", type=str, default=None, help="Directory for tokenized online replay episodes")
    parser.add_argument("--wm-refresh-batch-size", type=int, default=4, help="Episodes per supervised WM refresh step")
    parser.add_argument("--wm-refresh-lr", type=float, default=1e-4, help="Learning rate for supervised WM refresh")
    parser.add_argument("--wm-refresh-horizon", type=int, default=8, help="Fixed BPTT horizon for supervised WM refresh")
    parser.add_argument("--wm-refresh-warmup-steps", type=int, default=100, help="LR warmup steps for the WM refresh optimizer")
    parser.add_argument("--reward-mapping", choices=["sigmoid_affine", "raw_sigmoid", "clipped_logit"], default="raw_sigmoid")
    parser.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty coefficient against old policy (0 = disabled)")
    parser.add_argument("--lookahead", action="store_true", help="Enable lookahead planning for action selection")
    parser.add_argument("--lookahead-depth", type=int, default=1, help="Expansion depth for lookahead")
    parser.add_argument("--lookahead-width", type=int, default=4, help="Expansion width per node (num_actions = exhaustive)")
    parser.add_argument("--lookahead-horizon", type=int, default=4, help="Policy rollout steps after expansion")
    parser.add_argument("--wandb-project", type=str, default="mbrl-vlm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group (links Phase 1 + Phase 2)")
    parser.add_argument("--wandb-job-type", type=str, default="phase2", help="W&B job type")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50, help="Run real-env eval every N updates (0=disabled)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per eval round")
    parser.add_argument("--eval-max-steps", type=int, default=25, help="Max steps per eval episode")
    parser.add_argument("--collect-every", type=int, default=PPOConfig().collect_every, help="Collect real-env replay data every N updates (0=disabled)")
    parser.add_argument("--collect-episodes", type=int, default=PPOConfig().collect_episodes, help="Episodes per online replay collection round")
    parser.add_argument("--collect-max-steps", type=int, default=PPOConfig().collect_max_steps, help="Max steps per collected episode (<=0 uses adapter default)")
    parser.add_argument("--use-vjepa-eval", action="store_true", help="Load V-JEPA 2 for accurate eval tokenization")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.policy_family in {"vlm_freeform", "vlm_action_head"} and args.backbone == "mock":
        raise ValueError(f"{args.policy_family} policy requires the real Qwen backbone, not mock.")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1].")
    resolved_data_mode, resolved_online_ratio = resolve_wm_refresh_online_ratio(
        args.wm_refresh_data_mode,
        args.online_data_ratio,
    )
    if not 0.0 <= resolved_online_ratio <= 1.0:
        raise ValueError("--online-data-ratio must be in [0, 1].")
    if resolved_online_ratio > 0.0 and args.world_model_mode != "alternating_wm":
        raise ValueError("--online-data-ratio requires --world-model-mode alternating_wm.")
    if args.world_model_mode == "frozen_wm" and args.wm_checkpoint is None:
        raise ValueError("--world-model-mode frozen_wm requires --wm-checkpoint.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = phase2_single_env_config(
        env=args.env,
        backbone_model_name=args.backbone_model_name,
        attention_mode=args.attention_mode,
        action_conditioning_mode=args.action_conditioning_mode,
        anti_collapse=args.anti_collapse,
        policy_family=args.policy_family,
        freeform_update_mode=args.freeform_update_mode,
        world_model_mode=args.world_model_mode,
        warmup_ratio=args.warmup_ratio,
        total_updates=args.total_updates,
        rollout_batch_size=args.rollout_batch_size,
        rollout_horizon=args.rollout_horizon,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        normalize_advantages=args.normalize_advantages,
        reward_mapping=args.reward_mapping,
        kl_coef=args.kl_coef,
        wm_refresh_every=args.wm_refresh_every,
        wm_refresh_updates=args.wm_refresh_updates,
        wm_refresh_data_mode=resolved_data_mode,
        online_data_ratio=resolved_online_ratio,
        wm_refresh_batch_size=args.wm_refresh_batch_size,
        wm_refresh_lr=args.wm_refresh_lr,
        wm_refresh_horizon=args.wm_refresh_horizon,
        wm_refresh_warmup_steps=args.wm_refresh_warmup_steps,
        lookahead_enabled=args.lookahead,
        lookahead_expansion_depth=args.lookahead_depth,
        lookahead_expansion_width=args.lookahead_width,
        lookahead_rollout_horizon=args.lookahead_horizon,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        collect_every=args.collect_every,
        collect_episodes=args.collect_episodes,
        collect_max_steps=args.collect_max_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_enabled=not args.no_wandb,
    )

    dataset = TokenizedEpisodeDataset.from_directory(args.data_dir)
    data_source = OfflineDataSource(dataset, config)

    backbone = build_backbone(config, device, args.backbone)
    world_model = WorldModel(config, backbone)
    if args.wm_checkpoint is not None:
        phase1_step = load_world_model_checkpoint(world_model, args.wm_checkpoint, device)
        wm_init = f"checkpoint:{args.wm_checkpoint}"
    else:
        phase1_step = 0
        wm_init = "random"
        print("No WM checkpoint provided; initializing world model from scratch.")

    actor_critic = LatentActorCritic(config, device=device)
    online_replay = None
    if config.phase2.wm_refresh.online_data_ratio > 0.0:
        online_root = Path(args.online_data_dir) if args.online_data_dir else Path(args.checkpoint_dir) / "online_replay"
        online_replay = OnlineReplayBuffer(root=online_root, config=config)
    logger = PrintLogger()
    if config.wandb.enabled:
        try:
            logger = WandbLogger(
                config.wandb.project,
                config.wandb.run_name,
                group=config.wandb.group,
                job_type=config.wandb.job_type or "phase2",
                config={
                    "wm_checkpoint": str(args.wm_checkpoint) if args.wm_checkpoint is not None else None,
                    "wm_init": wm_init,
                    "policy_family": config.phase2.actor_critic.policy_family,
                    "world_model_mode": config.phase2.world_model_mode,
                    "training_total_steps": config.training.total_steps,
                    "wm_refresh_data_mode": config.phase2.wm_refresh.data_mix_mode,
                    "warmup_ratio": config.phase2.wm_refresh.warmup_ratio,
                    "wm_refresh_every": config.phase2.wm_refresh.refresh_every,
                    "wm_refresh_updates": config.phase2.wm_refresh.updates_per_refresh,
                    "wm_refresh_batch_size": config.phase2.wm_refresh.batch_size,
                    "wm_refresh_horizon": config.phase2.wm_refresh.horizon,
                    "wm_refresh_lr": config.phase2.wm_refresh.lr,
                    "online_data_ratio": config.phase2.wm_refresh.online_data_ratio,
                    "rollout_horizon": config.phase2.ppo.rollout_horizon,
                    "rollout_batch_size": config.phase2.ppo.rollout_batch_size,
                    "normalize_advantages": config.phase2.ppo.normalize_advantages,
                    "ppo_total_updates": config.phase2.ppo.total_updates,
                    "ppo_eval_every": config.phase2.ppo.eval_every,
                    "ppo_eval_episodes": config.phase2.ppo.eval_episodes,
                    "ppo_eval_max_steps": config.phase2.ppo.eval_max_steps,
                    "ppo_collect_every": config.phase2.ppo.collect_every,
                    "ppo_collect_episodes": config.phase2.ppo.collect_episodes,
                    "ppo_collect_max_steps": config.phase2.ppo.collect_max_steps,
                },
            )
        except Exception as exc:
            print(f"W&B disabled for this run: {exc}")

    trainer = Phase2Trainer(
        config=config,
        world_model=world_model,
        actor_critic=actor_critic,
        data_source=data_source,
        online_replay=online_replay,
        device=device,
        logger=logger,
    )

    start_update = 0
    if args.resume:
        start_update = trainer.load_checkpoint(args.resume)
        print(f"Resumed Phase 2 PPO checkpoint at update {start_update}")

    if args.wm_checkpoint is not None:
        print(f"Loaded WM checkpoint from step {phase1_step}: {args.wm_checkpoint}")
    else:
        print("Loaded no WM checkpoint; WM pretraining is controlled by warmup_ratio only.")
    print(f"Training Phase 2: {config.phase2.ppo.total_updates} PPO updates on {device}")
    print(f"  Backbone: {config.backbone.model_name}")
    print(f"  Action conditioning: {config.backbone.action_conditioning_mode}")
    print(f"  Policy family: {config.phase2.actor_critic.policy_family}")
    print(f"  World model mode: {config.phase2.world_model_mode}")
    if config.phase2.actor_critic.policy_family == "vlm_freeform":
        print(f"  Freeform update mode: {config.phase2.freeform_policy.update_mode}")
    elif config.phase2.actor_critic.policy_family == "vlm_action_head":
        print("  VLM policy mode: projector -> policy VLM -> categorical action head")
    print(f"  Rollout batch size: {config.phase2.ppo.rollout_batch_size}")
    print(f"  Rollout horizon: {config.phase2.ppo.rollout_horizon}")
    print(f"  Reward mapping: {config.phase2.ppo.reward_mapping}")
    if config.phase2.world_model_mode == "alternating_wm":
        wm_cfg = config.phase2.wm_refresh
        warmup_steps = int(config.training.total_steps * wm_cfg.warmup_ratio)
        print(
            "  WM refresh: "
            f"warmup_ratio={wm_cfg.warmup_ratio} ({warmup_steps}/{config.training.total_steps} supervised steps), "
            f"every {wm_cfg.refresh_every} updates, "
            f"{wm_cfg.updates_per_refresh} step(s), "
            f"batch={wm_cfg.batch_size}, horizon={wm_cfg.horizon}, "
            f"online_ratio={wm_cfg.online_data_ratio}"
        )
        if online_replay is not None:
            print(f"  Online replay dir: {online_replay.root}")
    if config.phase2.ppo.kl_coef > 0:
        print(f"  KL penalty coef: {config.phase2.ppo.kl_coef}")
    if config.phase2.lookahead.enabled:
        la = config.phase2.lookahead
        print(f"  Lookahead: depth={la.expansion_depth}, width={la.expansion_width}, horizon={la.rollout_horizon}")

    # Optional: load a real observation tokenizer for real-env eval and/or online replay collection.
    eval_tokenizer = None
    needs_real_tokenizer = (
        config.phase2.ppo.eval_every > 0
        or config.phase2.wm_refresh.online_data_ratio > 0.0
    )
    if needs_real_tokenizer:
        env_id = config.env.env_ids[0]
        if env_id == "habitat_objectnav":
            print("Loading Habitat ObjectNav tokenizer for real-env evaluation / replay collection...")
            eval_tokenizer = build_real_env_tokenizer(
                config,
                device,
                use_vjepa_eval=args.use_vjepa_eval,
            )
            print("  Habitat ObjectNav tokenizer loaded.")
        elif args.use_vjepa_eval:
            print("Loading V-JEPA 2 tokenizer for real-env evaluation / replay collection...")
            eval_tokenizer = build_real_env_tokenizer(
                config,
                device,
                use_vjepa_eval=args.use_vjepa_eval,
            )
            print("  V-JEPA 2 tokenizer loaded.")

    if config.phase2.ppo.eval_every > 0:
        print(f"  Real-env eval: every {config.phase2.ppo.eval_every} updates, {config.phase2.ppo.eval_episodes} episodes")
        if eval_tokenizer is None:
            print("  WARNING: Using fallback tokenizer for real-env eval. Eval quality may be degraded.")
    if config.phase2.ppo.collect_every > 0 and config.phase2.wm_refresh.online_data_ratio > 0.0:
        print(
            f"  Real-env collect: every {config.phase2.ppo.collect_every} updates, "
            f"{config.phase2.ppo.collect_episodes} episodes"
        )

    trainer.train(
        checkpoint_dir=args.checkpoint_dir,
        start_update=start_update,
        tokenizer=eval_tokenizer,
    )
    print("Phase 2 training complete.")


if __name__ == "__main__":
    main()
