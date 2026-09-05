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
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
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
    EnvRewardSpec,
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

@contextmanager
def isolated_torch_rng(seed: int):
    """Run deterministic diagnostics without perturbing CPU or CUDA RNG.

    ``torch.manual_seed`` seeds every visible CUDA generator as well as CPU.
    Saving only ``torch.random.get_rng_state()`` therefore silently changes
    later imagined PPO rollouts. ``fork_rng`` snapshots and restores all
    visible CUDA devices plus the CPU generator.
    """
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        yield

class PrintLogger:
    """Minimal stdout logger that also forwards to wandb if available.

    Also accumulates a per-key history so we can print a run-end summary
    (entropy min / collapse counts / clip-zero ratio / vloss max / ev min /
    SR mean±std). This makes run-to-run comparison diagnose *whether the
    entropy floor worked* rather than relying on noisy success_rate alone.
    """
    # Metrics tracked for the run-end summary. Each maps to a key in the
    # per-update metrics dict (or None if derived).
    _TRACKED = [
        "ppo/entropy", "ppo/clip_fraction", "ppo/value_loss",
        "ppo/explained_variance", "ppo/policy_loss",
        "ppo/num_minibatches", "ppo/actor_update",
        "ppo/behavior_bc_loss", "ppo/behavior_bc_accuracy",
        "rollout/reward_mean", "eval/success_rate",
    ]

    def __init__(self, wandb_run=None):
        self.wandb_run = wandb_run
        from collections import defaultdict
        self._history = defaultdict(list)
        if self.wandb_run is not None:
            if os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1":
                # The checkpoint/global pipeline update is provenance, not
                # the learning clock for a released Actor branch. Configure
                # every training metric to use accepted Actor PPO updates so
                # a branch resumed from (for example) update 268 starts at 0.
                self.wandb_run.define_metric("trainer/actor_ppo_update")
                self.wandb_run.define_metric(
                    "*", step_metric="trainer/actor_ppo_update"
                )
            # Evaluation is scheduled by accepted Actor PPO updates, not by
            # global pipeline updates.  Global updates also include Critic
            # warm-up and guarded WM work, so using them as the W&B x-axis
            # makes otherwise identical Actor learning curves incomparable.
            self.wandb_run.define_metric("eval/actor_ppo_update")
            self.wandb_run.define_metric(
                "eval/*", step_metric="eval/actor_ppo_update"
            )

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        wandb_step = step
        if os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1":
            metrics = dict(metrics)
            actor_updates = int(metrics.get("ppo/actor_update", 0.0))
            # A released PPO branch has its own accepted-Actor-update clock.
            # `step` is the resumed pipeline/checkpoint update (for example
            # 269) and is provenance only; it must not become W&B's _step.
            # Baseline evaluation is step 0; metrics emitted after the first
            # accepted Actor optimization are step 1. Rejected transactions
            # do not advance this clock.
            wandb_step = actor_updates
            metrics["trainer/actor_ppo_update"] = float(wandb_step)
            metrics["trainer/global_pipeline_update"] = float(step)
        parts = [
            f"{k}={v:.4f}"
            for k, v in sorted(metrics.items())
            if not k.startswith("_")
            and not (
                os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1"
                and k == "trainer/global_pipeline_update"
            )
        ]
        if os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1":
            display_step = actor_updates
            print(
                f"[actor_ppo_update {display_step:>4d}] "
                f"{' | '.join(parts)}"
            )
        else:
            print(f"[update {step:>5d}] {' | '.join(parts)}")
        for k in self._TRACKED:
            if k in metrics:
                try:
                    self._history[k].append(float(metrics[k]))
                except (TypeError, ValueError):
                    pass
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=wandb_step)

    def summarize(self) -> None:
        """Print run-end diagnostics for floor/collapse/reward diagnosis."""
        import statistics as _stat
        h = self._history
        def _stats(key):
            v = h.get(key, [])
            return _values_stats(v)
        def _values_stats(v):
            if not v:
                return "n=0"
            return (f"n={len(v)} min={min(v):.4f} max={max(v):.4f} "
                    f"mean={sum(v)/len(v):.4f}")
        actor_minibatches = h.get("ppo/num_minibatches", [])
        ent_all = h.get("ppo/entropy", [])
        ent = [
            value for value, minibatches in zip(ent_all, actor_minibatches)
            if minibatches > 0
        ]
        collapse = sum(1 for e in ent if e < 0.05)
        clip_all = h.get("ppo/clip_fraction", [])
        clip = [
            value for value, minibatches in zip(clip_all, actor_minibatches)
            if minibatches > 0
        ]
        clip_zero = sum(1 for c in clip if c == 0.0)
        sr = [s for s in h.get("eval/success_rate", []) if s >= 0]
        sr_thirds = ""
        if sr:
            t = max(1, len(sr) // 3)
            sr_thirds = (f" | SR first1/3={sum(sr[:t])/t:.3f}"
                         f" last1/3={sum(sr[-t:])/t:.3f}"
                         f" mean={sum(sr)/len(sr):.3f}"
                         f" (std={_stat.pstdev(sr) if len(sr)>1 else 0:.3f})")
        print("\n" + "=" * 60)
        print("=== RUN-END DIAGNOSTICS ===")
        print(
            "  actor PPO updates   : "
            f"{sum(1 for value in actor_minibatches if value > 0)}/"
            f"{len(actor_minibatches)} logged updates"
        )
        print(f"  ppo/entropy        : {_values_stats(ent)}")
        print(f"    → entropy<0.05 (collapse) count : {collapse}/{len(ent)} "
              f"({(100*collapse/max(1,len(ent))):.0f}%)")
        print(f"  ppo/clip_fraction  : {_values_stats(clip)}")
        print(f"    → clip==0 count                  : {clip_zero}/{len(clip)} "
              f"({(100*clip_zero/max(1,len(clip))):.0f}%)")
        print(f"  ppo/value_loss     : {_stats('ppo/value_loss')}")
        print(f"  ppo/explained_var  : {_stats('ppo/explained_variance')}")
        print(f"  ppo/expert_bc_loss : {_stats('ppo/behavior_bc_loss')}")
        print(f"  ppo/expert_bc_acc  : {_stats('ppo/behavior_bc_accuracy')}")
        print(f"  rollout/reward_mean: {_stats('rollout/reward_mean')}")
        print(f"  eval/success_rate  : {_stats('eval/success_rate')}{sr_thirds}")
        print("=" * 60)


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
    *,
    allow_phase2_world_model: bool = False,
) -> dict:
    """Load a Phase 1 WM checkpoint and return its metadata container."""
    print(f"Loading WM checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_phase2_checkpoint = (
        isinstance(ckpt, dict)
        and "world_model" in ckpt
        and "policy" in ckpt
        and "update" in ckpt
    )
    if is_phase2_checkpoint and not allow_phase2_world_model:
        raise ValueError(
            "A Phase-2 checkpoint was supplied as --wm-checkpoint. Pass "
            "--allow-self-contained-phase2-wm only when reproducing a "
            "provenance-locked historical checkpoint whose original Phase-1 "
            "artifact is unavailable."
        )
    model_state = (
        ckpt["world_model"]
        if is_phase2_checkpoint
        else ckpt.get("model", ckpt)
    )

    # Handle partial/missing keys gracefully
    missing, unexpected = world_model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (may be expected for new heads)")
    if unexpected:
        print(
            f"  Warning: {len(unexpected)} unexpected keys: "
            f"{list(unexpected)[:20]}"
        )

    step = ckpt.get("step", ckpt.get("update", 0))
    if is_phase2_checkpoint:
        ckpt["_self_contained_phase2_wm"] = True
        print(
            "  Recovered world model from self-contained Phase-2 checkpoint "
            f"at update {step}"
        )
    else:
        print(f"  Loaded at step {step}")
    return ckpt if isinstance(ckpt, dict) else {"model": ckpt, "step": step}


def _detect_hidden_dim(model_name: str) -> int:
    """Read hidden_size from a local VLM config.json so D auto-matches the backbone.

    3B Qwen2.5-VL → 2048; 7B → 3584. Falls back to 3584 (7B default) if the
    config can't be read locally.
    """
    import json
    from pathlib import Path

    candidate = Path(model_name) / "config.json"
    if candidate.is_file():
        try:
            with candidate.open() as fh:
                cfg = json.load(fh)
            hidden = cfg.get("hidden_size")
            if isinstance(hidden, int):
                return hidden
        except (OSError, json.JSONDecodeError):
            pass
    print(f"[hidden_dim] could not read config.json at {candidate}; falling back to 3584 (7B).")
    return 3584


def _validate_offline_action_encoding(
    dataset: TokenizedEpisodeDataset,
    *,
    num_actions: int,
    wm_action_id_offset: int,
) -> None:
    """Fail before CUDA/model loading when offline action ids are misconfigured."""
    if wm_action_id_offset != 0:
        raise ValueError(
            "wm_action_id_offset must be 0 for corrected training. The legacy "
            "Sokoban offset=1 path maps environment action 4 onto the WM's "
            "reserved null/start id 4, so it cannot represent all four "
            "actions. Migrate the tokenized dataset to canonical 0..3 and "
            "retrain Phase 1."
        )
    observed: set[int] = set()
    for episode in dataset.episodes:
        transition_count = int(episode.episode_length)
        observed.update(
            int(value)
            for value in episode.actions[:transition_count].tolist()
        )
    canonical = set(observed)
    invalid = sorted(
        value for value in canonical if value < 0 or value >= num_actions
    )
    if invalid:
        raise ValueError(
            "Offline action encoding is incompatible with the policy: "
            f"observed WM ids={sorted(observed)}, "
            f"wm_action_id_offset={wm_action_id_offset}, "
            f"canonical ids={sorted(canonical)}, invalid={invalid}. "
            "Run scripts/migrate_sokoban_action_ids.py and use "
            "--wm-action-id-offset 0."
        )
    print(
        "Offline action encoding: "
        f"WM ids={sorted(observed)} -> policy ids={sorted(canonical)} "
        f"(offset={wm_action_id_offset})"
    )


def _validate_vjepa_teacher_dataset(
    dataset: TokenizedEpisodeDataset,
    config: Config,
) -> None:
    """Reject unpaired or misaligned data before loading Qwen on GPU."""
    enabled = (
        config.training.vjepa_teacher_prior_coef > 0.0
        or config.training.vjepa_teacher_posterior_coef > 0.0
        or config.training.vjepa_teacher_delta_coef > 0.0
    )
    if not enabled:
        return
    missing = [
        episode.env_id
        for episode in dataset.episodes
        if episode.semantic_teacher_tokens is None
    ]
    if missing:
        raise ValueError(
            "V-JEPA semantic teacher is enabled but this dataset is not paired: "
            f"{len(missing)}/{len(dataset)} episodes lack semantic_teacher_tokens. "
            "Build a paired Qwen+V-JEPA dataset with "
            "scripts/build_qwen_vjepa_teacher_dataset.py."
        )
    expected_k = config.encoder.semantic_teacher_tokens
    expected_d = config.encoder.semantic_teacher_dim
    for episode in dataset.episodes:
        teacher = episode.semantic_teacher_tokens
        assert teacher is not None
        if episode.metadata.get("visual_input") != "qwen2.5_vl_native":
            raise ValueError(
                "Qwen-native WM requires paired replay produced by "
                "build_qwen_vjepa_teacher_dataset.py; missing or invalid "
                f"visual_input provenance for env={episode.env_id!r}."
            )
        if teacher.shape[1:] != (expected_k, expected_d):
            raise ValueError(
                "V-JEPA teacher tensor shape does not match config: "
                f"episode={tuple(teacher.shape)}, expected "
                f"(T, {expected_k}, {expected_d})."
            )
        if teacher.shape[1] != episode.obs_tokens.shape[1]:
            raise ValueError(
                "Qwen and V-JEPA token grids must use the same number of "
                f"spatial slots, got {episode.obs_tokens.shape[1]} and "
                f"{teacher.shape[1]}."
            )
    print(
        "Semantic teacher: frozen V-JEPA paired targets verified "
        f"(K={expected_k}, D={expected_d}, episodes={len(dataset)})"
    )


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
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else _detect_hidden_dim(args.backbone_model)
    return Config(
        hidden_dim=hidden_dim,
        belief=BeliefConfig(num_slots=args.belief_slots),
        encoder=EncoderConfig(
            encoder_type=args.encoder_type,
            compressed_tokens=args.belief_slots,
            vjepa2_raw_dim=1408,
            semantic_teacher_type=(
                "vjepa2"
                if (
                    args.vjepa_teacher_prior_coef > 0.0
                    or args.vjepa_teacher_posterior_coef > 0.0
                    or args.vjepa_teacher_delta_coef > 0.0
                )
                else "none"
            ),
            semantic_teacher_tokens=args.belief_slots,
        ),
        backbone=BackboneConfig(
            model_name=args.backbone_model,
            attention_mode="bidirectional",
            action_conditioning_mode=args.action_conditioning_mode,
            attn_implementation=args.attn_implementation,
        ),
        reward=RewardConfig(
            readout="mean_pool",
            supervision_source="posterior",
            head_hidden_dim=args.reward_head_hidden_dim,
            success_reward_threshold=args.success_reward_threshold,
        ),
        sigreg=SIGRegConfig(),
        ema=EMAConfig(),
        curriculum=CurriculumConfig(horizons=[1, 2, 4, 8], switch_steps=[0, 1, 2, 3]),
        training=TrainingConfig(
            total_steps=1,
            checkpoint_every=1,
            # The WorldModel is constructed before the Phase-2 refresher.
            # Propagate action-auxiliary settings here as well so optional
            # modules such as inverse_action_head exist at construction time.
            delta_cosine_coef=args.wm_delta_cosine_coef,
            inverse_action_coef=args.wm_inverse_action_coef,
            inverse_action_mode=args.wm_inverse_action_mode,
            inverse_action_lr=args.wm_inverse_action_lr,
            observation_anchor_coef=args.observation_anchor_coef,
            observation_delta_anchor_coef=args.observation_delta_anchor_coef,
            observation_delta_min_rms=args.observation_delta_min_rms,
            observation_anchor_projection_trainable=(
                not args.freeze_observation_anchor_projection
            ),
            vjepa_teacher_prior_coef=args.vjepa_teacher_prior_coef,
            vjepa_teacher_posterior_coef=args.vjepa_teacher_posterior_coef,
            vjepa_teacher_delta_coef=args.vjepa_teacher_delta_coef,
            vjepa_teacher_delta_min_rms=args.vjepa_teacher_delta_min_rms,
            prior_isolation_mode=args.prior_isolation_mode,
            isolated_prior_repair=args.wm_refresh_prior_lora_only,
            prior_residual_rank=args.prior_residual_rank,
            posterior_observation_residual_scale=(
                args.posterior_observation_residual_scale
            ),
            posterior_grounding_mode=args.posterior_grounding_mode,
            posterior_recurrent_residual_scale=(
                args.posterior_recurrent_residual_scale
            ),
            posterior_action_free=args.posterior_action_free,
        ),
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
            env_ids=[args.env_id],
            reward_specs={args.env_id: EnvRewardSpec(positive_value=args.positive_value, negative_value=-0.1)},
        ),
        phase2=Phase2Config(
            world_model_mode=args.world_model_mode,
            ppo=LegacyPPOConfig(
                total_updates=args.total_updates,
                rollout_batch_size=args.rollout_batch_size,
                rollout_horizon=args.rollout_horizon,
                use_value_bootstrap=args.value_bootstrap,
                imagination_termination_mode=(
                    args.imagination_termination_mode
                ),
                rollouts_per_update=args.rollouts_per_update,
                epochs_per_update=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                recompute_old_log_probs=args.recompute_old_log_probs,
                actor_lr=(
                    args.actor_lr if args.actor_lr is not None else args.lr
                ),
                critic_lr=(
                    args.critic_lr if args.critic_lr is not None else args.lr
                ),
                critic_warmup_min_updates=args.critic_warmup_min_updates,
                critic_warmup_ev_threshold=args.critic_warmup_ev_threshold,
                critic_warmup_ev_patience=args.critic_warmup_ev_patience,
                critic_warmup_validation_fraction=(
                    args.critic_warmup_validation_fraction
                ),
                critic_warmup_validation_size=args.critic_warmup_validation_size,
                critic_warmup_replay_capacity=args.critic_warmup_replay_capacity,
                critic_warmup_train_samples=args.critic_warmup_train_samples,
                critic_warmup_ev_ema_alpha=args.critic_warmup_ev_ema_alpha,
                critic_warmup_mse_improvement=args.critic_warmup_mse_improvement,
                entropy_coef=args.entropy_coef,
                kl_coef=args.kl_coef,
                target_kl=args.target_kl,
                behavior_kl_coef=args.behavior_kl_coef,
                behavior_bc_coef=args.behavior_bc_coef,
                behavior_bc_batch_size=args.behavior_bc_batch_size,
                offline_bc_steps=args.offline_bc_steps,
                offline_bc_batch_size=args.offline_bc_batch_size,
                offline_bc_cache_size=args.offline_bc_cache_size,
                offline_bc_strategies=tuple(
                    value.strip() for value in args.offline_bc_strategies.split(",")
                    if value.strip()
                ),
                offline_bc_lr=args.offline_bc_lr,
                wm_action_id_offset=args.wm_action_id_offset,
                clip_epsilon=args.clip_epsilon,
                normalize_advantages=args.normalize_advantages,
                target_entropy=args.target_entropy,
                entropy_floor_coef=args.entropy_floor_coef,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                eval_max_steps=args.eval_max_steps,
                checkpoint_every=args.checkpoint_every,
                collect_every=args.collect_every,
                collect_episodes=args.collect_episodes,
                collect_max_steps=args.collect_max_steps,
                reward_mapping=args.reward_mapping,
                reward_scale=args.reward_scale,
                reward_confidence_floor=(
                    args.reward_confidence_floor
                    if args.reward_confidence_floor is not None
                    else 0.5
                ),
                reward_low_confidence_scale=args.reward_low_confidence_scale,
                value_coef=args.value_coef,
            ),
            wm_refresh=WorldModelRefreshConfig(
                refresh_every=args.wm_refresh_every,
                updates_per_refresh=args.wm_refresh_updates,
                batch_size=args.wm_refresh_batch_size,
                lr=args.wm_refresh_lr,
                base_lr_factor=args.wm_refresh_base_lr_factor,
                weight_decay=args.wm_refresh_weight_decay,
                warmup_steps=args.wm_refresh_warmup_steps,
                grad_clip=args.wm_refresh_grad_clip,
                horizon=args.wm_refresh_horizon,
                reward_pos_weight=args.wm_refresh_reward_pos_weight,
                reward_loss_coef=args.wm_refresh_reward_loss_coef,
                freeze_reward_head=args.wm_refresh_freeze_reward_head,
                validation_batches=args.wm_refresh_validation_batches,
                open_dynamics_coef=args.wm_open_dynamics_coef,
                prior_reward_coef=args.wm_prior_reward_coef,
                open_loop_horizon=args.wm_open_loop_horizon,
                open_dynamics_decay=args.wm_open_dynamics_decay,
                prior_reward_decay=args.wm_prior_reward_decay,
                delta_cosine_coef=args.wm_delta_cosine_coef,
                inverse_action_coef=args.wm_inverse_action_coef,
                inverse_action_mode=args.wm_inverse_action_mode,
                inverse_action_lr=args.wm_inverse_action_lr,
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
    parser.add_argument(
        "--env-id", choices=["sokoban", "frozenlake"], default="sokoban",
        help="Environment adapter only; model/trainer architecture is shared.",
    )
    parser.add_argument(
        "--orchestration", choices=["standard", "lewm"], default="standard",
        help="Training state machine. lewm uses the isolated Le-WM package; "
             "it does not execute the generic MBRLPipeline.train loop.",
    )

    # Data & checkpoints
    parser.add_argument("--data-dir", type=str, default=None, help="Tokenized episodes directory")
    parser.add_argument("--wm-checkpoint", type=str, default=None, help="Phase 1 WM checkpoint path")
    parser.add_argument(
        "--critic-h2-cache",
        type=str,
        default=None,
        help="Fixed split-safe posterior panels used only by exact-H2 "
             "Critic pretraining.",
    )
    parser.add_argument(
        "--allow-self-contained-phase2-wm",
        action="store_true",
        help="Recover WM tensors from a provenance-locked Phase-2 checkpoint. "
             "This compatibility path exists only because the historical "
             "Phase-1 artifact was not retained.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Pipeline checkpoint to resume from")
    parser.add_argument(
        "--cross-eval-wm-checkpoint", type=str, default=None,
        help="Evaluation-only: retain --resume policy tensors but replace the "
             "assembled world model with the world_model state from this "
             "Phase-2 checkpoint, then evaluate and exit.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mbrl")
    parser.add_argument(
        "--frozenlake-spatial-audit",
        action="store_true",
        help="Read-only posterior/prior player-position audit; exits before training.",
    )
    parser.add_argument("--spatial-audit-train-episodes", type=int, default=400)
    parser.add_argument("--spatial-audit-val-episodes", type=int, default=200)
    parser.add_argument("--spatial-audit-probe-steps", type=int, default=400)
    parser.add_argument("--spatial-audit-output", type=str, default=None)
    parser.add_argument("--spatial-audit-enforce-gate", action="store_true")
    parser.add_argument("--spatial-audit-min-posterior", type=float, default=0.90)
    parser.add_argument("--spatial-audit-min-prior", type=float, default=0.75)
    parser.add_argument("--spatial-audit-min-counterfactual", type=float, default=0.70)
    parser.add_argument("--spatial-audit-min-baseline-multiple", type=float, default=2.0)

    # World model mode
    parser.add_argument(
        "--world-model-mode", choices=["frozen_wm", "alternating_wm"], default="frozen_wm",
    )
    parser.add_argument("--shared-backbone", action="store_true", default=False,
                        help="Reuse both the WM Qwen base and its default LoRA; "
                             "supported only for fully read-only VLM branches.")
    parser.add_argument("--independent-backbone", dest="shared_backbone", action="store_false",
                        help="Share one frozen Qwen base but create independent "
                             "WM/actor/critic LoRA adapters (default).")

    # PPO
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument(
        "--value-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bootstrap non-terminal imagined fragment leaves from the Critic. "
             "Use with per-transition termination; terminal leaves remain zero.",
    )
    parser.add_argument(
        "--imagination-termination-mode",
        choices=["fixed_horizon", "predicted_success"],
        default="fixed_horizon",
        help="How imagined rollouts terminate. fixed_horizon is the safe "
             "default while the prior reward head is not calibrated as a "
             "per-transition termination model.",
    )
    parser.add_argument(
        "--rollouts-per-update",
        type=int,
        default=1,
        help="Collect this many small imagined-rollout chunks before one PPO "
             "update; increases advantage sample size without increasing "
             "per-forward Qwen memory.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument(
        "--recompute-old-log-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute the unchanged rollout actor log-probabilities using "
             "the exact fixed PPO minibatch partition before optimization.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=None,
        help="Actor head/LoRA LR; defaults to --lr for compatibility.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Critic head/LoRA LR; defaults to --lr for compatibility.",
    )
    parser.add_argument("--critic-warmup-min-updates", type=int, default=0)
    parser.add_argument("--critic-warmup-ev-threshold", type=float, default=0.2)
    parser.add_argument("--critic-warmup-ev-patience", type=int, default=3)
    parser.add_argument("--critic-warmup-validation-fraction", type=float, default=0.2)
    parser.add_argument("--critic-warmup-validation-size", type=int, default=256)
    parser.add_argument("--critic-warmup-replay-capacity", type=int, default=4096)
    parser.add_argument("--critic-warmup-train-samples", type=int, default=512)
    parser.add_argument("--critic-warmup-ev-ema-alpha", type=float, default=0.2)
    parser.add_argument("--critic-warmup-mse-improvement", type=float, default=0.05)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.0,
        help="Approximate KL penalty against the rollout policy.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Stop remaining PPO minibatches when approximate KL exceeds this value.",
    )
    parser.add_argument(
        "--behavior-kl-coef",
        type=float,
        default=0.0,
        help="KL(pi || offline-BC policy) support penalty on imagined latent states.",
    )
    parser.add_argument(
        "--behavior-bc-coef", type=float, default=0.0,
        help="Expert cross-entropy rehearsal coefficient on real posterior beliefs.",
    )
    parser.add_argument("--behavior-bc-batch-size", type=int, default=32)
    parser.add_argument(
        "--offline-bc-steps",
        type=int,
        default=0,
        help="Pre-PPO behavior-cloning updates on offline posterior beliefs and logged actions.",
    )
    parser.add_argument("--offline-bc-batch-size", type=int, default=32)
    parser.add_argument("--offline-bc-cache-size", type=int, default=2048)
    parser.add_argument(
        "--offline-bc-strategies", type=str, default="",
        help="Comma-separated offline episode strategies used by BC/rehearsal.",
    )
    parser.add_argument("--offline-bc-lr", type=float, default=1e-4)
    parser.add_argument(
        "--wm-action-id-offset",
        type=int,
        choices=[0, 1],
        default=0,
        help="Compatibility mapping between policy ids and frozen-WM ids. "
             "Use 1 only for legacy Sokoban tokenized data/checkpoints "
             "trained on environment actions 1..4.",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--target-entropy", type=float, default=None,
        help="Entropy floor for anti-collapse. When mean policy entropy drops "
             "below this, a loss term pushes it back up. None = disabled. "
             "For 4-action Sokoban, ln(4)=1.386 is max; try 0.5 to stem collapse.",
    )
    parser.add_argument(
        "--entropy-floor-coef", type=float, default=0.1,
        help="Weight of the entropy-floor loss term (only active when --target-entropy is set).",
    )
    parser.add_argument("--normalize-advantages", action="store_true", default=True)
    parser.add_argument("--no-normalize-advantages", dest="normalize_advantages", action="store_false")
    parser.add_argument("--reward-mapping", choices=["sigmoid_affine", "raw_sigmoid", "clipped_logit", "terminal_success", "terminal_success_scaled", "terminal_success_conservative", "per_transition_success_conservative"],
                        default="sigmoid_affine")
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Success-range scale used by terminal_success_scaled (e.g. 0.1).",
    )
    parser.add_argument("--positive-value", type=float, default=10.9,
                        help="Positive reward magnitude for terminal_success/sigmoid_affine. Lower (e.g. 1.0) to ease value-function fitting.")
    parser.add_argument(
        "--reward-head-hidden-dim",
        type=int,
        default=None,
        help="Reward head architecture: omitted=legacy D->D->1, 0=linear, positive=compact bottleneck.",
    )
    parser.add_argument(
        "--require-injected-reward-head",
        action="store_true",
        help="Fail at startup unless the checkpoint contains a matching, "
             "explicitly injected compact reward head.",
    )
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Weight of value loss in PPO total loss.")
    parser.add_argument(
        "--reward-confidence-floor",
        type=float,
        default=None,
        help="Conservative reward floor. If omitted, read decision_threshold "
             "from an injected reward-head checkpoint.",
    )
    parser.add_argument(
        "--reward-low-confidence-scale",
        type=float,
        default=0.1,
        help="Residual ranking signal retained below the conservative floor.",
    )
    parser.add_argument(
        "--prior-isolation-mode",
        choices=["shared", "lora", "residual", "state_action"],
        default="shared",
        help="Prior-only transition parameterization. state_action requires a "
             "checkpoint containing a fitted constrained adapter.",
    )
    parser.add_argument("--prior-residual-rank", type=int, default=64)
    parser.add_argument(
        "--posterior-observation-residual-scale",
        type=float,
        default=0.0,
        help="Deterministic visual residual used while grounding posterior "
             "beliefs. Fitted prior/reward artifacts must declare and match "
             "this exact value.",
    )

    # Critic architecture
    parser.add_argument(
        "--critic-source", choices=["qwen_pooled", "qwen_slotwise_q", "latent_belief", "latent_ordered_v", "frozen_vlm"], default="qwen_pooled",
        help="Critic branch: latent_ordered_v preserves raw slot order and predicts scalar V(s) from explicit real returns.",
    )
    parser.add_argument(
        "--actor-source", choices=["qwen_pooled", "qwen_slotwise", "latent_belief", "frozen_vlm"], default="qwen_pooled",
        help="Actor branch: qwen_pooled=VLM+LoRA (policy loss trains LoRA); "
             "latent_belief=independent BeliefReadout+MLP (no VLM); "
             "frozen_vlm=Qwen+LoRA under no_grad + action_head (LoRA frozen w.r.t. policy loss).",
    )
    parser.add_argument("--actor-hidden-dim", type=int, default=1024, help="latent_belief actor MLP hidden dim")
    parser.add_argument("--actor-hidden-layers", type=int, default=2, help="latent_belief actor MLP hidden layers")
    parser.add_argument("--actor-slot-dim", type=int, default=64, help="qwen_slotwise per-token projection width")
    parser.add_argument(
        "--slotwise-actor-features", choices=["qwen", "raw"], default="qwen",
        help="Use Qwen token features or ordered raw belief slots for qwen_slotwise",
    )
    parser.add_argument(
        "--slotwise-behavior-scale", type=float, default=1.0,
        help="Legacy raw-slot behavior-logit scale for qwen_slotwise; use 0 "
             "for a direct Qwen actor.",
    )
    parser.add_argument("--critic-slot-dim", type=int, default=32, help="qwen_slotwise_q per-token projection width")
    parser.add_argument(
        "--slotwise-behavior-checkpoint",
        type=str,
        default=None,
        help="Historical slotwise policy checkpoint used as the frozen base "
             "for a zero-initialized qwen_slotwise residual actor.",
    )
    parser.add_argument(
        "--critic-hidden-dim", type=int, default=1024,
        help="Hidden width of the latent_belief critic MLP (ignored for qwen_pooled).",
    )
    parser.add_argument(
        "--critic-hidden-layers", type=int, default=2,
        help="Hidden layer count of the latent_belief critic MLP (ignored for qwen_pooled).",
    )

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
    parser.add_argument("--wm-refresh-base-lr-factor", type=float, default=0.01)
    parser.add_argument("--wm-refresh-weight-decay", type=float, default=0.01)
    parser.add_argument("--wm-refresh-warmup-steps", type=int, default=100)
    parser.add_argument("--wm-refresh-grad-clip", type=float, default=1.0)
    parser.add_argument("--wm-refresh-horizon", type=int, default=8)
    parser.add_argument("--wm-refresh-reward-pos-weight", type=float, default=None)
    parser.add_argument("--wm-refresh-reward-loss-coef", type=float, default=None)
    parser.add_argument("--wm-refresh-freeze-reward-head", action="store_true")
    parser.add_argument("--wm-refresh-validation-batches", type=int, default=0)
    parser.add_argument(
        "--wm-refresh-state-action-only",
        action="store_true",
        help="In alternating_wm with prior_isolation_mode=state_action, "
             "restrict the supervised WM refresh optimizer to only "
             "transition.prior_state_action.delta.weight. Posterior, Qwen WM, "
             "reward head, and all other transition parameters remain frozen.",
    )
    parser.add_argument(
        "--wm-refresh-prior-lora-only",
        action="store_true",
        help="Restrict supervised WM refresh to the independent prior LoRA. "
             "Requires --prior-isolation-mode lora. Posterior/default WM LoRA, "
             "transition modules, teacher projection, and Reward Head remain frozen.",
    )
    parser.add_argument("--wm-open-dynamics-coef", type=float, default=0.25)
    parser.add_argument("--wm-prior-reward-coef", type=float, default=0.5)
    parser.add_argument("--wm-open-loop-horizon", type=int, default=4)
    parser.add_argument("--wm-open-dynamics-decay", type=float, default=0.9)
    parser.add_argument("--wm-prior-reward-decay", type=float, default=1.0)
    parser.add_argument("--wm-delta-cosine-coef", type=float, default=0.0)
    parser.add_argument("--wm-inverse-action-coef", type=float, default=0.0)
    parser.add_argument("--wm-inverse-action-mode", choices=("joint", "prior_frozen"), default="joint")
    parser.add_argument("--wm-inverse-action-lr", type=float, default=None)
    parser.add_argument(
        "--wm-only-refresh-steps",
        type=int,
        default=0,
        help="Run only supervised WM refresh for this many optimizer steps, "
             "evaluate on a fixed held-out split, save, and exit before PPO.",
    )
    parser.add_argument("--wm-only-eval-every", type=int, default=5)
    parser.add_argument("--wm-only-val-batches", type=int, default=4)
    parser.add_argument("--wm-only-out-checkpoint", type=str, default=None)
    parser.add_argument(
        "--wm-only-eval-only",
        action="store_true",
        help="Evaluate --wm-checkpoint against the WM-only gate without training.",
    )
    parser.add_argument("--wm-only-max-dynamics", type=float, default=None)
    parser.add_argument("--wm-only-max-open-dynamics", type=float, default=None)
    parser.add_argument("--wm-only-min-inverse-action-acc", type=float, default=None)
    parser.add_argument("--wm-only-require-observation-anchor", action="store_true")
    parser.add_argument(
        "--core-probe",
        action="store_true",
        help="Run deterministic Phase-2 policy/PPO/critic correctness probes "
             "on the assembled production path, then exit before training.",
    )
    parser.add_argument("--core-probe-batch-size", type=int, default=4)
    parser.add_argument("--core-probe-sample-limit", type=int, default=16)
    parser.add_argument("--core-probe-critic-steps", type=int, default=20)
    parser.add_argument("--core-probe-critic-lr", type=float, default=1e-4)
    parser.add_argument("--core-probe-output", type=str, default=None)
    parser.add_argument(
        "--semantic-probe",
        action="store_true",
        help="Run fixed-belief reward/action-sensitivity and held-out "
             "prior-reward semantic probes, then exit before training.",
    )
    parser.add_argument("--semantic-probe-batch-size", type=int, default=8)
    parser.add_argument("--semantic-probe-horizon", type=int, default=4)
    parser.add_argument("--semantic-probe-sequences", type=int, default=32)
    parser.add_argument("--semantic-probe-rollout-repeats", type=int, default=4)
    parser.add_argument("--semantic-probe-seed", type=int, default=20260718)
    parser.add_argument("--semantic-probe-output", type=str, default=None)
    parser.add_argument(
        "--counterfactual-action-audit",
        action="store_true",
        help="Enumerate all four first actions from identical posterior "
             "beliefs and report H1/H2 relative-to-BC reward rankings.",
    )
    parser.add_argument("--counterfactual-audit-samples", type=int, default=512)
    parser.add_argument("--counterfactual-audit-batch-size", type=int, default=8)
    parser.add_argument("--counterfactual-audit-seed", type=int, default=20260802)
    parser.add_argument("--counterfactual-audit-output", type=str, default=None)
    parser.add_argument(
        "--train-ranking-reward-head", action="store_true",
        help="Jointly retain terminal calibration and train the compact Reward "
             "Head with solver-supervised H3 four-action rankings, then exit.",
    )
    parser.add_argument("--ranking-reward-output", type=str, default=None)
    parser.add_argument("--ranking-reward-cache", type=str, default=None)
    parser.add_argument("--ranking-terminal-cache", type=str, default=None)
    parser.add_argument("--ranking-train-states", type=int, default=12000)
    parser.add_argument("--ranking-val-states", type=int, default=2000)
    parser.add_argument("--ranking-test-states", type=int, default=2000)
    parser.add_argument("--ranking-epochs", type=int, default=80)
    parser.add_argument("--ranking-lr", type=float, default=1e-4)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-pairwise-weight", type=float, default=0.5)
    parser.add_argument("--ranking-terminal-weight", type=float, default=1.0)
    parser.add_argument(
        "--prior-reward-decomposition-audit",
        action="store_true",
        help="Compare base/local Prior four-action latent separation, decoded "
             "object outcomes, and Reward sensitivity, then exit.",
    )
    parser.add_argument("--decomposition-audit-samples", type=int, default=256)
    parser.add_argument("--decomposition-audit-batch-size", type=int, default=8)
    parser.add_argument("--decomposition-audit-seed", type=int, default=20260802)
    parser.add_argument("--decomposition-audit-output", type=str, default=None)

    # Model
    parser.add_argument("--backbone-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="World model hidden dim D, must match VLM backbone hidden_size "
             "(7B=3584 default; 3B=2048). If omitted, auto-detected from --backbone-model config.json.",
    )
    parser.add_argument("--encoder-type", choices=["vjepa2", "qwen"], default="vjepa2")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                        help="Attention implementation: sdpa, flash_attention_2, eager")
    parser.add_argument("--belief-slots", type=int, default=36)
    parser.add_argument(
        "--action-conditioning-mode",
        choices=["embedded", "text"],
        default="text",
    )
    parser.add_argument(
        "--posterior-grounding-mode",
        choices=["legacy_residual", "visual_anchor"],
        default="legacy_residual",
    )
    parser.add_argument(
        "--posterior-recurrent-residual-scale", type=float, default=0.25
    )
    parser.add_argument("--posterior-action-free", action="store_true")
    parser.add_argument("--observation-anchor-coef", type=float, default=0.0)
    parser.add_argument("--observation-delta-anchor-coef", type=float, default=0.0)
    parser.add_argument("--observation-delta-min-rms", type=float, default=1e-4)
    parser.add_argument("--freeze-observation-anchor-projection", action="store_true")
    parser.add_argument(
        "--vjepa-teacher-prior-coef", type=float, default=0.0,
        help="Coefficient for frozen V-JEPA supervision of action-conditioned future prior states.",
    )
    parser.add_argument(
        "--vjepa-teacher-posterior-coef", type=float, default=0.0,
        help="Optional frozen V-JEPA grounding loss for Qwen-observed posterior states.",
    )
    parser.add_argument(
        "--vjepa-teacher-delta-coef", type=float, default=0.0,
        help="Coefficient for slotwise prior state-change matching in frozen V-JEPA space.",
    )
    parser.add_argument("--vjepa-teacher-delta-min-rms", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--success-reward-threshold",
        type=float,
        default=1.0,
        help="Raw reward must exceed this threshold to label a transition successful.",
    )

    # Online learning / evaluation
    parser.add_argument("--online-ratio", type=float, default=0.5,
                        help="Fraction of online (real-collected) data in each WM refresh batch. "
                             "0=offline only, 1=online only. Only used when --collect-every>0.")
    parser.add_argument("--online-replay-root", type=str, default=None,
                        help="Directory for the online replay buffer. Default: <checkpoint-dir>/online_replay.")
    parser.add_argument("--online-replay-max", type=int, default=10000,
                        help="Max episodes retained in the online replay buffer.")
    parser.add_argument(
        "--eval-seeds-file", type=str, default=None,
        help="JSON file with a fixed list of eval seeds (e.g. VAGEN test set). "
             "When set, eval runs on these exact levels every time → reproducible "
             "success_rate comparable across runs. Format: {\"seeds\": [1,3,...]}.",
    )
    parser.add_argument(
        "--eval-levels-file", type=str, default=None,
        help="JSON file with precomputed eval layouts ({levels:[{room_state,room_fixed},...]}). "
             "Takes precedence over --eval-seeds-file. Layouts are injected directly, "
             "giving 100% reproducible levels (used to match VagenMirror0416 test set exactly).",
    )

    # Logging
    parser.add_argument("--wandb-project", type=str, default="mbrl-vlm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    if args.reward_scale < 0.0:
        parser.error("--reward-scale must be non-negative")
    teacher_coefficients = (
        args.vjepa_teacher_prior_coef,
        args.vjepa_teacher_posterior_coef,
        args.vjepa_teacher_delta_coef,
    )
    if any(value < 0.0 for value in teacher_coefficients):
        parser.error("V-JEPA teacher coefficients must be non-negative")
    if args.vjepa_teacher_delta_min_rms < 0.0:
        parser.error("--vjepa-teacher-delta-min-rms must be non-negative")
    if any(value > 0.0 for value in teacher_coefficients) and args.encoder_type != "qwen":
        parser.error(
            "V-JEPA semantic-teacher training requires --encoder-type qwen; "
            "V-JEPA may not be both the WM input and its teacher."
        )

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

    if args.mode == "full":
        _validate_offline_action_encoding(
            dataset,
            num_actions=config.env.num_actions,
            wm_action_id_offset=config.phase2.ppo.wm_action_id_offset,
        )
        _validate_vjepa_teacher_dataset(dataset, config)

    print(f"Dataset: {len(dataset)} episodes")

    # ── World Model ──
    backbone = build_backbone(config, device)
    world_model = WorldModel(config, backbone)
    world_model.to(device)
    if config.encoder.encoder_type == "qwen":
        print(
            "  Visual input: native Qwen2.5-VL image embeddings "
            f"({config.encoder.compressed_tokens} ordered spatial slots)",
            flush=True,
        )
    if config.encoder.semantic_teacher_type == "vjepa2":
        print(
            "  Spatial teacher: frozen compressed V-JEPA features; "
            f"prior={config.training.vjepa_teacher_prior_coef:g}, "
            f"posterior={config.training.vjepa_teacher_posterior_coef:g}, "
            f"delta={config.training.vjepa_teacher_delta_coef:g}",
            flush=True,
        )
    if config.training.inverse_action_coef > 0.0:
        if world_model.inverse_action_head is None:
            raise RuntimeError(
                "inverse-action supervision was requested, but the WorldModel "
                "did not construct inverse_action_head"
            )
        print(
            "  WM action auxiliary: inverse_action_head=enabled "
            f"coef={config.training.inverse_action_coef:g} "
            f"mode={config.training.inverse_action_mode} "
            f"features=slotwise_flat lr={config.training.inverse_action_lr or config.training.lr:g}",
            flush=True,
        )

    loaded_checkpoint: dict = {}
    if args.wm_checkpoint:
        loaded_checkpoint = load_wm_checkpoint(
            world_model,
            args.wm_checkpoint,
            device,
            allow_phase2_world_model=args.allow_self_contained_phase2_wm,
        )
        from sembelief_wm.model.checkpoint_semantics import (
            validate_world_model_semantics,
        )
        validate_world_model_semantics(
            loaded_checkpoint,
            attention_mode=config.backbone.attention_mode,
            context=f"Phase-1 WM checkpoint {args.wm_checkpoint}",
        )
        if args.critic_h2_cache:
            source_state = loaded_checkpoint.get("model")
            if not isinstance(source_state, dict):
                raise ValueError(
                    "Critic H2 warmup requires a self-contained Phase-1 "
                    "Reward-Head checkpoint"
                )
            runtime_state = world_model.state_dict()
            missing = sorted(set(runtime_state) - set(source_state))
            unexpected = sorted(set(source_state) - set(runtime_state))
            shape_mismatch = sorted(
                key for key in set(runtime_state) & set(source_state)
                if tuple(runtime_state[key].shape) != tuple(source_state[key].shape)
            )
            if missing or unexpected or shape_mismatch:
                raise RuntimeError(
                    "Critic H2 warmup refuses a partial/mismatched WM load: "
                    f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
                    f"shape_mismatch={shape_mismatch[:20]}"
                )
            print(
                "  Critic H2 source check: exact WM/Reward-Head key and shape "
                "match verified.",
                flush=True,
            )
        if args.wm_refresh_prior_lora_only:
            if args.prior_isolation_mode != "lora":
                raise ValueError(
                    "--wm-refresh-prior-lora-only requires "
                    "--prior-isolation-mode lora"
                )
            source_state = loaded_checkpoint.get("model", loaded_checkpoint)
            if not isinstance(source_state, dict):
                raise ValueError("Prior repair source has no World Model state")
            prior_name = world_model.transition.prior_lora_adapter_name
            prior_marker = f".{prior_name}."
            source_has_prior = any(
                "lora_" in name.lower() and prior_marker in name
                for name in source_state
            )
            current_keys = set(world_model.state_dict())
            source_keys = set(source_state)
            missing_non_prior = sorted(
                name for name in current_keys.difference(source_keys)
                if not ("lora_" in name.lower() and prior_marker in name)
            )
            unexpected = sorted(source_keys.difference(current_keys))
            if missing_non_prior or unexpected:
                raise RuntimeError(
                    "Prior repair checkpoint conversion is not architecture-safe: "
                    f"missing_non_prior={missing_non_prior[:20]}, "
                    f"unexpected={unexpected[:20]}"
                )
            if not source_has_prior:
                copy_adapter = getattr(
                    world_model.transition.backbone, "copy_lora_adapter", None
                )
                if not callable(copy_adapter):
                    raise TypeError(
                        "Qwen backbone cannot initialize the independent prior LoRA"
                    )
                # WorldModel construction happens before checkpoint loading, so
                # re-copy here to use the released, trained default adapter rather
                # than its random construction-time initialization.
                copy_adapter("default", prior_name)
                print(
                    "  Prior repair initialization: copied loaded default WM "
                    f"LoRA -> {prior_name}.",
                    flush=True,
                )
            else:
                print(
                    f"  Prior repair resume: retained checkpoint adapter {prior_name}.",
                    flush=True,
                )
        if args.require_injected_reward_head:
            injection = loaded_checkpoint.get("reward_head_injection")
            recovered_phase2_wm = bool(
                loaded_checkpoint.get("_self_contained_phase2_wm")
            )
            if (
                not recovered_phase2_wm
                and (
                    not loaded_checkpoint.get("injected_reward_head")
                    or not isinstance(injection, dict)
                )
            ):
                raise ValueError(
                    "Phase 2 requires an injected compact reward-head checkpoint; "
                    f"{args.wm_checkpoint} has no injection metadata."
                )
            checkpoint_hidden_dim = (
                args.reward_head_hidden_dim
                if recovered_phase2_wm
                else injection.get("head_hidden_dim")
            )
            if checkpoint_hidden_dim != args.reward_head_hidden_dim:
                raise ValueError(
                    "Reward-head architecture mismatch: checkpoint declares "
                    f"head_hidden_dim={checkpoint_hidden_dim}, CLI requested "
                    f"{args.reward_head_hidden_dim}."
                )
            checkpoint_state = (
                loaded_checkpoint["world_model"]
                if recovered_phase2_wm
                else loaded_checkpoint.get("model", loaded_checkpoint)
            )
            expected_reward_keys = {
                f"reward_head.{key}"
                for key in world_model.reward_head.state_dict()
            }
            missing_reward_keys = sorted(
                expected_reward_keys.difference(checkpoint_state)
            )
            if missing_reward_keys:
                raise ValueError(
                    "Injected checkpoint is missing reward-head tensors: "
                    f"{missing_reward_keys}."
                )
            print(
                "  Verified compact reward head"
                + (" from self-contained Phase-2 state: "
                   if recovered_phase2_wm else ": ")
                + f"hidden_dim={checkpoint_hidden_dim}, horizons="
                + (
                    "historical-manifest"
                    if recovered_phase2_wm
                    else str(
                        injection.get(
                            "horizons", [injection.get("horizon")]
                        )
                    )
                )
            )
            world_model._reward_head_injection = dict(injection or {})
            if (
                config.phase2.ppo.reward_mapping
                == "per_transition_success_conservative"
            ):
                trained_horizons = {
                    int(value) for value in injection.get("horizons", [])
                }
                required_horizons = set(
                    range(1, config.phase2.ppo.rollout_horizon + 1)
                )
                missing_horizons = sorted(
                    required_horizons.difference(trained_horizons)
                )
                if missing_horizons:
                    raise ValueError(
                        "per-transition termination requires Reward Head "
                        "coverage at every imagined step; checkpoint is "
                        f"missing horizons {missing_horizons}."
                    )
                if not injection.get("independent_horizon_starts", False):
                    raise ValueError(
                        "per-transition termination requires a Reward Head "
                        "trained with independent_horizon_starts=true so "
                        "short successful episodes are not excluded."
                    )
                if (
                    config.phase2.ppo.imagination_termination_mode
                    != "predicted_success"
                ):
                    raise ValueError(
                        "per-transition reward mapping requires "
                        "--imagination-termination-mode predicted_success."
                    )
                exact_h2_finite_returns = (
                    os.environ.get("COUNTERFACTUAL_H2_PPO", "0") == "1"
                    and os.environ.get(
                        "CRITIC_WARMUP_ZERO_BOOTSTRAP", "0"
                    ) == "1"
                    and config.phase2.ppo.rollout_horizon == 2
                )
                if (
                    not config.phase2.ppo.use_value_bootstrap
                    and not exact_h2_finite_returns
                ):
                    raise ValueError(
                        "per-transition finite fragments require "
                        "--value-bootstrap for non-terminal endpoints."
                    )
                if exact_h2_finite_returns:
                    print(
                        "  Exact-H2 PPO: using finite two-step returns with "
                        "zero leaf bootstrap.",
                        flush=True,
                    )
        if (
            config.phase2.ppo.reward_mapping
            in {
                "terminal_success_conservative",
                "per_transition_success_conservative",
            }
            and args.reward_confidence_floor is None
        ):
            injection = loaded_checkpoint.get("reward_head_injection", {})
            threshold = injection.get("decision_threshold")
            if threshold is None:
                raise ValueError(
                    "terminal_success_conservative requires either "
                    "--reward-confidence-floor or an injected checkpoint "
                    "with reward_head_injection.decision_threshold."
                )
            config.phase2.ppo.reward_confidence_floor = float(threshold)
            print(
                "  Conservative reward floor loaded from checkpoint: "
                f"{float(threshold):.6f}"
            )
        if args.prior_isolation_mode == "state_action":
            adapter = world_model.transition.prior_state_action
            assert adapter is not None
            if not bool(adapter.artifact_loaded):
                raise ValueError(
                    "--prior-isolation-mode state_action requires checkpoint "
                    "tensors for transition.prior_state_action"
                )
            if (
                args.world_model_mode != "frozen_wm"
                and not args.wm_refresh_state_action_only
            ):
                raise ValueError(
                    "state_action is currently audited only with frozen_wm; "
                    "alternating refresh requires "
                    "--wm-refresh-state-action-only"
                )
            adapter_metadata = loaded_checkpoint.get(
                "state_action_prior_injection", {}
            )
            expected_residual = adapter_metadata.get(
                "posterior_observation_residual_scale"
            )
            if (
                expected_residual is None
                and loaded_checkpoint.get("_self_contained_phase2_wm")
            ):
                expected_residual = (
                    config.training.posterior_observation_residual_scale
                )
                print(
                    "  State-action metadata recovered from locked historical "
                    "manifest; adapter tensors and artifact_loaded buffer were "
                    "verified in the checkpoint."
                )
            elif expected_residual is None:
                raise ValueError(
                    "state_action checkpoint does not declare "
                    "posterior_observation_residual_scale; rebuild it with "
                    "scripts/build_state_action_wm_checkpoint.py"
                )
            actual_residual = float(
                config.training.posterior_observation_residual_scale
            )
            if abs(float(expected_residual) - actual_residual) > 1e-12:
                raise ValueError(
                    "Posterior grounding mismatch: fitted state_action "
                    f"artifact requires residual_scale={expected_residual}, "
                    f"but Phase 2 requested {actual_residual}."
                )
            print(
                "  Verified fitted state-action prior adapter: "
                f"posterior_residual_scale={actual_residual:g}"
            )
    elif args.mode == "full":
        print("Warning: No WM checkpoint provided. Using random initialization.")

    if args.frozenlake_spatial_audit:
        if args.env_id != "frozenlake":
            raise ValueError("--frozenlake-spatial-audit requires --env-id frozenlake")
        if not args.wm_checkpoint:
            raise ValueError("--frozenlake-spatial-audit requires --wm-checkpoint")
        from dataclasses import asdict
        import json
        from sembelief_wm.diagnostics.frozenlake_spatial import (
            run_frozenlake_spatial_audit,
        )
        result = run_frozenlake_spatial_audit(
            world_model=world_model,
            dataset=dataset,
            config=config,
            device=device,
            seed=args.seed,
            train_episodes=args.spatial_audit_train_episodes,
            validation_episodes=args.spatial_audit_val_episodes,
            probe_steps=args.spatial_audit_probe_steps,
            train_indices=[
                int(value)
                for value in loaded_checkpoint.get("wm_only_refresh", {}).get(
                    "train_indices", ()
                )
            ] or None,
            validation_indices=[
                int(value)
                for value in loaded_checkpoint.get("wm_only_refresh", {}).get(
                    "val_indices", ()
                )
            ] or None,
            validation_exclude_indices=[
                int(value)
                for value in loaded_checkpoint.get("spatial_prior_repair", {})
                .get("split", {})
                .get("selection_indices", ())
            ] or None,
        )
        result_dict = asdict(result)
        gate_failures = []
        gate_checks = (
            ("posterior_accuracy", args.spatial_audit_min_posterior),
            ("prior_accuracy", args.spatial_audit_min_prior),
            ("counterfactual_accuracy", args.spatial_audit_min_counterfactual),
        )
        for key, threshold in gate_checks:
            if result_dict[key] < threshold:
                gate_failures.append(
                    f"{key}={result_dict[key]:.6g} < {threshold:.6g}"
                )
            relative_threshold = (
                result_dict["majority_baseline"]
                * args.spatial_audit_min_baseline_multiple
            )
            if result_dict[key] < relative_threshold:
                gate_failures.append(
                    f"{key}={result_dict[key]:.6g} < "
                    f"{args.spatial_audit_min_baseline_multiple:g}x majority "
                    f"({relative_threshold:.6g})"
                )
        result_dict["gate"] = {
            "passed": not gate_failures,
            "failures": gate_failures,
            "thresholds": {
                "posterior_accuracy": args.spatial_audit_min_posterior,
                "prior_accuracy": args.spatial_audit_min_prior,
                "counterfactual_accuracy": args.spatial_audit_min_counterfactual,
                "baseline_multiple": args.spatial_audit_min_baseline_multiple,
            },
        }
        result_dict["checkpoint"] = {
            "path": str(Path(args.wm_checkpoint).resolve()),
        }
        checkpoint_train_indices = loaded_checkpoint.get("wm_only_refresh", {}).get(
            "train_indices", ()
        )
        checkpoint_validation_indices = loaded_checkpoint.get(
            "wm_only_refresh", {}
        ).get("val_indices", ())
        checkpoint_selection_indices = loaded_checkpoint.get(
            "spatial_prior_repair", {}
        ).get("split", {}).get("selection_indices", ())
        result_dict["split"] = {
            "protocol": "checkpoint_fixed_stage1_split",
            "train_pool_episodes": len(checkpoint_train_indices),
            "validation_pool_episodes": len(checkpoint_validation_indices),
            "checkpoint_selection_episodes_excluded": len(
                checkpoint_selection_indices
            ),
            "train_validation_overlap": len(
                set(checkpoint_train_indices) & set(checkpoint_validation_indices)
            ),
        }
        result_dict["protocol"] = "frozenlake_spatial_counterfactual_v2"
        print(
            "Spatial Gate: " + (
                "PASS" if not gate_failures
                else "FAIL: " + "; ".join(gate_failures)
            ), flush=True,
        )
        if args.spatial_audit_output:
            report_path = Path(args.spatial_audit_output)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(result_dict, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Spatial audit report: {report_path}", flush=True)
        if args.spatial_audit_enforce_gate and gate_failures:
            raise RuntimeError(
                f"FrozenLake spatial gate failed; inspect {args.spatial_audit_output}"
            )
        return

    # Optional fixed WM validation for alternating Phase 2.  The held-out
    # episodes are also removed from the offline refresh/start-state source, so
    # the reported curves cannot improve merely because refresh trained on the
    # same episodes. Prefer the reward-head injection test split when present.
    wm_validation_source = None
    if (
        config.phase2.world_model_mode == "alternating_wm"
        and config.phase2.wm_refresh.validation_batches > 0
    ):
        split_indices = (
            loaded_checkpoint.get("reward_head_injection", {})
            .get("split_indices", {})
        )
        use_reward_splits = (
            os.environ.get("WM_REFRESH_USE_REWARD_SPLITS", "0") == "1"
        )
        if use_reward_splits:
            train_indices = sorted(set(split_indices.get("train", [])))
            val_indices = sorted(set(split_indices.get("validation", [])))
            if not train_indices or not val_indices:
                raise RuntimeError(
                    "WM_REFRESH_USE_REWARD_SPLITS requires non-empty train "
                    "and validation indices in reward_head_injection"
                )
            if set(train_indices) & set(val_indices):
                raise RuntimeError(
                    "Reward-Head train/validation indices overlap"
                )
        else:
            val_indices = sorted(set(split_indices.get("test", [])))
            train_indices = []
        if not val_indices:
            generator = torch.Generator().manual_seed(args.seed)
            permutation = torch.randperm(
                len(dataset), generator=generator
            ).tolist()
            split_at = max(1, int(0.9 * len(permutation)))
            val_indices = sorted(permutation[split_at:])
        if not val_indices:
            raise ValueError("Fixed WM validation split is empty")
        invalid_indices = [
            index for index in val_indices
            if index < 0 or index >= len(dataset)
        ]
        if invalid_indices:
            raise ValueError(
                "Checkpoint WM validation indices are incompatible with the "
                f"dataset: {invalid_indices[:5]}"
            )
        if not train_indices:
            val_index_set = set(val_indices)
            train_indices = [
                index for index in range(len(dataset))
                if index not in val_index_set
            ]
        if not train_indices:
            raise ValueError("Fixed WM validation consumed the entire dataset")
        refresh_dataset = TokenizedEpisodeDataset(
            [dataset.episodes[index] for index in train_indices]
        )
        validation_dataset = TokenizedEpisodeDataset(
            [dataset.episodes[index] for index in val_indices]
        )
        data_source = OfflineDataSource(refresh_dataset, config)
        wm_validation_source = OfflineDataSource(validation_dataset, config)
        print(
            "  Fixed WM validation enabled: "
            f"offline_train={len(refresh_dataset)}, "
            f"held_out={len(validation_dataset)}, "
            f"batches/refresh={config.phase2.wm_refresh.validation_batches}, "
            f"split_protocol={'reward_head_train_validation' if use_reward_splits else 'legacy'}"
        )

    # ── Action Adapter ──
    action_adapter = SokobanActionAdapter(
        hidden_dim=config.hidden_dim,
    )

    # ── WM Refresher (alternating mode) ──
    wm_refresher = None
    phase1_trainer = None
    if config.phase2.world_model_mode == "alternating_wm":
        from sembelief_wm.train import Phase1Trainer
        from sembelief_wm.trainers.wm_refresher import (
            WorldModelRefresher,
            make_phase1_refresh_config,
        )

        if args.wm_refresh_state_action_only and args.wm_refresh_prior_lora_only:
            raise ValueError(
                "--wm-refresh-state-action-only and "
                "--wm-refresh-prior-lora-only are mutually exclusive"
            )
        if args.wm_refresh_state_action_only:
            if args.prior_isolation_mode != "state_action":
                raise ValueError(
                    "--wm-refresh-state-action-only requires "
                    "--prior-isolation-mode state_action"
                )
            adapter = world_model.transition.prior_state_action
            if adapter is None or not bool(adapter.artifact_loaded):
                raise RuntimeError(
                    "state-action-only refresh requires a loaded fitted adapter"
                )
            world_model.requires_grad_(False)
            adapter.delta.weight.requires_grad_(True)
            trainable = [
                name for name, parameter in world_model.named_parameters()
                if parameter.requires_grad
            ]
            expected = ["transition.prior_state_action.delta.weight"]
            if trainable != expected:
                raise RuntimeError(
                    "unsafe state-action refresh ownership: "
                    f"expected={expected}, actual={trainable}"
                )
            print(
                "  Alternating WM isolation: optimizer owns only "
                "transition.prior_state_action.delta.weight; posterior, "
                "Qwen WM, and Reward Head are frozen.",
                flush=True,
            )
        elif args.wm_refresh_prior_lora_only:
            if args.prior_isolation_mode != "lora":
                raise ValueError(
                    "--wm-refresh-prior-lora-only requires "
                    "--prior-isolation-mode lora"
                )
            world_model.requires_grad_(False)
            prior_name = world_model.transition.prior_lora_adapter_name
            set_trainable = getattr(
                world_model.transition.backbone,
                "set_lora_adapter_trainable",
                None,
            )
            prior_parameters = getattr(
                world_model.transition.backbone,
                "lora_adapter_parameters",
                None,
            )
            if not callable(set_trainable) or not callable(prior_parameters):
                raise TypeError(
                    "prior-LoRA-only refresh requires named Qwen adapter ownership"
                )
            set_trainable(prior_name, True)
            expected_ids = {id(parameter) for parameter in prior_parameters(prior_name)}
            trainable_ids = {
                id(parameter) for parameter in world_model.parameters()
                if parameter.requires_grad
            }
            if not expected_ids or trainable_ids != expected_ids:
                trainable_names = [
                    name for name, parameter in world_model.named_parameters()
                    if parameter.requires_grad
                ]
                raise RuntimeError(
                    "unsafe prior-LoRA refresh ownership: optimizer candidates "
                    f"do not exactly match adapter {prior_name}; "
                    f"trainable={trainable_names[:20]}"
                )
            print(
                "  Prior repair isolation: optimizer owns only independent "
                f"Qwen adapter {prior_name!r} ({len(expected_ids)} tensors); "
                "posterior/default WM LoRA and all heads are frozen.",
                flush=True,
            )

        # The refresher shares the WM tensors but owns a Phase-2-specific
        # optimizer/scheduler/curriculum.  Do not let the original Phase-1
        # warmup or LR silently depend on the selected refresh frequency.
        refresh_trainer_config = make_phase1_refresh_config(config)
        if config.phase2.wm_refresh.freeze_reward_head:
            world_model.reward_head.requires_grad_(False)
        phase1_trainer = Phase1Trainer(
            config=refresh_trainer_config,
            world_model=world_model,
            data_source=data_source,
            device=device,
        )
        if refresh_trainer_config.training.inverse_action_coef > 0.0:
            assert world_model.inverse_action_head is not None
            optimizer_parameter_ids = {
                id(parameter)
                for group in phase1_trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            inverse_parameter_ids = {
                id(parameter)
                for parameter in world_model.inverse_action_head.parameters()
                if parameter.requires_grad
            }
            if not inverse_parameter_ids or not inverse_parameter_ids.issubset(
                optimizer_parameter_ids
            ):
                raise RuntimeError(
                    "inverse_action_head exists but is not owned by the WM optimizer"
                )
            print(
                "  Verified WM optimizer ownership: inverse_action_head parameters included.",
                flush=True,
            )
        if args.wm_refresh_state_action_only:
            adapter = world_model.transition.prior_state_action
            assert adapter is not None
            expected_ids = {id(adapter.delta.weight)}
            optimizer_ids = {
                id(parameter)
                for group in phase1_trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            if optimizer_ids != expected_ids:
                raise RuntimeError(
                    "state-action refresh optimizer ownership violation: "
                    f"expected 1 adapter tensor, got {len(optimizer_ids)}"
                )
            print(
                "  Verified alternating optimizer ownership: exactly one "
                "state_action delta tensor.",
                flush=True,
            )
        if args.wm_refresh_prior_lora_only:
            prior_name = world_model.transition.prior_lora_adapter_name
            expected_ids = {
                id(parameter) for parameter in
                world_model.transition.backbone.lora_adapter_parameters(prior_name)
            }
            optimizer_ids = {
                id(parameter)
                for group in phase1_trainer.optimizer.param_groups
                for parameter in group["params"]
            }
            if optimizer_ids != expected_ids:
                raise RuntimeError(
                    "prior-LoRA optimizer ownership violation: "
                    f"expected={len(expected_ids)} tensors, "
                    f"actual={len(optimizer_ids)}"
                )
            print(
                "  Verified prior repair optimizer ownership: exactly the "
                f"{prior_name!r} LoRA adapter.",
                flush=True,
            )
        fixed_validation_fn = None
        if wm_validation_source is not None:
            # Use identical sampled batches at every refresh without changing
            # the RNG stream used for training or PPO start-state sampling.
            validation_source = OfflineDataSource(
                wm_validation_source.dataset,
                refresh_trainer_config,
            )

            def fixed_validation_fn(step: int) -> dict[str, float]:
                with isolated_torch_rng(20260717):
                    return phase1_trainer.evaluate(
                        val_source=validation_source,
                        num_batches=(
                            config.phase2.wm_refresh.validation_batches
                        ),
                        global_step=step,
                        run_probing=False,
                    )

        wm_refresher = WorldModelRefresher(
            phase1_trainer=phase1_trainer,
            config=config.phase2.wm_refresh,
            validation_fn=fixed_validation_fn,
        )

    if args.wm_only_refresh_steps > 0 or args.wm_only_eval_only:
        if phase1_trainer is None:
            raise ValueError(
                "--wm-only-refresh-steps requires "
                "--world-model-mode alternating_wm"
            )
        if args.wm_only_eval_every <= 0:
            raise ValueError("--wm-only-eval-every must be positive")
        if args.wm_only_val_batches <= 0:
            raise ValueError("--wm-only-val-batches must be positive")
        if not args.wm_only_out_checkpoint:
            raise ValueError(
                "--wm-only-out-checkpoint is required for WM-only validation"
            )

        # A WM-only continuation must reuse the exact Stage-1 split stored by
        # its source checkpoint. Reward-head splits are only a legacy fallback.
        wm_split = loaded_checkpoint.get("wm_only_refresh", {})
        train_indices = list(wm_split.get("train_indices", []))
        val_indices = list(wm_split.get("val_indices", []))
        if not train_indices or not val_indices:
            split_indices = (
                loaded_checkpoint.get("reward_head_injection", {})
                .get("split_indices", {})
            )
            train_indices = list(split_indices.get("train", []))
            val_indices = list(split_indices.get("test", []))
        if not train_indices or not val_indices:
            generator = torch.Generator().manual_seed(args.seed)
            permutation = torch.randperm(
                len(dataset), generator=generator
            ).tolist()
            split_at = max(1, int(0.9 * len(permutation)))
            train_indices = permutation[:split_at]
            val_indices = permutation[split_at:]
        train_dataset = TokenizedEpisodeDataset(
            [dataset.episodes[index] for index in train_indices]
        )
        val_dataset = TokenizedEpisodeDataset(
            [dataset.episodes[index] for index in val_indices]
        )
        train_source = OfflineDataSource(train_dataset, refresh_trainer_config)
        val_source = OfflineDataSource(val_dataset, refresh_trainer_config)

        # This branch exits before the main PPO logger is constructed, so it
        # must own its telemetry lifecycle.  Without this, WM-only launchers
        # accepted W&B arguments but silently produced stdout-only runs.
        wm_wandb_run = None
        if config.wandb.enabled:
            try:
                import wandb
                wm_wandb_run = wandb.init(
                    project=config.wandb.project,
                    name=config.wandb.run_name,
                    group=config.wandb.group,
                    job_type=(
                        "wm_prior_repair"
                        if args.wm_refresh_prior_lora_only
                        else "wm_only_refresh"
                    ),
                    config={
                        "source_wm": args.wm_checkpoint,
                        "output_checkpoint": args.wm_only_out_checkpoint,
                        "train_episodes": len(train_dataset),
                        "validation_episodes": len(val_dataset),
                        "refresh_steps": args.wm_only_refresh_steps,
                        "refresh_lr": config.phase2.wm_refresh.lr,
                        "refresh_batch_size": config.phase2.wm_refresh.batch_size,
                        "refresh_horizon": config.phase2.wm_refresh.horizon,
                        "open_loop_horizon": config.phase2.wm_refresh.open_loop_horizon,
                        "prior_lora_only": args.wm_refresh_prior_lora_only,
                        "prior_adapter": world_model.transition.prior_lora_adapter_name,
                        "vjepa_prior_coef": refresh_trainer_config.training.vjepa_teacher_prior_coef,
                        "vjepa_delta_coef": refresh_trainer_config.training.vjepa_teacher_delta_coef,
                        "latent_delta_coef": refresh_trainer_config.training.delta_cosine_coef,
                    },
                )
            except Exception as exc:
                if os.environ.get("REQUIRE_WANDB", "0") == "1":
                    raise RuntimeError(
                        "W&B is required for this WM-only run but initialization "
                        f"failed: {type(exc).__name__}: {exc}"
                    ) from exc
                print(
                    "Warning: wandb disabled for WM-only run "
                    f"({type(exc).__name__}: {exc}); logging to stdout only",
                    flush=True,
                )

        def evaluate_fixed(step: int) -> dict[str, float]:
            # OfflineDataSource samples with torch.randint. Preserve both CPU
            # and CUDA RNG because evaluation runs through the CUDA WM.
            with isolated_torch_rng(20260716):
                return phase1_trainer.evaluate(
                    val_source=val_source,
                    num_batches=args.wm_only_val_batches,
                    global_step=step,
                    run_probing=False,
                )

        def print_wm_metrics(prefix: str, metrics: dict[str, float]) -> None:
            keys = (
                "val/loss/total",
                "val/loss/dynamics",
                "val/loss/reward",
                "val/loss/open_dynamics",
                "val/loss/open_prior_reward",
                "val/metric/reward_auroc_post",
                "val/metric/reward_auroc_pri",
                "val/metric/reward_brier_post",
                "val/metric/reward_brier_pri",
                "val/metric/grad_norm",
                "val/metric/inverse_action_acc_prior",
                "val/metric/inverse_action_acc_post",
                "val/metric/action_aux_valid_count",
                "val/loss/observation_anchor",
                "val/metric/observation_anchor_valid_count",
                "val/loss/vjepa_teacher_prior",
                "val/loss/vjepa_teacher_posterior",
                "val/loss/vjepa_teacher_delta",
                "val/metric/vjepa_teacher_prior_valid_count",
                "val/metric/vjepa_teacher_posterior_valid_count",
                "val/metric/vjepa_teacher_delta_valid_count",
            )
            values = [
                f"{key}={metrics[key]:.6g}" for key in keys if key in metrics
            ]
            print(f"[{prefix}] " + " | ".join(values), flush=True)
            if any(key.startswith("val/diagnostic/prior_confusion_") for key in metrics):
                for branch in ("post", "prior"):
                    rows = []
                    for true_action in range(config.env.num_actions):
                        rows.append([
                            int(metrics.get(
                                f"val/diagnostic/{branch}_confusion_{true_action}_{predicted_action}",
                                0.0,
                            ))
                            for predicted_action in range(config.env.num_actions)
                        ])
                    print(
                        f"[{prefix}] {branch} confusion rows=true cols=pred: {rows}",
                        flush=True,
                    )

        output = Path(args.wm_only_out_checkpoint)
        output.parent.mkdir(parents=True, exist_ok=True)
        latest_path = output.parent / "latest.pt"
        best_path = output.parent / "best.pt"
        base_step = int(loaded_checkpoint.get("step", 0))
        best_dynamics_score = float("inf")
        best_updated_this_run = False

        def wm_gate(metrics: dict[str, float]) -> tuple[bool, list[str]]:
            import math
            failures: list[str] = []
            # Stage-1 follows the Sokoban base-grounding protocol: this gate
            # checks that the supervised objective is numerically usable, not
            # that an unrelated task-independent absolute loss is reached.
            # Spatial semantics are audited by the following decoder stage.
            dynamics = metrics.get("val/loss/dynamics")
            if dynamics is None or not math.isfinite(dynamics):
                failures.append("dynamics loss is missing or non-finite")
            checks = (
                ("val/loss/dynamics", args.wm_only_max_dynamics, "max"),
                ("val/loss/open_dynamics", args.wm_only_max_open_dynamics, "max"),
                ("val/metric/inverse_action_acc_prior", args.wm_only_min_inverse_action_acc, "min"),
            )
            for key, threshold, direction in checks:
                if threshold is None:
                    continue
                value = metrics.get(key)
                if value is None:
                    failures.append(f"missing {key}")
                elif direction == "max" and value > threshold:
                    failures.append(f"{key}={value:.6g} > {threshold:.6g}")
                elif direction == "min" and value < threshold:
                    failures.append(f"{key}={value:.6g} < {threshold:.6g}")
            if args.wm_only_require_observation_anchor:
                anchor = metrics.get("val/loss/observation_anchor")
                count = metrics.get("val/metric/observation_anchor_valid_count", 0.0)
                if anchor is None or not math.isfinite(anchor):
                    failures.append("observation anchor loss is missing or non-finite")
                if count <= 0.0:
                    failures.append("observation anchor has no valid samples")
            teacher_checks = (
                (
                    "prior",
                    refresh_trainer_config.training.vjepa_teacher_prior_coef,
                ),
                (
                    "posterior",
                    refresh_trainer_config.training.vjepa_teacher_posterior_coef,
                ),
                (
                    "delta",
                    refresh_trainer_config.training.vjepa_teacher_delta_coef,
                ),
            )
            for name, coefficient in teacher_checks:
                if coefficient <= 0.0:
                    continue
                loss = metrics.get(f"val/loss/vjepa_teacher_{name}")
                count = metrics.get(
                    f"val/metric/vjepa_teacher_{name}_valid_count", 0.0
                )
                if loss is None or not math.isfinite(loss):
                    failures.append(f"V-JEPA teacher {name} loss is missing or non-finite")
                if count <= 0.0:
                    failures.append(
                        f"V-JEPA teacher {name} has no valid held-out samples"
                    )
            return not failures, failures

        def save_wm_checkpoint(
            path: Path,
            *,
            step: int,
            validation_metrics: dict[str, float],
        ) -> None:
            """Save the two formal WM artifacts without periodic-file sprawl."""
            from sembelief_wm.model.checkpoint_semantics import (
                world_model_semantics,
            )

            saved = {
                key: value
                for key, value in loaded_checkpoint.items()
                if key not in {"model", "optimizer", "scheduler", "step"}
            }
            saved.update({
                "model": world_model.state_dict(),
                "optimizer": phase1_trainer.optimizer.state_dict(),
                "scheduler": phase1_trainer.scheduler.state_dict(),
                "step": base_step + step,
                "config": refresh_trainer_config,
                "wm_semantics": world_model_semantics(
                    refresh_trainer_config.backbone.attention_mode
                ),
                "validation_metrics": dict(validation_metrics),
                "wm_only_refresh": {
                    "steps": step,
                    "lr": config.phase2.wm_refresh.lr,
                    "warmup_steps": config.phase2.wm_refresh.warmup_steps,
                    "horizon": config.phase2.wm_refresh.horizon,
                    "open_loop_horizon": config.phase2.wm_refresh.open_loop_horizon,
                    "open_dynamics_coef": config.phase2.wm_refresh.open_dynamics_coef,
                    "prior_reward_coef": config.phase2.wm_refresh.prior_reward_coef,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "stage1_gate_protocol": f"{args.env_id}_base_grounding",
                },
            })
            if args.wm_refresh_prior_lora_only:
                prior_name = world_model.transition.prior_lora_adapter_name
                saved["prior_repair"] = {
                    "format": "qwen_independent_prior_lora_v1",
                    "base_checkpoint": str(Path(args.wm_checkpoint).resolve()),
                    "prior_adapter": prior_name,
                    "posterior_adapter": "default",
                    "posterior_frozen": True,
                    "teacher_projection_frozen": True,
                    "reward_head_frozen": True,
                    "trainable_tensor_count": len(
                        world_model.transition.backbone.lora_adapter_parameters(
                            prior_name
                        )
                    ),
                }
            temporary = path.with_name(f".{path.name}.tmp")
            torch.save(saved, temporary)
            os.replace(temporary, path)

        print(
            "=== WM-only refresh validation ===\n"
            f"  train episodes={len(train_dataset)} held-out={len(val_dataset)}\n"
            f"  steps={args.wm_only_refresh_steps} "
            f"eval_every={args.wm_only_eval_every}",
            flush=True,
        )
        initial_validation = evaluate_fixed(0)
        print_wm_metrics("wm-only step 0", initial_validation)
        if wm_wandb_run is not None:
            wm_wandb_run.log(initial_validation, step=base_step)
        if args.wm_only_eval_only:
            gate_passed, gate_failures = wm_gate(initial_validation)
            print(
                "WM gate: " + (
                    "PASS" if gate_passed
                    else "FAIL: " + "; ".join(gate_failures)
                ),
                flush=True,
            )
            if not gate_passed:
                raise RuntimeError(
                    "Existing WM checkpoint did not pass the WM gate"
                )
            # Preserve the exact trained optimizer/scheduler state.  Only add
            # the corrected validation metrics and promote this same model to
            # best.pt; no optimizer step is taken in eval-only mode.
            promoted = dict(loaded_checkpoint)
            promoted["validation_metrics"] = dict(initial_validation)
            temporary = best_path.with_name(f".{best_path.name}.tmp")
            torch.save(promoted, temporary)
            os.replace(temporary, best_path)
            print(f"Existing WM checkpoint promoted: {best_path}", flush=True)
            if wm_wandb_run is not None:
                wm_wandb_run.finish()
            return
        for step in range(1, args.wm_only_refresh_steps + 1):
            batch = train_source.sample_batch(
                config.phase2.wm_refresh.batch_size
            )
            train_metrics = phase1_trainer.train_one_step(
                global_step=step - 1,
                batch=batch,
            ).as_dict()
            if step == 1 or step % args.wm_only_eval_every == 0:
                train_summary = {
                    f"train/{key}": value
                    for key, value in train_metrics.items()
                    if key in {
                        "loss/total",
                        "loss/dynamics",
                        "loss/reward",
                        "loss/open_dynamics",
                        "loss/open_prior_reward",
                        "loss/delta_cosine",
                        "loss/inverse_action",
                        "metric/inverse_action_acc_prior",
                        "metric/inverse_action_acc_post",
                        "metric/action_aux_valid_count",
                        "loss/observation_anchor",
                        "metric/observation_anchor_valid_count",
                        "loss/vjepa_teacher_prior",
                        "loss/vjepa_teacher_posterior",
                        "loss/vjepa_teacher_delta",
                        "metric/vjepa_teacher_prior_valid_count",
                        "metric/vjepa_teacher_posterior_valid_count",
                        "metric/vjepa_teacher_delta_valid_count",
                        "metric/grad_norm",
                    }
                }
                print(
                    f"[wm-only train {step}] "
                    + " | ".join(
                        f"{key}={value:.6g}"
                        for key, value in sorted(train_summary.items())
                    ),
                    flush=True,
                )
                validation_metrics = evaluate_fixed(step - 1)
                print_wm_metrics(f"wm-only step {step}", validation_metrics)
                if wm_wandb_run is not None:
                    wm_wandb_run.log(
                        {**train_summary, **validation_metrics},
                        step=base_step + step,
                    )
                save_wm_checkpoint(
                    latest_path,
                    step=step,
                    validation_metrics=validation_metrics,
                )
                gate_passed, gate_failures = wm_gate(validation_metrics)
                dynamics_score = float(validation_metrics["val/loss/dynamics"])
                dynamics_score += float(
                    refresh_trainer_config.training.open_dynamics_coef
                    * validation_metrics.get("val/loss/open_dynamics", 0.0)
                )
                dynamics_score += float(
                    refresh_trainer_config.training.delta_cosine_coef
                    * validation_metrics.get("val/loss/delta_cosine", 0.0)
                )
                dynamics_score += float(
                    refresh_trainer_config.training.observation_anchor_coef
                    * validation_metrics.get("val/loss/observation_anchor", 0.0)
                )
                dynamics_score += float(
                    refresh_trainer_config.training.vjepa_teacher_prior_coef
                    * validation_metrics.get("val/loss/vjepa_teacher_prior", 0.0)
                )
                dynamics_score += float(
                    refresh_trainer_config.training.vjepa_teacher_posterior_coef
                    * validation_metrics.get("val/loss/vjepa_teacher_posterior", 0.0)
                )
                dynamics_score += float(
                    refresh_trainer_config.training.vjepa_teacher_delta_coef
                    * validation_metrics.get("val/loss/vjepa_teacher_delta", 0.0)
                )
                print(
                    f"WM latest checkpoint saved: {latest_path}", flush=True
                )
                print(
                    "WM gate: " + ("PASS" if gate_passed else "FAIL: " + "; ".join(gate_failures)),
                    flush=True,
                )
                if gate_passed and dynamics_score < best_dynamics_score:
                    best_updated_this_run = True
                    best_dynamics_score = dynamics_score
                    save_wm_checkpoint(
                        best_path,
                        step=step,
                        validation_metrics=validation_metrics,
                    )
                    print(
                        "WM best checkpoint updated: "
                        f"{best_path} dynamics_score={dynamics_score:.6g}",
                        flush=True,
                    )

        if not best_updated_this_run:
            raise RuntimeError(
                "WM-only training finished without a checkpoint passing the WM gate; "
                f"inspect {latest_path} and the training log"
            )
        print(
            f"WM-only training complete: latest={latest_path}, best={best_path}",
            flush=True,
        )
        if wm_wandb_run is not None:
            wm_wandb_run.finish()
        return

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
                    "rollouts_per_update": config.phase2.ppo.rollouts_per_update,
                    "ppo_epochs": config.phase2.ppo.epochs_per_update,
                    "minibatch_size": config.phase2.ppo.minibatch_size,
                    "recompute_old_log_probs": (
                        config.phase2.ppo.recompute_old_log_probs
                    ),
                    "actor_lr": config.phase2.ppo.actor_lr,
                    "critic_lr": config.phase2.ppo.critic_lr,
                    "entropy_coef": config.phase2.ppo.entropy_coef,
                    "target_kl": config.phase2.ppo.target_kl,
                    "behavior_kl_coef": config.phase2.ppo.behavior_kl_coef,
                    "offline_bc_steps": config.phase2.ppo.offline_bc_steps,
                    "wm_action_id_offset": config.phase2.ppo.wm_action_id_offset,
                    "reward_mapping": config.phase2.ppo.reward_mapping,
                    "reward_scale": config.phase2.ppo.reward_scale,
                    "wm_open_dynamics_coef": config.phase2.wm_refresh.open_dynamics_coef,
                    "wm_prior_reward_coef": config.phase2.wm_refresh.prior_reward_coef,
                    "wm_open_loop_horizon": config.phase2.wm_refresh.open_loop_horizon,
                    "wm_refresh_horizon": config.phase2.wm_refresh.horizon,
                    "wm_refresh_reward_pos_weight": config.phase2.wm_refresh.reward_pos_weight,
                    "wm_refresh_reward_loss_coef": config.phase2.wm_refresh.reward_loss_coef,
                    "wm_refresh_freeze_reward_head": config.phase2.wm_refresh.freeze_reward_head,
                    "wm_refresh_validation_batches": config.phase2.wm_refresh.validation_batches,
                    "wm_open_dynamics_decay": config.phase2.wm_refresh.open_dynamics_decay,
                    "wm_prior_reward_decay": config.phase2.wm_refresh.prior_reward_decay,
                    "success_reward_threshold": config.reward.success_reward_threshold,
                    "normalize_advantages": config.phase2.ppo.normalize_advantages,
                    "critic_source": args.critic_source,
                    "actor_source": args.actor_source,
                    "critic_hidden_dim": args.critic_hidden_dim,
                    "eval_episodes": config.phase2.ppo.eval_episodes,
                },
            )
        except Exception as exc:
            if os.environ.get("REQUIRE_WANDB", "0") == "1":
                raise RuntimeError(
                    "W&B is required for this experiment but initialization "
                    f"failed: {type(exc).__name__}: {exc}"
                ) from exc
            # Optional telemetry may degrade gracefully to stdout-only.
            print(f"Warning: wandb disabled ({type(exc).__name__}: {exc}); logging to stdout only")

    logger = PrintLogger(wandb_run=wandb_run)

    # ── Assemble Pipeline ──
    if args.train_ranking_reward_head:
        from sembelief_wm.train.ranking_reward import train_ranking_reward_head
        if not args.ranking_reward_output or not args.ranking_terminal_cache:
            raise ValueError(
                "--train-ranking-reward-head requires --ranking-reward-output "
                "and --ranking-terminal-cache"
            )
        train_ranking_reward_head(
            world_model=world_model,
            dataset=dataset,
            source_checkpoint=loaded_checkpoint,
            output_path=args.ranking_reward_output,
            cache_path=args.ranking_reward_cache,
            terminal_cache_path=args.ranking_terminal_cache,
            train_states=args.ranking_train_states,
            validation_states=args.ranking_val_states,
            test_states=args.ranking_test_states,
            epochs=args.ranking_epochs,
            learning_rate=args.ranking_lr,
            ranking_loss_weight=args.ranking_loss_weight,
            pairwise_loss_weight=args.ranking_pairwise_weight,
            terminal_loss_weight=args.ranking_terminal_weight,
            device=device,
            null_action_id=config.env.null_action_id,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    print("\n=== Assembling MBRL Pipeline ===")
    print(f"  Mode: {config.phase2.world_model_mode}")
    print("  Qwen base: one frozen instance shared by WM/actor/critic")
    print(
        "  LoRA ownership: "
        + ("WM default adapter, read-only policy view" if args.shared_backbone
           else "independent WM/actor/critic adapters")
    )
    print(
        f"  Rollout: batch={config.phase2.ppo.rollout_batch_size}, "
        f"horizon={config.phase2.ppo.rollout_horizon}, "
        f"chunks/update={config.phase2.ppo.rollouts_per_update}, "
        "termination="
        f"{config.phase2.ppo.imagination_termination_mode}"
    )
    print(
        f"  PPO: epochs={config.phase2.ppo.epochs_per_update}, "
        f"actor_lr={config.phase2.ppo.actor_lr}, "
        f"critic_lr={config.phase2.ppo.critic_lr}, "
        f"entropy={config.phase2.ppo.entropy_coef}"
    )
    print(
        f"  Reward: mapping={config.phase2.ppo.reward_mapping}, "
        f"scale={config.phase2.ppo.reward_scale}"
    )
    print(
        f"  Stability: target_kl={config.phase2.ppo.target_kl}, "
        f"offline_bc_steps={config.phase2.ppo.offline_bc_steps}, "
        f"behavior_kl_coef={config.phase2.ppo.behavior_kl_coef}"
    )
    if config.phase2.ppo.wm_action_id_offset:
        print(
            "  Legacy action compatibility: policy 0..3 -> frozen WM 1..4; "
            "offline BC labels 1..4 -> policy 0..3"
        )
    if args.actor_source == "latent_belief":
        print(f"  Actor: latent_belief (BeliefReadout+MLP {config.hidden_dim}->{args.actor_hidden_dim}x{args.actor_hidden_layers}->num_actions, NO VLM)")
    elif args.actor_source == "frozen_vlm":
        print(f"  Actor: frozen_vlm (Qwen+LoRA under no_grad → action_adapter; LoRA frozen w.r.t. policy loss)")
    elif args.actor_source == "qwen_slotwise":
        print(
            f"  Actor: qwen_slotwise ({args.slotwise_actor_features} ordered "
            f"features; 36x{args.actor_slot_dim} readout; no mean pooling; "
            f"legacy-base-scale={args.slotwise_behavior_scale})"
        )
    else:
        ownership = "WM read-only" if args.shared_backbone else "actor-owned"
        print(f"  Actor: qwen_pooled ({ownership} LoRA over shared Qwen → action_adapter)")
    if args.critic_source == "latent_belief":
        print(f"  Critic: latent_belief (independent BeliefReadout+MLP "
              f"{config.hidden_dim}->{args.critic_hidden_dim}x{args.critic_hidden_layers}->1, "
              f"NO VLM forward; actor still Qwen+LoRA)")
    elif args.critic_source == "latent_ordered_v":
        print(
            f"  Critic: latent_ordered_v (real-return scalar V; ordered raw "
            f"36x{args.critic_slot_dim} readout → 1; no action input, no VLM)"
        )
    elif args.critic_source == "frozen_vlm":
        print(f"  Critic: frozen_vlm (Qwen+LoRA forward under no_grad → MLP value_head "
              f"{config.hidden_dim}->{256}→1 trainable; LoRA frozen w.r.t. critic; "
              f"actor still Qwen+LoRA via policy loss)")
    elif args.critic_source == "qwen_slotwise_q":
        ownership = "WM read-only" if args.shared_backbone else "critic-owned"
        print(
            f"  Critic: qwen_slotwise_q ({ownership} Qwen LoRA; ordered "
            f"36x{args.critic_slot_dim} readout → Q(s,·); no mean pooling)"
        )
    else:
        ownership = "WM read-only" if args.shared_backbone else "critic-owned"
        print(f"  Critic: qwen_pooled ({ownership} LoRA over shared Qwen → value_head)")
    print(f"  Eval: every={config.phase2.ppo.eval_every}, episodes={config.phase2.ppo.eval_episodes}")
    if config.phase2.world_model_mode == "alternating_wm":
        wm_r = config.phase2.wm_refresh
        print(
            f"  WM refresh: every={wm_r.refresh_every}, "
            f"steps={wm_r.updates_per_refresh}, lr={wm_r.lr}, "
            f"warmup={wm_r.warmup_steps}, grad_clip={wm_r.grad_clip}, "
            f"horizon={wm_r.horizon}; open_loop="
            f"H{wm_r.open_loop_horizon}, dynamics_coef={wm_r.open_dynamics_coef}, "
            f"prior_reward_coef={wm_r.prior_reward_coef}"
        )
    if config.phase2.ppo.collect_every > 0:
        print(f"  Online collect: every={config.phase2.ppo.collect_every}, episodes={config.phase2.ppo.collect_episodes}")
    print()

    # ── Online learning + real-env evaluation wiring ──
    # RealCollector needs: env_factory, image tokenizer (real RGB → tokens),
    # world_model (posterior refresh), policy (the LLM actor-critic built above).
    # We construct the LLM policy here first so both collector and evaluator can
    # share it; assemble_llm_pipeline will rebuild it from the same backbone.
    online_buffer = None
    real_collector = None
    evaluator = None
    online_ratio = 0.0
    want_online = config.phase2.ppo.collect_every > 0
    # eval_enabled stays True whenever eval_every > 0: even without online collect,
    # we still build a RealCollector purely for periodic success_rate evaluation.
    want_real_env = want_online or config.phase2.ppo.eval_every > 0

    if want_real_env:
        from sembelief_wm.collectors.real import RealCollector, RealCollectorConfig, RealEnvEvaluator
        from sembelief_wm.data.replay import OnlineReplayBuffer

        if config.encoder.encoder_type == "qwen":
            from sembelief_wm.data.tokenizers import QwenVisionTokenizer

            print("  Using the loaded Qwen2.5-VL vision tower for real-env posterior...", flush=True)
            qwen_tokenizer = QwenVisionTokenizer.from_transition_backbone(
                config, backbone, device=device
            )
            if config.encoder.semantic_teacher_type == "vjepa2":
                from sembelief_wm.data.tokenizers import (
                    ImageTokenizer,
                    QwenVJEPAObservationTokenizer,
                )

                print("  Loading frozen V-JEPA 2 semantic teacher for online replay...", flush=True)
                vjepa_teacher = ImageTokenizer(config, device=str(device))
                image_tokenizer = QwenVJEPAObservationTokenizer(
                    qwen_tokenizer, vjepa_teacher
                )
            else:
                image_tokenizer = qwen_tokenizer
        else:
            from sembelief_wm.data.tokenizers.image import ImageTokenizer

            # Legacy V-JEPA-input route. New Qwen-native runs never enter this
            # branch, so V-JEPA cannot accidentally become the WM input again.
            print("  Loading V-JEPA 2 image tokenizer for real-env rollout...")
            image_tokenizer = ImageTokenizer(config, device=str(device))
            tokenizer_weights = os.environ.get("IMAGE_TOKENIZER_WEIGHTS")
            if not tokenizer_weights:
                raise RuntimeError(
                    "real-env posterior requires IMAGE_TOKENIZER_WEIGHTS; "
                    "random VisualTokenProjector initialization is forbidden"
                )
            tokenizer_path = Path(tokenizer_weights).resolve()
            if not tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"IMAGE_TOKENIZER_WEIGHTS does not exist: {tokenizer_path}"
                )
            image_tokenizer.load_weights(str(tokenizer_path))
            tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            image_tokenizer.provenance = {
                "path": str(tokenizer_path),
                "sha256": tokenizer_sha256,
                "size": tokenizer_path.stat().st_size,
            }
            print(
                "  Loaded locked ImageTokenizer weights: "
                f"{tokenizer_path} sha256={tokenizer_sha256}",
                flush=True,
            )

        # Online replay buffer (only needed when collecting).
        if want_online:
            replay_root = args.online_replay_root or str(Path(args.checkpoint_dir) / "online_replay")
            online_buffer = OnlineReplayBuffer(root=replay_root, max_episodes=args.online_replay_max)
            online_ratio = args.online_ratio
            print(f"  Online replay: root={replay_root}, max={args.online_replay_max}, online_ratio={online_ratio}")

        # Sokoban env factory. We need the LLM policy to act; build it here via
        # the same shared-backbone path assemble_llm_pipeline uses, so the
        # collector/evaluator share the *same* policy object the pipeline trains.
        # (assemble_llm_pipeline returns its own llm_policy; we patch the
        # collector/evaluator to reference that one after assembly — see below.)
        if args.env_id == "sokoban":
            from sembelief_wm.data.adapters.sokoban import SokobanAdapter
            env_adapter = SokobanAdapter(require_real=True)
        else:
            from sembelief_wm.data.adapters.frozenlake import FrozenLakeAdapter
            env_adapter = FrozenLakeAdapter(vagen_root=os.environ.get("VAGEN_ROOT"))
        # Fail before BC/training if the production environment cannot render.
        env_adapter.make_env(seed=0).reset(seed=0)
        env_id_tensor = torch.tensor([0], device=device, dtype=torch.long)
        max_steps = config.phase2.ppo.collect_max_steps or config.phase2.ppo.eval_max_steps or 25

        # Placeholder policy: will be replaced with the real llm_policy after
        # assemble_llm_pipeline builds it. Set to None; RealCollector must tolerate
        # late binding. We instead construct the collector after assembly below.
        print(f"  Real-env rollout: max_steps={max_steps}, env={args.env_id}")
        _real_cfg = RealCollectorConfig(
            max_steps=max_steps,
            deterministic=False,
            exploration_epsilon=float(os.environ.get(
                "ONLINE_COLLECTION_EPSILON", "0.0"
            )),
            capture_policy_trajectory=(
                os.environ.get("REAL_RETURN_CRITIC_ANCHOR", "0") == "1"
            ),
        )

        # Optional fixed eval set. Two formats:
        #   --eval-levels-file: JSON {levels:[{room_state,room_fixed},...]}
        #     → injects layouts directly (100% reproducible, e.g. VagenMirror export).
        #   --eval-seeds-file: JSON {seeds:[...]} → reseeds generate_room per level.
        # eval_levels takes precedence when both given.
        _eval_seeds: list[int] | None = None
        _eval_levels: list[dict] | None = None
        if args.eval_levels_file:
            import json as _json
            with open(args.eval_levels_file) as _fh:
                _data = _json.load(_fh)
            _eval_levels = list(_data["levels"])
            eval_level_limit = int(os.environ.get("EVAL_LEVEL_LIMIT", "0"))
            if eval_level_limit > 0:
                _eval_levels = _eval_levels[:eval_level_limit]
                print(
                    "  Diagnostic eval level limit: "
                    f"{len(_eval_levels)} (full file remains unchanged)"
                )
            print(f"  Eval levels: {len(_eval_levels)} fixed layouts from {args.eval_levels_file}")
        elif args.eval_seeds_file:
            import json as _json
            with open(args.eval_seeds_file) as _fh:
                _data = _json.load(_fh)
            _eval_seeds = list(_data["seeds"])
            print(f"  Eval seeds: {len(_eval_seeds)} fixed levels from {args.eval_seeds_file}")

    # ── LLM policy config (critic branch selectable) ──
    llm_policy_config = LLMPolicyConfig(
        hidden_dim=config.hidden_dim,
        num_slots=config.belief.num_slots,
        actor_source=args.actor_source,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_hidden_layers=args.actor_hidden_layers,
        actor_slot_dim=args.actor_slot_dim,
        slotwise_actor_features=args.slotwise_actor_features,
        slotwise_behavior_scale=args.slotwise_behavior_scale,
        critic_source=args.critic_source,
        critic_slot_dim=args.critic_slot_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_hidden_layers=args.critic_hidden_layers,
    )

    pipeline_class = None
    if args.orchestration == "lewm":
        from sembelief_wm.lewm import LeWMOrchestrator
        pipeline_class = LeWMOrchestrator
        print(
            "  Orchestration: isolated Le-WM state machine "
            "(generic MBRLPipeline.train is not used).",
            flush=True,
        )
    pipeline, llm_policy = assemble_llm_pipeline(
        config=config,
        world_model=world_model,
        action_adapter=action_adapter,
        data_source=data_source,
        device=device,
        shared_backbone=args.shared_backbone,
        logger=logger,
        wm_refresher=wm_refresher,
        online_buffer=online_buffer,
        online_ratio=online_ratio,
        llm_policy_config=llm_policy_config,
        pipeline_class=pipeline_class,
    )

    if args.critic_h2_cache:
        if not args.wm_checkpoint:
            raise ValueError("--critic-h2-cache requires --wm-checkpoint")
        from sembelief_wm.rl.critic_h2_cache import install_critic_h2_cache

        install_critic_h2_cache(
            pipeline=pipeline,
            world_model=world_model,
            cache_path=args.critic_h2_cache,
            source_checkpoint=args.wm_checkpoint,
            device=device,
            seed=args.seed,
        )

    if args.slotwise_behavior_checkpoint:
        if args.actor_source != "qwen_slotwise":
            raise ValueError(
                "--slotwise-behavior-checkpoint requires --actor-source "
                "qwen_slotwise"
            )
        behavior_payload = torch.load(
            args.slotwise_behavior_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        behavior_policy = behavior_payload.get("policy")
        if not isinstance(behavior_policy, dict):
            raise RuntimeError(
                "slotwise behavior checkpoint has no policy state_dict"
            )
        llm_policy.load_slotwise_behavior_prior(behavior_policy)
        if args.slotwise_behavior_scale == 0.0:
            print(
                "  Historical slotwise behavior checkpoint loaded for audit "
                "only; scale=0 keeps it out of Actor logits.",
                flush=True,
            )
        else:
            print(
                "  Qwen slotwise residual initialized over frozen historical "
                f"behavior policy: {args.slotwise_behavior_checkpoint}",
                flush=True,
            )

    # Resume before any probe. Previously core_probe silently inspected the
    # freshly assembled Phase-1 policy even when --resume named a Phase-2
    # checkpoint, invalidating checkpoint comparisons.
    start_update = 0
    if args.resume:
        start_update = pipeline.load_checkpoint(args.resume)
        print(f"Resumed from update {start_update}")
    if os.environ.get("RUNTIME_H2_CACHE_RESUME_SMOKE_ONLY", "0") == "1":
        prior = pipeline.world_model.transition.prior_state_action
        architecture = getattr(prior, "architecture", type(prior).__name__)
        print(
            "RUNTIME_H2_CACHE_RESUME_SMOKE PASS "
            f"update={start_update} prior={architecture} ", flush=True,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    fixed_scalar_v_output = os.environ.get("FIXED_SCALAR_V_PROBE_OUTPUT")
    if fixed_scalar_v_output:
        from sembelief_wm.rl.fixed_scalar_v_probe import run_fixed_scalar_v_probe

        report = run_fixed_scalar_v_probe(
            pipeline=pipeline,
            policy=llm_policy,
            output_dir=fixed_scalar_v_output,
            starts=int(os.environ.get("FIXED_SCALAR_V_STARTS", "2048")),
            continuations=int(os.environ.get(
                "FIXED_SCALAR_V_CONTINUATIONS", "4"
            )),
            collection_batch_size=int(os.environ.get(
                "FIXED_SCALAR_V_COLLECTION_BATCH", "8"
            )),
            feature_batch_size=int(os.environ.get(
                "FIXED_SCALAR_V_FEATURE_BATCH", "32"
            )),
            steps=int(os.environ.get("FIXED_SCALAR_V_STEPS", "1000")),
            train_batch_size=int(os.environ.get(
                "FIXED_SCALAR_V_TRAIN_BATCH", "128"
            )),
            learning_rate=float(os.environ.get(
                "FIXED_SCALAR_V_LR", "1e-4"
            )),
            eval_every=int(os.environ.get(
                "FIXED_SCALAR_V_EVAL_EVERY", "25"
            )),
            patience=int(os.environ.get(
                "FIXED_SCALAR_V_PATIENCE", "10"
            )),
            seed=int(os.environ.get("FIXED_SCALAR_V_SEED", "20260816")),
            start_dataset_path=os.environ.get("FIXED_SCALAR_V_START_DATASET"),
        )
        print("FIXED_SCALAR_V_RESULT " + json.dumps(report, sort_keys=True))
        if wandb_run is not None:
            wandb_run.finish()
        return

    fixed_q_output = os.environ.get("FIXED_COUNTERFACTUAL_Q_PROBE_OUTPUT")
    if fixed_q_output:
        from sembelief_wm.rl.fixed_counterfactual_q_probe import (
            run_fixed_counterfactual_q_probe,
        )
        report = run_fixed_counterfactual_q_probe(
            pipeline=pipeline, policy=llm_policy,
            scalar_dataset_path=os.environ["FIXED_COUNTERFACTUAL_Q_SOURCE"],
            output_dir=fixed_q_output,
            repeats=int(os.environ.get("FIXED_COUNTERFACTUAL_Q_REPEATS", "4")),
            collection_batch_size=int(os.environ.get("FIXED_COUNTERFACTUAL_Q_BATCH", "8")),
            steps=int(os.environ.get("FIXED_COUNTERFACTUAL_Q_STEPS", "1000")),
            train_batch_size=int(os.environ.get("FIXED_COUNTERFACTUAL_Q_TRAIN_BATCH", "128")),
            learning_rate=float(os.environ.get("FIXED_COUNTERFACTUAL_Q_LR", "1e-4")),
            eval_every=int(os.environ.get("FIXED_COUNTERFACTUAL_Q_EVAL_EVERY", "25")),
            patience=int(os.environ.get("FIXED_COUNTERFACTUAL_Q_PATIENCE", "10")),
            counterfactual_dataset_path=os.environ.get("FIXED_COUNTERFACTUAL_Q_DATASET"),
            ranking_margin=float(os.environ.get("FIXED_COUNTERFACTUAL_Q_PAIR_MARGIN", "0.01")),
            ranking_temperature=float(os.environ.get("FIXED_COUNTERFACTUAL_Q_RANK_TEMPERATURE", "0.01")),
            regression_coefficient=float(os.environ.get("FIXED_COUNTERFACTUAL_Q_REGRESSION_COEF", "0.10")),
        )
        print("FIXED_COUNTERFACTUAL_Q_REPORT " + json.dumps(report, sort_keys=True))
        if wandb_run is not None:
            wandb_run.finish()
        return

    if args.core_probe:
        from sembelief_wm.rl.core_probe import run_phase2_core_probe

        if config.phase2.ppo.offline_bc_steps > 0:
            raise ValueError(
                "--core-probe requires --offline-bc-steps 0 so it diagnoses "
                "the assembled policy without running a hidden initialization "
                "phase; run BC as a separate downstream experiment."
            )
        run_phase2_core_probe(
            pipeline=pipeline,
            policy=llm_policy,
            batch_size=args.core_probe_batch_size,
            sample_limit=args.core_probe_sample_limit,
            critic_steps=args.core_probe_critic_steps,
            critic_lr=args.core_probe_critic_lr,
            output_path=args.core_probe_output,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    if args.semantic_probe:
        from sembelief_wm.rl.semantic_probe import run_phase2_semantic_probe

        if config.phase2.ppo.offline_bc_steps > 0:
            raise ValueError("--semantic-probe requires --offline-bc-steps 0")
        if not args.semantic_probe_output:
            raise ValueError("--semantic-probe-output is required")
        run_phase2_semantic_probe(
            pipeline=pipeline,
            policy=llm_policy,
            batch_size=args.semantic_probe_batch_size,
            horizon=args.semantic_probe_horizon,
            num_sequences=args.semantic_probe_sequences,
            rollout_repeats=args.semantic_probe_rollout_repeats,
            seed=args.semantic_probe_seed,
            checkpoint_path=args.resume or args.wm_checkpoint,
            output_path=args.semantic_probe_output,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    if args.counterfactual_action_audit:
        from sembelief_wm.rl.counterfactual_action_audit import (
            run_same_start_four_action_audit,
        )

        if config.phase2.ppo.offline_bc_steps > 0:
            raise ValueError(
                "--counterfactual-action-audit requires --offline-bc-steps 0"
            )
        if not args.resume:
            raise ValueError(
                "--counterfactual-action-audit requires a BC/policy --resume checkpoint"
            )
        if not args.counterfactual_audit_output:
            raise ValueError("--counterfactual-audit-output is required")
        run_same_start_four_action_audit(
            pipeline=pipeline,
            policy=llm_policy,
            world_model=world_model,
            output_path=args.counterfactual_audit_output,
            num_samples=args.counterfactual_audit_samples,
            batch_size=args.counterfactual_audit_batch_size,
            seed=args.counterfactual_audit_seed,
            action_id_offset=config.phase2.ppo.wm_action_id_offset,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    if args.prior_reward_decomposition_audit:
        from sembelief_wm.rl.prior_reward_decomposition_audit import (
            run_prior_reward_decomposition_audit,
        )

        if config.phase2.ppo.offline_bc_steps > 0:
            raise ValueError(
                "--prior-reward-decomposition-audit requires --offline-bc-steps 0"
            )
        if not args.resume:
            raise ValueError(
                "--prior-reward-decomposition-audit requires --resume"
            )
        if not args.decomposition_audit_output:
            raise ValueError("--decomposition-audit-output is required")
        run_prior_reward_decomposition_audit(
            pipeline=pipeline,
            policy=llm_policy,
            world_model=world_model,
            output_path=args.decomposition_audit_output,
            num_samples=args.decomposition_audit_samples,
            batch_size=args.decomposition_audit_batch_size,
            seed=args.decomposition_audit_seed,
            action_id_offset=config.phase2.ppo.wm_action_id_offset,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    # Now that llm_policy exists, finish wiring RealCollector + Evaluator so they
    # share the trained policy. This must happen AFTER assemble_llm_pipeline.
    if want_real_env:
        real_collector = RealCollector(
            env_factory=lambda seed: env_adapter.make_env(seed=seed),
            tokenizer=image_tokenizer,
            world_model=world_model,
            policy=llm_policy,
            config=_real_cfg,
            env_id_tensor=env_id_tensor,
            env_id=args.env_id,
            replay_buffer=online_buffer,
            device=device,
            wm_action_id_offset=config.phase2.ppo.wm_action_id_offset,
            # Sokoban exposes environment actions 1..4 while its canonical
            # model actions are 0..3. FrozenLake already uses canonical 0..3
            # actions, so applying Sokoban's historical +1 mapping would send
            # invalid/wrong actions during collection and evaluation.
            model_to_env_action=(
                (lambda action: action + 1)
                if args.env_id == "sokoban"
                else (lambda action: action)
            ),
        )
        evaluator = RealEnvEvaluator(real_collector, eval_seeds=_eval_seeds, eval_levels=_eval_levels)
        # Inject into the already-built pipeline (assemble_llm_pipeline couldn't
        # set these because llm_policy didn't exist yet at assembly time).
        pipeline.real_collector = real_collector
        pipeline.evaluator = evaluator
        if pipeline.policy is not llm_policy or real_collector.policy is not pipeline.policy:
            raise RuntimeError(
                "Actor/eval identity violation: PPO and RealEnvEvaluator do not "
                "share the exact same policy object"
            )
        print("  Wired RealCollector + RealEnvEvaluator into pipeline (shared llm_policy).")
        print("  Verified Actor -> real eval policy object identity.")

        if args.cross_eval_wm_checkpoint:
            if not args.resume:
                raise ValueError(
                    "--cross-eval-wm-checkpoint requires --resume to supply "
                    "the Actor/Critic checkpoint"
                )
            cross_path = Path(args.cross_eval_wm_checkpoint)
            if not cross_path.is_file():
                raise FileNotFoundError(
                    f"cross-evaluation WM checkpoint not found: {cross_path}"
                )
            cross_checkpoint = torch.load(
                cross_path, map_location="cpu", weights_only=False
            )
            cross_world_model = cross_checkpoint.get("world_model")
            if cross_world_model is None:
                raise ValueError(
                    "--cross-eval-wm-checkpoint must be a Phase-2 checkpoint "
                    "containing world_model tensors"
                )
            incompatible = world_model.load_state_dict(cross_world_model, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "cross-evaluation WM load was not exact: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            if _eval_levels is None and _eval_seeds is None:
                raise ValueError(
                    "--cross-eval-wm-checkpoint requires a fixed evaluation "
                    "set via --eval-levels-file or --eval-seeds-file"
                )
            metrics = evaluator.evaluate(config.phase2.ppo.eval_episodes)
            print(
                "CROSS_EVAL_RESULT "
                f"actor_checkpoint={args.resume} "
                f"actor_update={start_update} "
                f"wm_checkpoint={cross_path} "
                f"wm_update={cross_checkpoint.get('update')} "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in sorted(metrics.items())
                    if isinstance(value, (int, float))
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.finish()
            return

        if os.environ.get("ONLINE_RENDERED_ACTOR_WARMUP", "0") == "1":
            from sembelief_wm.rl.online_actor_warmup import (
                run_online_rendered_actor_warmup,
            )
            warmup_cache = os.environ.get(
                "ONLINE_ACTOR_WARMUP_CACHE",
                str(Path(args.checkpoint_dir) / "online_actor_warmup_cache.pt"),
            )
            pipeline.online_actor_warmup_fn = lambda: run_online_rendered_actor_warmup(
                policy=llm_policy,
                collector=real_collector,
                episodes=list(dataset.episodes),
                eval_levels=_eval_levels,
                cache_path=warmup_cache,
                train_size=int(os.environ.get("ONLINE_ACTOR_WARMUP_TRAIN_SIZE", "1500")),
                validation_size=int(os.environ.get("ONLINE_ACTOR_WARMUP_VAL_SIZE", "300")),
                steps=int(os.environ.get("ONLINE_ACTOR_WARMUP_STEPS", "1000")),
                batch_size=int(os.environ.get("ONLINE_ACTOR_WARMUP_BATCH_SIZE", "64")),
                learning_rate=float(os.environ.get("ONLINE_ACTOR_WARMUP_LR", "5e-5")),
                seed=int(os.environ.get("ONLINE_ACTOR_WARMUP_SEED", "20260812")),
                target_top1=float(os.environ.get(
                    "ONLINE_ACTOR_WARMUP_TARGET_TOP1", "0.60"
                )),
                target_max_fraction=float(os.environ.get(
                    "ONLINE_ACTOR_WARMUP_TARGET_MAX_FRACTION", "0.40"
                )),
                gate_patience=int(os.environ.get(
                    "ONLINE_ACTOR_WARMUP_GATE_PATIENCE", "2"
                )),
            )
            print(
                "  Enabled leakage-free online-rendered solver Actor warm-up: "
                f"cache={warmup_cache}",
                flush=True,
            )

    posterior_target_output = os.environ.get(
        "POSTERIOR_TARGET_BUCKET_DIAGNOSTIC_OUTPUT"
    )
    if posterior_target_output:
        if evaluator is None:
            raise RuntimeError(
                "POSTERIOR_TARGET_BUCKET_DIAGNOSTIC_OUTPUT requires the fixed "
                "real-env evaluator"
            )
        from sembelief_wm.rl.posterior_target_bucket_diagnostic import (
            run_posterior_target_bucket_diagnostic,
        )
        run_posterior_target_bucket_diagnostic(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=posterior_target_output,
            level_offset=int(os.environ.get(
                "POSTERIOR_TARGET_LEVEL_OFFSET", "32"
            )),
            level_limit=int(os.environ.get(
                "POSTERIOR_TARGET_LEVEL_LIMIT", "64"
            )),
            batch_size=int(os.environ.get(
                "POSTERIOR_TARGET_BATCH_SIZE", "32"
            )),
            reward_scale=float(os.environ.get(
                "POSTERIOR_TARGET_REAL_REWARD_SCALE", "0.1"
            )),
            checkpoint_update=start_update,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    transition_target_output = os.environ.get(
        "TRANSITION_TARGET_CAUSAL_DIAGNOSTIC_OUTPUT"
    )
    if transition_target_output:
        if evaluator is None:
            raise RuntimeError(
                "TRANSITION_TARGET_CAUSAL_DIAGNOSTIC_OUTPUT requires the "
                "fixed real-env evaluator"
            )
        from sembelief_wm.rl.transition_target_causal_diagnostic import (
            run_transition_target_causal_diagnostic,
        )
        run_transition_target_causal_diagnostic(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=transition_target_output,
            level_offset=int(os.environ.get(
                "TRANSITION_TARGET_LEVEL_OFFSET", "32"
            )),
            level_limit=int(os.environ.get(
                "TRANSITION_TARGET_LEVEL_LIMIT", "64"
            )),
            reward_scale=float(os.environ.get(
                "TRANSITION_TARGET_REAL_REWARD_SCALE", "0.1"
            )),
            checkpoint_update=start_update,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    long_horizon_critic_output = os.environ.get(
        "LONG_HORIZON_CRITIC_GROUNDING_OUTPUT_DIR"
    )
    if long_horizon_critic_output:
        if evaluator is None:
            raise RuntimeError(
                "LONG_HORIZON_CRITIC_GROUNDING_OUTPUT_DIR requires the fixed "
                "real-env evaluator"
            )
        from sembelief_wm.rl.long_horizon_critic_grounding import (
            run_long_horizon_critic_grounding,
        )
        run_long_horizon_critic_grounding(
            pipeline=pipeline,
            evaluator=evaluator,
            output_dir=long_horizon_critic_output,
            train_offset=int(os.environ.get(
                "LONG_HORIZON_CRITIC_TRAIN_OFFSET", "64"
            )),
            train_count=int(os.environ.get(
                "LONG_HORIZON_CRITIC_TRAIN_COUNT", "192"
            )),
            validation_offset=int(os.environ.get(
                "LONG_HORIZON_CRITIC_VALIDATION_OFFSET", "32"
            )),
            validation_count=int(os.environ.get(
                "LONG_HORIZON_CRITIC_VALIDATION_COUNT", "32"
            )),
            max_updates=int(os.environ.get(
                "LONG_HORIZON_CRITIC_MAX_UPDATES", "1200"
            )),
            eval_every=int(os.environ.get(
                "LONG_HORIZON_CRITIC_EVAL_EVERY", "25"
            )),
            level_groups_per_update=int(os.environ.get(
                "LONG_HORIZON_CRITIC_GROUPS_PER_UPDATE", "16"
            )),
            endpoint_samples_per_update=int(os.environ.get(
                "LONG_HORIZON_CRITIC_ENDPOINT_SAMPLES", "64"
            )),
            gamma=float(os.environ.get(
                "LONG_HORIZON_CRITIC_GAMMA", str(pipeline.config.gamma)
            )),
            reward_scale=float(os.environ.get(
                "LONG_HORIZON_CRITIC_REWARD_SCALE", "0.1"
            )),
            ranking_coef=float(os.environ.get(
                "LONG_HORIZON_CRITIC_RANKING_COEF", "0.05"
            )),
            ranking_temperature=float(os.environ.get(
                "LONG_HORIZON_CRITIC_RANKING_TEMPERATURE", "0.05"
            )),
            ev_threshold=float(os.environ.get(
                "LONG_HORIZON_CRITIC_EV_THRESHOLD", "0.10"
            )),
            required_streak=int(os.environ.get(
                "LONG_HORIZON_CRITIC_REQUIRED_STREAK", "3"
            )),
            margin=float(os.environ.get(
                "LONG_HORIZON_CRITIC_MARGIN", "0.01"
            )),
            seed=int(os.environ.get(
                "LONG_HORIZON_CRITIC_SEED", "20260818"
            )),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    spatial_decoder_output = os.environ.get(
        "SPATIAL_DECODER_REAL_POSTERIOR_AUDIT_OUTPUT"
    )
    if spatial_decoder_output:
        if evaluator is None:
            raise RuntimeError(
                "SPATIAL_DECODER_REAL_POSTERIOR_AUDIT_OUTPUT requires the "
                "fixed real-env evaluator"
            )
        from sembelief_wm.rl.spatial_decoder_real_posterior_audit import (
            run_spatial_decoder_real_posterior_audit,
        )
        run_spatial_decoder_real_posterior_audit(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=spatial_decoder_output,
            level_offset=int(os.environ.get(
                "SPATIAL_DECODER_AUDIT_LEVEL_OFFSET", "32"
            )),
            level_limit=int(os.environ.get(
                "SPATIAL_DECODER_AUDIT_LEVEL_LIMIT", "64"
            )),
            batch_size=int(os.environ.get(
                "SPATIAL_DECODER_AUDIT_BATCH_SIZE", "64"
            )),
            checkpoint_update=start_update,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    prior_repair_output = os.environ.get("RUNTIME_MATCHED_PRIOR_REPAIR_OUTPUT_DIR")
    if prior_repair_output:
        from sembelief_wm.rl.runtime_matched_state_action_prior_repair import (
            repair_runtime_matched_state_action_prior,
        )
        required = {
            "cache_path": os.environ.get("TERMINAL_REWARD_NEAR_CACHE"),
            "decoder_path": os.environ.get("TERMINAL_REWARD_DECODER"),
            "reward_path": os.environ.get("TERMINAL_REWARD_CHECKPOINT"),
            "initial_prior_path": os.environ.get("TERMINAL_REWARD_PRIOR"),
        }
        if any(value is None for value in required.values()):
            raise RuntimeError("runtime-matched prior repair requires Reward/cache/prior artifacts")
        repair_runtime_matched_state_action_prior(
            pipeline=pipeline, output_dir=prior_repair_output,
            source_checkpoint=args.resume or args.wm_checkpoint,
            stage1_steps=int(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE1_STEPS", "1200")),
            stage2_steps=int(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE2_STEPS", "800")),
            batch_size=int(os.environ.get("RUNTIME_MATCHED_PRIOR_REPAIR_BATCH_SIZE", "16")),
            eval_batch_size=int(os.environ.get("RUNTIME_MATCHED_PRIOR_REPAIR_EVAL_BATCH_SIZE", "32")),
            eval_every=int(os.environ.get("RUNTIME_MATCHED_PRIOR_REPAIR_EVAL_EVERY", "50")),
            stage1_lr=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE1_LR", "2e-5")),
            stage2_lr=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE2_LR", "5e-6")),
            stage1_latent_coef=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE1_LATENT_COEF", "10.0")),
            stage2_latent_coef=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE2_LATENT_COEF", "5.0")),
            stage1_h1_coef=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE1_H1_COEF", "1.0")),
            stage1_h2_coef=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE1_H2_COEF", "1.0")),
            stage1_required_streak=int(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE1_REQUIRED_STREAK", "3")),
            stage2_reward_coef=float(os.environ.get("RUNTIME_MATCHED_PRIOR_STAGE2_REWARD_COEF", "0.25")),
            teacher_rehearsal_coef=float(os.environ.get("RUNTIME_MATCHED_PRIOR_TEACHER_REHEARSAL_COEF", "0.5")),
            stage2_static_layout_coef=float(os.environ.get(
                "RUNTIME_MATCHED_PRIOR_STAGE2_STATIC_LAYOUT_COEF", "1.0"
            )),
            seed=int(os.environ.get("RUNTIME_MATCHED_PRIOR_REPAIR_SEED", "0")),
            **required,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    reward_coordinate_output = os.environ.get(
        "TERMINAL_REWARD_COORDINATE_AUDIT_OUTPUT"
    )
    if reward_coordinate_output:
        from sembelief_wm.rl.terminal_reward_coordinate_audit import (
            run_terminal_reward_coordinate_audit,
        )
        cache_path = os.environ.get("TERMINAL_REWARD_COORDINATE_AUDIT_CACHE")
        reward_checkpoint = os.environ.get("TERMINAL_REWARD_CHECKPOINT")
        if not cache_path or not reward_checkpoint:
            raise RuntimeError(
                "Reward coordinate audit requires "
                "TERMINAL_REWARD_COORDINATE_AUDIT_CACHE and TERMINAL_REWARD_CHECKPOINT"
            )
        run_terminal_reward_coordinate_audit(
            pipeline=pipeline, cache_path=cache_path,
            reward_checkpoint=reward_checkpoint, output_path=reward_coordinate_output,
            batch_size=int(os.environ.get("TERMINAL_REWARD_COORDINATE_AUDIT_BATCH", "64")),
            require_pass=(
                os.environ.get("TERMINAL_REWARD_COORDINATE_REQUIRE_PASS", "1") == "1"
            ),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    frozen_posterior_probe_output = os.environ.get(
        "FROZEN_POSTERIOR_SPATIAL_PROBE_OUTPUT"
    )
    if frozen_posterior_probe_output:
        if evaluator is None:
            raise RuntimeError(
                "FROZEN_POSTERIOR_SPATIAL_PROBE_OUTPUT requires the "
                "fixed real-env evaluator"
            )
        from sembelief_wm.rl.frozen_posterior_spatial_probe import (
            run_frozen_posterior_spatial_probe,
        )
        run_frozen_posterior_spatial_probe(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=frozen_posterior_probe_output,
            level_offset=int(os.environ.get(
                "FROZEN_POSTERIOR_PROBE_LEVEL_OFFSET", "32"
            )),
            level_limit=int(os.environ.get(
                "FROZEN_POSTERIOR_PROBE_LEVEL_LIMIT", "64"
            )),
            steps=int(os.environ.get(
                "FROZEN_POSTERIOR_PROBE_STEPS", "800"
            )),
            batch_size=int(os.environ.get(
                "FROZEN_POSTERIOR_PROBE_BATCH_SIZE", "128"
            )),
            lr=float(os.environ.get(
                "FROZEN_POSTERIOR_PROBE_LR", "3e-4"
            )),
            seed=int(os.environ.get(
                "FROZEN_POSTERIOR_PROBE_SEED", "0"
            )),
            checkpoint_update=start_update,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    runtime_decoder_output = os.environ.get(
        "RUNTIME_MATCHED_SPATIAL_DECODER_OUTPUT_DIR"
    )
    if runtime_decoder_output:
        if evaluator is None:
            raise RuntimeError(
                "RUNTIME_MATCHED_SPATIAL_DECODER_OUTPUT_DIR requires fixed real evaluation"
            )
        from sembelief_wm.rl.runtime_matched_spatial_decoder import (
            run_runtime_matched_spatial_decoder,
        )
        run_runtime_matched_spatial_decoder(
            pipeline=pipeline, evaluator=evaluator, output_dir=runtime_decoder_output,
            steps=int(os.environ.get("RUNTIME_SPATIAL_DECODER_STEPS", "2000")),
            batch_size=int(os.environ.get("RUNTIME_SPATIAL_DECODER_BATCH_SIZE", "128")),
            eval_every=int(os.environ.get("RUNTIME_SPATIAL_DECODER_EVAL_EVERY", "100")),
            lr=float(os.environ.get("RUNTIME_SPATIAL_DECODER_LR", "3e-4")),
            seed=int(os.environ.get("RUNTIME_SPATIAL_DECODER_SEED", "0")),
            checkpoint_update=start_update,
            base_wm_checkpoint=args.wm_checkpoint,
            source_checkpoint=args.resume or args.wm_checkpoint,
            bfs_dataset_path=os.environ.get("RUNTIME_SPATIAL_DECODER_BFS_DATASET"),
            actor_augmentation=(
                os.environ.get("RUNTIME_SPATIAL_DECODER_ACTOR_AUGMENTATION", "1") == "1"
            ),
            actor_train_levels=int(os.environ.get(
                "RUNTIME_SPATIAL_DECODER_ACTOR_TRAIN_LEVELS", "64"
            )),
            actor_validation_levels=int(os.environ.get(
                "RUNTIME_SPATIAL_DECODER_ACTOR_VALIDATION_LEVELS", "16"
            )),
            actor_test_levels=int(os.environ.get(
                "RUNTIME_SPATIAL_DECODER_ACTOR_TEST_LEVELS", "16"
            )),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    runtime_h2_cache_output = os.environ.get(
        "RUNTIME_MATCHED_H2_SEQUENCE_CACHE_OUTPUT"
    )
    if runtime_h2_cache_output:
        if evaluator is None:
            raise RuntimeError("runtime H1/H2 cache builder requires fixed real evaluation")
        decoder_path = os.environ.get("TERMINAL_REWARD_DECODER")
        if not decoder_path:
            raise RuntimeError("runtime H1/H2 cache builder requires TERMINAL_REWARD_DECODER")
        from sembelief_wm.rl.runtime_matched_h2_sequence_cache import (
            build_runtime_matched_h2_sequence_cache,
        )
        build_runtime_matched_h2_sequence_cache(
            evaluator=evaluator, decoder_path=decoder_path,
            output_path=runtime_h2_cache_output,
            batch_size=int(os.environ.get("RUNTIME_MATCHED_H2_CACHE_BATCH_SIZE", "128")),
            source_checkpoint=args.resume or args.wm_checkpoint,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    initial_h12_cache_output = os.environ.get(
        "RUNTIME_MATCHED_INITIAL_H12_CACHE_OUTPUT"
    )
    if initial_h12_cache_output:
        if evaluator is None:
            raise RuntimeError(
                "initial H1/H2 posterior cache builder requires fixed real evaluation"
            )
        from sembelief_wm.rl.initial_h12_posterior_cache import (
            build_initial_h12_posterior_cache,
        )
        build_initial_h12_posterior_cache(
            evaluator=evaluator,
            output_path=initial_h12_cache_output,
            source_checkpoint=args.resume or args.wm_checkpoint,
            batch_size=int(os.environ.get(
                "RUNTIME_MATCHED_INITIAL_H12_CACHE_BATCH_SIZE", "32"
            )),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    online_counterfactual_output = os.environ.get(
        "ONLINE_COUNTERFACTUAL_256_OUTPUT"
    )
    if online_counterfactual_output:
        if evaluator is None:
            raise RuntimeError(
                "ONLINE_COUNTERFACTUAL_256_OUTPUT requires the real evaluator"
            )
        from sembelief_wm.rl.online_counterfactual_256_audit import (
            run_online_counterfactual_256_audit,
        )
        run_online_counterfactual_256_audit(
            pipeline=pipeline,
            policy=llm_policy,
            evaluator=evaluator,
            output_path=online_counterfactual_output,
            batch_size=int(os.environ.get(
                "ONLINE_COUNTERFACTUAL_256_BATCH_SIZE", "16"
            )),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    imagined_real_output = os.environ.get("IMAGINED_REAL_RANKING_OUTPUT")
    if imagined_real_output:
        if evaluator is None:
            raise RuntimeError(
                "IMAGINED_REAL_RANKING_OUTPUT requires fixed real-env evaluation"
            )
        from sembelief_wm.rl.imagined_real_action_ranking_audit import (
            run_imagined_real_action_ranking_audit,
        )
        run_imagined_real_action_ranking_audit(
            pipeline=pipeline,
            policy=llm_policy,
            evaluator=evaluator,
            output_path=imagined_real_output,
            real_repeats=int(os.environ.get("IMAGINED_REAL_REAL_REPEATS", "16")),
            imagined_repeats=int(os.environ.get("IMAGINED_REAL_MODEL_REPEATS", "32")),
            reward_scale=float(os.environ.get("IMAGINED_REAL_REWARD_SCALE", "0.1")),
            seed=int(os.environ.get("IMAGINED_REAL_SEED", "20260813")),
            tie_epsilon=float(os.environ.get("IMAGINED_REAL_TIE_EPS", "1e-4")),
            advantage_margin=float(os.environ.get("IMAGINED_REAL_ADV_MARGIN", "0.01")),
            deterministic_continuation=(
                os.environ.get("IMAGINED_REAL_DETERMINISTIC_CONTINUATION", "0") == "1"
            ),
            ordered_q_checkpoint=os.environ.get("IMAGINED_REAL_ORDERED_Q_CHECKPOINT"),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    near_terminal_output = os.environ.get("NEAR_TERMINAL_RANKING_OUTPUT")
    if near_terminal_output:
        from sembelief_wm.rl.near_terminal_ranking_audit import (
            run_near_terminal_ranking_audit,
        )
        cache_path = os.environ.get("NEAR_TERMINAL_RANKING_CACHE")
        if not cache_path:
            raise RuntimeError("NEAR_TERMINAL_RANKING_CACHE is required")
        run_near_terminal_ranking_audit(
            pipeline=pipeline, policy=llm_policy, cache_path=cache_path,
            output_path=near_terminal_output,
            starts=int(os.environ.get("NEAR_TERMINAL_RANKING_STARTS", "256")),
            batch_size=int(os.environ.get("NEAR_TERMINAL_RANKING_BATCH_SIZE", "16")),
            seed=int(os.environ.get("NEAR_TERMINAL_RANKING_SEED", "20260813")),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    critic_repair_output = os.environ.get("NEAR_TERMINAL_CRITIC_REPAIR_OUTPUT")
    if critic_repair_output:
        from sembelief_wm.rl.near_terminal_critic_repair import repair_near_terminal_critic
        cache_path=os.environ.get("NEAR_TERMINAL_RANKING_CACHE")
        if not cache_path: raise RuntimeError("NEAR_TERMINAL_RANKING_CACHE is required")
        pipeline.load_checkpoint_update=start_update
        report=repair_near_terminal_critic(
            pipeline=pipeline,policy=llm_policy,cache_path=cache_path,output_dir=critic_repair_output,
            steps=int(os.environ.get("NEAR_TERMINAL_CRITIC_REPAIR_STEPS","600")),
            batch_size=int(os.environ.get("NEAR_TERMINAL_CRITIC_REPAIR_BATCH_SIZE","64")),
            lr=float(os.environ.get("NEAR_TERMINAL_CRITIC_REPAIR_LR","1e-4")),
        )
        print("NEAR_TERMINAL_CRITIC_REPAIR "+json.dumps(report,sort_keys=True),flush=True)
        if wandb_run is not None: wandb_run.finish()
        return

    # Diagnostic-only cache construction through the exact production eval
    # renderer and the exact same ImageTokenizer instance.  This hook lives
    # here (rather than in an assembly wrapper) because the tokenizer is only
    # created for real-env evaluation after pipeline assembly.
    online_h8_output = os.environ.get("BUILD_ONLINE_FULL_H8_CACHE_OUTPUT")
    if online_h8_output:
        if not want_real_env or real_collector is None:
            raise RuntimeError(
                "BUILD_ONLINE_FULL_H8_CACHE_OUTPUT requires real-env eval "
                "and its production ImageTokenizer"
            )
        from sembelief_wm.rl.full_trajectory_h8 import (
            build_online_rendered_full_trajectory_h8_cache,
        )
        manifest = build_online_rendered_full_trajectory_h8_cache(
            world_model=world_model,
            tokenizer=image_tokenizer,
            env_factory=lambda seed: env_adapter.make_env(seed=seed),
            episodes=data_source.dataset.episodes,
            output_dir=online_h8_output,
            device=device,
            max_episodes=(
                int(os.environ["BUILD_ONLINE_FULL_H8_MAX_EPISODES"])
                if os.environ.get("BUILD_ONLINE_FULL_H8_MAX_EPISODES") else None
            ),
        )
        print("ONLINE_FULL_TRAJECTORY_H8_CACHE " + json.dumps(manifest), flush=True)
        if wandb_run is not None:
            wandb_run.finish()
        return

    # Read-only frozen 2x2 diagnostic. Panels differ only by Actor while both
    # use the release Critic for bootstrap; matrix rows then differ only by
    # the evaluated Critic checkpoint.
    frozen_cross_output = os.environ.get("FROZEN_CROSS_DIAGNOSTIC_OUTPUT")
    if frozen_cross_output:
        from sembelief_wm.rl.frozen_actor_critic_cross_diagnostic import (
            run_frozen_actor_critic_cross_diagnostic,
        )
        run_frozen_actor_critic_cross_diagnostic(
            pipeline=pipeline,
            actor0_checkpoint=os.environ["FROZEN_CROSS_ACTOR0_CHECKPOINT"],
            actorN_checkpoint=os.environ["FROZEN_CROSS_ACTORN_CHECKPOINT"],
            output_path=frozen_cross_output,
            panel_batches=int(os.environ.get(
                "FROZEN_CROSS_PANEL_BATCHES", "12"
            )),
            panel_batch_size=int(os.environ.get(
                "FROZEN_CROSS_PANEL_BATCH_SIZE", "128"
            )),
            seed=int(os.environ.get("FROZEN_CROSS_SEED", "20260815")),
            ev_threshold=float(os.environ.get(
                "FROZEN_CROSS_EV_THRESHOLD", "0.10"
            )),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    # Read-only causal ablation for the joint Critic failure.  Every branch is
    # run from the same snapshot and the diagnostic restores both model and
    # optimizer before returning, so this hook cannot silently continue
    # training from an ablation branch.
    joint_critic_diagnostic_output = os.environ.get(
        "JOINT_CRITIC_CAUSAL_DIAGNOSTIC_OUTPUT"
    )
    if joint_critic_diagnostic_output:
        if evaluator is None or real_collector is None:
            raise RuntimeError(
                "JOINT_CRITIC_CAUSAL_DIAGNOSTIC_OUTPUT requires the fixed "
                "real evaluator and RealCollector"
            )
        from sembelief_wm.rl.joint_critic_causal_diagnostic import (
            run_joint_critic_causal_diagnostic,
        )
        run_joint_critic_causal_diagnostic(
            pipeline=pipeline,
            output_path=joint_critic_diagnostic_output,
            h2_batch_size=int(os.environ.get(
                "JOINT_DIAGNOSTIC_H2_BATCH_SIZE", "128"
            )),
            h2_validation_batches=int(os.environ.get(
                "JOINT_DIAGNOSTIC_H2_VALIDATION_BATCHES", "4"
            )),
            real_level_offset=int(os.environ.get(
                "JOINT_DIAGNOSTIC_REAL_LEVEL_OFFSET", "32"
            )),
            real_level_count=int(os.environ.get(
                "JOINT_DIAGNOSTIC_REAL_LEVEL_COUNT", "32"
            )),
            mixed_real_fraction=float(os.environ.get(
                "JOINT_DIAGNOSTIC_MIXED_REAL_FRACTION", "0.25"
            )),
            seed=int(os.environ.get(
                "JOINT_DIAGNOSTIC_SEED", "20260814"
            )),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    # ── Train ──
    print(f"=== Starting training: {config.phase2.ppo.total_updates} updates ===\n")
    t0 = time.time()

    pipeline.train(
        checkpoint_dir=args.checkpoint_dir,
        start_update=start_update,
    )

    elapsed = time.time() - t0
    print(f"\n=== Training complete in {elapsed:.1f}s ===")

    # Run-end diagnostics: entropy min / collapse count / clip-zero ratio /
    # vloss max / SR thirds. Lets us judge whether the entropy floor worked
    # rather than relying on noisy success_rate alone.
    if hasattr(logger, "summarize"):
        logger.summarize()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
