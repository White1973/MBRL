"""Standalone production bootstrap for the isolated Le-WM state machine.

This module intentionally does not import ``scripts.train_mbrl``.  Ordinary-WM
CLI diagnostics may evolve independently while Le-WM keeps a small, explicit
and fail-closed construction path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch

from ..collectors.real import RealCollector, RealCollectorConfig, RealEnvEvaluator
from ..config import (
    BackboneConfig,
    BeliefConfig,
    Config,
    CurriculumConfig,
    EMAConfig,
    EncoderConfig,
    EnvironmentConfig,
    EnvRewardSpec,
    Phase2Config,
    PPOConfig,
    RewardConfig,
    SIGRegConfig,
    TrainingConfig,
    WandbConfig,
)
from ..data.adapters.sokoban import SokobanAdapter
from ..data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from ..data.tokenizers.image import ImageTokenizer
from ..envs.sokoban.action_adapter import SokobanActionAdapter
from ..model import QwenTransitionBackbone, WorldModel
from ..model.checkpoint_semantics import validate_world_model_semantics
from ..pipelines.assemble import assemble_llm_pipeline
from ..rl.llm_policy import LLMPolicyConfig
from .config import LeWMStage
from .pipeline import LeWMOrchestrator


_REPOSITORY = Path(__file__).resolve().parents[2]
_DEFAULT_BACKBONE = Path("/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class LeWMPrintLogger:
    """Le-WM-owned stdout logger; no ordinary-WM diagnostics are imported."""

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        values = " | ".join(
            f"{key}={value:.4f}"
            for key, value in sorted(metrics.items())
            if not key.startswith("_")
        )
        print(f"[update {step:>5d}] {values}", flush=True)


def _config() -> Config:
    backbone = Path(os.environ.get("LEWM_BACKBONE_MODEL", str(_DEFAULT_BACKBONE)))
    hidden_dim = _env_int("LEWM_HIDDEN_DIM", 2048)
    reward_mapping = os.environ.get(
        "REWARD_MAPPING", "per_transition_success_conservative"
    )
    return Config(
        hidden_dim=hidden_dim,
        belief=BeliefConfig(num_slots=_env_int("LEWM_BELIEF_SLOTS", 36)),
        encoder=EncoderConfig(
            encoder_type="vjepa2",
            compressed_tokens=_env_int("LEWM_BELIEF_SLOTS", 36),
            vjepa2_raw_dim=1408,
        ),
        backbone=BackboneConfig(
            model_name=str(backbone),
            attention_mode="bidirectional",
            action_conditioning_mode="embedded",
            attn_implementation=os.environ.get("LEWM_ATTN_IMPLEMENTATION", "sdpa"),
        ),
        reward=RewardConfig(
            readout="mean_pool",
            supervision_source="posterior",
            head_hidden_dim=_env_int("LEWM_REWARD_HEAD_HIDDEN_DIM", 128),
            success_reward_threshold=1.0,
        ),
        sigreg=SIGRegConfig(),
        ema=EMAConfig(),
        curriculum=CurriculumConfig(
            horizons=[1, 2, 4, 8], switch_steps=[0, 1, 2, 3]
        ),
        training=TrainingConfig(
            total_steps=1,
            checkpoint_every=1,
            prior_isolation_mode="shared",
            prior_residual_rank=64,
            posterior_observation_residual_scale=0.0,
            posterior_grounding_mode="visual_anchor",
            posterior_recurrent_residual_scale=0.25,
            posterior_action_free=True,
        ),
        wandb=WandbConfig(enabled=False),
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
            env_ids=["sokoban"],
            reward_specs={
                "sokoban": EnvRewardSpec(positive_value=1.0, negative_value=-0.1)
            },
        ),
        phase2=Phase2Config(
            world_model_mode="frozen_wm",
            ppo=PPOConfig(
                total_updates=_env_int("TOTAL_UPDATES", 113),
                rollout_batch_size=_env_int("ROLLOUT_BATCH_SIZE", 8),
                rollout_horizon=_env_int("ROLLOUT_HORIZON", 3),
                use_value_bootstrap=True,
                imagination_termination_mode="predicted_success",
                rollouts_per_update=_env_int("ROLLOUTS_PER_UPDATE", 8),
                epochs_per_update=_env_int("PPO_EPOCHS", 2),
                minibatch_size=_env_int("MINIBATCH_SIZE", 64),
                recompute_old_log_probs=True,
                actor_lr=_env_float("ACTOR_LR", 2e-5),
                critic_lr=_env_float("CRITIC_LR", 3e-5),
                critic_warmup_min_updates=0,
                critic_warmup_ev_threshold=0.10,
                critic_warmup_ev_patience=3,
                critic_warmup_validation_size=256,
                critic_warmup_replay_capacity=4096,
                critic_warmup_train_samples=512,
                entropy_coef=_env_float("ENTROPY_COEF", 0.01),
                target_kl=_env_float("TARGET_KL", 0.01),
                behavior_kl_coef=0.0,
                behavior_bc_coef=0.0,
                offline_bc_steps=0,
                wm_action_id_offset=0,
                clip_epsilon=_env_float("CLIP_EPSILON", 0.1),
                target_entropy=_env_float("TARGET_ENTROPY", 0.5),
                entropy_floor_coef=_env_float("ENTROPY_FLOOR_COEF", 0.05),
                reward_mapping=reward_mapping,
                reward_scale=_env_float("REWARD_SCALE", 0.1),
                reward_low_confidence_scale=0.1,
                collect_every=_env_int("COLLECT_EVERY", 1),
                collect_episodes=_env_int("COLLECT_EPISODES", 32),
                eval_every=_env_int("EVAL_EVERY", 0),
                eval_episodes=256,
                eval_max_steps=25,
                checkpoint_every=_env_int("CHECKPOINT_EVERY", 5),
            ),
        ),
    )


def _validate_dataset(dataset: TokenizedEpisodeDataset) -> None:
    observed: set[int] = set()
    for episode in dataset.episodes:
        observed.update(
            int(value)
            for value in episode.actions[: int(episode.episode_length)].tolist()
        )
    invalid = sorted(value for value in observed if value < 0 or value >= 4)
    if invalid:
        raise ValueError(
            "Le-WM requires canonical Sokoban action ids 0..3; "
            f"observed={sorted(observed)}, invalid={invalid}"
        )
    print(
        "Offline action encoding: "
        f"WM ids={sorted(observed)} -> policy ids={sorted(observed)} (offset=0)"
    )


def _load_world_model(
    world_model: WorldModel, checkpoint_path: Path, config: Config,
) -> None:
    print(f"Loading WM checkpoint: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path, map_location=next(world_model.parameters()).device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Le-WM requires a metadata-bearing Phase-1 WM checkpoint")
    if "world_model" in payload and "policy" in payload:
        raise RuntimeError("Le-WM WM_CKPT must be Phase-1, not a Phase-2 checkpoint")
    state = payload.get("model", payload)
    missing, unexpected = world_model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys: {list(missing)[:20]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys: {list(unexpected)[:20]}")
    validate_world_model_semantics(
        payload,
        attention_mode=config.backbone.attention_mode,
        context=f"Le-WM Phase-1 checkpoint {checkpoint_path}",
    )
    injection = payload.get("reward_head_injection")
    if not payload.get("injected_reward_head") or not isinstance(injection, dict):
        raise RuntimeError("Le-WM requires an explicitly injected compact Reward Head")
    expected_hidden = config.reward.head_hidden_dim
    if injection.get("head_hidden_dim") != expected_hidden:
        raise RuntimeError(
            "Le-WM Reward Head architecture mismatch: "
            f"checkpoint={injection.get('head_hidden_dim')}, expected={expected_hidden}"
        )
    expected_keys = {
        f"reward_head.{key}" for key in world_model.reward_head.state_dict()
    }
    missing_reward = sorted(expected_keys.difference(state))
    if missing_reward:
        raise RuntimeError(f"Le-WM WM checkpoint lacks Reward tensors: {missing_reward}")
    required_horizons = set(range(1, config.phase2.ppo.rollout_horizon + 1))
    trained_horizons = {int(value) for value in injection.get("horizons", [])}
    if not required_horizons.issubset(trained_horizons):
        raise RuntimeError(
            "Le-WM Reward Head horizon coverage mismatch: "
            f"required={sorted(required_horizons)}, trained={sorted(trained_horizons)}"
        )
    if not injection.get("independent_horizon_starts", False):
        raise RuntimeError("Le-WM requires independent_horizon_starts=true")
    threshold = injection.get("decision_threshold")
    if threshold is None:
        raise RuntimeError("Le-WM Reward checkpoint has no decision_threshold")
    config.phase2.ppo.reward_confidence_floor = float(threshold)
    world_model._reward_head_injection = dict(injection)
    print(
        "  Verified compact Reward Head: "
        f"hidden_dim={expected_hidden}, horizons={sorted(trained_horizons)}, "
        f"confidence_floor={float(threshold):.6f}",
        flush=True,
    )


def _load_levels(path: Path) -> list[dict]:
    with path.open() as handle:
        payload = json.load(handle)
    levels = list(payload["levels"])
    limit = _env_int("EVAL_LEVEL_LIMIT", 0)
    return levels[:limit] if limit > 0 else levels


def main() -> None:
    stage = LeWMStage(os.environ.get("LEWM_STAGE", "real_critic_probe"))
    if os.environ.get("CRITIC_SOURCE", "latent_ordered_v") != "latent_ordered_v":
        raise RuntimeError("released Le-WM entrypoint requires CRITIC_SOURCE=latent_ordered_v")
    if _env_int("OFFLINE_BC_STEPS", 0) != 0:
        raise RuntimeError("Le-WM entrypoint forbids Offline BC")
    if _env_float("ONLINE_RATIO", 0.0) != 0.0:
        raise RuntimeError("Le-WM entrypoint forbids online replay mixing")

    seed = _env_int("SEED", 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(os.environ.get("LEWM_DEVICE", "cuda:0"))
    config = _config()

    dataset_path = Path(os.environ.get(
        "DATA_DIR",
        str(_REPOSITORY / "data/sokoban_10k_tokenized_3B_canonical/tokenized"),
    ))
    wm_checkpoint = Path(os.environ["WM_CKPT"])
    resume = Path(os.environ["RESUME"])
    output = Path(os.environ["OUTPUT_DIR"])
    tokenizer_weights = Path(os.environ["IMAGE_TOKENIZER_WEIGHTS"])
    levels_path = Path(os.environ.get(
        "EVAL_LEVELS_FILE",
        str(_REPOSITORY / "data/vagen_mirror_testset/sokoban_256.json"),
    ))
    for path in (dataset_path, wm_checkpoint, resume, tokenizer_weights, levels_path):
        if not path.exists():
            raise FileNotFoundError(f"Le-WM required input does not exist: {path}")

    dataset = TokenizedEpisodeDataset.from_directory(str(dataset_path))
    _validate_dataset(dataset)
    data_source = OfflineDataSource(dataset, config)
    print(f"Dataset: {len(dataset)} episodes")

    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    world_model = WorldModel(config, backbone).to(device)
    _load_world_model(world_model, wm_checkpoint, config)

    print("\n=== Assembling isolated Le-WM Pipeline ===")
    print("  World Model: frozen")
    print(
        "  Actor: qwen_slotwise raw ordered features; "
        + (
            "guarded Actor-only PPO"
            if stage is LeWMStage.ACTOR_PPO else
            "frozen by orchestration"
        )
    )
    print(
        "  Critic: latent_ordered_v, "
        f"lr={config.phase2.ppo.critic_lr}, "
        f"updates/collection={os.environ.get('LEWM_REAL_CRITIC_UPDATES', '10')}"
    )
    print(f"  Stage: {stage.value}")

    policy_config = LLMPolicyConfig(
        hidden_dim=config.hidden_dim,
        num_slots=config.belief.num_slots,
        actor_source="qwen_slotwise",
        actor_hidden_dim=256,
        actor_hidden_layers=1,
        actor_slot_dim=64,
        slotwise_actor_features="raw",
        slotwise_behavior_scale=0.0,
        critic_source="latent_ordered_v",
        critic_slot_dim=32,
    )
    pipeline, policy = assemble_llm_pipeline(
        config=config,
        world_model=world_model,
        action_adapter=SokobanActionAdapter(hidden_dim=config.hidden_dim),
        data_source=data_source,
        device=device,
        shared_backbone=False,
        logger=LeWMPrintLogger(),
        llm_policy_config=policy_config,
        pipeline_class=LeWMOrchestrator,
    )

    print("  Loading locked V-JEPA tokenizer...")
    tokenizer = ImageTokenizer(config, device=str(device))
    tokenizer.load_weights(str(tokenizer_weights))
    tokenizer.provenance = {
        "path": str(tokenizer_weights.resolve()),
        "sha256": hashlib.sha256(tokenizer_weights.read_bytes()).hexdigest(),
        "size": tokenizer_weights.stat().st_size,
    }
    levels = _load_levels(levels_path)
    adapter = SokobanAdapter(require_real=True)
    adapter.make_env(seed=0)
    collector = RealCollector(
        env_factory=lambda local_seed: adapter.make_env(seed=local_seed),
        tokenizer=tokenizer,
        world_model=world_model,
        policy=policy,
        config=RealCollectorConfig(
            max_steps=25,
            deterministic=False,
            exploration_epsilon=0.0,
            capture_policy_trajectory=False,
        ),
        env_id_tensor=torch.tensor([0], device=device, dtype=torch.long),
        env_id="sokoban",
        replay_buffer=None,
        device=device,
        wm_action_id_offset=0,
    )
    evaluator = RealEnvEvaluator(collector, eval_levels=levels)
    pipeline.real_collector = collector
    pipeline.evaluator = evaluator
    if pipeline.policy is not policy or collector.policy is not policy:
        raise RuntimeError("Le-WM Actor/collector identity contract failed")
    print(
        f"  Fixed evaluation layouts: {len(levels)} from {levels_path}",
        flush=True,
    )

    start_update = pipeline.load_checkpoint(resume)
    print(f"Resumed Le-WM source update: {start_update}")
    action_grounding_output = os.environ.get("LEWM_ACTION_GROUNDING_AUDIT_OUTPUT")
    if action_grounding_output:
        from .action_grounding_audit import run_action_grounding_audit

        run_action_grounding_audit(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=action_grounding_output,
            level_count=_env_int("LEWM_ACTION_GROUNDING_LEVELS", 32),
            states_per_episode=_env_int("LEWM_ACTION_GROUNDING_STATES_PER_EPISODE", 3),
            real_repeats=_env_int("LEWM_ACTION_GROUNDING_REAL_REPEATS", 8),
            imagined_repeats=_env_int("LEWM_ACTION_GROUNDING_IMAGINED_REPEATS", 16),
            real_batch_size=_env_int("LEWM_ACTION_GROUNDING_REAL_BATCH_SIZE", 8),
            imagined_batch_size=_env_int("LEWM_ACTION_GROUNDING_IMAGINED_BATCH_SIZE", 8),
            bootstrap_repeats=_env_int("LEWM_ACTION_GROUNDING_BOOTSTRAPS", 2000),
            seed=_env_int("LEWM_ACTION_GROUNDING_SEED", 20261218),
            reward_scale=_env_float("LEWM_REAL_RETURN_REWARD_SCALE", 0.1),
            tie_epsilon=_env_float("LEWM_ACTION_GROUNDING_TIE_EPS", 0.01),
            advantage_margin=_env_float("LEWM_ACTION_GROUNDING_ADV_MARGIN", 0.01),
            progress_every=_env_int("LEWM_ACTION_GROUNDING_PROGRESS_EVERY", 4),
        )
        return
    intermediate_output = os.environ.get("LEWM_INTERMEDIATE_RETURN_AUDIT_OUTPUT")
    if intermediate_output:
        from .intermediate_return_audit import run_intermediate_return_audit

        run_intermediate_return_audit(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=intermediate_output,
            level_count=_env_int("LEWM_INTERMEDIATE_LEVELS", 32),
            states_per_episode=_env_int("LEWM_INTERMEDIATE_STATES_PER_EPISODE", 3),
            continuation_repeats=_env_int("LEWM_INTERMEDIATE_CONTINUATION_REPEATS", 12),
            bootstrap_repeats=_env_int("LEWM_INTERMEDIATE_BOOTSTRAPS", 2000),
            seed=_env_int("LEWM_INTERMEDIATE_SEED", 20261118),
            reward_scale=_env_float("LEWM_REAL_RETURN_REWARD_SCALE", 0.1),
            critic_batch_size=_env_int("LEWM_INTERMEDIATE_CRITIC_BATCH_SIZE", 64),
            continuation_batch_size=_env_int(
                "LEWM_INTERMEDIATE_CONTINUATION_BATCH_SIZE", 8
            ),
            progress_every=_env_int("LEWM_INTERMEDIATE_PROGRESS_EVERY", 8),
        )
        return
    grounding_output = os.environ.get("LEWM_GROUNDING_AUDIT_OUTPUT")
    if grounding_output:
        from .grounding_audit import run_grounding_audit

        run_grounding_audit(
            pipeline=pipeline,
            evaluator=evaluator,
            output_path=grounding_output,
            level_count=_env_int("LEWM_GROUNDING_LEVELS", 64),
            rollout_repeats=_env_int("LEWM_GROUNDING_ROLLOUT_REPEATS", 8),
            bootstrap_repeats=_env_int("LEWM_GROUNDING_BOOTSTRAPS", 2000),
            seed=_env_int("LEWM_GROUNDING_SEED", 20261018),
            reward_scale=_env_float("LEWM_REAL_RETURN_REWARD_SCALE", 0.1),
            batch_size=_env_int("LEWM_GROUNDING_BATCH_SIZE", 64),
        )
        return
    started = time.time()
    pipeline.train(checkpoint_dir=output, start_update=start_update)
    print(f"\n=== Le-WM complete in {time.time() - started:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
