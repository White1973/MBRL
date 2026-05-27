"""Assembly functions that wire concrete components into the MBRL pipeline.

This module bridges the generic pipeline (mbrl_train.py) with the specific
SemBelief-WM components (WorldModel, LatentActorCritic, etc.). It is the
ONLY place where concrete model classes are imported alongside RL modules.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from ..data.datasource import OfflineDataSource
from ..data.replay import OnlineReplayBuffer, MixedDataSampler
from ..model.world_model import WorldModel
from ..types import BeliefState, SequenceBatch

from ..rl.ppo import PPOUpdater, PPOConfig
from ..rl.llm_policy import LLMActorCritic, LLMPolicyConfig
from ..rl.action_adapter import ActionAdapter
from ..collectors.imagined import ImaginedCollector, ImaginedCollectorConfig
from ..collectors.belief_sampler import sample_start_beliefs, PosteriorGrounder
from ..collectors.reward_transforms import make_reward_transform
from ..collectors.adapters import (
    LatentActorCriticAdapter,
    make_dynamics_step,
    make_predict_reward,
    make_get_belief_slots,
)
from ..model.policy_backbone import QwenPolicyBackbone
from .mbrl_train import MBRLPipeline, PipelineConfig, Logger


def make_dynamics_step_fn(
    world_model: WorldModel,
    config: Config,
) -> Any:
    """Create dynamics step callable for ImaginedCollector."""
    return make_dynamics_step(world_model)


def make_reward_fns(
    world_model: WorldModel,
    config: Config,
) -> tuple[Any, Any]:
    """Create predict_reward and reward_transform callables for ImaginedCollector.

    Returns:
        (predict_reward, reward_transform) — two separate callables.
        predict_reward: (belief) -> reward_logits (raw)
        reward_transform: (reward_logits, **kwargs) -> scalar reward
    """
    predict_reward = make_predict_reward(world_model)

    # Get reward spec for the default env
    spec = config.env.reward_spec_for(config.env.env_ids[0])
    logit_to_scalar = make_reward_transform(
        mapping=config.phase2.ppo.reward_mapping,
        positive_value=spec.positive_value,
        negative_value=spec.negative_value,
    )

    def reward_transform(reward_logits: Tensor, **kwargs: Any) -> Tensor:
        return logit_to_scalar(reward_logits)

    return predict_reward, reward_transform


def make_posterior_grounder(
    world_model: WorldModel,
    config: Config,
    device: torch.device,
) -> PosteriorGrounder:
    """Create a PosteriorGrounder matching the belief_sampler.PosteriorGrounder protocol.

    Returns objects with .slots attribute (BeliefState), accepts keyword args.
    """
    param_dtype = next(world_model.parameters()).dtype

    class _Grounder:
        def get_initial_belief(
            self,
            batch_size: int,
            *,
            device: torch.device = device,
            dtype: torch.dtype = param_dtype,
        ) -> BeliefState:
            return world_model.get_initial_belief(
                batch_size, device=device, dtype=dtype
            )

        def posterior_step(
            self,
            *,
            prev_belief: BeliefState,
            prev_actions: Tensor,
            observation_tokens: Tensor,
            env_ids: Tensor | None,
        ) -> BeliefState:
            return world_model.posterior_step(
                prev_belief=prev_belief,
                prev_actions=prev_actions,
                observation_tokens=observation_tokens.to(device, param_dtype),
                env_ids=env_ids,
            )

    return _Grounder()


def assemble_pipeline(
    *,
    config: Config,
    world_model: WorldModel,
    actor_critic: nn.Module,
    data_source: OfflineDataSource,
    device: torch.device,
    logger: Logger | None = None,
    evaluator: Any | None = None,
    real_collector: Any | None = None,
    wm_refresher: Any | None = None,
    online_buffer: OnlineReplayBuffer | None = None,
    online_ratio: float = 0.0,
    pipeline_config: PipelineConfig | None = None,
) -> MBRLPipeline:
    """Wire all components into a ready-to-train MBRLPipeline.

    This is the main entry point for assembling a model-based RL experiment.

    Args:
        online_buffer: If provided, belief sampling and WM refresh will mix
                       online replay data with offline data at the given ratio.
        online_ratio: Fraction of online data in mixed sampling (0.0 = offline only).
    """
    ppo_cfg = config.phase2.ppo

    # --- Build pipeline config from legacy config if not provided ---
    if pipeline_config is None:
        pipeline_config = PipelineConfig(
            total_updates=ppo_cfg.total_updates,
            eval_every=ppo_cfg.eval_every,
            eval_episodes=ppo_cfg.eval_episodes,
            checkpoint_every=ppo_cfg.checkpoint_every,
            rollout_batch_size=ppo_cfg.rollout_batch_size,
            rollout_horizon=ppo_cfg.rollout_horizon,
            ppo=PPOConfig(
                lr=ppo_cfg.actor_lr,
                clip_epsilon=ppo_cfg.clip_epsilon,
                value_coef=ppo_cfg.value_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                max_grad_norm=ppo_cfg.max_grad_norm,
                epochs=ppo_cfg.epochs_per_update,
                minibatch_size=ppo_cfg.minibatch_size,
                normalize_advantages=ppo_cfg.normalize_advantages,
            ),
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            wm_refresh_every=(
                config.phase2.wm_refresh.refresh_every
                if config.phase2.world_model_mode == "alternating_wm"
                else 0
            ),
            wm_refresh_steps=config.phase2.wm_refresh.updates_per_refresh,
            collect_every=ppo_cfg.collect_every,
            collect_episodes=ppo_cfg.collect_episodes,
        )

    # --- Freeze world model for imagined rollout ---
    world_model.eval()
    world_model.requires_grad_(False)

    # --- Build imagined collector ---
    adapted_policy = LatentActorCriticAdapter(actor_critic)
    dynamics_step = make_dynamics_step_fn(world_model, config)
    predict_reward, reward_transform = make_reward_fns(world_model, config)
    get_belief_slots = make_get_belief_slots()

    imagined_collector = ImaginedCollector(
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        policy=adapted_policy,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(
            horizon=pipeline_config.rollout_horizon,
            batch_size=pipeline_config.rollout_batch_size,
            bootstrap_with_value=pipeline_config.use_value_bootstrap,
        ),
    )

    # --- Build mixed data sampler (for beliefs + WM refresh) ---
    mixed_sampler = MixedDataSampler(
        offline_source=data_source,
        online_buffer=online_buffer,
        online_ratio=online_ratio,
    )

    # --- Build belief sampler (uses mixed data) ---
    param_dtype = next(world_model.parameters()).dtype
    grounder = make_posterior_grounder(world_model, config, device)

    def sample_beliefs_fn(batch_size: int) -> BeliefState:
        batch = mixed_sampler.sample(batch_size, config)
        batch = _to_device(batch, device)
        slots = sample_start_beliefs(
            batch, grounder, device=device, dtype=param_dtype
        )
        return BeliefState(slots=slots)

    # --- Build WM refresh sample function ---
    wm_refresh_sample_fn = None
    if wm_refresher is not None:
        def wm_refresh_sample_fn(batch_size: int) -> SequenceBatch:
            batch = mixed_sampler.sample(batch_size, config)
            return _to_device(batch, device)

    # --- Build PPO updater ---
    ppo_updater = PPOUpdater(
        policy=adapted_policy,
        config=pipeline_config.ppo,
    )

    # --- Assemble ---
    return MBRLPipeline(
        config=pipeline_config,
        imagine_fn=imagined_collector.collect,
        sample_beliefs_fn=sample_beliefs_fn,
        ppo_updater=ppo_updater,
        policy=actor_critic,
        evaluator=evaluator,
        wm_refresher=wm_refresher,
        wm_refresh_sample_fn=wm_refresh_sample_fn,
        world_model=world_model,
        real_collector=real_collector,
        logger=logger,
    )


def assemble_llm_pipeline(
    *,
    config: Config,
    world_model: WorldModel,
    action_adapter: ActionAdapter,
    data_source: OfflineDataSource,
    device: torch.device,
    shared_backbone: bool = True,
    logger: Logger | None = None,
    evaluator: Any | None = None,
    real_collector: Any | None = None,
    wm_refresher: Any | None = None,
    online_buffer: OnlineReplayBuffer | None = None,
    online_ratio: float = 0.0,
    pipeline_config: PipelineConfig | None = None,
    llm_policy_config: LLMPolicyConfig | None = None,
) -> tuple[MBRLPipeline, LLMActorCritic]:
    """Wire components into a pipeline using LLM policy (Qwen + LoRA + ActionAdapter).

    This is the main assembly for the LLM-as-policy architecture. The backbone
    is shared with the world model by default — policy gradients flow through
    the same LoRA adapters.

    Args:
        action_adapter: Environment-specific action head + text parser.
        shared_backbone: If True (default), policy reuses the WM's Qwen backbone.
                         If False, creates an independent Qwen + LoRA.

    Returns:
        (pipeline, llm_policy) — the pipeline and the LLM policy for inspection.
    """
    ppo_cfg = config.phase2.ppo

    # --- Build pipeline config ---
    if pipeline_config is None:
        pipeline_config = PipelineConfig(
            total_updates=ppo_cfg.total_updates,
            eval_every=ppo_cfg.eval_every,
            eval_episodes=ppo_cfg.eval_episodes,
            checkpoint_every=ppo_cfg.checkpoint_every,
            rollout_batch_size=ppo_cfg.rollout_batch_size,
            rollout_horizon=ppo_cfg.rollout_horizon,
            ppo=PPOConfig(
                lr=ppo_cfg.actor_lr,
                clip_epsilon=ppo_cfg.clip_epsilon,
                value_coef=ppo_cfg.value_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                max_grad_norm=ppo_cfg.max_grad_norm,
                epochs=ppo_cfg.epochs_per_update,
                minibatch_size=ppo_cfg.minibatch_size,
                normalize_advantages=ppo_cfg.normalize_advantages,
            ),
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            wm_refresh_every=(
                config.phase2.wm_refresh.refresh_every
                if config.phase2.world_model_mode == "alternating_wm"
                else 0
            ),
            wm_refresh_steps=config.phase2.wm_refresh.updates_per_refresh,
            collect_every=ppo_cfg.collect_every,
            collect_episodes=ppo_cfg.collect_episodes,
        )

    # --- Freeze world model FIRST, before building backbone ---
    # This must happen before from_shared() so we can selectively
    # re-enable gradients on shared LoRA params afterward.
    world_model.eval()
    world_model.requires_grad_(False)

    # --- Build backbone ---
    if shared_backbone:
        backbone = QwenPolicyBackbone.from_shared(world_model.transition.backbone)
        # Re-enable gradients on the shared LoRA params that PPO will update.
        # world_model.requires_grad_(False) froze them, but policy needs them trainable.
        for p in backbone.trainable_parameters():
            p.requires_grad_(True)
    else:
        backbone = QwenPolicyBackbone.from_config(config)

    # --- Build LLM policy ---
    llm_cfg = llm_policy_config or LLMPolicyConfig(
        hidden_dim=config.hidden_dim,
        num_slots=config.belief.num_slots,
    )
    llm_policy = LLMActorCritic(
        backbone=backbone,
        action_adapter=action_adapter,
        config=llm_cfg,
    )
    llm_policy.to(device)

    # --- Build imagined collector ---
    # LLMActorCritic directly satisfies Policy protocol — no adapter needed.
    dynamics_step = make_dynamics_step_fn(world_model, config)
    predict_reward, reward_transform = make_reward_fns(world_model, config)
    get_belief_slots = make_get_belief_slots()

    imagined_collector = ImaginedCollector(
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        policy=llm_policy,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(
            horizon=pipeline_config.rollout_horizon,
            batch_size=pipeline_config.rollout_batch_size,
            bootstrap_with_value=pipeline_config.use_value_bootstrap,
        ),
    )

    # --- Build mixed data sampler ---
    mixed_sampler = MixedDataSampler(
        offline_source=data_source,
        online_buffer=online_buffer,
        online_ratio=online_ratio,
    )

    param_dtype = next(world_model.parameters()).dtype
    grounder = make_posterior_grounder(world_model, config, device)

    def sample_beliefs_fn(batch_size: int) -> BeliefState:
        batch = mixed_sampler.sample(batch_size, config)
        batch = _to_device(batch, device)
        slots = sample_start_beliefs(
            batch, grounder, device=device, dtype=param_dtype
        )
        return BeliefState(slots=slots)

    # --- Build WM refresh sample function ---
    wm_refresh_sample_fn = None
    if wm_refresher is not None:
        def wm_refresh_sample_fn(batch_size: int) -> SequenceBatch:
            batch = mixed_sampler.sample(batch_size, config)
            return _to_device(batch, device)

    # --- Build PPO updater (with shared backbone params if needed) ---
    extra_params = backbone.trainable_parameters() if shared_backbone else None
    ppo_updater = PPOUpdater(
        config=pipeline_config.ppo,
        policy=llm_policy,
        extra_params=extra_params,
    )

    # --- Assemble ---
    pipeline = MBRLPipeline(
        config=pipeline_config,
        imagine_fn=imagined_collector.collect,
        sample_beliefs_fn=sample_beliefs_fn,
        ppo_updater=ppo_updater,
        policy=llm_policy,
        evaluator=evaluator,
        wm_refresher=wm_refresher,
        wm_refresh_sample_fn=wm_refresh_sample_fn,
        world_model=world_model,
        real_collector=real_collector,
        logger=logger,
    )

    return pipeline, llm_policy


def _to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        obs_tokens=batch.obs_tokens.to(device),
        actions=batch.actions.to(device),
        rewards=batch.rewards.to(device),
        episode_lengths=batch.episode_lengths.to(device),
        env_ids=None if batch.env_ids is None else batch.env_ids.to(device),
    )
