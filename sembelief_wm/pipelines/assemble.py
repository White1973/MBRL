"""Assembly functions that wire concrete components into the MBRL pipeline.

This module bridges the generic pipeline (mbrl_train.py) with the specific
SemBelief-WM components (WorldModel, LatentActorCritic, etc.). It is the
ONLY place where concrete model classes are imported alongside RL modules.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from ..data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from ..data.replay import (
    MixedDataSampler,
    OnlineReplayBuffer,
    UnifiedRandomReplayPool,
)
from ..model.world_model import WorldModel
from ..types import BeliefState, SequenceBatch

from ..rl.ppo import PPOUpdater, PPOConfig
from ..rl.offline_bc import OfflineBCConfig, OfflineBehaviorCloner
from ..rl.llm_policy import LLMActorCritic, LLMPolicyConfig
from ..rl.action_adapter import ActionAdapter
from ..collectors.imagined import ImaginedCollector, ImaginedCollectorConfig
from ..collectors.belief_sampler import (
    sample_start_beliefs,
    sample_start_beliefs_and_actions,
    PosteriorGrounder,
)
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
    return make_dynamics_step(
        world_model,
        action_id_offset=config.phase2.ppo.wm_action_id_offset,
    )


def _assert_optimizer_does_not_own_world_model(
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    policy_adapter_names: frozenset[str] = frozenset(),
) -> None:
    """Fail fast if any WM-owned parameter leaked into the PPO optimizer.

    Phase-2 ownership is intentionally strict: supervised WM refresh and PPO
    must never keep separate Adam states for the same Parameter object. Actor
    and critic adapters live inside the shared PEFT container and are excluded
    explicitly; the WM default adapter, Qwen base, and WM heads may not overlap.
    """
    def owned_by_policy_adapter(name: str) -> bool:
        lowered = name.lower()
        return "lora_" in lowered and any(
            f".{adapter_name}." in name
            for adapter_name in policy_adapter_names
        )

    wm_parameters = {
        p.data_ptr(): name
        for name, p in world_model.named_parameters()
        if not owned_by_policy_adapter(name)
    }
    overlap = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = wm_parameters.get(parameter.data_ptr())
            if name is not None:
                overlap.append(name)
    if overlap:
        sample = ", ".join(overlap[:8])
        raise RuntimeError(
            "PPO optimizer owns world-model parameters: "
            f"{sample}. WM refresh and PPO must optimize disjoint Parameter "
            "objects; use an independent policy backbone for trainable Qwen "
            "branches."
        )


def _assert_lora_groups_disjoint(
    first: QwenPolicyBackbone,
    second: QwenPolicyBackbone,
    *,
    first_name: str,
    second_name: str,
) -> None:
    """Verify independently optimized actor/critic LoRA objects do not alias."""
    first_ptrs = {parameter.data_ptr() for parameter in first.trainable_parameters()}
    second_ptrs = {
        parameter.data_ptr() for parameter in second.trainable_parameters()
    }
    overlap = first_ptrs & second_ptrs
    if overlap:
        raise RuntimeError(
            f"{first_name} and {second_name} share {len(overlap)} trainable "
            "LoRA tensors. Trainable actor and critic Qwen branches must own "
            "independent LoRA Parameter objects."
        )


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
        scale=config.phase2.ppo.reward_scale,
        confidence_floor=config.phase2.ppo.reward_confidence_floor,
        low_confidence_scale=config.phase2.ppo.reward_low_confidence_scale,
    )
    if (
        config.phase2.ppo.reward_mapping
        == "per_transition_success_conservative"
    ):
        floor = config.phase2.ppo.reward_confidence_floor
        epsilon = min(1e-4, floor / 2.0, (1.0 - floor) / 2.0)
        probabilities = torch.tensor([floor - epsilon, floor + epsilon])
        probe_rewards = logit_to_scalar(torch.logit(probabilities))
        if not (probe_rewards[0] < 0 < probe_rewards[1]):
            raise RuntimeError(
                "Per-transition reward/termination invariant failed: a "
                "state below the success threshold must receive a negative "
                "step reward and a state above it must receive a positive "
                "terminal reward."
            )

    def reward_transform(reward_logits: Tensor, **kwargs: Any) -> Tensor:
        return logit_to_scalar(reward_logits, **kwargs)

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
            eval_at_start=(
                ppo_cfg.eval_every > 0
                and os.environ.get("SKIP_BASELINE_EVAL", "0") != "1"
            ),
            checkpoint_every=ppo_cfg.checkpoint_every,
            rollout_batch_size=ppo_cfg.rollout_batch_size,
            rollout_horizon=ppo_cfg.rollout_horizon,
            use_value_bootstrap=ppo_cfg.use_value_bootstrap,
            rollouts_per_update=ppo_cfg.rollouts_per_update,
            critic_warmup_min_updates=ppo_cfg.critic_warmup_min_updates,
            critic_warmup_ev_threshold=ppo_cfg.critic_warmup_ev_threshold,
            critic_warmup_ev_patience=ppo_cfg.critic_warmup_ev_patience,
            critic_warmup_validation_fraction=(
                ppo_cfg.critic_warmup_validation_fraction
            ),
            critic_warmup_validation_size=ppo_cfg.critic_warmup_validation_size,
            critic_warmup_replay_capacity=ppo_cfg.critic_warmup_replay_capacity,
            critic_warmup_train_samples=ppo_cfg.critic_warmup_train_samples,
            critic_warmup_ev_ema_alpha=ppo_cfg.critic_warmup_ev_ema_alpha,
            critic_warmup_mse_improvement=ppo_cfg.critic_warmup_mse_improvement,
            ppo=PPOConfig(
                lr=ppo_cfg.actor_lr,
                critic_lr=ppo_cfg.critic_lr,
                clip_epsilon=ppo_cfg.clip_epsilon,
                value_coef=ppo_cfg.value_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                kl_coef=ppo_cfg.kl_coef,
                target_kl=ppo_cfg.target_kl,
                behavior_kl_coef=ppo_cfg.behavior_kl_coef,
                behavior_bc_coef=ppo_cfg.behavior_bc_coef,
                behavior_bc_batch_size=ppo_cfg.behavior_bc_batch_size,
                max_grad_norm=ppo_cfg.max_grad_norm,
                epochs=ppo_cfg.epochs_per_update,
                minibatch_size=ppo_cfg.minibatch_size,
                recompute_old_log_probs=ppo_cfg.recompute_old_log_probs,
                normalize_advantages=ppo_cfg.normalize_advantages,
                target_entropy=ppo_cfg.target_entropy,
                entropy_floor_coef=ppo_cfg.entropy_floor_coef,
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
            reward_success_threshold=(
                ppo_cfg.reward_confidence_floor
                if ppo_cfg.reward_mapping in {
                    "terminal_success_conservative",
                    "per_transition_success_conservative",
                }
                else 0.5
            ),
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
            success_threshold=pipeline_config.reward_success_threshold,
            termination_mode=(
                config.phase2.ppo.imagination_termination_mode
            ),
        ),
    )

    # --- Build mixed data sampler (for beliefs + WM refresh) ---
    if os.environ.get("UNIFIED_RANDOM_REPLAY", "0") == "1":
        if online_buffer is None:
            raise RuntimeError("UNIFIED_RANDOM_REPLAY requires online collection")
        mixed_sampler = UnifiedRandomReplayPool(
            data_source,
            online_buffer,
            offline_episodes=int(os.environ.get(
                "UNIFIED_REPLAY_OFFLINE_EPISODES", "800"
            )),
            online_target=int(os.environ.get(
                "UNIFIED_REPLAY_ONLINE_TARGET", "800"
            )),
            seed=int(os.environ.get("UNIFIED_REPLAY_SEED", "20260901")),
        )
    else:
        mixed_sampler = MixedDataSampler(
            offline_source=data_source,
            online_buffer=online_buffer,
            online_ratio=online_ratio,
        )

    # --- Build belief sampler (uses mixed data) ---
    param_dtype = next(world_model.parameters()).dtype
    grounder = make_posterior_grounder(world_model, config, device)

    def sample_beliefs_fn(batch_size: int) -> BeliefState:
        slots = _sample_horizon_eligible_beliefs(
            batch_size=batch_size,
            rollout_horizon=pipeline_config.rollout_horizon,
            null_action_id=config.env.null_action_id,
            mixed_sampler=mixed_sampler,
            config=config,
            grounder=grounder,
            device=device,
            dtype=param_dtype,
        )
        return BeliefState(slots=slots)

    # --- Build WM refresh sample function ---
    wm_refresh_sample_fn = None
    if wm_refresher is not None:
        def wm_refresh_sample_fn(batch_size: int) -> SequenceBatch:
            batch = mixed_sampler.sample(batch_size, config)
            return _to_device(batch, device)
        wm_refresh_sample_fn.mixed_sampler = mixed_sampler

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
    shared_backbone: bool = False,
    logger: Logger | None = None,
    evaluator: Any | None = None,
    real_collector: Any | None = None,
    wm_refresher: Any | None = None,
    online_buffer: OnlineReplayBuffer | None = None,
    online_ratio: float = 0.0,
    pipeline_config: PipelineConfig | None = None,
    llm_policy_config: LLMPolicyConfig | None = None,
    pipeline_class: type | None = None,
) -> tuple[Any, LLMActorCritic]:
    """Wire components into a pipeline using LLM policy (Qwen + LoRA + ActionAdapter).

    This is the main assembly for the LLM-as-policy architecture. One frozen
    Qwen base is shared by WM/actor/critic, while their LoRA adapters remain
    independent by default.

    Args:
        action_adapter: Environment-specific action head + text parser.
        shared_backbone: Compatibility mode that also reuses the WM's default
                         LoRA and is therefore read-only. If False (default),
                         shares only the frozen Qwen base and creates separate
                         policy LoRA adapters initialized from the WM LoRA.

    Returns:
        (pipeline, llm_policy) — the pipeline and the LLM policy for inspection.
    """
    ppo_cfg = config.phase2.ppo
    llm_cfg = llm_policy_config or LLMPolicyConfig(
        hidden_dim=config.hidden_dim,
        num_slots=config.belief.num_slots,
    )

    # Trainable qwen_pooled branches must never share LoRA Parameter objects
    # with the world model.  This applies to alternating_wm as well as frozen_wm:
    # otherwise PPO and Phase1Trainer keep independent Adam states for the same
    # LoRA tensors and alternately optimize incompatible objectives.
    actor_trains_qwen = (
        llm_cfg.actor_source == "qwen_pooled"
        or (
            llm_cfg.actor_source == "qwen_slotwise"
            and llm_cfg.slotwise_actor_features == "qwen"
        )
    )
    qwen_used_by_ppo = (
        actor_trains_qwen
        or llm_cfg.critic_source in {"qwen_pooled", "qwen_slotwise_q"}
    )
    vlm_used_by_policy = (
        llm_cfg.actor_source in {"qwen_pooled", "qwen_slotwise", "frozen_vlm"}
        or llm_cfg.critic_source in {
            "qwen_pooled", "qwen_slotwise_q", "frozen_vlm"
        }
    )
    shared_would_drift = (
        config.phase2.world_model_mode == "alternating_wm"
        and vlm_used_by_policy
    )
    if shared_backbone and (qwen_used_by_ppo or shared_would_drift):
        raise ValueError(
            "Invalid Phase 2 ownership: shared_backbone=True would couple the "
            "policy and world-model LoRA parameters. Trainable qwen_pooled "
            "would give PPO and WM overlapping ownership; alternating_wm would "
            "also move a frozen_vlm policy representation during WM refresh. "
            "Use --independent-backbone. Shared mode is supported only when "
            "the WM and all VLM policy branches remain read-only."
        )

    # --- Build pipeline config ---
    if pipeline_config is None:
        pipeline_config = PipelineConfig(
            total_updates=ppo_cfg.total_updates,
            eval_every=ppo_cfg.eval_every,
            eval_episodes=ppo_cfg.eval_episodes,
            eval_at_start=(
                ppo_cfg.eval_every > 0
                and os.environ.get("SKIP_BASELINE_EVAL", "0") != "1"
            ),
            checkpoint_every=ppo_cfg.checkpoint_every,
            rollout_batch_size=ppo_cfg.rollout_batch_size,
            rollout_horizon=ppo_cfg.rollout_horizon,
            use_value_bootstrap=ppo_cfg.use_value_bootstrap,
            rollouts_per_update=ppo_cfg.rollouts_per_update,
            critic_warmup_min_updates=ppo_cfg.critic_warmup_min_updates,
            critic_warmup_ev_threshold=ppo_cfg.critic_warmup_ev_threshold,
            critic_warmup_ev_patience=ppo_cfg.critic_warmup_ev_patience,
            critic_warmup_validation_fraction=(
                ppo_cfg.critic_warmup_validation_fraction
            ),
            critic_warmup_validation_size=ppo_cfg.critic_warmup_validation_size,
            critic_warmup_replay_capacity=ppo_cfg.critic_warmup_replay_capacity,
            critic_warmup_train_samples=ppo_cfg.critic_warmup_train_samples,
            critic_warmup_ev_ema_alpha=ppo_cfg.critic_warmup_ev_ema_alpha,
            critic_warmup_mse_improvement=ppo_cfg.critic_warmup_mse_improvement,
            ppo=PPOConfig(
                lr=ppo_cfg.actor_lr,
                critic_lr=ppo_cfg.critic_lr,
                clip_epsilon=ppo_cfg.clip_epsilon,
                value_coef=ppo_cfg.value_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                kl_coef=ppo_cfg.kl_coef,
                target_kl=ppo_cfg.target_kl,
                behavior_kl_coef=ppo_cfg.behavior_kl_coef,
                behavior_bc_coef=ppo_cfg.behavior_bc_coef,
                behavior_bc_batch_size=ppo_cfg.behavior_bc_batch_size,
                max_grad_norm=ppo_cfg.max_grad_norm,
                epochs=ppo_cfg.epochs_per_update,
                minibatch_size=ppo_cfg.minibatch_size,
                recompute_old_log_probs=ppo_cfg.recompute_old_log_probs,
                normalize_advantages=ppo_cfg.normalize_advantages,
                target_entropy=ppo_cfg.target_entropy,
                entropy_floor_coef=ppo_cfg.entropy_floor_coef,
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
            reward_success_threshold=(
                ppo_cfg.reward_confidence_floor
                if ppo_cfg.reward_mapping in {
                    "terminal_success_conservative",
                    "per_transition_success_conservative",
                }
                else 0.5
            ),
        )

    # --- Freeze world model FIRST, before building backbone ---
    # This must happen before adding policy adapters. Their LoRA groups are
    # selectively enabled below; the WM default adapter remains frozen with
    # respect to PPO.
    world_model.eval()
    world_model.requires_grad_(False)

    # --- Build policy adapter views over the single WM-owned Qwen base ---
    wm_backbone = getattr(world_model.transition, "backbone", None)
    if wm_backbone is None:
        wm_backbone = getattr(world_model, "backbone", None)
    if shared_backbone:
        # Real WorldModel stores it under transition.backbone; lightweight
        # protocol-compatible models may expose backbone directly.
        if wm_backbone is None:
            raise AttributeError(
                "shared_backbone=True requires world_model.transition.backbone "
                "or world_model.backbone."
            )
        backbone = QwenPolicyBackbone.from_shared(wm_backbone)
        # Shared means read-only in Phase 2. Any trainable qwen_pooled branch was
        # rejected above, so the WM LoRA remains frozen with respect to PPO.
    else:
        if wm_backbone is None:
            raise AttributeError(
                "Independent policy LoRA requires a world-model backbone."
            )
        add_adapter = getattr(wm_backbone, "add_lora_adapter", None)
        if not callable(add_adapter):
            raise TypeError(
                "Independent policy LoRA requires a multi-adapter Qwen "
                "backbone exposing add_lora_adapter()."
            )

        actor_uses_vlm = llm_cfg.actor_source in {
            "qwen_pooled",
            "qwen_slotwise",
            "frozen_vlm",
        }
        critic_uses_vlm = llm_cfg.critic_source in {
            "qwen_pooled", "qwen_slotwise_q",
            "frozen_vlm",
        }
        if actor_uses_vlm:
            add_adapter(
                "actor",
                initialize_from="default",
                lora_dropout=0.0,
            )
            backbone = QwenPolicyBackbone.from_shared_adapter(
                wm_backbone,
                adapter_name="actor",
                trainable=actor_trains_qwen,
            )
        elif critic_uses_vlm:
            add_adapter(
                "critic",
                initialize_from="default",
                lora_dropout=0.0,
            )
            backbone = QwenPolicyBackbone.from_shared_adapter(
                wm_backbone,
                adapter_name="critic",
                trainable=llm_cfg.critic_source in {"qwen_pooled", "qwen_slotwise_q"},
            )
        else:
            # The backbone is unused when both branches consume latent belief
            # slots directly. Keep a read-only default-adapter view without
            # creating unnecessary policy LoRA tensors.
            backbone = QwenPolicyBackbone.from_shared(wm_backbone)

    # --- If actor and critic both consume VLM features, give the critic its own
    #     backbone even when one branch is frozen_vlm. This prevents the frozen
    #     branch's feature distribution from moving when the other branch's
    #     LoRA is optimized.
    critic_backbone: QwenPolicyBackbone | None = None
    vlm_sources = {
        "qwen_pooled", "qwen_slotwise", "qwen_slotwise_q", "frozen_vlm"
    }
    decouple_vlm_branches = (
        not shared_backbone
        and llm_cfg.actor_source in vlm_sources
        and llm_cfg.critic_source in vlm_sources
    )
    if decouple_vlm_branches:
        if wm_backbone is None:
            raise AttributeError(
                "Decoupled VLM critic requires a world-model backbone."
            )
        assert wm_backbone is not None
        add_adapter = getattr(wm_backbone, "add_lora_adapter", None)
        if not callable(add_adapter):
            raise TypeError(
                "Decoupled VLM critic requires multi-adapter Qwen support."
            )
        add_adapter(
            "critic",
            initialize_from="default",
            lora_dropout=0.0,
        )
        critic_backbone = QwenPolicyBackbone.from_shared_adapter(
            wm_backbone,
            adapter_name="critic",
            trainable=llm_cfg.critic_source in {"qwen_pooled", "qwen_slotwise_q"},
        )
        _assert_lora_groups_disjoint(
            backbone,
            critic_backbone,
            first_name="actor",
            second_name="critic",
        )
        print(
            "  VLM actor/critic decoupled: critic has an independent "
            "LoRA adapter initialized from WM; actor, critic, and WM share one "
            "frozen Qwen base.",
            flush=True,
        )

    # --- Build LLM policy ---
    llm_policy = LLMActorCritic(
        backbone=backbone,
        action_adapter=action_adapter,
        config=llm_cfg,
        critic_backbone=critic_backbone,
    )
    llm_policy.to(device)
    # The shared Qwen is a non-owning reference inside QwenPolicyBackbone, so
    # ``llm_policy.eval()`` alone cannot recurse into it. Adapters are also
    # created after the initial ``world_model.eval()`` above and therefore
    # start in PyTorch's default training mode. Make the production rollout,
    # PPO re-evaluation, and real-env evaluation path deterministic without
    # changing any parameter's requires-grad ownership.
    llm_policy.set_deterministic_forward_mode()
    forward_mode = llm_policy.forward_mode_diagnostics()
    if forward_mode["active_dropout_modules"] != 0.0:
        raise RuntimeError(
            "Phase-2 policy forward remains stochastic after stabilization: "
            f"{forward_mode}. PPO old/current log-probabilities would not be "
            "comparable."
        )
    print(
        "  Policy forward stabilized: active dropout=0 "
        f"(audited modules={int(forward_mode['dropout_modules'])}); "
        "actor/critic gradients remain enabled.",
        flush=True,
    )

    offline_bc_enabled = ppo_cfg.offline_bc_steps > 0
    behavior_support_enabled = ppo_cfg.behavior_kl_coef > 0.0
    behavior_rehearsal_enabled = ppo_cfg.behavior_bc_coef > 0.0
    if behavior_support_enabled and not offline_bc_enabled:
        raise ValueError(
            "behavior_kl_coef > 0 requires offline_bc_steps > 0 so the "
            "reference policy represents offline behavior rather than the "
            "random actor initialization"
        )
    if behavior_rehearsal_enabled and not offline_bc_enabled:
        raise ValueError(
            "behavior_bc_coef > 0 requires offline_bc_steps > 0 so PPO "
            "rehearses a fixed expert posterior cache"
        )
    if behavior_support_enabled and config.phase2.world_model_mode != "frozen_wm":
        raise ValueError(
            "behavior KL snapshots are restricted to frozen_wm: after an "
            "alternating WM refresh the old-policy latent coordinate system "
            "is stale. Use direct expert rehearsal instead."
        )
    if behavior_support_enabled:
        # Register a frozen placeholder before PPOUpdater and checkpoint-load
        # construction. OfflineBehaviorCloner replaces it with the fitted
        # snapshot before update 0; resume loads the saved fitted snapshot.
        # Plain BC initialization needs no reference and therefore remains
        # valid for qwen_pooled actors with trainable LoRA representations.
        llm_policy.capture_behavior_reference()

    # --- Build imagined collector ---
    # LLMActorCritic directly satisfies Policy protocol — no adapter needed.
    dynamics_step = make_dynamics_step_fn(world_model, config)
    predict_reward, reward_transform = make_reward_fns(world_model, config)
    get_belief_slots = make_get_belief_slots()

    relative_action_value = None
    relative_scale = float(os.environ.get("LEWM_RELATIVE_H3_REWARD_SCALE", "0"))
    relative_margin_ref = float(
        os.environ.get("LEWM_RELATIVE_H3_MARGIN_REF", "0.10")
    )
    if relative_scale > 0.0:
        if relative_margin_ref <= 0.0:
            raise ValueError("LEWM_RELATIVE_H3_MARGIN_REF must be positive")
        if pipeline_config.rollout_horizon < 3:
            raise ValueError("Le-WM relative H3 reward requires rollout_horizon>=3")

        @torch.no_grad()
        def relative_action_value(belief, selected_action):
            slots = get_belief_slots(belief)
            batch = slots.shape[0]
            candidate = dynamics_step(
                BeliefState(slots=slots.repeat_interleave(4, dim=0)),
                torch.arange(4, device=slots.device).repeat(batch),
            )
            for _ in range(2):
                follow = llm_policy.actor_logits(
                    get_belief_slots(candidate)
                ).argmax(-1)
                candidate = dynamics_step(candidate, follow)
            scores = torch.sigmoid(predict_reward(candidate)).reshape(batch, 4)
            centered = scores - scores.mean(-1, keepdim=True)
            sorted_scores = scores.sort(dim=-1, descending=True).values
            top1_top2_margin = sorted_scores[:, 0] - sorted_scores[:, 1]
            confidence = (top1_top2_margin / relative_margin_ref).clamp(0.0, 1.0)
            selected_scores = scores.gather(
                1, selected_action.long()[:, None]
            ).squeeze(1)
            # Rank is one-based: 1 means the actor selected the H3 top action.
            selected_rank = 1 + (scores > selected_scores[:, None]).sum(-1)
            bonus = relative_scale * confidence * centered.gather(
                1, selected_action.long()[:, None]
            ).squeeze(1).clamp(-0.5, 0.5)
            return bonus, {
                "score_gap": scores.max(-1).values - scores.min(-1).values,
                "top1_top2_margin": top1_top2_margin,
                "selected_rank": selected_rank.to(scores.dtype),
                "selected_is_top1": (selected_rank == 1).to(scores.dtype),
            }

        print(
            "  Le-WM relative H3 action-value shaping enabled: "
            f"scale={relative_scale:g}, baseline=four-action mean, "
            f"margin_ref={relative_margin_ref:g}",
            flush=True,
        )

    imagined_collector = ImaginedCollector(
        dynamics_step=dynamics_step,
        predict_reward=predict_reward,
        reward_transform=reward_transform,
        relative_action_value=relative_action_value,
        policy=llm_policy,
        get_belief_slots=get_belief_slots,
        config=ImaginedCollectorConfig(
            horizon=pipeline_config.rollout_horizon,
            batch_size=pipeline_config.rollout_batch_size,
            bootstrap_with_value=pipeline_config.use_value_bootstrap,
            success_threshold=pipeline_config.reward_success_threshold,
            termination_mode=(
                config.phase2.ppo.imagination_termination_mode
            ),
        ),
    )

    # --- Build mixed data sampler ---
    if os.environ.get("UNIFIED_RANDOM_REPLAY", "0") == "1":
        if online_buffer is None:
            raise RuntimeError("UNIFIED_RANDOM_REPLAY requires online collection")
        mixed_sampler = UnifiedRandomReplayPool(
            data_source,
            online_buffer,
            offline_episodes=int(os.environ.get(
                "UNIFIED_REPLAY_OFFLINE_EPISODES", "800"
            )),
            online_target=int(os.environ.get(
                "UNIFIED_REPLAY_ONLINE_TARGET", "800"
            )),
            seed=int(os.environ.get("UNIFIED_REPLAY_SEED", "20260901")),
        )
    else:
        mixed_sampler = MixedDataSampler(
            offline_source=data_source,
            online_buffer=online_buffer,
            online_ratio=online_ratio,
        )

    param_dtype = next(world_model.parameters()).dtype
    grounder = make_posterior_grounder(world_model, config, device)

    def sample_beliefs_fn(batch_size: int) -> BeliefState:
        slots = _sample_horizon_eligible_beliefs(
            batch_size=batch_size,
            rollout_horizon=pipeline_config.rollout_horizon,
            null_action_id=config.env.null_action_id,
            mixed_sampler=mixed_sampler,
            config=config,
            grounder=grounder,
            device=device,
            dtype=param_dtype,
        )
        return BeliefState(slots=slots)

    bc_data_source = data_source
    bc_validation_data_source = None
    if ppo_cfg.offline_bc_strategies:
        allowed = set(ppo_cfg.offline_bc_strategies)
        episodes = [
            episode for episode in data_source.dataset.episodes
            if str(episode.metadata.get("strategy", "")) in allowed
        ]
        if not episodes:
            raise ValueError(
                f"offline BC strategy filter selected no episodes: {sorted(allowed)}"
            )
        bc_data_source = OfflineDataSource(
            TokenizedEpisodeDataset(episodes), data_source.config,
            env_id_to_index=data_source.env_id_to_index,
        )
        print(
            f"  Offline BC strategy filter: {sorted(allowed)} -> "
            f"{len(episodes)} episodes", flush=True,
        )
    if offline_bc_enabled:
        episodes = list(bc_data_source.dataset.episodes)
        if len(episodes) < 2:
            raise ValueError("episode-disjoint offline BC needs at least 2 episodes")
        generator = torch.Generator().manual_seed(20260720)
        order = torch.randperm(len(episodes), generator=generator).tolist()
        validation_count = max(1, len(episodes) // 5)
        validation_ids = set(order[:validation_count])
        train_episodes = [e for i, e in enumerate(episodes) if i not in validation_ids]
        validation_episodes = [e for i, e in enumerate(episodes) if i in validation_ids]
        bc_data_source = OfflineDataSource(
            TokenizedEpisodeDataset(train_episodes), data_source.config,
            env_id_to_index=data_source.env_id_to_index,
        )
        bc_validation_data_source = OfflineDataSource(
            TokenizedEpisodeDataset(validation_episodes), data_source.config,
            env_id_to_index=data_source.env_id_to_index,
        )
        print(
            f"  Offline BC episode-disjoint split: train={len(train_episodes)}, "
            f"validation={len(validation_episodes)}", flush=True,
        )

    def sample_behavior_batch_fn(batch_size: int) -> tuple[Tensor, Tensor]:
        return _sample_horizon_eligible_behavior_examples(
            batch_size=batch_size,
            rollout_horizon=pipeline_config.rollout_horizon,
            null_action_id=config.env.null_action_id,
            data_source=bc_data_source,
            grounder=grounder,
            device=device,
            dtype=param_dtype,
            action_id_offset=ppo_cfg.wm_action_id_offset,
            num_policy_actions=config.env.num_actions,
        )

    def sample_behavior_validation_batch_fn(batch_size: int) -> tuple[Tensor, Tensor]:
        if bc_validation_data_source is None:
            raise RuntimeError("offline BC validation source is unavailable")
        return _sample_horizon_eligible_behavior_examples(
            batch_size=batch_size,
            rollout_horizon=pipeline_config.rollout_horizon,
            null_action_id=config.env.null_action_id,
            data_source=bc_validation_data_source,
            grounder=grounder,
            device=device,
            dtype=param_dtype,
            action_id_offset=ppo_cfg.wm_action_id_offset,
            num_policy_actions=config.env.num_actions,
        )

    offline_behavior_cloner = None
    if offline_bc_enabled:
        offline_behavior_cloner = OfflineBehaviorCloner(
            policy=llm_policy,
            sample_batch_fn=sample_behavior_batch_fn,
            validation_sample_batch_fn=sample_behavior_validation_batch_fn,
            capture_behavior_reference=behavior_support_enabled,
            config=OfflineBCConfig(
                steps=ppo_cfg.offline_bc_steps,
                batch_size=ppo_cfg.offline_bc_batch_size,
                cache_size=ppo_cfg.offline_bc_cache_size,
                lr=ppo_cfg.offline_bc_lr,
                weight_decay=ppo_cfg.weight_decay,
                max_grad_norm=ppo_cfg.max_grad_norm,
            ),
        )

    # --- Build WM refresh sample function ---
    wm_refresh_sample_fn = None
    if wm_refresher is not None:
        def wm_refresh_sample_fn(batch_size: int) -> SequenceBatch:
            batch = mixed_sampler.sample(batch_size, config)
            return _to_device(batch, device)
        wm_refresh_sample_fn.mixed_sampler = mixed_sampler

    # --- Build PPO updater ---
    # Adapter views are non-owning: Qwen base and every LoRA tensor are stored
    # once under the WM PEFT container. Only qwen_pooled adapter tensors are
    # enabled and passed explicitly to PPO below.
    actor_optimizer_params = list(llm_policy.actor_parameters())
    critic_optimizer_params = list(llm_policy.critic_parameters())
    main_backbone_trains_lora = (
        actor_trains_qwen
        or (
            llm_cfg.critic_source in {"qwen_pooled", "qwen_slotwise_q"}
            and critic_backbone is None
        )
    )
    if shared_backbone or not main_backbone_trains_lora:
        for p in backbone.trainable_parameters():
            p.requires_grad_(False)
    else:
        for p in backbone.trainable_parameters():
            p.requires_grad_(True)
    if (
        critic_backbone is not None
        and llm_cfg.critic_source not in {"qwen_pooled", "qwen_slotwise_q"}
    ):
        for p in critic_backbone.trainable_parameters():
            p.requires_grad_(False)
    elif critic_backbone is not None:
        for p in critic_backbone.trainable_parameters():
            p.requires_grad_(True)

    # Adapter views are deliberately non-owning so policy.state_dict() does not
    # duplicate the shared Qwen base. Pass only PPO-owned LoRA tensors as
    # explicit optimizer parameters; heads remain policy-owned submodules.
    policy_adapter_names: set[str] = set()
    if actor_trains_qwen:
        actor_optimizer_params.extend(backbone.trainable_parameters())
        policy_adapter_names.add(backbone.adapter_name)
    if llm_cfg.critic_source in {"qwen_pooled", "qwen_slotwise_q"}:
        critic_lora_backbone = critic_backbone or backbone
        critic_optimizer_params.extend(
            critic_lora_backbone.trainable_parameters()
        )
        policy_adapter_names.add(critic_lora_backbone.adapter_name)
    ppo_updater = PPOUpdater(
        config=pipeline_config.ppo,
        policy=llm_policy,
        actor_params=actor_optimizer_params,
        critic_params=critic_optimizer_params,
    )
    # Enforce disjoint optimizer ownership in every WM mode. In alternating_wm
    # the Phase1 refresher has its own optimizer, making overlap even more
    # dangerous than in the frozen case.
    _assert_optimizer_does_not_own_world_model(
        world_model,
        ppo_updater.optimizer,
        policy_adapter_names=frozenset(policy_adapter_names),
    )

    if behavior_rehearsal_enabled:
        if config.phase2.world_model_mode == "alternating_wm":
            print(
                "  Expert Rehearsal: fixed episode/action source; posterior "
                "beliefs re-grounded with the current WM each update.",
                flush=True,
            )
        else:
            print(
                "  Expert Rehearsal: fixed episode-disjoint posterior/action "
                "latent cache (safe because WM is frozen).",
                flush=True,
            )

    # --- Assemble ---
    orchestration_class = pipeline_class or MBRLPipeline
    pipeline = orchestration_class(
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
        offline_behavior_cloner=offline_behavior_cloner,
        behavior_sample_fn=(
            # A frozen WM can safely reuse the fixed posterior-latent cache.
            # In alternating mode the fixed episode/action supervision is
            # re-grounded through the current posterior after every refresh;
            # reusing old latent tensors would train in a stale coordinate
            # system and silently corrupt the actor.
            sample_behavior_batch_fn
            if (
                behavior_rehearsal_enabled
                and config.phase2.world_model_mode == "alternating_wm"
            )
            else (
                offline_behavior_cloner.sample_rehearsal_batch
                if offline_behavior_cloner is not None else None
            )
        ),
    )
    pipeline.replay_sampler = mixed_sampler

    return pipeline, llm_policy


def _to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        obs_tokens=batch.obs_tokens.to(device),
        actions=batch.actions.to(device),
        rewards=batch.rewards.to(device),
        episode_lengths=batch.episode_lengths.to(device),
        env_ids=None if batch.env_ids is None else batch.env_ids.to(device),
        episode_success=(
            None
            if batch.episode_success is None
            else batch.episode_success.to(device)
        ),
        semantic_teacher_tokens=(
            None
            if batch.semantic_teacher_tokens is None
            else batch.semantic_teacher_tokens.to(device)
        ),
        semantic_teacher_mask=(
            None
            if batch.semantic_teacher_mask is None
            else batch.semantic_teacher_mask.to(device)
        ),
    )


def _sample_horizon_eligible_beliefs(
    *,
    batch_size: int,
    rollout_horizon: int,
    null_action_id: int,
    mixed_sampler: MixedDataSampler,
    config: Config,
    grounder: PosteriorGrounder,
    device: torch.device,
    dtype: torch.dtype,
    max_attempts: int = 32,
) -> Tensor:
    """Sample and ground a full batch with H real transitions remaining.

    Short episodes are common in Sokoban (including one-step successes). They
    cannot support an H-step endpoint objective and are therefore filtered,
    with replacement samples drawn until the requested batch size is filled.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")

    chunks: list[Tensor] = []
    remaining = batch_size
    attempts = 0
    while remaining > 0 and attempts < max_attempts:
        attempts += 1
        batch = mixed_sampler.sample(remaining, config)
        eligible = batch.episode_lengths >= rollout_horizon + 1
        if not bool(eligible.any()):
            continue

        batch = _index_sequence_batch(batch, eligible)
        batch = _to_device(batch, device)
        chunks.append(
            sample_start_beliefs(
                batch,
                grounder,
                device=device,
                dtype=dtype,
                rollout_horizon=rollout_horizon,
                null_action_id=null_action_id,
            )
        )
        remaining -= batch.batch_size

    if remaining > 0:
        raise RuntimeError(
            f"Could not sample {batch_size} episodes with at least "
            f"{rollout_horizon} transitions after {max_attempts} attempts; "
            f"still missing {remaining}"
        )
    return torch.cat(chunks, dim=0)


def _sample_horizon_eligible_behavior_examples(
    *,
    batch_size: int,
    rollout_horizon: int,
    null_action_id: int,
    data_source: OfflineDataSource,
    grounder: PosteriorGrounder,
    device: torch.device,
    dtype: torch.dtype,
    action_id_offset: int = 0,
    num_policy_actions: int = 4,
    max_attempts: int = 32,
) -> tuple[Tensor, Tensor]:
    """Sample aligned offline ``(posterior belief, logged action)`` pairs.

    This deliberately reads from ``OfflineDataSource`` directly rather than
    ``MixedDataSampler`` so behavior cloning cannot silently consume online
    replay even if a different experiment enables collection elsewhere.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    belief_chunks: list[Tensor] = []
    action_chunks: list[Tensor] = []
    remaining = batch_size
    attempts = 0
    while remaining > 0 and attempts < max_attempts:
        attempts += 1
        batch = data_source.sample_batch(remaining)
        eligible = batch.episode_lengths >= rollout_horizon + 1
        if not bool(eligible.any()):
            continue
        batch = _to_device(_index_sequence_batch(batch, eligible), device)
        beliefs, logged_actions = sample_start_beliefs_and_actions(
            batch,
            grounder,
            device=device,
            dtype=dtype,
            rollout_horizon=rollout_horizon,
            null_action_id=null_action_id,
            early_state_probability=float(os.environ.get(
                "BC_EARLY_STATE_PROBABILITY", "0"
            )),
            early_state_max_t=int(os.environ.get("BC_EARLY_STATE_MAX_T", "2")),
        )
        logged_actions = logged_actions - action_id_offset
        invalid = (logged_actions < 0) | (
            logged_actions >= num_policy_actions
        )
        if bool(invalid.any()):
            values = sorted(
                set(logged_actions[invalid].detach().cpu().tolist())
            )
            raise ValueError(
                "Offline BC action labels are outside the policy range "
                f"[0, {num_policy_actions - 1}] after applying "
                f"wm_action_id_offset={action_id_offset}; invalid={values}. "
                "Use --wm-action-id-offset 1 for the legacy Sokoban 1..4 "
                "dataset/checkpoint pair, otherwise regenerate tokenized data "
                "with model_actions 0..3."
            )
        belief_chunks.append(beliefs)
        action_chunks.append(logged_actions)
        remaining -= batch.batch_size

    if remaining > 0:
        raise RuntimeError(
            f"Could not sample {batch_size} offline behavior examples with "
            f"horizon={rollout_horizon} after {max_attempts} attempts; "
            f"still missing {remaining}"
        )
    return torch.cat(belief_chunks, dim=0), torch.cat(action_chunks, dim=0)


def _index_sequence_batch(batch: SequenceBatch, mask: Tensor) -> SequenceBatch:
    """Select episodes from a SequenceBatch without changing time padding."""
    return SequenceBatch(
        obs_tokens=batch.obs_tokens[mask],
        actions=batch.actions[mask],
        rewards=batch.rewards[mask],
        episode_lengths=batch.episode_lengths[mask],
        env_ids=None if batch.env_ids is None else batch.env_ids[mask],
        episode_success=(
            None
            if batch.episode_success is None
            else batch.episode_success[mask]
        ),
        semantic_teacher_tokens=(
            None
            if batch.semantic_teacher_tokens is None
            else batch.semantic_teacher_tokens[mask]
        ),
        semantic_teacher_mask=(
            None
            if batch.semantic_teacher_mask is None
            else batch.semantic_teacher_mask[mask]
        ),
    )
