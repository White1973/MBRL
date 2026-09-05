"""PPO updater.

Takes a PPOBatch + a Policy, runs clipped PPO updates.
Does not know where the batch came from (real env, imagined rollout, etc.).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .trajectory import PPOBatch
from .policy import Policy


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    epochs: int = 4
    minibatch_size: int = 512
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.0
    target_kl: float | None = None
    behavior_kl_coef: float = 0.0
    behavior_bc_coef: float = 0.0
    behavior_bc_batch_size: int = 32
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    lr: float = 3e-4
    critic_lr: float | None = None
    weight_decay: float = 0.0
    # Re-evaluate the unchanged rollout actor once using the exact PPO
    # minibatch partition. This removes false ratios caused by Qwen/bf16
    # batch-shape numerics (rollout uses B, PPO normally uses B*H), while the
    # sampled actions and latent trajectories remain unchanged.
    recompute_old_log_probs: bool = True
    # Entropy floor (anti-collapse): when mean policy entropy drops below
    # target_entropy, add a loss term that pushes entropy back up. Stabilizes
    # the "intermittent collapse + recovery" oscillation seen in long runs.
    # None = disabled (legacy behavior). entropy_floor_coef scales the term.
    target_entropy: float | None = None
    entropy_floor_coef: float = 0.1


@dataclass
class PPOMetrics:
    """Metrics from one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    kl_divergence: float
    explained_variance: float
    value_delta: float
    post_update_entropy: float
    entropy_deficit: float
    entropy_floor_active_fraction: float
    target_kl_early_stop: float
    behavior_kl: float
    behavior_bc_loss: float
    behavior_bc_accuracy: float
    num_minibatches: float
    sample_coverage: float
    rollout_reference_logprob_mean_abs: float
    rollout_reference_logprob_max_abs: float
    old_log_probs_recomputed: float
    initial_clip_fraction: float
    initial_kl_divergence: float
    attempted_minibatches: float
    max_minibatch_kl: float
    last_minibatch_kl: float
    rejected_minibatch_kl: float
    post_update_kl_divergence: float
    post_update_clip_fraction: float
    critic_num_minibatches: float
    critic_sample_coverage: float
    actor_grad_norm: float
    critic_grad_norm: float


@dataclass
class JointCriticMetrics:
    """Metrics from one group-aware H2 + real Critic update."""

    loss: float
    h2_mse: float
    real_mse: float
    ranking_loss: float
    h2_explained_variance: float
    real_explained_variance: float
    grad_norm: float
    h2_samples: float
    real_samples: float
    real_groups: float
    minibatches: float
    gradient_cosine: float
    gradient_conflict_fraction: float


class PPOUpdater:
    """Stateful PPO updater that owns the optimizer.

    Uses a single optimizer over all policy.parameters() plus any
    extra_params (e.g., shared backbone LoRA parameters that are not
    registered as submodules of the policy).

    Usage:
        updater = PPOUpdater(config, policy)
        metrics = updater.update(batch)

    For shared backbone setups:
        shared_params = backbone.trainable_parameters()
        updater = PPOUpdater(config, policy, extra_params=shared_params)
    """

    def __init__(
        self,
        config: PPOConfig,
        policy: Policy,
        extra_params: list[nn.Parameter] | None = None,
        actor_params: list[nn.Parameter] | None = None,
        critic_params: list[nn.Parameter] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        if config.target_kl is not None and config.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when enabled")
        if config.behavior_kl_coef < 0.0:
            raise ValueError("behavior_kl_coef must be non-negative")
        if config.behavior_bc_coef < 0.0:
            raise ValueError("behavior_bc_coef must be non-negative")
        if config.behavior_bc_batch_size <= 0:
            raise ValueError("behavior_bc_batch_size must be positive")
        if config.behavior_bc_coef > 0.0 and not callable(
            getattr(policy, "actor_logits_trainable", None)
        ):
            raise TypeError(
                "behavior_bc_coef requires policy.actor_logits_trainable()"
            )
        if config.behavior_kl_coef > 0.0 and not callable(
            getattr(policy, "evaluate_actions_with_behavior", None)
        ):
            raise TypeError(
                "behavior_kl_coef requires "
                "policy.evaluate_actions_with_behavior()"
            )

        explicit_split = actor_params is not None or critic_params is not None
        if explicit_split and (actor_params is None or critic_params is None):
            raise ValueError(
                "actor_params and critic_params must be provided together"
            )
        if explicit_split and extra_params:
            raise ValueError(
                "extra_params cannot be combined with explicit actor/critic groups"
            )

        # Explicit groups keep Adam state, LR, gradient clipping and update
        # scheduling disjoint while retaining one serializable optimizer
        # container. A critic-only step has actor grads=None, so Adam does not
        # advance or decay actor parameters.
        if explicit_split:
            assert actor_params is not None and critic_params is not None
            self._actor_trainable = _unique_trainable(actor_params)
            self._critic_trainable = _unique_trainable(critic_params)
            actor_ids = {id(parameter) for parameter in self._actor_trainable}
            overlap = actor_ids & {
                id(parameter) for parameter in self._critic_trainable
            }
            if overlap:
                raise ValueError(
                    "actor and critic optimizer groups overlap; ownership must "
                    "be disjoint"
                )
            if not self._actor_trainable or not self._critic_trainable:
                raise ValueError(
                    "explicit actor and critic optimizer groups must be non-empty"
                )
            self._all_trainable = (
                self._actor_trainable + self._critic_trainable
            )
            warmup_head_ids = {
                id(parameter)
                for parameter in getattr(policy, "critic_parameters")()
            }
            self._critic_warmup_head_ids = warmup_head_ids
            self._separate_actor_critic = True
            critic_lr = config.critic_lr or config.lr
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self._actor_trainable,
                        "lr": config.lr,
                        "group_name": "actor",
                    },
                    {
                        "params": self._critic_trainable,
                        "lr": critic_lr,
                        "group_name": "critic",
                    },
                ],
                weight_decay=config.weight_decay,
            )
            return

        # Compatibility path for generic policies without ownership helpers.
        all_params = list(policy.parameters())
        if extra_params:
            # Deduplicate by data_ptr to avoid issues with overlapping param sets
            seen = {p.data_ptr() for p in all_params}
            for p in extra_params:
                if p.data_ptr() not in seen:
                    all_params.append(p)
                    seen.add(p.data_ptr())
        self._all_trainable = [p for p in all_params if p.requires_grad]
        self._actor_trainable = self._all_trainable
        self._critic_trainable = self._all_trainable
        self._separate_actor_critic = False

        self.optimizer = torch.optim.AdamW(
            self._all_trainable,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def update(
        self,
        batch: PPOBatch,
        behavior_batch: tuple[Tensor, Tensor] | None = None,
        *,
        critic_only: bool = False,
        actor_enabled: bool = True,
        critic_enabled: bool = True,
        critic_bucket_ids: Tensor | None = None,
    ) -> PPOMetrics:
        """Run multi-epoch minibatch PPO on a flattened batch."""

        cfg = self.config
        states = batch.states
        actions = batch.actions
        advantages = batch.advantages.clone()
        returns = batch.returns
        old_log_probs = batch.old_log_probs
        old_values = batch.old_values
        if (
            not critic_only
            and actor_enabled
            and self.config.behavior_bc_coef > 0.0
            and behavior_batch is None
        ):
            raise ValueError(
                "behavior_bc_coef > 0 requires a posterior expert batch"
            )

        N = actions.shape[0]
        if critic_bucket_ids is not None and len(critic_bucket_ids) != N:
            raise ValueError("critic_bucket_ids must align with PPO batch")
        if N == 0:
            return PPOMetrics(
                policy_loss=0.0, value_loss=0.0, entropy=0.0,
                clip_fraction=0.0, kl_divergence=0.0,
                explained_variance=0.0, value_delta=0.0,
                post_update_entropy=0.0, entropy_deficit=0.0,
                entropy_floor_active_fraction=0.0,
                target_kl_early_stop=0.0,
                behavior_kl=0.0,
                behavior_bc_loss=0.0,
                behavior_bc_accuracy=0.0,
                num_minibatches=0.0,
                sample_coverage=0.0,
                rollout_reference_logprob_mean_abs=0.0,
                rollout_reference_logprob_max_abs=0.0,
                old_log_probs_recomputed=float(cfg.recompute_old_log_probs),
                initial_clip_fraction=0.0,
                initial_kl_divergence=0.0,
                attempted_minibatches=0.0,
                max_minibatch_kl=0.0,
                last_minibatch_kl=0.0,
                rejected_minibatch_kl=0.0,
                post_update_kl_divergence=0.0,
                post_update_clip_fraction=0.0,
                critic_num_minibatches=0.0,
                critic_sample_coverage=0.0,
                actor_grad_norm=0.0,
                critic_grad_norm=0.0,
            )

        def expert_rehearsal() -> tuple[Tensor, Tensor]:
            if behavior_batch is None:
                zero = torch.zeros((), device=states.device)
                return zero, zero
            behavior_states, behavior_actions = behavior_batch
            logits_fn = getattr(self.policy, "actor_logits_trainable")
            behavior_logits = logits_fn(
                behavior_states.to(device=states.device)
            )
            targets = behavior_actions.to(
                device=behavior_logits.device, dtype=torch.long
            )
            loss = F.cross_entropy(behavior_logits, targets)
            accuracy = (
                behavior_logits.argmax(dim=-1) == targets
            ).float().mean()
            return loss, accuracy

        if cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)

        minibatch_size = min(cfg.minibatch_size, N)

        # Use one fixed partition for reference evaluation and all PPO epochs.
        # The actor parameters are still untouched here. In particular, this
        # avoids comparing rollout log-probs computed as (B,K,D) forwards with
        # PPO log-probs computed as a different flattened batch shape. Qwen
        # bf16 kernels can otherwise differ enough to create fake clipping.
        grouped_critic_warmup = bool(
            critic_only
            and os.environ.get("COUNTERFACTUAL_H2_PPO", "0") == "1"
        )
        if grouped_critic_warmup:
            if N % 4 != 0 or minibatch_size % 4 != 0:
                raise RuntimeError(
                    "exact H2 Critic minibatches must contain complete "
                    "four-action groups"
                )
            group_perm = torch.randperm(N // 4, device=states.device)
            fixed_perm = (
                group_perm[:, None] * 4
                + torch.arange(4, device=states.device)[None, :]
            ).reshape(-1)
        else:
            fixed_perm = torch.randperm(N, device=states.device)
        fixed_minibatches = [
            fixed_perm[start : start + minibatch_size]
            for start in range(0, N, minibatch_size)
        ]
        rollout_old_log_probs = old_log_probs.detach().float()
        rollout_reference_mean_abs = 0.0
        rollout_reference_max_abs = 0.0
        if cfg.recompute_old_log_probs:
            stabilized = getattr(
                self.policy,
                "set_deterministic_forward_mode",
                None,
            )
            if callable(stabilized):
                stabilized()
            reference = torch.empty_like(rollout_old_log_probs)
            evaluate_actor = getattr(
                self.policy,
                "evaluate_actor_actions",
                None,
            )
            with torch.no_grad():
                for idx in fixed_minibatches:
                    if callable(evaluate_actor):
                        ref_log_probs, _ = evaluate_actor(
                            states[idx], actions[idx]
                        )
                    else:
                        ref_log_probs, _, _ = self.policy.evaluate_actions(
                            states[idx], actions[idx]
                        )
                    reference[idx] = ref_log_probs.float()
            reference_delta = reference - rollout_old_log_probs
            rollout_reference_mean_abs = float(
                reference_delta.abs().mean().item()
            )
            rollout_reference_max_abs = float(
                reference_delta.abs().max().item()
            )
            old_log_probs = reference

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_kl = 0.0
        total_entropy_deficit = 0.0
        total_behavior_kl = 0.0
        total_behavior_bc_loss = 0.0
        total_behavior_bc_accuracy = 0.0
        entropy_floor_active_count = 0
        num_minibatches = 0
        num_samples_processed = 0
        critic_num_minibatches = 0
        critic_samples_processed = 0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        target_kl_early_stop = False
        actor_updates_enabled = not critic_only and actor_enabled
        if not critic_enabled and critic_only:
            raise ValueError("critic_only requires critic_enabled=True")
        if not critic_enabled and not self._separate_actor_critic:
            raise ValueError(
                "actor-only PPO requires explicit actor/critic parameter groups"
            )
        initial_clip_fraction = 0.0
        initial_kl_divergence = 0.0
        attempted_minibatches = 0
        max_minibatch_kl = 0.0
        last_minibatch_kl = 0.0
        rejected_minibatch_kl = 0.0
        separate_adapter_backward = bool(
            getattr(
                self.policy,
                "requires_separate_actor_critic_backward",
                False,
            )
        )
        if separate_adapter_backward and cfg.behavior_kl_coef > 0.0:
            # LLMActorCritic evaluates the frozen behavior adapter first and
            # restores the current actor adapter last.  Actor backward thus
            # remains safe with gradient checkpointing; critic adapter
            # switching still happens only after actor backward completes.
            pass

        for _ in range(cfg.epochs):
            if cfg.recompute_old_log_probs:
                minibatches = fixed_minibatches
            elif grouped_critic_warmup:
                epoch_groups = torch.randperm(N // 4, device=states.device)
                epoch_perm = (
                    epoch_groups[:, None] * 4
                    + torch.arange(4, device=states.device)[None, :]
                ).reshape(-1)
                minibatches = list(epoch_perm.split(minibatch_size))
            else:
                minibatches = list(
                    torch.randperm(N, device=states.device).split(
                        minibatch_size
                    )
                )
            for idx in minibatches:

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_old_log_probs = old_log_probs[idx]

                if self._separate_actor_critic:
                    # Actor trust-region update. Once target KL is exceeded,
                    # skip all remaining actor forwards/steps but continue the
                    # critic path below on every sample.
                    if actor_updates_enabled:
                        self.optimizer.zero_grad(set_to_none=True)
                        behavior_kl = torch.zeros(
                            (), device=mb_states.device
                        )
                        if cfg.behavior_kl_coef > 0.0:
                            evaluate_with_behavior = getattr(
                                self.policy,
                                "evaluate_actor_actions_with_behavior",
                            )
                            (
                                new_log_probs,
                                entropy,
                                action_log_probs,
                                behavior_log_probs,
                            ) = evaluate_with_behavior(
                                mb_states,
                                mb_actions,
                            )
                            action_probabilities = action_log_probs.exp()
                            behavior_kl = (
                                action_probabilities
                                * (action_log_probs - behavior_log_probs)
                            ).sum(dim=-1).mean()
                        else:
                            evaluate_actor = getattr(
                                self.policy,
                                "evaluate_actor_actions",
                            )
                            new_log_probs, entropy = evaluate_actor(
                                mb_states,
                                mb_actions,
                            )
                        entropy_mean = entropy.mean()
                        log_ratio = new_log_probs - mb_old_log_probs
                        ratio = torch.exp(log_ratio)
                        clipped = torch.clamp(
                            ratio,
                            1.0 - cfg.clip_epsilon,
                            1.0 + cfg.clip_epsilon,
                        )
                        surr = torch.min(
                            ratio * mb_advantages,
                            clipped * mb_advantages,
                        )
                        policy_loss = -surr.mean()
                        clipfrac = (
                            torch.abs(ratio - 1.0) > cfg.clip_epsilon
                        ).float().mean()
                        kl_div = ((ratio - 1.0) - log_ratio).mean()
                        kl_value = float(kl_div.detach().item())
                        attempted_minibatches += 1
                        max_minibatch_kl = max(
                            max_minibatch_kl,
                            kl_value,
                        )
                        last_minibatch_kl = kl_value
                        if attempted_minibatches == 1:
                            initial_clip_fraction = float(
                                clipfrac.detach().item()
                            )
                            initial_kl_divergence = kl_value
                        if (
                            cfg.target_kl is not None
                            and kl_value > cfg.target_kl
                        ):
                            target_kl_early_stop = True
                            rejected_minibatch_kl = kl_value
                            actor_updates_enabled = False
                        else:
                            actor_loss = (
                                policy_loss
                                - cfg.entropy_coef * entropy_mean
                                + cfg.kl_coef * kl_div
                                + cfg.behavior_kl_coef * behavior_kl
                            )
                            # Weakly remove only the batch-global action bias.
                            # State-dependent deviations remain untouched, so
                            # this cannot prescribe an expert action or force
                            # every individual state to be uniform.
                            global_bias_coef = float(os.environ.get(
                                "GLOBAL_ACTION_BIAS_COEF", "0"
                            ))
                            if global_bias_coef > 0.0:
                                logits_fn = getattr(
                                    self.policy, "actor_logits_trainable"
                                )
                                mean_logits = logits_fn(mb_states).float().mean(0)
                                mean_logits = mean_logits - mean_logits.mean()
                                actor_loss = actor_loss + global_bias_coef * (
                                    mean_logits.pow(2).mean()
                                )
                            behavior_bc_loss, behavior_bc_accuracy = (
                                expert_rehearsal()
                            )
                            actor_loss = (
                                actor_loss
                                + cfg.behavior_bc_coef * behavior_bc_loss
                            )
                            entropy_deficit = torch.zeros(
                                (), device=mb_states.device
                            )
                            if cfg.target_entropy is not None:
                                entropy_deficit = torch.relu(
                                    cfg.target_entropy - entropy_mean
                                )
                                actor_loss = (
                                    actor_loss
                                    + cfg.entropy_floor_coef
                                    * entropy_deficit
                                )
                            actor_loss.backward()
                            actor_grad_norm = nn.utils.clip_grad_norm_(
                                self._actor_trainable,
                                cfg.max_grad_norm,
                            )
                            self.optimizer.step()
                            total_actor_grad_norm += float(actor_grad_norm)
                            total_policy_loss += float(policy_loss.item())
                            total_entropy += float(entropy_mean.item())
                            total_clipfrac += float(clipfrac.item())
                            total_kl += float(kl_div.item())
                            total_behavior_kl += float(behavior_kl.item())
                            total_behavior_bc_loss += float(
                                behavior_bc_loss.item()
                            )
                            total_behavior_bc_accuracy += float(
                                behavior_bc_accuracy.item()
                            )
                            total_entropy_deficit += float(
                                entropy_deficit.item()
                            )
                            entropy_floor_active_count += int(
                                entropy_deficit.item() > 0.0
                            )
                            num_minibatches += 1
                            num_samples_processed += int(idx.numel())

                    if critic_enabled:
                        # Critic normally consumes the full configured sample
                        # budget even after Actor KL early-stop. Grounded Actor
                        # diagnostics may disable this synthetic-return path so
                        # only real-return anchor updates can change Critic.
                        self.optimizer.zero_grad(set_to_none=True)
                        evaluate_values = getattr(
                            self.policy,
                            "evaluate_values",
                        )
                        values = evaluate_values(mb_states, mb_actions)
                        squared_error = (values - mb_returns).pow(2)
                        value_loss = squared_error.mean()
                        if grouped_critic_warmup:
                            grouped_values = values.float().view(-1, 4)
                            grouped_returns = mb_returns.float().view(-1, 4)
                            centered_values = (
                                grouped_values
                                - grouped_values.mean(dim=1, keepdim=True)
                            )
                            centered_returns = (
                                grouped_returns
                                - grouped_returns.mean(dim=1, keepdim=True)
                            )
                            raw_mse_coef = float(os.environ.get(
                                "CRITIC_WARMUP_RAW_MSE_COEF", "0.25"
                            ))
                            centered_mse_coef = float(os.environ.get(
                                "CRITIC_WARMUP_CENTERED_MSE_COEF", "1.0"
                            ))
                            if raw_mse_coef < 0.0 or centered_mse_coef <= 0.0:
                                raise ValueError(
                                    "exact H2 Critic MSE coefficients require "
                                    "raw>=0 and centered>0"
                                )
                            group_bucket_ids = (
                                None if critic_bucket_ids is None
                                else critic_bucket_ids[idx].view(-1, 4)[:, 0]
                            )
                            raw_per_group = squared_error.view(-1, 4).mean(dim=1)
                            centered_per_group = (
                                centered_values - centered_returns
                            ).pow(2).mean(dim=1)
                            regression_per_group = (
                                raw_mse_coef * raw_per_group
                                + centered_mse_coef * centered_per_group
                            )
                            worst_bucket_coef = float(os.environ.get(
                                "CRITIC_WORST_BUCKET_COEF", "0.50"
                            ))
                            if worst_bucket_coef < 0.0:
                                raise ValueError(
                                    "CRITIC_WORST_BUCKET_COEF must be non-negative"
                                )
                            if group_bucket_ids is None:
                                value_loss = regression_per_group.mean()
                            else:
                                bucket_losses = torch.stack([
                                    regression_per_group[group_bucket_ids == bucket_id].mean()
                                    for bucket_id in group_bucket_ids.unique(sorted=True)
                                ])
                                value_loss = (
                                    bucket_losses.mean()
                                    + worst_bucket_coef * bucket_losses.max()
                                )
                            ranking_terms = []
                            temperature = float(os.environ.get(
                                "CRITIC_WARMUP_RANKING_TEMPERATURE", "0.10"
                            ))
                            if temperature <= 0.0:
                                raise ValueError(
                                    "CRITIC_WARMUP_RANKING_TEMPERATURE must "
                                    "be positive"
                                )
                            for left in range(4):
                                for right in range(left + 1, 4):
                                    target_delta = (
                                        grouped_returns[:, left]
                                        - grouped_returns[:, right]
                                    )
                                    informative = target_delta.abs() > 1e-6
                                    if bool(informative.any()):
                                        predicted_delta = (
                                            grouped_values[:, left]
                                            - grouped_values[:, right]
                                        )
                                        signed_margin = (
                                            target_delta[informative].sign()
                                            * predicted_delta[informative]
                                            / temperature
                                        )
                                        ranking_terms.append(
                                            F.softplus(-signed_margin).mean()
                                        )
                            if ranking_terms:
                                ranking_coef = float(os.environ.get(
                                    "CRITIC_WARMUP_RANKING_COEF", "0.10"
                                ))
                                value_loss = value_loss + ranking_coef * (
                                    torch.stack(ranking_terms).mean()
                                )
                        (cfg.value_coef * value_loss).backward()
                        if (
                            critic_only
                            and os.environ.get(
                                "CRITIC_WARMUP_HEAD_ONLY", "1"
                            ) == "1"
                        ):
                            for parameter in self._critic_trainable:
                                if id(parameter) not in self._critic_warmup_head_ids:
                                    parameter.grad = None
                        critic_grad_norm = nn.utils.clip_grad_norm_(
                            self._critic_trainable,
                            cfg.max_grad_norm,
                        )
                        self.optimizer.step()
                        total_critic_grad_norm += float(critic_grad_norm)
                        total_value_loss += float(value_loss.item())
                        critic_num_minibatches += 1
                        critic_samples_processed += int(idx.numel())
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                behavior_kl = torch.zeros((), device=mb_states.device)
                if separate_adapter_backward:
                    evaluate_actor = getattr(
                        self.policy,
                        "evaluate_actor_actions",
                    )
                    new_log_probs, entropy = evaluate_actor(
                        mb_states,
                        mb_actions,
                    )
                    values = None
                elif cfg.behavior_kl_coef > 0.0:
                    (
                        new_log_probs,
                        entropy,
                        values,
                        action_log_probs,
                        behavior_log_probs,
                    ) = self.policy.evaluate_actions_with_behavior(
                        mb_states, mb_actions
                    )
                    action_probabilities = action_log_probs.exp()
                    behavior_kl = (
                        action_probabilities
                        * (action_log_probs - behavior_log_probs)
                    ).sum(dim=-1).mean()
                else:
                    new_log_probs, entropy, values = self.policy.evaluate_actions(
                        mb_states, mb_actions
                    )
                entropy_mean = entropy.mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                )
                surr = torch.min(ratio * mb_advantages, clipped * mb_advantages)
                policy_loss = -surr.mean()

                clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float().mean()
                # Non-negative approximate reverse KL used by PPO diagnostics
                # and early stopping: E[(ratio - 1) - log(ratio)].
                kl_div = ((ratio - 1.0) - log_ratio).mean()
                kl_value = float(kl_div.detach().item())
                attempted_minibatches += 1
                max_minibatch_kl = max(max_minibatch_kl, kl_value)
                last_minibatch_kl = kl_value
                if attempted_minibatches == 1:
                    initial_clip_fraction = float(clipfrac.detach().item())
                    initial_kl_divergence = kl_value

                # The KL describes the current actor *before* this minibatch's
                # optimizer step. If it is already outside the trust region,
                # reject the minibatch without backward/step. The old behavior
                # applied one additional, already-invalid update and only then
                # stopped, systematically overshooting target_kl.
                if cfg.target_kl is not None and kl_value > cfg.target_kl:
                    target_kl_early_stop = True
                    rejected_minibatch_kl = kl_value
                    break

                actor_loss = (
                    policy_loss
                    - cfg.entropy_coef * entropy_mean
                    + cfg.kl_coef * kl_div
                    + cfg.behavior_kl_coef * behavior_kl
                )
                behavior_bc_loss, behavior_bc_accuracy = expert_rehearsal()
                actor_loss = (
                    actor_loss + cfg.behavior_bc_coef * behavior_bc_loss
                )
                # Entropy floor: when entropy < target, add a positive loss
                # whose gradient increases entropy (d relu(target-H)/dH = -1
                # when H<target → loss decreases as H increases).
                if cfg.target_entropy is not None:
                    entropy_deficit = torch.relu(cfg.target_entropy - entropy_mean)
                    actor_loss = (
                        actor_loss
                        + cfg.entropy_floor_coef * entropy_deficit
                    )
                    total_entropy_deficit += float(entropy_deficit.item())
                    entropy_floor_active_count += int(entropy_deficit.item() > 0.0)

                if separate_adapter_backward:
                    # Backpropagate while the actor adapter is still active.
                    # This is required for gradient-checkpoint recomputation;
                    # switching to the critic adapter first would silently
                    # recompute actor layers with the wrong LoRA.
                    actor_loss.backward()
                    evaluate_values = getattr(self.policy, "evaluate_values")
                    values = evaluate_values(mb_states, mb_actions)
                    value_loss = (values - mb_returns).pow(2).mean()
                    (cfg.value_coef * value_loss).backward()
                else:
                    assert values is not None
                    value_loss = (values - mb_returns).pow(2).mean()
                    loss = actor_loss + cfg.value_coef * value_loss
                    loss.backward()
                combined_grad_norm = nn.utils.clip_grad_norm_(
                    self._all_trainable,
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_clipfrac += clipfrac.item()
                total_kl += kl_div.item()
                total_behavior_kl += behavior_kl.item()
                total_behavior_bc_loss += behavior_bc_loss.item()
                total_behavior_bc_accuracy += behavior_bc_accuracy.item()
                num_minibatches += 1
                num_samples_processed += int(idx.numel())
                critic_num_minibatches += 1
                critic_samples_processed += int(idx.numel())
                total_actor_grad_norm += float(combined_grad_norm)
                total_critic_grad_norm += float(combined_grad_norm)

            if target_kl_early_stop and not self._separate_actor_critic:
                break

        n = max(1, num_minibatches)
        critic_n = max(1, critic_num_minibatches)

        # Post-update diagnostics
        with torch.no_grad():
            # Do not re-evaluate the entire accumulated rollout batch at once.
            # With a frozen VLM actor, N can be hundreds of belief sequences;
            # a single Qwen forward here can exceed the peak memory of the
            # actual minibatched PPO update even though gradients are disabled.
            updated_log_probs = torch.empty_like(old_log_probs.float())
            updated_entropy = torch.empty_like(old_log_probs.float())
            updated_values = torch.empty_like(returns.float())
            # Reuse the same fixed groups as the old-policy reference. This
            # keeps post-update KL free of Qwen/bf16 batch-shape artifacts.
            for idx in fixed_minibatches:
                log_prob_chunk, entropy_chunk, value_chunk = (
                    self.policy.evaluate_actions(
                        states[idx], actions[idx]
                    )
                )
                updated_log_probs[idx] = log_prob_chunk.float()
                updated_entropy[idx] = entropy_chunk.float()
                updated_values[idx] = value_chunk.float()
            post_log_ratio = updated_log_probs - old_log_probs.float()
            post_ratio = post_log_ratio.exp()
            post_update_kl = float(
                (((post_ratio - 1.0) - post_log_ratio).mean()).item()
            )
            post_update_clip = float(
                (
                    (post_ratio - 1.0).abs() > cfg.clip_epsilon
                ).float().mean().item()
            )
            ev = _explained_variance(updated_values, returns)
            vdelta = (updated_values - old_values).abs().mean().item()
            post_update_entropy = float(updated_entropy.mean().item())

        return PPOMetrics(
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / critic_n,
            entropy=total_entropy / n,
            clip_fraction=total_clipfrac / n,
            kl_divergence=total_kl / n,
            explained_variance=ev,
            value_delta=vdelta,
            post_update_entropy=post_update_entropy,
            entropy_deficit=total_entropy_deficit / n,
            entropy_floor_active_fraction=entropy_floor_active_count / n,
            target_kl_early_stop=float(target_kl_early_stop),
            behavior_kl=total_behavior_kl / n,
            behavior_bc_loss=total_behavior_bc_loss / n,
            behavior_bc_accuracy=total_behavior_bc_accuracy / n,
            num_minibatches=float(num_minibatches),
            # Fraction of the configured epoch×sample budget actually consumed.
            # This exposes target-KL early stops that otherwise look like a
            # normal PPO update after metrics are averaged over minibatches.
            sample_coverage=min(
                1.0,
                num_samples_processed / max(N * cfg.epochs, 1),
            ),
            rollout_reference_logprob_mean_abs=(
                rollout_reference_mean_abs
            ),
            rollout_reference_logprob_max_abs=(
                rollout_reference_max_abs
            ),
            old_log_probs_recomputed=float(cfg.recompute_old_log_probs),
            initial_clip_fraction=initial_clip_fraction,
            initial_kl_divergence=initial_kl_divergence,
            attempted_minibatches=float(attempted_minibatches),
            max_minibatch_kl=max_minibatch_kl,
            last_minibatch_kl=last_minibatch_kl,
            rejected_minibatch_kl=rejected_minibatch_kl,
            post_update_kl_divergence=post_update_kl,
            post_update_clip_fraction=post_update_clip,
            critic_num_minibatches=float(critic_num_minibatches),
            critic_sample_coverage=min(
                1.0,
                critic_samples_processed / max(N * cfg.epochs, 1),
            ),
            actor_grad_norm=total_actor_grad_norm / n,
            critic_grad_norm=total_critic_grad_norm / critic_n,
        )

    def update_joint_critic(
        self,
        h2_batch: PPOBatch,
        real_batch: PPOBatch,
        *,
        train_samples: int = 512,
        real_fraction: float = 0.25,
        ranking_coef: float = 0.05,
        ranking_temperature: float = 0.05,
        project_conflicting_gradients: bool = True,
    ) -> JointCriticMetrics:
        """One source-balanced Critic update with intact four-action groups.

        The two regression losses are averaged within source before weighting,
        so a larger real-return gradient cannot dominate merely because of
        sample count. Real samples are selected and minibatched as complete
        ordered ``[0, 1, 2, 3]`` counterfactual groups. Pairwise ordering is
        optimized in the same backward pass as both value objectives.
        """
        if not self._separate_actor_critic:
            raise RuntimeError(
                "joint Critic update requires disjoint Actor/Critic parameters"
            )
        if train_samples < 8:
            raise ValueError("joint Critic train_samples must be at least 8")
        if not 0.0 < real_fraction < 0.5:
            raise ValueError("joint Critic real_fraction must be in (0, 0.5)")
        if ranking_coef < 0.0 or ranking_temperature <= 0.0:
            raise ValueError(
                "ranking_coef must be non-negative and temperature positive"
            )
        if len(h2_batch.actions) == 0 or len(real_batch.actions) == 0:
            raise ValueError("joint Critic batches must be non-empty")
        if len(real_batch.actions) % 4 != 0:
            raise RuntimeError(
                "real Critic replay must contain complete four-action groups"
            )
        grouped_actions = real_batch.actions.long().view(-1, 4)
        expected = torch.arange(
            4, device=grouped_actions.device
        ).expand_as(grouped_actions)
        if not torch.equal(grouped_actions, expected):
            raise RuntimeError(
                "real Critic replay lost ordered [0,1,2,3] action groups"
            )

        # Round the real budget to complete groups. Each minibatch also uses
        # complete groups, preventing a random split from losing ranking
        # supervision for a state.
        desired_real = max(4, round(train_samples * real_fraction / 4) * 4)
        real_count = min(len(real_batch.actions), desired_real)
        real_count -= real_count % 4
        real_groups = real_count // 4
        h2_count = min(len(h2_batch.actions), train_samples - real_count)
        if real_groups <= 0 or h2_count <= 0:
            raise RuntimeError("joint Critic sampling produced an empty source")

        real_group_perm = torch.randperm(
            len(real_batch.actions) // 4,
            device=real_batch.actions.device,
        )[:real_groups]
        real_index = (
            real_group_perm[:, None] * 4
            + torch.arange(4, device=real_batch.actions.device)[None, :]
        ).reshape(-1)
        h2_index = torch.randperm(
            len(h2_batch.actions), device=h2_batch.actions.device
        )[:h2_count]

        # Keep the requested source fraction in every optimizer step. The real
        # share is rounded to groups of four.
        minibatch_size = min(self.config.minibatch_size, h2_count + real_count)
        real_per_mb = max(4, round(minibatch_size * real_fraction / 4) * 4)
        h2_per_mb = max(1, minibatch_size - real_per_mb)
        evaluate_values = getattr(self.policy, "evaluate_values")
        totals = {
            "loss": 0.0,
            "h2_mse": 0.0,
            "real_mse": 0.0,
            "ranking": 0.0,
            "grad_norm": 0.0,
            "gradient_cosine": 0.0,
            "gradient_conflicts": 0.0,
        }
        steps = 0

        def pairwise_loss(prediction: Tensor, target: Tensor) -> Tensor:
            prediction = prediction.view(-1, 4)
            target = target.view(-1, 4)
            losses: list[Tensor] = []
            for left in range(4):
                for right in range(left + 1, 4):
                    target_delta = target[:, left] - target[:, right]
                    informative = target_delta.abs() > 1e-6
                    if bool(informative.any()):
                        predicted_delta = (
                            prediction[:, left] - prediction[:, right]
                        )
                        signed_margin = (
                            target_delta[informative].sign()
                            * predicted_delta[informative]
                            / ranking_temperature
                        )
                        losses.append(F.softplus(-signed_margin).mean())
            if not losses:
                return prediction.sum() * 0.0
            return torch.stack(losses).mean()

        for _ in range(self.config.epochs):
            epoch_h2 = h2_index[torch.randperm(
                h2_count, device=h2_index.device
            )]
            epoch_groups = real_group_perm[torch.randperm(
                real_groups, device=real_group_perm.device
            )]
            h2_cursor = group_cursor = 0
            while h2_cursor < h2_count or group_cursor < real_groups:
                h_index = epoch_h2[h2_cursor:h2_cursor + h2_per_mb]
                groups_per_mb = max(1, real_per_mb // 4)
                groups = epoch_groups[
                    group_cursor:group_cursor + groups_per_mb
                ]
                if len(h_index) == 0 or len(groups) == 0:
                    # Do not create a source-only tail step; those are exactly
                    # the alternating updates this method is meant to remove.
                    break
                r_index = (
                    groups[:, None] * 4
                    + torch.arange(4, device=groups.device)[None, :]
                ).reshape(-1)
                h2_cursor += len(h_index)
                group_cursor += len(groups)

                h_prediction = evaluate_values(
                    h2_batch.states[h_index], h2_batch.actions[h_index]
                ).float()
                r_prediction = evaluate_values(
                    real_batch.states[r_index], real_batch.actions[r_index]
                ).float()
                h_target = h2_batch.returns[h_index].float()
                r_target = real_batch.returns[r_index].float()
                h2_mse = F.mse_loss(h_prediction, h_target)
                real_mse = F.mse_loss(r_prediction, r_target)
                ranking = pairwise_loss(r_prediction, r_target)
                h2_objective = (
                    self.config.value_coef
                    * (1.0 - real_fraction)
                    * h2_mse
                )
                real_objective = self.config.value_coef * (
                    real_fraction * real_mse + ranking_coef * ranking
                )
                value_loss = h2_objective + real_objective
                self.optimizer.zero_grad(set_to_none=True)
                h2_gradients = torch.autograd.grad(
                    h2_objective,
                    self._critic_trainable,
                    allow_unused=True,
                )
                real_gradients = torch.autograd.grad(
                    real_objective,
                    self._critic_trainable,
                    allow_unused=True,
                )
                dot = h2_norm = real_norm = 0.0
                for h_gradient, r_gradient in zip(
                    h2_gradients, real_gradients
                ):
                    if h_gradient is not None:
                        h2_norm += float(h_gradient.detach().float().square().sum())
                    if r_gradient is not None:
                        real_norm += float(r_gradient.detach().float().square().sum())
                    if h_gradient is not None and r_gradient is not None:
                        dot += float((
                            h_gradient.detach().float()
                            * r_gradient.detach().float()
                        ).sum())
                cosine = dot / max(math.sqrt(h2_norm * real_norm), 1e-12)
                conflict = project_conflicting_gradients and dot < 0.0
                for parameter, h_gradient, r_gradient in zip(
                    self._critic_trainable,
                    h2_gradients,
                    real_gradients,
                ):
                    if h_gradient is None and r_gradient is None:
                        parameter.grad = None
                        continue
                    h_value = (
                        torch.zeros_like(r_gradient)
                        if h_gradient is None else h_gradient
                    )
                    r_value = (
                        torch.zeros_like(h_gradient)
                        if r_gradient is None else r_gradient
                    )
                    if conflict:
                        # Symmetric PCGrad: remove only the mutually harmful
                        # component. Source weights are already included in
                        # h2_objective/real_objective above.
                        h_projected = h_value - (
                            dot / max(real_norm, 1e-12)
                        ) * r_value
                        r_projected = r_value - (
                            dot / max(h2_norm, 1e-12)
                        ) * h_value
                        parameter.grad = h_projected + r_projected
                    else:
                        parameter.grad = h_value + r_value
                grad_norm = nn.utils.clip_grad_norm_(
                    self._critic_trainable, self.config.max_grad_norm
                )
                self.optimizer.step()
                totals["loss"] += float(
                    ((1.0 - real_fraction) * h2_mse
                     + real_fraction * real_mse
                     + ranking_coef * ranking).detach()
                )
                totals["h2_mse"] += float(h2_mse.detach())
                totals["real_mse"] += float(real_mse.detach())
                totals["ranking"] += float(ranking.detach())
                totals["grad_norm"] += float(grad_norm)
                totals["gradient_cosine"] += cosine
                totals["gradient_conflicts"] += float(conflict)
                steps += 1

        if steps == 0:
            raise RuntimeError("joint Critic update produced no mixed minibatch")

        @torch.no_grad()
        def explained_variance(batch: PPOBatch, index: Tensor) -> float:
            predictions = []
            for begin in range(0, len(index), self.config.minibatch_size):
                selected = index[begin:begin + self.config.minibatch_size]
                predictions.append(evaluate_values(
                    batch.states[selected], batch.actions[selected]
                ).float())
            prediction = torch.cat(predictions)
            target = batch.returns[index].float()
            variance = target.var(unbiased=False)
            if float(variance) < 1e-8:
                return 0.0
            return float(
                1.0 - (target - prediction).var(unbiased=False) / variance
            )

        return JointCriticMetrics(
            loss=totals["loss"] / steps,
            h2_mse=totals["h2_mse"] / steps,
            real_mse=totals["real_mse"] / steps,
            ranking_loss=totals["ranking"] / steps,
            h2_explained_variance=explained_variance(h2_batch, h2_index),
            real_explained_variance=explained_variance(real_batch, real_index),
            grad_norm=totals["grad_norm"] / steps,
            h2_samples=float(h2_count),
            real_samples=float(real_count),
            real_groups=float(real_groups),
            minibatches=float(steps),
            gradient_cosine=totals["gradient_cosine"] / steps,
            gradient_conflict_fraction=(
                totals["gradient_conflicts"] / steps
            ),
        )

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    @staticmethod
    def _clone_optimizer_value(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: PPOUpdater._clone_optimizer_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [PPOUpdater._clone_optimizer_value(item) for item in value]
        return copy.deepcopy(value)

    def snapshot_actor_transaction(self) -> dict:
        """Snapshot only Actor parameters and their Adam state."""
        return {
            "parameters": [
                parameter.detach().clone()
                for parameter in self._actor_trainable
            ],
            "optimizer_state": [
                self._clone_optimizer_value(
                    self.optimizer.state.get(parameter, {})
                )
                for parameter in self._actor_trainable
            ],
        }

    def restore_actor_transaction(self, snapshot: dict) -> None:
        parameters = snapshot["parameters"]
        optimizer_state = snapshot["optimizer_state"]
        if len(parameters) != len(self._actor_trainable):
            raise RuntimeError("Actor parameter set changed during transaction")
        with torch.no_grad():
            for parameter, saved in zip(self._actor_trainable, parameters):
                parameter.copy_(saved)
        for parameter, saved in zip(self._actor_trainable, optimizer_state):
            self.optimizer.state[parameter] = self._clone_optimizer_value(saved)

    def snapshot_training_transaction(self) -> dict:
        """Snapshot all PPO-owned parameters and optimizer state."""
        return {
            "parameters": [
                parameter.detach().clone()
                for parameter in self._all_trainable
            ],
            "optimizer": self._clone_optimizer_value(
                self.optimizer.state_dict()
            ),
        }

    def restore_training_transaction(self, snapshot: dict) -> None:
        parameters = snapshot["parameters"]
        if len(parameters) != len(self._all_trainable):
            raise RuntimeError("PPO parameter set changed during transaction")
        with torch.no_grad():
            for parameter, saved in zip(self._all_trainable, parameters):
                parameter.copy_(saved)
        self.optimizer.load_state_dict(snapshot["optimizer"])

    def load_state_dict(self, state: dict) -> None:
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])


def _explained_variance(values: Tensor, targets: Tensor) -> float:
    target_var = torch.var(targets)
    if not torch.isfinite(target_var) or target_var.item() < 1e-8:
        return 0.0
    residual = torch.var(targets - values)
    return float((1.0 - residual / target_var).item())


def _unique_trainable(
    parameters: list[nn.Parameter],
) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result
