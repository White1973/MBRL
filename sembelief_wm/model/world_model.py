"""World-model assembly for SemBelief-WM.

This module wires together the transition core, reward head, target
construction, and Phase 1 loss computation. It intentionally does not own
the concrete visual encoder or Qwen backbone wrapper; those remain injected
dependencies behind the `TransitionBackbone` interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .belief import LearnedInitialBelief
from ..config import Config
from ..ema import EMATeacher
from ..train.losses import (
    Phase1Losses,
    compose_phase1_losses,
    dynamics_loss,
    dynamics_loss_sum,
    local_time_weights,
    reward_bce_loss,
    reward_bce_loss_sum,
)
from .reward import RewardHead
from ..train.sigreg import SIGRegLoss, SIGRegTerms
from ..train.targets import (
    dynamics_mask_from_window,
    posterior_valid_mask_from_window,
    reward_targets_from_window,
)
from .transition import TransitionBackbone, TransitionCore
from ..types import BeliefState, BeliefTrajectory, TrainingWindow


@dataclass(frozen=True)
class Phase1Rollout:
    """Aligned prior/posterior rollout over one training window."""

    posterior: BeliefTrajectory
    prior: BeliefTrajectory


@dataclass(frozen=True)
class Phase1Outputs:
    """Structured Phase 1 outputs for training and logging."""

    rollout: Phase1Rollout
    posterior_reward_logits: Tensor
    prior_reward_logits: Tensor
    reward_targets: Tensor
    dynamics_mask: Tensor          # (B, T) — valid dynamics supervision steps
    posterior_valid_mask: Tensor    # (B, T) — valid posterior beliefs (for SIGReg)
    reward_mask: Tensor
    sigreg_terms: SIGRegTerms
    losses: Phase1Losses


class WorldModel(nn.Module):
    """Minimal world-model assembly for Phase 1 training."""

    def __init__(self, config: Config, backbone: TransitionBackbone) -> None:
        super().__init__()
        self.config = config
        self.transition = TransitionCore(config, backbone)
        self.reward_head = RewardHead(config)
        self.initial_belief = LearnedInitialBelief(config)
        self.sigreg = SIGRegLoss(config) if config.anti_collapse.use_sigreg else None
        self.ema_teacher = (
            EMATeacher(config, self.transition)
            if config.anti_collapse.use_ema_target
            else None
        )

    def get_initial_belief(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BeliefState:
        """Return the learned initial belief for the start of an episode."""
        return self.initial_belief.get(batch_size, device=device, dtype=dtype)

    def posterior_step(
        self,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        observation_tokens: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefState:
        return self.transition.posterior_step(
            prev_belief=prev_belief,
            prev_actions=prev_actions,
            observation_tokens=observation_tokens,
            env_ids=env_ids,
        )

    def prior_step(
        self,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefState:
        return self.transition.prior_step(
            prev_belief=prev_belief,
            prev_actions=prev_actions,
            env_ids=env_ids,
        )

    def rollout_prior(
        self,
        start_belief: BeliefState,
        action_seq: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefTrajectory:
        return self.transition.rollout_prior(
            start_belief=start_belief,
            action_seq=action_seq,
            env_ids=env_ids,
        )

    def predict_reward(self, belief: BeliefState | Tensor) -> Tensor:
        return self._predict_reward_sequence(belief)

    def flush_sigreg_buffer(self) -> None:
        if self.sigreg is not None:
            self.sigreg.flush()

    def update_ema(self) -> None:
        """Update EMA teacher after optimizer step (call from trainer)."""
        if self.ema_teacher is not None:
            self.ema_teacher.update(self.transition)

    def compute_phase1_outputs(
        self,
        window: TrainingWindow,
    ) -> Phase1Outputs:
        """Compute supervision losses for one window.

        Regularization (SIGReg / EMA variance) is NOT computed here.
        The trainer handles regularization at step-level to avoid
        per-window dilution and ensure correct scaling.
        """
        # Compute EMA targets before the online rollout builds an autograd graph.
        # The EMA teacher temporarily swaps trainable transition parameters to
        # their shadow values, so doing this first avoids mutating parameters
        # that are still referenced by the online graph.
        ema_rollout = self._rollout_window_ema(window) if self.ema_teacher is not None else None
        rollout = self._rollout_window(window)
        reward_targets, reward_mask = reward_targets_from_window(window)
        dynamics_mask = dynamics_mask_from_window(window)
        posterior_valid_mask = posterior_valid_mask_from_window(window)

        posterior_reward_logits = self._predict_reward_sequence(rollout.posterior.beliefs)
        prior_reward_logits = self._predict_reward_sequence(rollout.prior.beliefs)

        time_weights = local_time_weights(
            valid_lengths=window.valid_lengths,
            decay=self.config.curriculum.horizon_decay,
        )

        # Dynamics target can come from either the online posterior or the
        # EMA posterior teacher, independent of regularization choice.
        if ema_rollout is not None:
            dynamics_target = ema_rollout.posterior.beliefs.detach()
        else:
            dynamics_target = rollout.posterior.beliefs.detach()

        # Dummy SIGRegTerms — real regularization is computed at step-level by trainer
        zero = rollout.posterior.beliefs.new_zeros(())
        sigreg_terms = SIGRegTerms(
            ep=zero, var=zero,
            ep_unscaled=zero, var_unscaled=zero,
            mean_std=zero, min_std=zero,
            num_current=0, num_total=0,
        )

        # Mean-reduced losses (for logging)
        dynamics = dynamics_loss(
            rollout.prior.beliefs,
            dynamics_target,
            weights=time_weights,
            mask=dynamics_mask,
        )
        reward = self._reward_loss_from_source(
            posterior_reward_logits=posterior_reward_logits,
            prior_reward_logits=prior_reward_logits,
            reward_targets=reward_targets,
            reward_mask=reward_mask,
            time_weights=time_weights,
        )

        # Sum losses (for per-transition normalization in trainer)
        dyn_sum, dyn_wsum = dynamics_loss_sum(
            rollout.prior.beliefs,
            dynamics_target,
            weights=time_weights,
            mask=dynamics_mask,
        )
        rew_sum, rew_wsum = self._reward_loss_sum_from_source(
            posterior_reward_logits=posterior_reward_logits,
            prior_reward_logits=prior_reward_logits,
            reward_targets=reward_targets,
            reward_mask=reward_mask,
            time_weights=time_weights,
        )

        losses = compose_phase1_losses(
            dynamics=dynamics,
            reward=reward,
            config=self.config,
            sigreg_terms=sigreg_terms,
            dynamics_sum=dyn_sum,
            reward_sum=rew_sum,
            dynamics_weight_sum=dyn_wsum,
            reward_weight_sum=rew_wsum,
        )

        return Phase1Outputs(
            rollout=rollout,
            posterior_reward_logits=posterior_reward_logits,
            prior_reward_logits=prior_reward_logits,
            reward_targets=reward_targets,
            dynamics_mask=dynamics_mask,
            posterior_valid_mask=posterior_valid_mask,
            reward_mask=reward_mask,
            sigreg_terms=sigreg_terms,
            losses=losses,
        )

    @staticmethod
    def _extract_valid_beliefs(
        beliefs: Tensor, valid_mask: Tensor,
    ) -> Tensor:
        """Select valid posterior beliefs using (B, T) mask, returning (N_valid, K, D)."""
        # beliefs: (B, T, K, D), valid_mask: (B, T)
        flat_beliefs = beliefs.reshape(-1, *beliefs.shape[2:])  # (B*T, K, D)
        flat_mask = valid_mask.reshape(-1)  # (B*T,)
        return flat_beliefs[flat_mask]  # (N_valid, K, D)

    def _rollout_window(self, window: TrainingWindow) -> Phase1Rollout:
        posterior_slots: list[Tensor] = []
        prior_slots: list[Tensor] = []

        belief = window.prev_belief
        prev_actions = window.prev_actions

        for step in range(window.horizon):
            observation_tokens = window.obs_tokens[:, step]
            posterior = self.posterior_step(
                prev_belief=belief,
                prev_actions=prev_actions,
                observation_tokens=observation_tokens,
                env_ids=window.env_ids,
            )
            prior = self.prior_step(
                prev_belief=belief,
                prev_actions=prev_actions,
                env_ids=window.env_ids,
            )
            posterior_slots.append(posterior.slots)
            prior_slots.append(prior.slots)

            belief = posterior
            prev_actions = window.actions[:, step]

        posterior_beliefs = torch.stack(posterior_slots, dim=1)
        prior_beliefs = torch.stack(prior_slots, dim=1)
        return Phase1Rollout(
            posterior=BeliefTrajectory(
                beliefs=posterior_beliefs,
                actions=window.actions,
            ),
            prior=BeliefTrajectory(
                beliefs=prior_beliefs,
                actions=window.actions,
            ),
        )

    @torch.no_grad()
    def _rollout_window_ema(self, window: TrainingWindow) -> Phase1Rollout:
        """Rollout EMA posterior targets over one window.

        This keeps the existing online `window.prev_belief` as the rollout
        start, then applies EMA updates for the posterior steps within the
        current window. That gives a stabilized target without maintaining a
        second full hidden-state trajectory across the whole episode.
        """
        assert self.ema_teacher is not None
        posterior_slots: list[Tensor] = []
        belief = window.prev_belief
        prev_actions = window.prev_actions
        for step in range(window.horizon):
            observation_tokens = window.obs_tokens[:, step]
            posterior = self.ema_teacher.posterior_step(
                prev_belief=belief,
                prev_actions=prev_actions,
                observation_tokens=observation_tokens,
                env_ids=window.env_ids,
            )
            posterior_slots.append(posterior.slots)
            belief = posterior
            prev_actions = window.actions[:, step]
        posterior_beliefs = torch.stack(posterior_slots, dim=1)
        return Phase1Rollout(
            posterior=BeliefTrajectory(beliefs=posterior_beliefs, actions=window.actions),
            prior=BeliefTrajectory(beliefs=posterior_beliefs, actions=window.actions),  # unused
        )

    def _predict_reward_sequence(self, belief: BeliefState | Tensor) -> Tensor:
        slots = belief.slots if isinstance(belief, BeliefState) else belief

        if slots.ndim == 3:
            return self.reward_head(slots)
        if slots.ndim != 4:
            raise ValueError(
                "predict_reward expects belief tensors with shape (B, K, D) or (B, T, K, D), "
                f"got {tuple(slots.shape)}."
            )

        batch_size, horizon, num_slots, hidden_dim = slots.shape
        flat_slots = slots.reshape(batch_size * horizon, num_slots, hidden_dim)
        flat_logits = self.reward_head(flat_slots)
        return flat_logits.reshape(batch_size, horizon)

    def _reward_loss_from_source(
        self,
        *,
        posterior_reward_logits: Tensor,
        prior_reward_logits: Tensor,
        reward_targets: Tensor,
        reward_mask: Tensor,
        time_weights: Tensor,
    ) -> Tensor:
        source = self.config.reward.supervision_source
        pw = self.config.reward.pos_weight
        if source == "posterior":
            return reward_bce_loss(
                posterior_reward_logits,
                reward_targets,
                pos_weight=pw,
                weights=time_weights,
                mask=reward_mask,
            )
        if source == "prior":
            return reward_bce_loss(
                prior_reward_logits,
                reward_targets,
                pos_weight=pw,
                weights=time_weights,
                mask=reward_mask,
            )
        if source == "both":
            posterior_loss = reward_bce_loss(
                posterior_reward_logits,
                reward_targets,
                pos_weight=pw,
                weights=time_weights,
                mask=reward_mask,
            )
            prior_loss = reward_bce_loss(
                prior_reward_logits,
                reward_targets,
                pos_weight=pw,
                weights=time_weights,
                mask=reward_mask,
            )
            return 0.5 * (posterior_loss + prior_loss)

        raise ValueError(f"Unsupported reward supervision source: {source}")

    def _reward_loss_sum_from_source(
        self,
        *,
        posterior_reward_logits: Tensor,
        prior_reward_logits: Tensor,
        reward_targets: Tensor,
        reward_mask: Tensor,
        time_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (weighted_sum, weight_sum) for reward loss."""
        source = self.config.reward.supervision_source
        pw = self.config.reward.pos_weight
        if source == "posterior":
            return reward_bce_loss_sum(
                posterior_reward_logits, reward_targets,
                pos_weight=pw, weights=time_weights, mask=reward_mask,
            )
        if source == "prior":
            return reward_bce_loss_sum(
                prior_reward_logits, reward_targets,
                pos_weight=pw, weights=time_weights, mask=reward_mask,
            )
        if source == "both":
            ps, pw1 = reward_bce_loss_sum(
                posterior_reward_logits, reward_targets,
                pos_weight=pw, weights=time_weights, mask=reward_mask,
            )
            prs, pw2 = reward_bce_loss_sum(
                prior_reward_logits, reward_targets,
                pos_weight=pw, weights=time_weights, mask=reward_mask,
            )
            return 0.5 * (ps + prs), pw1  # same mask, same weight sum
        raise ValueError(f"Unsupported reward supervision source: {source}")
