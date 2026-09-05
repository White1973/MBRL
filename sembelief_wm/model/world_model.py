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
import torch.nn.functional as F
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
    terminal_reward_loss,
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
class OpenLoopOutputs:
    """Action-only rollout and aligned real-trajectory supervision.

    ``prior.beliefs[:, k]`` is generated recursively from the grounded first
    posterior and actions only.  Observations are used exclusively by the
    detached ``posterior_targets`` branch.
    """

    prior: BeliefTrajectory
    posterior_targets: Tensor
    reward_logits: Tensor
    reward_targets: Tensor
    dynamics_mask: Tensor
    reward_mask: Tensor
    dynamics_weights: Tensor
    reward_weights: Tensor


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
    open_loop: OpenLoopOutputs | None = None


@dataclass(frozen=True)
class ActionAuxiliaryOutputs:
    """Action-aware losses and sufficient statistics for aggregation."""

    delta_cosine_sum: Tensor
    inverse_action_sum: Tensor
    weight_sum: Tensor
    prior_correct: Tensor
    posterior_correct: Tensor
    prior_confusion: Tensor
    posterior_confusion: Tensor


@dataclass(frozen=True)
class ObservationAnchorOutputs:
    """Frozen-observation anchoring losses and normalizers."""

    state_sum: Tensor
    state_weight: Tensor
    delta_sum: Tensor
    delta_weight: Tensor


@dataclass(frozen=True)
class VJEPATeacherOutputs:
    """Per-slot frozen-V-JEPA teacher losses and normalizers.

    ``prior`` is the actual future-state objective: Z^pri_t is produced from
    Z_{t-1} and a_{t-1}, then compared to the V-JEPA features of o_t.  The
    posterior statistic is useful for verifying that Qwen observations retain
    the same spatial semantics, but it is independent from the prior target.
    """

    prior_sum: Tensor
    prior_weight: Tensor
    posterior_sum: Tensor
    posterior_weight: Tensor
    delta_sum: Tensor
    delta_weight: Tensor


class WorldModel(nn.Module):
    """Minimal world-model assembly for Phase 1 training."""

    def __init__(self, config: Config, backbone: TransitionBackbone) -> None:
        super().__init__()
        self.config = config
        self.transition = TransitionCore(config, backbone)
        self.reward_head = RewardHead(config)
        self.inverse_action_head = (
            nn.Linear(
                config.belief.num_slots * config.hidden_dim,
                config.env.num_actions,
            )
            if config.training.inverse_action_coef > 0
            else None
        )
        anchor_enabled = (
            config.training.observation_anchor_coef > 0
            or config.training.observation_delta_anchor_coef > 0
        )
        self.observation_anchor_head = (
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            if anchor_enabled
            else None
        )
        if self.observation_anchor_head is not None:
            # Tokens and belief slots already share D. Identity initialization
            # makes the new branch a conservative continuation from a Phase-1
            # checkpoint while still allowing basis adaptation.
            nn.init.eye_(self.observation_anchor_head.weight)
            self.observation_anchor_head.weight.requires_grad_(
                config.training.observation_anchor_projection_trainable
            )
        teacher_enabled = (
            config.training.vjepa_teacher_prior_coef > 0.0
            or config.training.vjepa_teacher_posterior_coef > 0.0
            or config.training.vjepa_teacher_delta_coef > 0.0
        )
        if teacher_enabled and config.encoder.semantic_teacher_type != "vjepa2":
            raise ValueError(
                "V-JEPA teacher coefficients require "
                "encoder.semantic_teacher_type='vjepa2'."
            )
        self.vjepa_teacher_head = (
            nn.Linear(
                config.hidden_dim,
                config.encoder.semantic_teacher_dim,
                bias=False,
            )
            if teacher_enabled
            else None
        )
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

    def compute_action_auxiliaries(
        self,
        *,
        start_beliefs: Tensor,
        prior_beliefs: Tensor,
        posterior_beliefs: Tensor,
        actions: Tensor,
        mask: Tensor,
    ) -> ActionAuxiliaryOutputs:
        """Supervise action-dependent change without an absolute-state shortcut."""
        target_delta = posterior_beliefs.detach() - start_beliefs.detach()
        prior_delta = prior_beliefs - start_beliefs
        target_rms = target_delta.float().pow(2).mean(dim=(-2, -1)).sqrt()
        action_valid = (actions >= 0) & (actions < self.config.env.num_actions)
        informative = mask.bool() & action_valid & (
            target_rms >= float(self.config.training.delta_min_rms)
        )
        weight = informative.float()
        weight_sum = weight.sum()
        prior_flat = prior_delta.float().flatten(start_dim=2)
        target_flat = target_delta.float().flatten(start_dim=2)
        cosine = F.cosine_similarity(prior_flat, target_flat, dim=-1, eps=1e-6)
        delta_cosine_sum = ((1.0 - cosine) * weight).sum()

        zero = delta_cosine_sum.new_zeros(())
        if self.inverse_action_head is None or weight_sum.item() == 0:
            confusion = zero.new_zeros(
                (self.config.env.num_actions, self.config.env.num_actions)
            )
            return ActionAuxiliaryOutputs(
                delta_cosine_sum=delta_cosine_sum,
                inverse_action_sum=zero,
                weight_sum=weight_sum,
                prior_correct=zero,
                posterior_correct=zero,
                prior_confusion=confusion,
                posterior_confusion=confusion.clone(),
            )

        # The detached posterior branch teaches the small classifier what a
        # real action delta looks like; the prior branch then pushes dynamics
        # to make the logged action identifiable without moving the teacher.
        # Preserve the fixed belief-slot order.  Mean pooling erased which
        # spatial slot changed and made opposite directions indistinguishable.
        post_features = target_delta.flatten(start_dim=2)
        prior_features = prior_delta.flatten(start_dim=2)
        post_logits = self.inverse_action_head(post_features.detach())
        prior_logits = self.inverse_action_head(prior_features)
        safe_actions = actions.clamp(0, self.config.env.num_actions - 1)
        post_ce = F.cross_entropy(
            post_logits.flatten(0, 1), safe_actions.flatten(), reduction="none"
        ).reshape_as(weight)
        prior_ce = F.cross_entropy(
            prior_logits.flatten(0, 1), safe_actions.flatten(), reduction="none"
        ).reshape_as(weight)
        mode = self.config.training.inverse_action_mode
        if mode == "joint":
            inverse_action_sum = (0.5 * (post_ce + prior_ce) * weight).sum()
        elif mode == "prior_frozen":
            if any(parameter.requires_grad for parameter in self.inverse_action_head.parameters()):
                raise RuntimeError(
                    "prior_frozen requires a frozen inverse_action_head"
                )
            # Stage B: the decoder was trained exclusively on detached real
            # posterior deltas.  Freeze it and use its fixed decision geometry
            # to push only the predicted prior delta toward the logged action.
            inverse_action_sum = (prior_ce * weight).sum()
        else:
            raise ValueError(f"Unsupported inverse_action_mode: {mode}")
        prior_correct = (
            (prior_logits.argmax(dim=-1) == safe_actions).float() * weight
        ).sum()
        posterior_correct = (
            (post_logits.argmax(dim=-1) == safe_actions).float() * weight
        ).sum()
        valid_flat = informative.flatten()
        target_flat = safe_actions.flatten()[valid_flat]
        prior_pred_flat = prior_logits.argmax(dim=-1).flatten()[valid_flat]
        post_pred_flat = post_logits.argmax(dim=-1).flatten()[valid_flat]
        num_actions = self.config.env.num_actions
        prior_confusion = torch.bincount(
            target_flat * num_actions + prior_pred_flat,
            minlength=num_actions * num_actions,
        ).reshape(num_actions, num_actions)
        posterior_confusion = torch.bincount(
            target_flat * num_actions + post_pred_flat,
            minlength=num_actions * num_actions,
        ).reshape(num_actions, num_actions)
        return ActionAuxiliaryOutputs(
            delta_cosine_sum=delta_cosine_sum,
            inverse_action_sum=inverse_action_sum,
            weight_sum=weight_sum,
            prior_correct=prior_correct,
            posterior_correct=posterior_correct,
            prior_confusion=prior_confusion,
            posterior_confusion=posterior_confusion,
        )

    def compute_observation_anchor(
        self,
        *,
        posterior_beliefs: Tensor,
        observation_tokens: Tensor,
        posterior_mask: Tensor,
    ) -> ObservationAnchorOutputs:
        """Preserve frozen V-JEPA state and change information in posteriors.

        Alignment is per slot, not mean pooled. Observation tokens are frozen
        targets and never receive gradient. Delta supervision is restricted to
        consecutive observations within the current BPTT window.
        """
        zero = posterior_beliefs.new_zeros(())
        if self.observation_anchor_head is None:
            return ObservationAnchorOutputs(zero, zero, zero, zero)
        if posterior_beliefs.shape != observation_tokens.shape:
            raise ValueError(
                "Observation anchoring requires aligned posterior/visual slots, "
                f"got {tuple(posterior_beliefs.shape)} and "
                f"{tuple(observation_tokens.shape)}"
            )

        predicted = self.observation_anchor_head(posterior_beliefs)
        targets = observation_tokens.detach().to(dtype=predicted.dtype)
        state_per_slot = 1.0 - F.cosine_similarity(
            predicted.float(), targets.float(), dim=-1, eps=1e-6
        )
        state_mask = posterior_mask.bool().unsqueeze(-1).expand_as(state_per_slot)
        state_sum = (state_per_slot * state_mask.float()).sum()
        state_weight = state_mask.float().sum()

        if posterior_beliefs.shape[1] < 2:
            return ObservationAnchorOutputs(
                state_sum, state_weight, zero, zero
            )
        predicted_delta = predicted[:, 1:] - predicted[:, :-1]
        target_delta = targets[:, 1:] - targets[:, :-1]
        target_rms = target_delta.float().pow(2).mean(dim=(-2, -1)).sqrt()
        consecutive = posterior_mask[:, 1:].bool() & posterior_mask[:, :-1].bool()
        informative = consecutive & (
            target_rms >= float(
                self.config.training.observation_delta_min_rms
            )
        )
        delta_per_slot = 1.0 - F.cosine_similarity(
            predicted_delta.float(), target_delta.float(), dim=-1, eps=1e-6
        )
        delta_mask = informative.unsqueeze(-1).expand_as(delta_per_slot)
        delta_sum = (delta_per_slot * delta_mask.float()).sum()
        delta_weight = delta_mask.float().sum()
        return ObservationAnchorOutputs(
            state_sum, state_weight, delta_sum, delta_weight
        )

    def compute_vjepa_teacher_losses(
        self,
        *,
        prior_beliefs: Tensor,
        posterior_beliefs: Tensor,
        start_beliefs: Tensor,
        teacher_tokens: Tensor | None,
        teacher_mask: Tensor | None,
        prev_teacher_tokens: Tensor | None,
        prev_teacher_mask: Tensor | None,
        dynamics_mask: Tensor,
        posterior_mask: Tensor,
    ) -> VJEPATeacherOutputs:
        """Match Qwen-belief states to frozen V-JEPA spatial semantics.

        The teacher is a detached `(K, D_vjepa)` spatial grid.  No teacher
        tokens are concatenated to Qwen inputs and no learned query token is
        introduced.  The delta branch compares the predicted prior change
        from the *grounded previous belief* with the matched V-JEPA change,
        which makes it specifically action-conditioned rather than another
        low-latent-MSE objective.
        """
        zero = prior_beliefs.new_zeros(())
        if self.vjepa_teacher_head is None:
            return VJEPATeacherOutputs(zero, zero, zero, zero, zero, zero)
        if teacher_tokens is None:
            raise RuntimeError(
                "Frozen V-JEPA teacher loss is enabled, but this training "
                "window has no semantic_teacher_tokens. Refuse to silently "
                "fall back to latent-only WM optimization."
            )
        if teacher_mask is None:
            teacher_mask = torch.ones(
                teacher_tokens.shape[:2], dtype=torch.bool,
                device=teacher_tokens.device,
            )
        if prior_beliefs.shape[:3] != teacher_tokens.shape[:3]:
            raise ValueError(
                "V-JEPA teacher requires time- and slot-aligned grids, got "
                f"belief={tuple(prior_beliefs.shape)} "
                f"teacher={tuple(teacher_tokens.shape)}."
            )
        if teacher_tokens.shape[-1] != self.config.encoder.semantic_teacher_dim:
            raise ValueError(
                "V-JEPA teacher feature dimension does not match config: "
                f"tokens={teacher_tokens.shape[-1]} expected="
                f"{self.config.encoder.semantic_teacher_dim}."
            )
        if start_beliefs.shape != prior_beliefs.shape:
            raise ValueError(
                "start_beliefs must align with prior beliefs for V-JEPA "
                f"delta supervision, got {tuple(start_beliefs.shape)} and "
                f"{tuple(prior_beliefs.shape)}."
            )

        targets = teacher_tokens.detach().to(dtype=prior_beliefs.dtype)
        prior_pred = self.vjepa_teacher_head(prior_beliefs)
        posterior_pred = self.vjepa_teacher_head(posterior_beliefs)
        state_shape = prior_pred.shape[:-1]
        if tuple(teacher_mask.shape) != tuple(state_shape[:2]):
            raise ValueError(
                "teacher_mask must have shape (B, T), got "
                f"{tuple(teacher_mask.shape)} for {tuple(state_shape)}."
            )

        prior_per_slot = 1.0 - F.cosine_similarity(
            prior_pred.float(), targets.float(), dim=-1, eps=1e-6
        )
        prior_valid = dynamics_mask.bool() & teacher_mask.bool()
        prior_slot_mask = prior_valid.unsqueeze(-1).expand_as(prior_per_slot)
        prior_sum = (prior_per_slot * prior_slot_mask.float()).sum()
        prior_weight = prior_slot_mask.float().sum()

        posterior_per_slot = 1.0 - F.cosine_similarity(
            posterior_pred.float(), targets.float(), dim=-1, eps=1e-6
        )
        posterior_valid = posterior_mask.bool() & teacher_mask.bool()
        posterior_slot_mask = posterior_valid.unsqueeze(-1).expand_as(
            posterior_per_slot
        )
        posterior_sum = (
            posterior_per_slot * posterior_slot_mask.float()
        ).sum()
        posterior_weight = posterior_slot_mask.float().sum()

        if prev_teacher_tokens is None:
            return VJEPATeacherOutputs(
                prior_sum,
                prior_weight,
                posterior_sum,
                posterior_weight,
                zero,
                zero,
            )
        if prev_teacher_mask is None:
            prev_teacher_mask = torch.ones(
                prior_beliefs.shape[0], dtype=torch.bool,
                device=prior_beliefs.device,
            )
        if prev_teacher_tokens.shape != teacher_tokens[:, 0].shape:
            raise ValueError(
                "prev_teacher_tokens must align with one teacher frame, got "
                f"{tuple(prev_teacher_tokens.shape)} and "
                f"{tuple(teacher_tokens[:, 0].shape)}."
            )

        teacher_starts = torch.cat(
            [prev_teacher_tokens.unsqueeze(1), targets[:, :-1]], dim=1
        )
        teacher_start_mask = torch.cat(
            [prev_teacher_mask.bool().unsqueeze(1), teacher_mask[:, :-1].bool()],
            dim=1,
        )
        # Detaching the grounded predecessor prevents the posterior branch
        # from moving just to make a prior delta look easier. Gradients flow
        # only through Z^pri_t and the compact belief->teacher predictor.
        predicted_delta = prior_pred - self.vjepa_teacher_head(
            start_beliefs
        ).detach()
        target_delta = targets - teacher_starts
        target_rms = target_delta.float().pow(2).mean(dim=(-2, -1)).sqrt()
        delta_valid = (
            dynamics_mask.bool()
            & teacher_mask.bool()
            & teacher_start_mask
            & (
                target_rms
                >= float(self.config.training.vjepa_teacher_delta_min_rms)
            )
        )
        delta_per_slot = 1.0 - F.cosine_similarity(
            predicted_delta.float(), target_delta.float(), dim=-1, eps=1e-6
        )
        delta_slot_mask = delta_valid.unsqueeze(-1).expand_as(delta_per_slot)
        delta_sum = (delta_per_slot * delta_slot_mask.float()).sum()
        delta_weight = delta_slot_mask.float().sum()
        return VJEPATeacherOutputs(
            prior_sum,
            prior_weight,
            posterior_sum,
            posterior_weight,
            delta_sum,
            delta_weight,
        )

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
        reward_targets, reward_mask = reward_targets_from_window(
            window, threshold=self.config.reward.success_reward_threshold
        )
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

        open_loop = self._compute_open_loop_outputs(
            window=window,
            rollout=rollout,
            dynamics_target=dynamics_target,
            reward_targets=reward_targets,
            reward_mask=reward_mask,
            dynamics_mask=dynamics_mask,
        )

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

        # Auxiliary terminal-step reward BCE. Only the window that actually
        # contains Z_T participates, and its label is the aligned r_{T-1}.
        aux_reward, aux_loss_sum, aux_weight_sum, aux_num_pos, aux_num_total = terminal_reward_loss(
            posterior_reward_logits,
            reward_targets=reward_targets,
            reward_mask=reward_mask,
            valid_lengths=window.valid_lengths,
            terminal_mask=window.terminal_mask,
            pos_weight=self.config.reward.pos_weight,
        )
        # compose/trainer applies lambda_reward outside reward_sum. Keep the
        # auxiliary multiplier independent to avoid lambda_reward being applied
        # twice (the previous implementation scaled it by lambda_reward here).
        # Legacy code hard-coded this to 20, which combined with reward
        # ``pos_weight`` pushed sparse Sokoban reward heads toward predicting
        # success everywhere.  Keep the auxiliary explicit and independently
        # configurable; new training defaults to zero.
        aux_weight = float(
            getattr(self.config.reward, "terminal_aux_weight", 0.0)
        )
        reward = reward + aux_weight * aux_reward

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
        # Fold the auxiliary terminal-step loss into reward_sum so the trainer's
        # per-transition normalized total (total_sup_sum) actually includes it
        # in backward. Without this, the aux term added to `reward` (mean) is
        # bypassed by the trainer's total computation and never trains the head.
        rew_sum = rew_sum + aux_weight * aux_loss_sum
        rew_wsum = rew_wsum + aux_weight * aux_weight_sum

        if open_loop is None:
            open_dynamics = zero
            prior_reward = zero
            open_dyn_sum = zero
            open_dyn_wsum = zero
            prior_rew_sum = zero
            prior_rew_wsum = zero
        else:
            open_dynamics = dynamics_loss(
                open_loop.prior.beliefs,
                open_loop.posterior_targets,
                weights=open_loop.dynamics_weights,
                mask=open_loop.dynamics_mask,
            )
            open_dyn_sum, open_dyn_wsum = dynamics_loss_sum(
                open_loop.prior.beliefs,
                open_loop.posterior_targets,
                weights=open_loop.dynamics_weights,
                mask=open_loop.dynamics_mask,
            )
            prior_reward = reward_bce_loss(
                open_loop.reward_logits,
                open_loop.reward_targets,
                pos_weight=self.config.reward.pos_weight,
                weights=open_loop.reward_weights,
                mask=open_loop.reward_mask,
            )
            prior_rew_sum, prior_rew_wsum = reward_bce_loss_sum(
                open_loop.reward_logits,
                open_loop.reward_targets,
                pos_weight=self.config.reward.pos_weight,
                weights=open_loop.reward_weights,
                mask=open_loop.reward_mask,
            )

        losses = compose_phase1_losses(
            dynamics=dynamics,
            reward=reward,
            open_dynamics=open_dynamics,
            prior_reward=prior_reward,
            config=self.config,
            sigreg_terms=sigreg_terms,
            dynamics_sum=dyn_sum,
            reward_sum=rew_sum,
            dynamics_weight_sum=dyn_wsum,
            reward_weight_sum=rew_wsum,
            open_dynamics_sum=open_dyn_sum,
            prior_reward_sum=prior_rew_sum,
            open_dynamics_weight_sum=open_dyn_wsum,
            prior_reward_weight_sum=prior_rew_wsum,
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
            open_loop=open_loop,
        )

    def _compute_open_loop_outputs(
        self,
        *,
        window: TrainingWindow,
        rollout: Phase1Rollout,
        dynamics_target: Tensor,
        reward_targets: Tensor,
        reward_mask: Tensor,
        dynamics_mask: Tensor,
    ) -> OpenLoopOutputs | None:
        """Build a true open-loop branch starting from grounded ``Z_0``.

        The first posterior is detached before it becomes the rollout start,
        preventing the open-loop objective from changing the observation
        grounder merely to make future prediction easier.  Reward logits are
        evaluated on detached prior beliefs, so their BCE updates only the
        reward head; the dynamics objective remains the sole gradient path
        from this branch into the transition model.
        """
        requested = self.config.training.open_loop_horizon
        enabled = (
            self.config.training.open_dynamics_coef > 0.0
            or self.config.training.prior_reward_coef > 0.0
        )
        num_steps = min(max(requested, 0), max(window.horizon - 1, 0))
        if not enabled or num_steps == 0:
            return None

        start = BeliefState(slots=rollout.posterior.beliefs[:, 0].detach())
        actions = window.actions[:, :num_steps]
        prior = self.rollout_prior(start, actions, env_ids=window.env_ids)
        posterior_targets = dynamics_target[:, 1 : num_steps + 1].detach()

        # Deliberately stop reward gradients at imagined beliefs.  This avoids
        # a shortcut in which dynamics produces easy-to-classify but physically
        # inaccurate latent states.
        prior_reward_logits = self._predict_reward_sequence(
            prior.beliefs.detach()
        )
        aligned_reward_targets = reward_targets[:, 1 : num_steps + 1]
        aligned_dynamics_mask = dynamics_mask[:, 1 : num_steps + 1]
        aligned_reward_mask = reward_mask[:, 1 : num_steps + 1]

        steps = torch.arange(
            num_steps,
            device=prior.beliefs.device,
            dtype=torch.float32,
        )
        dynamics_weights = torch.pow(
            torch.full_like(steps, self.config.training.open_dynamics_decay),
            steps,
        ).unsqueeze(0).expand(window.batch_size, -1)
        reward_weights = torch.pow(
            torch.full_like(steps, self.config.training.prior_reward_decay),
            steps,
        ).unsqueeze(0).expand(window.batch_size, -1)

        return OpenLoopOutputs(
            prior=prior,
            posterior_targets=posterior_targets,
            reward_logits=prior_reward_logits,
            reward_targets=aligned_reward_targets,
            dynamics_mask=aligned_dynamics_mask,
            reward_mask=aligned_reward_mask,
            dynamics_weights=dynamics_weights,
            reward_weights=reward_weights,
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
