"""Phase 1 loss functions for SemBelief-WM."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from .sigreg import SIGRegTerms
from ..types import BeliefState


@dataclass(frozen=True)
class Phase1Losses:
    """Structured Phase 1 loss outputs.

    Contains both mean-reduced values (for logging) and weighted sums
    (for per-transition normalization in the trainer).
    """

    total: Tensor         # mean-reduced total (for logging only)
    dynamics: Tensor      # mean-reduced dynamics loss
    reward: Tensor        # mean-reduced reward loss
    sigreg_ep: Tensor
    sigreg_var: Tensor
    # Weighted sums for per-transition normalization
    dynamics_sum: Tensor  # sum of (weighted) per-step dynamics losses
    reward_sum: Tensor    # sum of (weighted) per-step reward losses
    dynamics_weight_sum: Tensor  # sum of weights for dynamics
    reward_weight_sum: Tensor    # sum of weights for reward


def compose_phase1_losses(
    *,
    dynamics: Tensor,
    reward: Tensor,
    config: Config,
    sigreg_terms: SIGRegTerms | None = None,
    dynamics_sum: Tensor | None = None,
    reward_sum: Tensor | None = None,
    dynamics_weight_sum: Tensor | None = None,
    reward_weight_sum: Tensor | None = None,
) -> Phase1Losses:
    """Compose scalar Phase 1 terms into the total weighted objective."""

    if sigreg_terms is None:
        sigreg_ep = dynamics.new_zeros(())
        sigreg_var = dynamics.new_zeros(())
    else:
        sigreg_ep = sigreg_terms.ep
        sigreg_var = sigreg_terms.var

    reg_total = dynamics.new_zeros(())
    if config.anti_collapse.use_sigreg:
        reg_total = (
            reg_total
            + config.sigreg.lambda_ep * sigreg_ep
            + config.sigreg.lambda_var * sigreg_var
        )
    if config.anti_collapse.use_ema_variance:
        reg_total = reg_total + config.ema.lambda_var * sigreg_var

    total = dynamics + config.training.lambda_reward * reward + reg_total

    # Default sum/weight fields to zero if not provided (backward compat)
    zero = dynamics.new_zeros(())
    return Phase1Losses(
        total=total,
        dynamics=dynamics,
        reward=reward,
        sigreg_ep=sigreg_ep,
        sigreg_var=sigreg_var,
        dynamics_sum=dynamics_sum if dynamics_sum is not None else zero,
        reward_sum=reward_sum if reward_sum is not None else zero,
        dynamics_weight_sum=dynamics_weight_sum if dynamics_weight_sum is not None else zero,
        reward_weight_sum=reward_weight_sum if reward_weight_sum is not None else zero,
    )


def valid_time_mask(valid_lengths: Tensor, horizon: int | None = None) -> Tensor:
    """Build a `(B, T)` mask from per-sample valid lengths."""

    if valid_lengths.ndim != 1:
        raise ValueError(
            f"valid_lengths must have shape (B,), got {tuple(valid_lengths.shape)}."
        )
    if horizon is None:
        horizon = int(valid_lengths.max().item())

    steps = torch.arange(horizon, device=valid_lengths.device)
    return steps.unsqueeze(0) < valid_lengths.unsqueeze(1)


def local_time_weights(valid_lengths: Tensor, decay: float) -> Tensor:
    """Return window-local decay weights `decay^t_local` masked by validity."""

    mask = valid_time_mask(valid_lengths)
    steps = torch.arange(mask.shape[1], device=valid_lengths.device, dtype=torch.float32)
    weights = torch.pow(
        torch.full_like(steps, fill_value=decay),
        steps,
    )
    return weights.unsqueeze(0) * mask.to(dtype=weights.dtype)


def binary_reward_targets(rewards: Tensor) -> Tensor:
    """Convert raw environment rewards to binary success labels."""

    return (rewards > 0).to(dtype=rewards.dtype)


def compute_pos_weight(
    reward_targets: Tensor,
    mask: Tensor | None = None,
    clamp_range: tuple[float, float] = (1.0, 20.0),
) -> float:
    """Compute data-driven pos_weight = neg_count / pos_count from masked targets."""
    if mask is not None:
        flat = reward_targets[mask.bool()]
    else:
        flat = reward_targets.flatten()
    pos = flat.sum().item()
    neg = flat.numel() - pos
    if pos < 1:
        return clamp_range[1]
    return max(clamp_range[0], min(neg / pos, clamp_range[1]))


def _belief_slots_tensor(belief: BeliefState | Tensor) -> Tensor:
    return belief.slots if isinstance(belief, BeliefState) else belief


def _masked_weighted_mean(
    values: Tensor,
    *,
    weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    if mask is not None:
        values = values * mask.to(dtype=values.dtype)

    if weights is None:
        if mask is None:
            return values.mean()
        denom = mask.sum().clamp_min(1).to(dtype=values.dtype)
        return values.sum() / denom

    if mask is not None:
        weights = weights * mask.to(dtype=weights.dtype)

    if weights.shape != values.shape:
        raise ValueError(
            f"weights must match values shape, got weights={tuple(weights.shape)} "
            f"values={tuple(values.shape)}."
        )

    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def _masked_weighted_sum_and_denom(
    values: Tensor,
    *,
    weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (weighted_sum, weight_sum) for per-transition normalization."""
    if mask is not None:
        values = values * mask.to(dtype=values.dtype)

    if weights is None:
        if mask is None:
            count = torch.tensor(
                float(values.numel()), device=values.device, dtype=values.dtype,
            )
            return values.sum(), count
        count = mask.sum().clamp_min(1).to(dtype=values.dtype)
        return values.sum(), count

    if mask is not None:
        weights = weights * mask.to(dtype=weights.dtype)

    return (values * weights).sum(), weights.sum().clamp_min(1e-8)


def _dynamics_per_step(
    z_prior: BeliefState | Tensor,
    z_posterior_target: BeliefState | Tensor,
) -> Tensor:
    """Compute per-step squared L2 distances, returning (B, T) tensor."""
    prior = _belief_slots_tensor(z_prior)
    target = _belief_slots_tensor(z_posterior_target)

    if prior.shape != target.shape:
        raise ValueError(
            f"dynamics_loss requires matching shapes, got prior={tuple(prior.shape)} "
            f"target={tuple(target.shape)}."
        )

    if prior.ndim == 3:
        prior = prior.unsqueeze(1)
        target = target.unsqueeze(1)
    elif prior.ndim != 4:
        raise ValueError(
            "dynamics_loss expects belief tensors with shape (B, K, D) or (B, T, K, D), "
            f"got {tuple(prior.shape)}."
        )

    return (prior - target).pow(2).mean(dim=(-2, -1))


def dynamics_loss(
    z_prior: BeliefState | Tensor,
    z_posterior_target: BeliefState | Tensor,
    *,
    weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Squared L2 dynamics loss on aligned prior/posterior beliefs (mean-reduced)."""
    per_step = _dynamics_per_step(z_prior, z_posterior_target)
    return _masked_weighted_mean(per_step, weights=weights, mask=mask)


def dynamics_loss_sum(
    z_prior: BeliefState | Tensor,
    z_posterior_target: BeliefState | Tensor,
    *,
    weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Squared L2 dynamics loss returning (weighted_sum, weight_sum)."""
    per_step = _dynamics_per_step(z_prior, z_posterior_target)
    return _masked_weighted_sum_and_denom(per_step, weights=weights, mask=mask)


def _reward_bce_per_step(
    reward_logits: Tensor,
    reward_targets: Tensor,
    *,
    pos_weight: float,
) -> Tensor:
    """Compute per-step BCE, returning (B, T) tensor."""
    if reward_logits.shape != reward_targets.shape:
        raise ValueError(
            f"reward_bce_loss requires matching shapes, got logits={tuple(reward_logits.shape)} "
            f"targets={tuple(reward_targets.shape)}."
        )

    logits = reward_logits
    targets = reward_targets.to(dtype=logits.dtype)

    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
        targets = targets.unsqueeze(1)
    elif logits.ndim != 2:
        raise ValueError(
            "reward_bce_loss expects tensors with shape (B,) or (B, T), "
            f"got {tuple(reward_logits.shape)}."
        )

    pos_weight_tensor = torch.tensor(
        pos_weight,
        dtype=logits.dtype,
        device=logits.device,
    )
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight_tensor,
    )


def reward_bce_loss(
    reward_logits: Tensor,
    reward_targets: Tensor,
    *,
    pos_weight: float,
    weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """BCEWithLogits reward loss on aligned binary targets (mean-reduced)."""
    per_step = _reward_bce_per_step(reward_logits, reward_targets, pos_weight=pos_weight)
    return _masked_weighted_mean(per_step, weights=weights, mask=mask)


def reward_bce_loss_sum(
    reward_logits: Tensor,
    reward_targets: Tensor,
    *,
    pos_weight: float,
    weights: Tensor | None = None,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """BCEWithLogits reward loss returning (weighted_sum, weight_sum)."""
    per_step = _reward_bce_per_step(reward_logits, reward_targets, pos_weight=pos_weight)
    return _masked_weighted_sum_and_denom(per_step, weights=weights, mask=mask)


def covariance_loss(embeddings: Tensor) -> Tensor:
    """Penalize off-diagonal covariance to resist dimensional collapse.

    The input must be flattened valid posterior samples with shape `(N, D)`.
    This follows the VICReg-style covariance penalty: square the off-diagonal
    entries of the centered covariance matrix and average them.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"covariance_loss expects (N, D), got {tuple(embeddings.shape)}."
        )
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(embeddings.shape[0] - 1, 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diag.pow(2).mean()


def compute_phase1_loss(
    *,
    z_prior: BeliefState | Tensor,
    z_posterior_target: BeliefState | Tensor,
    reward_logits: Tensor,
    reward_targets: Tensor,
    dynamics_mask: Tensor,
    reward_mask: Tensor,
    config: Config,
    sigreg_terms: SIGRegTerms | None = None,
) -> Phase1Losses:
    """Compute the current Phase 1 loss.

    The caller is responsible for:
    - aligning `reward_targets` with belief time (`Z_t -> r_{t-1}`);
      `targets.reward_targets_from_window(...)` is the canonical helper
    - applying stop-gradient to `z_posterior_target`
    - setting `reward_mask[:, 0] = False` when the first state has no reward label
    """

    time_weights = local_time_weights(
        valid_lengths=dynamics_mask.sum(dim=1),
        decay=config.curriculum.horizon_decay,
    )
    dynamics = dynamics_loss(
        z_prior,
        z_posterior_target,
        weights=time_weights,
        mask=dynamics_mask,
    )
    reward = reward_bce_loss(
        reward_logits,
        reward_targets,
        pos_weight=config.reward.pos_weight,
        weights=time_weights,
        mask=reward_mask,
    )
    return compose_phase1_losses(
        dynamics=dynamics,
        reward=reward,
        config=config,
        sigreg_terms=sigreg_terms,
    )
