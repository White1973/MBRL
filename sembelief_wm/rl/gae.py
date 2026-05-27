"""Generalized Advantage Estimation.

Pure functions, no state, no side effects. Can be used with any trajectory
source (real env, imagined rollout, etc.).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .trajectory import Trajectory, PPOBatch


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute GAE advantages and discounted returns.

    Args:
        rewards: (B, H) per-step rewards.
        values:  (B, H+1) value estimates. values[:, -1] is the bootstrap value.
        dones:   (B, H) episode termination flags (1.0 = done).
        gamma:   discount factor.
        gae_lambda: GAE lambda.
        mask:    (B, H) optional — 1.0 for valid steps, 0.0 for padding.
                 Padding steps get advantages=0, returns=0.

    Returns:
        advantages: (B, H)
        returns:    (B, H)
    """
    B, H = rewards.shape
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in range(H - 1, -1, -1):
        not_done = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage

        # Gate by mask *inside* the loop so padding steps can't leak into
        # valid steps via the next_advantage carry.  A padding step produces
        # delta=0 and next_advantage=0, preventing backward contamination.
        if mask is not None:
            step_valid = mask[:, t]
            delta = delta * step_valid
            next_advantage = next_advantage * step_valid

        advantages[:, t] = next_advantage

    returns = advantages + values[:, :-1]

    # Also zero returns on padding (values[:, :-1] may be non-zero on padding)
    if mask is not None:
        returns = returns * mask

    return advantages, returns


def trajectory_to_ppo_batch(
    trajectory: Trajectory,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> PPOBatch:
    """Convert a Trajectory into a flattened PPOBatch with computed advantages.

    This is the bridge between trajectory producers and the PPO updater.
    If the trajectory has a mask, only valid steps are included in the batch
    (padding is stripped, not passed through).
    """
    advantages, returns = compute_gae(
        trajectory.rewards,
        trajectory.values,
        trajectory.dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        mask=trajectory.mask,
    )

    B, H = trajectory.actions.shape[:2]
    state_shape = trajectory.states.shape[2:]

    # Flatten everything to (B*H, ...)
    flat_states = trajectory.states.reshape(B * H, *state_shape)
    flat_actions = trajectory.actions.reshape(B * H, *trajectory.actions.shape[2:])
    flat_advantages = advantages.reshape(B * H)
    flat_returns = returns.reshape(B * H)
    flat_old_log_probs = trajectory.log_probs.reshape(B * H)
    flat_old_values = trajectory.values[:, :-1].reshape(B * H)

    # If mask exists, strip padding steps so PPO never sees them
    if trajectory.mask is not None:
        valid = trajectory.mask.reshape(B * H).bool()
        flat_states = flat_states[valid]
        flat_actions = flat_actions[valid]
        flat_advantages = flat_advantages[valid]
        flat_returns = flat_returns[valid]
        flat_old_log_probs = flat_old_log_probs[valid]
        flat_old_values = flat_old_values[valid]

    return PPOBatch(
        states=flat_states,
        actions=flat_actions,
        advantages=flat_advantages,
        returns=flat_returns,
        old_log_probs=flat_old_log_probs,
        old_values=flat_old_values,
    )
