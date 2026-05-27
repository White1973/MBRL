"""Target-construction helpers for Phase 1 training."""
from __future__ import annotations

import torch
from torch import Tensor

from .losses import binary_reward_targets, valid_time_mask
from ..types import TrainingWindow


def _window_valid_mask(window: TrainingWindow) -> Tensor:
    """Return the base `(B, H)` mask for real posterior timesteps."""
    return valid_time_mask(window.valid_lengths, horizon=window.horizon)


def posterior_valid_mask_from_window(window: TrainingWindow) -> Tensor:
    """Build the `(B, H)` mask for all valid posterior beliefs.

    A posterior is valid wherever a real observation exists.  This mask
    is used by SIGReg (all real posteriors participate in regularization),
    including `Z_0` and the final posterior aligned to the terminal
    observation.
    """
    return _window_valid_mask(window)


def dynamics_mask_from_window(window: TrainingWindow) -> Tensor:
    """Build the `(B, H)` validity mask for dynamics supervision.

    Dynamics supervision applies to real transitions only. Under the
    `T_env + 1` time axis, `Z_0` is a valid posterior but not a real
    transition target, so the first timestep of the initial window is
    excluded.
    """
    mask = _window_valid_mask(window)
    if window.is_initial_window and mask.shape[1] > 0:
        mask = mask.clone()
        mask[:, 0] = False
    return mask


def reward_valid_mask_from_window(window: TrainingWindow) -> Tensor:
    """Build the `(B, H)` validity mask for reward supervision.

    Reward labels align to `Z_t -> r_{t-1}`. Step 0 of non-initial windows
    is valid because the previous reward comes from the preceding window;
    step 0 of the initial window is invalid because `r_{-1}` does not
    exist.
    """
    mask = _window_valid_mask(window)
    if mask.shape[1] > 0:
        mask = mask.clone()
        mask[:, 0] = mask[:, 0] & window.has_prev_reward.to(dtype=torch.bool)
    return mask


def reward_targets_from_window(window: TrainingWindow) -> tuple[Tensor, Tensor]:
    """Construct aligned binary reward targets and mask from a training window.

    Output contract:
    - `targets[:, t]` is the binary label aligned to `Z_t`
    - `mask[:, t]` says whether that label is valid

    In particular:
    - `targets[:, 0] = 1[prev_rewards > 0]`
    - `mask[:, 0] = has_prev_reward`
    - for `t >= 1`, `targets[:, t] = 1[rewards[:, t-1] > 0]`
    """

    if window.prev_rewards.ndim != 1:
        raise ValueError(
            f"prev_rewards must have shape (B,), got {tuple(window.prev_rewards.shape)}."
        )
    if window.has_prev_reward.ndim != 1:
        raise ValueError(
            "has_prev_reward must have shape (B,), got "
            f"{tuple(window.has_prev_reward.shape)}."
        )

    batch_size, horizon = window.batch_size, window.horizon
    if window.prev_rewards.shape[0] != batch_size:
        raise ValueError(
            "prev_rewards batch size must match window batch size, got "
            f"{window.prev_rewards.shape[0]} vs {batch_size}."
        )
    if window.has_prev_reward.shape[0] != batch_size:
        raise ValueError(
            "has_prev_reward batch size must match window batch size, got "
            f"{window.has_prev_reward.shape[0]} vs {batch_size}."
        )

    reward_binary = binary_reward_targets(window.rewards)
    targets = torch.zeros_like(window.rewards)
    if horizon > 1:
        targets[:, 1:] = reward_binary[:, :-1]
    targets[:, 0] = binary_reward_targets(window.prev_rewards)

    mask = reward_valid_mask_from_window(window)
    return targets, mask
