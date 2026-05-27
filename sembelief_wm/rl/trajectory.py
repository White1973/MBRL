"""Trajectory data structures for RL.

These are the universal data contracts between trajectory producers
(real env collectors, imagined rollout) and consumers (GAE, PPO).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class Trajectory:
    """A batch of trajectories, possibly variable-length (padded to H).

    All tensors share a (B, H) prefix:
        B = number of parallel trajectories
        H = max horizon (padded if variable-length)

    states:        (B, H, *state_shape) — arbitrary state representation
    actions:       (B, H) discrete or (B, H, A) continuous
    rewards:       (B, H) scalar per step
    dones:         (B, H) bool — episode termination flags
    log_probs:     (B, H) log pi(a|s) at collection time
    values:        (B, H+1) V(s) estimates, last entry is bootstrap value
    mask:          (B, H) optional — 1.0 for valid steps, 0.0 for padding.
                   None means all steps are valid (fixed-horizon, no padding).
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    log_probs: Tensor
    values: Tensor       # (B, H+1) — includes bootstrap value at index H
    mask: Tensor | None = None  # (B, H) — None means all valid

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]


@dataclass(frozen=True)
class PPOBatch:
    """Flattened, ready-to-train batch for PPO.

    All tensors have shape (N, ...) where N = total number of valid steps.
    No time structure — just a bag of transitions.
    Padding steps are stripped during construction — PPO never sees them.
    """

    states: Tensor           # (N, *state_shape)
    actions: Tensor          # (N,) or (N, A)
    advantages: Tensor       # (N,)
    returns: Tensor          # (N,)
    old_log_probs: Tensor    # (N,)
    old_values: Tensor       # (N,)
