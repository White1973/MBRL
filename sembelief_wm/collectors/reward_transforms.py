"""Reward transform functions for converting world model logits to scalars.

These are used by ImaginedCollector's reward_transform callable.
Extracted from agent/phase2.py to decouple the new RL stack from legacy code.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sigmoid_affine(
    reward_logits: Tensor,
    positive_value: float,
    negative_value: float,
) -> Tensor:
    """r = negative + sigmoid(logit) * (positive - negative)."""
    probs = torch.sigmoid(reward_logits)
    return negative_value + probs * (positive_value - negative_value)


def raw_sigmoid(reward_logits: Tensor) -> Tensor:
    """r = sigmoid(logit) - 0.5, range [-0.5, 0.5]."""
    return torch.sigmoid(reward_logits) - 0.5


def clipped_logit(reward_logits: Tensor, lo: float = -5.0, hi: float = 5.0) -> Tensor:
    """r = clamp(logit, lo, hi)."""
    return torch.clamp(reward_logits, lo, hi)


def make_reward_transform(
    mapping: str,
    positive_value: float = 1.0,
    negative_value: float = -1.0,
):
    """Create a reward transform callable from a mapping name.

    Args:
        mapping: One of 'sigmoid_affine', 'raw_sigmoid', 'clipped_logit'.
        positive_value: Positive reward value (for sigmoid_affine).
        negative_value: Negative reward value (for sigmoid_affine).

    Returns:
        Callable: (reward_logits: Tensor) -> Tensor
    """
    if mapping == "raw_sigmoid":
        return raw_sigmoid
    if mapping == "clipped_logit":
        return clipped_logit
    # sigmoid_affine (default)
    def _transform(reward_logits: Tensor) -> Tensor:
        return sigmoid_affine(reward_logits, positive_value, negative_value)
    return _transform
