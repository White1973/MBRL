"""Reward transform functions for converting world model logits to scalars.

These are used by ImaginedCollector's reward_transform callable.
Extracted from agent/phase2.py to decouple the new RL stack from legacy code.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def sigmoid_affine(
    reward_logits: Tensor,
    positive_value: float,
    negative_value: float,
    **kwargs: Any,
) -> Tensor:
    """r = negative + sigmoid(logit) * (positive - negative)."""
    probs = torch.sigmoid(reward_logits)
    return negative_value + probs * (positive_value - negative_value)


def raw_sigmoid(reward_logits: Tensor, **kwargs: Any) -> Tensor:
    """r = sigmoid(logit) - 0.5, range [-0.5, 0.5]."""
    return torch.sigmoid(reward_logits) - 0.5


def clipped_logit(reward_logits: Tensor, lo: float = -5.0, hi: float = 5.0, **kwargs: Any) -> Tensor:
    """r = clamp(logit, lo, hi)."""
    return torch.clamp(reward_logits, lo, hi)


def terminal_success(
    reward_logits: Tensor,
    positive_value: float,
    negative_value: float,
    *,
    step_index: int,
    horizon: int,
    **kwargs: Any,
) -> Tensor:
    """Sparse terminal reward: step penalty until final step, then success reward.

    Intermediate steps always receive ``negative_value`` (e.g. -0.1). The final
    step uses ``sigmoid_affine`` so the reward head's success-probability logit
    is mapped to ``negative_value`` / ``positive_value``.
    """
    if step_index < horizon - 1:
        return torch.full_like(reward_logits, negative_value)
    return sigmoid_affine(reward_logits, positive_value, negative_value)


def terminal_success_scaled(
    reward_logits: Tensor,
    positive_value: float,
    negative_value: float,
    *,
    step_index: int,
    horizon: int,
    scale: float = 1.0,
    **kwargs: Any,
) -> Tensor:
    """Sparse terminal reward with configurable magnitude scaling.

    Same as ``terminal_success`` but scales the success reward magnitude by
    ``scale`` (keeping the step penalty at negative_value). This keeps the
    reward head's discrimination (sigmoid(logit)) intact while reducing the
    reward magnitude so the value function can fit the lower-variance returns.

    Example: positive_value=10.9, negative_value=-0.1, scale=0.1 makes the
    endpoint range [-0.1, +1.0] instead of [-0.1, +10.9].
    """
    if step_index < horizon - 1:
        return torch.full_like(reward_logits, negative_value)
    scaled_pos = negative_value + (positive_value - negative_value) * scale
    return negative_value + torch.sigmoid(reward_logits) * (scaled_pos - negative_value)


def terminal_success_conservative(
    reward_logits: Tensor,
    positive_value: float,
    negative_value: float,
    *,
    step_index: int,
    horizon: int,
    confidence_floor: float,
    low_confidence_scale: float = 0.1,
    **kwargs: Any,
) -> Tensor:
    """Endpoint reward that suppresses low-confidence false positives.

    Below ``confidence_floor`` only a small fraction of the calibrated
    probability is retained for ranking.  Above the floor a linear ramp adds
    the main success signal.  This remains continuous for PPO while preventing
    probabilities near the natural base rate from becoming positive rewards.
    """
    if step_index < horizon - 1:
        return torch.full_like(reward_logits, negative_value)
    if not 0.0 <= confidence_floor < 1.0:
        raise ValueError("confidence_floor must be in [0, 1)")
    if not 0.0 <= low_confidence_scale <= 1.0:
        raise ValueError("low_confidence_scale must be in [0, 1]")
    probabilities = torch.sigmoid(reward_logits)
    high_confidence = torch.clamp(
        (probabilities - confidence_floor) / (1.0 - confidence_floor),
        min=0.0,
        max=1.0,
    )
    conservative_probability = (
        low_confidence_scale * probabilities
        + (1.0 - low_confidence_scale) * high_confidence
    )
    return negative_value + conservative_probability * (
        positive_value - negative_value
    )


def per_transition_success_conservative(
    reward_logits: Tensor,
    positive_value: float,
    negative_value: float,
    *,
    confidence_floor: float,
    low_confidence_scale: float = 0.1,
    scale: float = 1.0,
    **kwargs: Any,
) -> Tensor:
    """Threshold-consistent success reward evaluated on every transition.

    Unlike ``terminal_success_conservative``, this transform has no knowledge
    of the imagined fragment index.  A calibrated per-transition terminal
    detector may therefore reward an H1/H2 success immediately.  Reward and
    termination deliberately use the *same* calibrated threshold: every state
    that terminates as success receives a positive reward, and every state
    below the threshold receives the step penalty.  ``scale`` applies to the
    complete environment reward, preserving its sign and positive/negative
    ratio (for example +1/-0.1 becomes +0.1/-0.01 at scale=0.1).
    """
    del kwargs
    if not 0.0 <= confidence_floor < 1.0:
        raise ValueError("confidence_floor must be in [0, 1)")
    if not 0.0 <= low_confidence_scale <= 1.0:
        raise ValueError("low_confidence_scale must be in [0, 1]")
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    probabilities = torch.sigmoid(reward_logits)
    return torch.where(
        probabilities >= confidence_floor,
        torch.full_like(probabilities, positive_value * scale),
        torch.full_like(probabilities, negative_value * scale),
    )


def make_reward_transform(
    mapping: str,
    positive_value: float = 1.0,
    negative_value: float = -1.0,
    scale: float = 1.0,
    confidence_floor: float = 0.5,
    low_confidence_scale: float = 0.1,
):
    """Create a reward transform callable from a mapping name.

    Args:
        mapping: One of 'sigmoid_affine', 'raw_sigmoid', 'clipped_logit',
            'terminal_success', 'terminal_success_scaled',
            'terminal_success_conservative',
            'per_transition_success_conservative'.
        positive_value: Positive reward value (for sigmoid_affine / terminal_success).
        negative_value: Negative reward value (for sigmoid_affine / terminal_success).

    Returns:
        Callable: (reward_logits: Tensor, *, step_index, horizon) -> Tensor
    """
    if scale < 0.0:
        raise ValueError("scale must be non-negative")
    if mapping == "raw_sigmoid":
        return raw_sigmoid
    if mapping == "clipped_logit":
        return clipped_logit
    if mapping == "terminal_success":
        def _terminal_transform(reward_logits: Tensor, **kwargs: Any) -> Tensor:
            return terminal_success(
                reward_logits, positive_value, negative_value, **kwargs
            )
        return _terminal_transform
    if mapping == "terminal_success_scaled":
        def _terminal_scaled_transform(reward_logits: Tensor, **kwargs: Any) -> Tensor:
            return terminal_success_scaled(
                reward_logits,
                positive_value,
                negative_value,
                scale=scale,
                **kwargs,
            )
        return _terminal_scaled_transform
    if mapping == "terminal_success_conservative":
        def _terminal_conservative_transform(
            reward_logits: Tensor, **kwargs: Any
        ) -> Tensor:
            return terminal_success_conservative(
                reward_logits,
                positive_value,
                negative_value,
                confidence_floor=confidence_floor,
                low_confidence_scale=low_confidence_scale,
                **kwargs,
            )
        return _terminal_conservative_transform
    if mapping == "per_transition_success_conservative":
        def _per_transition_conservative_transform(
            reward_logits: Tensor, **kwargs: Any
        ) -> Tensor:
            return per_transition_success_conservative(
                reward_logits,
                positive_value,
                negative_value,
                confidence_floor=confidence_floor,
                low_confidence_scale=low_confidence_scale,
                scale=scale,
                **kwargs,
            )
        return _per_transition_conservative_transform
    # sigmoid_affine (default)
    def _transform(reward_logits: Tensor, **kwargs: Any) -> Tensor:
        return sigmoid_affine(reward_logits, positive_value, negative_value)
    return _transform
