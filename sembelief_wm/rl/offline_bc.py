"""Compatibility boundary for the frozen-H2 baseline.

The trusted run used ``offline_bc_steps=0``.  The later, substantially larger
offline-BC implementation is intentionally not part of this baseline tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor, nn


@dataclass(frozen=True)
class OfflineBCConfig:
    steps: int = 0
    batch_size: int = 32
    cache_size: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("offline BC steps must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("offline BC batch_size must be positive")
        if self.cache_size < 0:
            raise ValueError("offline BC cache_size must be non-negative")
        if self.lr <= 0.0:
            raise ValueError("offline BC learning rate must be positive")


class OfflineBehaviorCloner:
    """Reject BC experiments that are outside the locked baseline."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        sample_batch_fn: Callable[[int], tuple[Tensor, Tensor]],
        validation_sample_batch_fn: (
            Callable[[int], tuple[Tensor, Tensor]] | None
        ) = None,
        config: OfflineBCConfig,
        capture_behavior_reference: bool = True,
    ) -> None:
        del validation_sample_batch_fn
        self.sample_batch_fn = sample_batch_fn
        self.policy = policy
        self.config = config
        self.capture_behavior_reference = capture_behavior_reference

    def fit(self) -> dict[str, float]:
        if self.config.steps != 0:
            raise RuntimeError(
                "Offline BC is intentionally excluded from the trusted "
                "frozen-H2 baseline. Use a separate experimental worktree."
            )
        if self.capture_behavior_reference:
            capture = getattr(self.policy, "capture_behavior_reference", None)
            if callable(capture):
                capture()
        return {
            "bc/loss": 0.0,
            "bc/accuracy": 0.0,
            "bc/entropy": 0.0,
            "bc/steps": 0.0,
        }

    def sample_rehearsal_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        return self.sample_batch_fn(batch_size)
