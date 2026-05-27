"""Protocols for trajectory collectors.

These define what a collector needs from its dependencies (world model,
reward predictor, belief grounding). The concrete WorldModel class
satisfies DynamicsModel + RewardPredictor; adapters bridge the gap
when signatures differ.
"""
from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor


class DynamicsModel(Protocol):
    """Latent dynamics: advance belief by one action step."""

    def prior_step(
        self,
        prev_belief: Any,
        prev_actions: Tensor,
        **kwargs: Any,
    ) -> Any:
        """(belief, action) -> next_belief in latent space."""
        ...


class RewardPredictor(Protocol):
    """Predict scalar reward from a latent belief."""

    def predict_reward(self, belief: Any) -> Tensor:
        """belief -> reward logits (B,) or (B, 1)."""
        ...


class BeliefGrounder(Protocol):
    """Ground initial beliefs from observation data for rollout starts.

    This bridges the gap between the replay buffer (which stores tokenized
    observations) and the imagination engine (which needs belief states).
    """

    def ground_beliefs(
        self,
        obs_batch: Any,
        **kwargs: Any,
    ) -> Any:
        """Observation data -> grounded belief states (B, K, D)."""
        ...
