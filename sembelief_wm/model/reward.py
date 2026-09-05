"""Reward prediction head for SemBelief-WM.

This module closes the minimal world-model public API by providing
`predict_reward(belief) -> reward_logits`.

It intentionally stays small:
- read out slot beliefs to a pooled vector `(B, D)`
- map pooled vectors to scalar reward logits `(B,)`

Loss construction remains outside this module.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .belief import BeliefReadout, BeliefReadoutMode
from ..config import Config
from ..types import BeliefState


class RewardHead(nn.Module):
    """Predict reward logits from belief slots.

    The default path uses the configured reward readout, but callers may
    override the readout mode to support later ablations without changing
    the rest of the head contract.
    """

    def __init__(
        self,
        config: Config,
        *,
        readout_mode: BeliefReadoutMode | None = None,
    ) -> None:
        super().__init__()
        self.readout = BeliefReadout.from_config(config, mode=readout_mode)
        D = config.hidden_dim
        head_hidden_dim = config.reward.head_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        if head_hidden_dim is None:
            # Legacy architecture retained for old Phase 1 checkpoints.
            self.net = nn.Sequential(
                nn.Linear(D, D),
                nn.GELU(),
                nn.Linear(D, 1),
            )
            self.compact_net = None
        elif head_hidden_dim == 0:
            self.net = None
            self.compact_net = nn.Linear(D, 1)
        elif head_hidden_dim > 0:
            self.net = None
            self.compact_net = nn.Sequential(
                nn.Linear(D, head_hidden_dim),
                nn.GELU(),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            raise ValueError(
                "reward.head_hidden_dim must be None, 0, or a positive integer, "
                f"got {head_hidden_dim}."
            )

    @property
    def spec(self) -> str:
        """Human-readable summary of the reward head contract."""
        return (
            f"RewardHead(readout={self.readout.spec.mode}, "
            f"slots={self.readout.spec.input_slots}, "
            f"hidden_dim={self.readout.spec.hidden_dim}, "
            f"head_hidden_dim={self.head_hidden_dim})"
        )

    def forward_pooled(self, pooled: Tensor) -> Tensor:
        """Classify already-pooled belief features of shape ``(B, D)``."""
        if pooled.ndim != 2:
            raise ValueError(
                f"forward_pooled expects (B, D), got {tuple(pooled.shape)}."
            )
        classifier = self.net if self.net is not None else self.compact_net
        assert classifier is not None
        return classifier(pooled).squeeze(-1)

    def forward(self, belief: BeliefState | Tensor) -> Tensor:
        """Map belief slots to scalar reward logits.

        Inputs:
            belief: `(B, K_belief, D)` or `BeliefState`

        Returns:
            reward_logits: `(B,)`
        """
        pooled = self.readout(belief)
        return self.forward_pooled(pooled)
