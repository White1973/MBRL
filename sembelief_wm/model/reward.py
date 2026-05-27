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
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1),
        )

    @property
    def spec(self) -> str:
        """Human-readable summary of the reward head contract."""
        return (
            f"RewardHead(readout={self.readout.spec.mode}, "
            f"slots={self.readout.spec.input_slots}, "
            f"hidden_dim={self.readout.spec.hidden_dim})"
        )

    def forward(self, belief: BeliefState | Tensor) -> Tensor:
        """Map belief slots to scalar reward logits.

        Inputs:
            belief: `(B, K_belief, D)` or `BeliefState`

        Returns:
            reward_logits: `(B,)`
        """
        pooled = self.readout(belief)
        logits = self.net(pooled)
        return logits.squeeze(-1)
