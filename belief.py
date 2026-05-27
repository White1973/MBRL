"""Belief initialization and readout utilities.

This module intentionally stays small:
- belief initialization defines how a fresh episode/window starts
- belief readout maps slot beliefs (B, K, D) -> pooled vectors (B, D)

Transition dynamics are implemented elsewhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config
from ..types import BeliefState

BeliefReadoutMode = Literal["mean_pool", "attention_pool", "learned_query"]


def zero_belief(
    batch_size: int,
    config: Config,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> BeliefState:
    """Create an explicit zero-initialized belief state.

    Kept for backward compatibility. Prefer LearnedInitialBelief for training.
    """
    slots = torch.zeros(
        batch_size,
        config.belief.num_slots,
        config.hidden_dim,
        device=device,
        dtype=dtype,
    )
    return BeliefState(slots=slots)


class LearnedInitialBelief(nn.Module):
    """Learned initial belief state (matches main branch).

    A trainable parameter initialized with truncated normal, providing a
    non-degenerate starting point for SIGReg statistics and prior prediction.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.initial_belief = nn.Parameter(
            torch.empty(1, config.belief.num_slots, config.hidden_dim)
        )
        nn.init.trunc_normal_(
            self.initial_belief, std=1.0 / math.sqrt(config.hidden_dim)
        )

    def get(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BeliefState:
        """Return the learned initial belief expanded to batch size."""
        belief = self.initial_belief.expand(batch_size, -1, -1)
        if dtype is not None and belief.dtype != dtype:
            belief = belief.to(dtype=dtype)
        # Device is handled by nn.Module.to() — the parameter lives on the
        # module's device already.
        return BeliefState(slots=belief)


@dataclass(frozen=True)
class BeliefReadoutSpec:
    """Static description of the slot-to-vector readout contract."""

    mode: BeliefReadoutMode
    input_slots: int
    hidden_dim: int


class BeliefReadout(nn.Module):
    """Map slot beliefs (B, K, D) to pooled vectors (B, D).

    `mean_pool` is the current baseline. Other modes remain available behind
    the same contract so later ablations do not change downstream module APIs.
    """

    def __init__(
        self,
        *,
        mode: BeliefReadoutMode,
        num_slots: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self._spec = BeliefReadoutSpec(
            mode=mode,
            input_slots=num_slots,
            hidden_dim=hidden_dim,
        )

        if self.mode == "mean_pool":
            self.score = None
            self.query = None
        elif self.mode == "attention_pool":
            self.score = nn.Linear(hidden_dim, 1)
            self.query = None
        elif self.mode == "learned_query":
            self.score = None
            self.query = nn.Parameter(torch.randn(hidden_dim))
        else:
            raise ValueError(f"Unsupported belief readout mode: {self.mode}")

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        mode: BeliefReadoutMode | None = None,
    ) -> BeliefReadout:
        """Build a readout using the shared belief geometry from config.

        `mode` is passed explicitly by the caller when a head wants a readout
        contract different from the default reward readout.
        """
        return cls(
            mode=config.reward.readout if mode is None else mode,
            num_slots=config.belief.num_slots,
            hidden_dim=config.hidden_dim,
        )

    @property
    def spec(self) -> BeliefReadoutSpec:
        """Expose the static slot-to-vector contract for downstream inspection."""
        return self._spec

    def forward(self, belief: BeliefState | Tensor) -> Tensor:
        slots = belief.slots if isinstance(belief, BeliefState) else belief

        if slots.ndim != 3:
            raise ValueError(
                f"Belief readout expects slots with shape (B, K, D), got {tuple(slots.shape)}."
            )

        if self.mode == "mean_pool":
            return slots.mean(dim=1)

        if self.mode == "attention_pool":
            # Align dtype with score layer parameters (slots may be bfloat16)
            score_dtype = next(self.score.parameters()).dtype
            slots_cast = slots.to(dtype=score_dtype)
            logits = self.score(slots_cast).squeeze(-1)
            weights = torch.softmax(logits, dim=1)
            return torch.einsum("bk,bkd->bd", weights, slots_cast)

        query = self.query.to(dtype=slots.dtype, device=slots.device)
        scale = slots.shape[-1] ** -0.5
        logits = torch.einsum("d,bkd->bk", query, slots) * scale
        weights = torch.softmax(logits, dim=1)
        return torch.einsum("bk,bkd->bd", weights, slots)
