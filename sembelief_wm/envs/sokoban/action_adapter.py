"""Sokoban action adapter.

Sokoban has 4 discrete push actions: up, down, left, right.
Env actions are 1-4; model actions are 0-3.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
from torch import Tensor


# Model action index → name
_ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

# Text → model action index (case-insensitive, supports common variants)
_TEXT_TO_ACTION: dict[str, int] = {
    "up": 0, "move up": 0, "push up": 0,
    "down": 1, "move down": 1, "push down": 1,
    "left": 2, "move left": 2, "push left": 2,
    "right": 3, "move right": 3, "push right": 3,
}

# Regex fallback: find first occurrence of up/down/left/right
_DIRECTION_RE = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)


class SokobanActionAdapter(nn.Module):
    """Action adapter for Sokoban (4-way discrete push).

    Supports both action-head mode (forward_logits) and free-decoding
    mode (decode_text). The action head is a single linear layer from
    the LLM hidden dimension to 4 logits.
    """

    def __init__(self, hidden_dim: int) -> None:
        """
        Args:
            hidden_dim: LLM backbone hidden dimension (e.g. 3584 for Qwen2.5-VL-7B).
        """
        super().__init__()
        self._num_actions = 4
        self.action_head = nn.Linear(hidden_dim, self._num_actions)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def forward_logits(self, hidden_states: Tensor) -> Tensor:
        """(B, D) → (B, 4) action logits."""
        return self.action_head(hidden_states)

    def decode_text(self, text: str) -> int | None:
        """Parse free-form text into action index 0-3.

        Tries exact match first, then regex fallback.
        Returns None if no valid action found.
        """
        normalized = text.strip().lower()

        # Exact match
        if normalized in _TEXT_TO_ACTION:
            return _TEXT_TO_ACTION[normalized]

        # Regex fallback: find first direction keyword
        match = _DIRECTION_RE.search(text)
        if match:
            return _TEXT_TO_ACTION[match.group(1).lower()]

        return None

    def format_action(self, action: int) -> str:
        """Action index 0-3 → "up"/"down"/"left"/"right"."""
        return _ACTION_NAMES[action]

    @staticmethod
    def model_to_env_action(model_action: int) -> int:
        """Convert model action (0-3) to env action (1-4)."""
        return model_action + 1

    @staticmethod
    def env_to_model_action(env_action: int) -> int:
        """Convert env action (1-4) to model action (0-3)."""
        return env_action - 1
