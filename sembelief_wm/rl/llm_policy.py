"""LLM-based actor-critic policy.

Uses an LLM backbone (e.g. Qwen2.5-VL + LoRA) to process belief slots
as soft prefix tokens, then delegates action output to an environment-
specific ActionAdapter.

This module implements the Policy protocol from rl/policy.py.
It does NOT know about specific environments — that knowledge lives
entirely in the ActionAdapter.

Belief slots are fed directly into the LLM backbone as soft prefix tokens.
No separate projection layer is needed because belief_dim is always aligned
with llm_hidden_dim (both = backbone hidden_size). The LoRA adapters in the
backbone learn to read the latent slots during SFT/PPO training.

Two output modes:
  - Action head: belief → LLM hidden → adapter.forward_logits → discrete dist
  - Free decoding: belief → LLM generate → adapter.decode_text → action index
    (not yet implemented — requires token-level PPO, deferred to later sprint)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from .action_adapter import ActionAdapter


@dataclass
class LLMPolicyConfig:
    """Configuration for LLM actor-critic."""

    hidden_dim: int = 3584            # belief slot dim == LLM hidden dim (always aligned)
    num_slots: int = 36               # belief slot count
    value_hidden_dim: int = 256       # value head hidden layer
    output_mode: str = "action_head"  # "action_head" or "free_decoding"


class LLMActorCritic(nn.Module):
    """LLM-based actor-critic that reads belief slots.

    Architecture:
        belief_slots (B, K, D)
            → LLM backbone forward (with LoRA) → hidden_states (B, K, D)
            → pool → (B, D)
            → action_adapter.forward_logits → (B, num_actions)
            → value_head → (B,)

    Belief slots are fed directly as soft prefix tokens — no projection
    needed since belief_dim == llm_hidden_dim by design. LoRA learns to
    read the latent slots during SFT/PPO.

    The LLM backbone is injected, not owned. This allows sharing
    the backbone with the world model's posterior update if desired.

    Implements rl.policy.Policy protocol (compatible via adapter).
    """

    def __init__(
        self,
        backbone: nn.Module,
        action_adapter: ActionAdapter,
        config: LLMPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or LLMPolicyConfig()

        self.backbone = backbone
        self.action_adapter = action_adapter
        self.config = cfg

        # Value head: pool → hidden → scalar
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.value_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.value_hidden_dim, 1),
        )

    def _encode_beliefs(self, states: Tensor) -> Tensor:
        """Convert flat states back to slots, run through backbone, pool.

        Args:
            states: (B, K*D) flat belief vector, or (B, K, D) slots.

        Returns:
            pooled: (B, D) — pooled hidden representation.
        """
        cfg = self.config

        # Reshape to slots if flat
        if states.ndim == 2:
            slots = states.view(-1, cfg.num_slots, cfg.hidden_dim)
        else:
            slots = states  # already (B, K, D)

        # Forward through backbone — belief slots as soft prefix tokens
        hidden = self.backbone(slots)  # expected: (B, K, D) or (B, D)

        # Pool to (B, D) if backbone returns per-token hidden states
        if hidden.ndim == 3:
            hidden = hidden.mean(dim=1)

        return hidden

    def act(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select actions given belief states.

        Returns: (actions, log_probs, entropy, values)
        """
        hidden = self._encode_beliefs(states)

        # Action distribution
        logits = self.action_adapter.forward_logits(hidden)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()

        # Value estimate
        values = self.value_head(hidden).squeeze(-1)

        return actions, dist.log_prob(actions), dist.entropy(), values

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate given (state, action) pairs.

        Returns: (log_probs, entropy, values)
        """
        hidden = self._encode_beliefs(states)

        logits = self.action_adapter.forward_logits(hidden)
        dist = Categorical(logits=logits)
        values = self.value_head(hidden).squeeze(-1)

        return dist.log_prob(actions), dist.entropy(), values
