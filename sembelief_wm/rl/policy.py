"""Policy interfaces and implementations for RL.

This module defines the Policy protocol that PPO operates on,
plus concrete implementations (MLP actor-critic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


class Policy(Protocol):
    """Minimal policy interface for PPO.

    The PPO updater only calls evaluate_actions() and parameters().
    The act() method is used by collectors during rollout.

    Design notes:
    - act() returns a 4-tuple. Implementations with different signatures
      (e.g. LatentActorCritic returns 5 values) need a thin adapter to
      conform — this protocol is NOT directly plug-and-play with arbitrary
      existing actor-critics.
    - **kwargs on both methods allows implementations to accept extra
      context (env_ids, etc.) without breaking the protocol.
    - parameters() returns all trainable params. PPO builds a single
      optimizer over all of them — this supports both separate and
      shared-trunk actor-critics without assuming disjoint param sets.
    """

    def act(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select actions given states.

        Returns:
            actions:   (B,) or (B, A)
            log_probs: (B,)
            entropy:   (B,)
            values:    (B,)
        """
        ...

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate given (state, action) pairs.

        Returns:
            log_probs: (B,)
            entropy:   (B,)
            values:    (B,)
        """
        ...

    def parameters(self) -> Any:
        """All trainable parameters for the optimizer."""
        ...


@dataclass
class MLPSpec:
    """Configuration for an MLP actor-critic."""

    state_dim: int
    num_actions: int
    hidden_dim: int = 256
    hidden_layers: int = 2
    activation: str = "gelu"


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    activation: str,
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """MLP actor-critic for discrete actions on flat state vectors.

    Takes (B, state_dim) states, outputs discrete action distributions
    and scalar value estimates. No knowledge of beliefs, world models,
    or environments.
    """

    def __init__(self, spec: MLPSpec) -> None:
        super().__init__()
        self.spec = spec

        self.policy_net = _build_mlp(
            input_dim=spec.state_dim,
            hidden_dim=spec.hidden_dim,
            hidden_layers=spec.hidden_layers,
            activation=spec.activation,
            output_dim=spec.num_actions,
        )

        self.value_net = _build_mlp(
            input_dim=spec.state_dim,
            hidden_dim=spec.hidden_dim,
            hidden_layers=spec.hidden_layers,
            activation=spec.activation,
            output_dim=1,
        )

    def act(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits = self.policy_net(states)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        values = self.value_net(states).squeeze(-1)
        return actions, dist.log_prob(actions), dist.entropy(), values

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.policy_net(states)
        dist = Categorical(logits=logits)
        values = self.value_net(states).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), values
