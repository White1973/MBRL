"""Discrete state tokenizer for environments like FrozenLake.

Converts integer state observations into (K, D_model) token tensors.
The state is one-hot encoded then projected to D_model via a fixed linear
layer (deterministic from seed), and replicated to K tokens to match the
unified K_vis contract.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
from torch import Tensor

from ..schema import Observation


class DiscreteStateTokenizer(nn.Module):
    """Tokenize discrete integer states into (K, D_model) observation tokens.

    For FrozenLake with 16 states and K=36, D=3584:
      int state → one-hot (16,) → Linear → (D,) → repeat K → (36, 3584)

    The projection weights are initialized from a fixed seed to ensure
    reproducibility. The same seed always produces the same tokenization.
    """

    def __init__(
        self,
        num_states: int,
        output_dim: int,
        output_tokens: int = 36,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._num_states = num_states
        self._output_dim = output_dim
        self._output_tokens = output_tokens
        self._seed = seed
        self.projection = nn.Linear(num_states, output_dim)
        # Deterministic init from seed so re-tokenization is consistent
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.projection.weight.normal_(0, 0.02, generator=gen)
            self.projection.bias.zero_()

    @property
    def config_hash(self) -> str:
        """Deterministic hash of tokenizer configuration for manifest tracking."""
        key = f"discrete_state:n={self._num_states}:d={self._output_dim}:k={self._output_tokens}:s={self._seed}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def tokenize(self, obs: Observation) -> Tensor:
        """Single observation → (K, D_model)."""
        state_id = int(obs)
        one_hot = torch.zeros(self._num_states, dtype=torch.float32)
        one_hot[state_id] = 1.0
        projected = self.projection(one_hot)  # (D,)
        return projected.unsqueeze(0).expand(self._output_tokens, -1)  # (K, D)

    def batch_tokenize(self, observations: list[Observation]) -> Tensor:
        """Batch of observations → (N, K, D_model)."""
        state_ids = [int(obs) for obs in observations]
        one_hot = torch.zeros(len(state_ids), self._num_states, dtype=torch.float32)
        for i, sid in enumerate(state_ids):
            one_hot[i, sid] = 1.0
        projected = self.projection(one_hot)  # (N, D)
        return projected.unsqueeze(1).expand(-1, self._output_tokens, -1)  # (N, K, D)

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def output_dim(self) -> int:
        return self._output_dim
