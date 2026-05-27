"""Base protocols for raw-observation tokenizers."""
from __future__ import annotations

from typing import Protocol

from torch import Tensor

from ..schema import Observation


class ObservationTokenizer(Protocol):
    """Convert raw observations into world-model observation tokens."""

    def tokenize(self, obs: Observation) -> Tensor:
        """Return one tokenized observation with shape (K, D_model)."""

    def batch_tokenize(self, observations: list[Observation]) -> Tensor:
        """Return batched tokens with shape (N, K, D_model)."""

    @property
    def output_tokens(self) -> int: ...

    @property
    def output_dim(self) -> int: ...
