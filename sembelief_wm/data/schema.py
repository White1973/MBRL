"""Generic multi-environment data schemas for SemBelief-WM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

Observation = Any
Action = Any
ModelAction = int | Tensor


@dataclass(frozen=True)
class ActionSpaceSpec:
    """World-model-facing action-space description."""

    type: Literal["discrete", "continuous"]
    num_actions: int | None = None
    action_dim: int | None = None
    null_action_id: int | None = None

    def __post_init__(self) -> None:
        if self.type == "discrete":
            if self.num_actions is None:
                raise ValueError("discrete ActionSpaceSpec requires num_actions.")
            if self.null_action_id is None:
                object.__setattr__(self, "null_action_id", self.num_actions)
        elif self.type == "continuous":
            if self.action_dim is None:
                raise ValueError("continuous ActionSpaceSpec requires action_dim.")
        else:
            raise ValueError(f"Unsupported action space type: {self.type}")


@dataclass(frozen=True)
class ObservationSpaceSpec:
    """Raw observation-space description used by adapters and tokenizers."""

    modality: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    fields: dict[str, "ObservationSpaceSpec"] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardSpec:
    """Environment-specific reward semantics."""

    positive_threshold: float = 0.0
    negative_value: float | None = None
    positive_value: float | None = None


@dataclass
class RawEpisode:
    """Raw environment rollout before observation tokenization.

    observations has length T+1; actions/rewards/dones/infos have length T.
    model_actions are the environment actions mapped into the global
    world-model action space.
    """

    env_id: str
    observations: list[Observation]
    actions: list[Action]
    model_actions: list[ModelAction]
    rewards: list[float]
    dones: list[bool]
    infos: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = len(self.actions)
        if len(self.observations) != length + 1:
            raise ValueError(
                "RawEpisode requires len(observations) == len(actions) + 1, got "
                f"{len(self.observations)} observations and {length} actions."
            )
        for name, values in (
            ("model_actions", self.model_actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
            ("infos", self.infos),
        ):
            if len(values) != length:
                raise ValueError(
                    f"RawEpisode requires len({name}) == len(actions), got "
                    f"{len(values)} vs {length}."
                )

    @property
    def length(self) -> int:
        return len(self.actions)


@dataclass
class TokenizedEpisode:
    """Tokenized episode consumed by the world-model DataSource.

    obs_tokens/actions/rewards share T_seq = T_env + 1. The final action and
    reward entries are dummy padding so the trainer can keep a shared time axis.
    """

    obs_tokens: Tensor
    actions: Tensor
    rewards: Tensor
    episode_length: int
    env_id: str
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.obs_tokens.ndim != 3:
            raise ValueError(
                "TokenizedEpisode.obs_tokens must have shape (T, K, D), got "
                f"{tuple(self.obs_tokens.shape)}."
            )
        if self.actions.shape[0] != self.obs_tokens.shape[0]:
            raise ValueError("TokenizedEpisode actions and obs_tokens must share T.")
        if self.rewards.shape[0] != self.obs_tokens.shape[0]:
            raise ValueError("TokenizedEpisode rewards and obs_tokens must share T.")
        if not 0 < self.episode_length <= self.obs_tokens.shape[0]:
            raise ValueError(
                "episode_length must be within token sequence length, got "
                f"{self.episode_length} for T={self.obs_tokens.shape[0]}."
            )

    @property
    def seq_len(self) -> int:
        return int(self.obs_tokens.shape[0])


@dataclass(frozen=True)
class StrategySpec:
    """Sampling strategy entry in collector config."""

    name: str
    weight: int
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Strategy weight must be positive, got {self.weight}.")


@dataclass(frozen=True)
class CollectorConfig:
    """Config for pluggable strategy scheduling and episode collection."""

    num_episodes: int
    max_steps: int
    strategies: list[StrategySpec]
    seed: int
    split: str = "train"

    def __post_init__(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if not self.strategies:
            raise ValueError("CollectorConfig requires at least one strategy.")


def episode_to_dict(episode: TokenizedEpisode) -> dict[str, Any]:
    """Convert a TokenizedEpisode to a torch-serializable dict."""

    return {
        "obs_tokens": episode.obs_tokens.detach().cpu(),
        "actions": episode.actions.detach().cpu(),
        "rewards": episode.rewards.detach().cpu(),
        "episode_length": int(episode.episode_length),
        "env_id": episode.env_id,
        "split": episode.split,
        "metadata": episode.metadata,
    }


def episode_from_dict(data: dict[str, Any]) -> TokenizedEpisode:
    """Load a TokenizedEpisode from a torch-serialized dict."""

    return TokenizedEpisode(
        obs_tokens=torch.as_tensor(data["obs_tokens"]),
        actions=torch.as_tensor(data["actions"]),
        rewards=torch.as_tensor(data["rewards"], dtype=torch.float32),
        episode_length=int(data["episode_length"]),
        env_id=str(data["env_id"]),
        split=str(data.get("split", "train")),
        metadata=dict(data.get("metadata", {})),
    )
