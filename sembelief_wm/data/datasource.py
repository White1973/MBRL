"""Manifest-driven tokenized datasets and DataSource implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from ..config import Config
from ..types import SequenceBatch
from .schema import TokenizedEpisode
from .storage import load_tokenized_episode, read_manifest


class TokenizedEpisodeDataset:
    """In-memory view of tokenized episodes.

    This is intentionally manifest-oriented, but also keeps a random factory
    for smoke tests and quick trainer validation.
    """

    def __init__(self, episodes: list[TokenizedEpisode]) -> None:
        if not episodes:
            raise ValueError("TokenizedEpisodeDataset requires at least one episode.")
        self.episodes = episodes

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        env_id: str | None = None,
    ) -> TokenizedEpisodeDataset:
        """Load tokenized episodes listed in a manifest.jsonl file."""

        manifest = Path(manifest_path)
        root = manifest.parent
        episodes: list[TokenizedEpisode] = []
        for entry in read_manifest(manifest):
            if split is not None and entry.get("split") != split:
                continue
            if env_id is not None and entry.get("env_id") != env_id:
                continue
            episode_path = root / str(entry.get("path", f"{entry['episode_id']}.pt"))
            episodes.append(load_tokenized_episode(episode_path))
        return cls(episodes)

    @classmethod
    def from_directory(cls, path: str | Path) -> TokenizedEpisodeDataset:
        """Load tokenized episodes from a directory.

        Accepts either a tokenized directory containing manifest.jsonl or a
        legacy flat directory with episode_*.pt files.
        """

        root = Path(path)
        manifest = root / "manifest.jsonl"
        if manifest.exists():
            return cls.from_manifest(manifest)

        files = sorted(root.glob("episode_*.pt"))
        if not files:
            files = sorted(root.glob("ep_*.pt"))
        if not files:
            raise FileNotFoundError(f"No tokenized episode files found in {root}")
        return cls([load_tokenized_episode(path) for path in files])

    @classmethod
    def from_random(
        cls,
        num_episodes: int,
        seq_len: int,
        config: Config,
        *,
        env_id: str | None = None,
    ) -> TokenizedEpisodeDataset:
        """Create random tokenized episodes for smoke testing."""

        if env_id is None:
            env_id = config.env.env_ids[0]
        episodes = []
        t_env = seq_len  # number of transitions
        t_seq = t_env + 1  # number of observations (including terminal)
        for _ in range(num_episodes):
            episodes.append(
                TokenizedEpisode(
                    obs_tokens=torch.randn(
                        t_seq,
                        config.encoder.compressed_tokens,
                        config.hidden_dim,
                    ),
                    actions=torch.randint(0, config.env.num_actions, (t_seq,)),
                    rewards=torch.zeros(t_seq),
                    episode_length=t_env,
                    env_id=env_id,
                    split="train",
                    metadata={"source": "random"},
                )
            )
        return cls(episodes)

    def filter(
        self,
        *,
        split: str | None = None,
        env_id: str | None = None,
    ) -> TokenizedEpisodeDataset:
        """Return a filtered dataset view."""

        return TokenizedEpisodeDataset([
            episode
            for episode in self.episodes
            if (split is None or episode.split == split)
            and (env_id is None or episode.env_id == env_id)
        ])

    def __len__(self) -> int:
        return len(self.episodes)


class OfflineDataSource:
    """Sample and collate tokenized episodes into SequenceBatch."""

    def __init__(
        self,
        dataset: TokenizedEpisodeDataset,
        config: Config,
        *,
        env_id_to_index: dict[str, int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.env_id_to_index = (
            {env_id: index for index, env_id in enumerate(config.env.env_ids)}
            if env_id_to_index is None
            else dict(env_id_to_index)
        )

    def filter(
        self,
        *,
        split: str | None = None,
        env_id: str | None = None,
    ) -> OfflineDataSource:
        """Return a filtered data source."""

        return OfflineDataSource(
            self.dataset.filter(split=split, env_id=env_id),
            self.config,
            env_id_to_index=self.env_id_to_index,
        )

    def sample_batch(self, batch_size: int) -> SequenceBatch:
        """Sample episodes with replacement and pad to uniform length."""

        indices = torch.randint(0, len(self.dataset), (batch_size,)).tolist()
        selected = [self.dataset.episodes[i] for i in indices]
        return self._collate(selected)

    def _collate(self, episodes: Sequence[TokenizedEpisode]) -> SequenceBatch:
        """Collate tokenized episodes into a padded SequenceBatch.

        Time alignment contract:
          - episode_length = T_env (number of real transitions)
          - obs_tokens has T_env + 1 frames: o_0, o_1, ..., o_{T_env}
          - actions has T_env + 1 entries: a_0, ..., a_{T_env-1}, dummy
          - rewards has T_env + 1 entries: r_0, ..., r_{T_env-1}, dummy
          - SequenceBatch.episode_lengths stores T_env + 1 real posterior
            steps, not transition count
          - dynamics/reward supervision masks are derived later and exclude
            the initial posterior step `Z_0`
        """
        # T_env = episode.episode_length = number of real transitions
        # T_seq = T_env + 1 = number of observations (including terminal)
        # episode_lengths in SequenceBatch = T_seq = number of posterior steps
        t_envs = [episode.episode_length for episode in episodes]
        t_seqs = [t + 1 for t in t_envs]
        max_seq_len = max(t_seqs)
        batch_size = len(episodes)

        sample = episodes[0].obs_tokens
        k_vis, hidden_dim = sample.shape[1], sample.shape[2]

        obs_tokens = torch.zeros(batch_size, max_seq_len, k_vis, hidden_dim, dtype=sample.dtype)
        actions = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        rewards = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        env_ids = torch.zeros(batch_size, dtype=torch.long)
        episode_lengths = torch.tensor(t_seqs, dtype=torch.long)
        episode_success = torch.zeros(batch_size, dtype=torch.float32)

        teacher_shape: tuple[int, int] | None = None
        for episode in episodes:
            if episode.semantic_teacher_tokens is not None:
                candidate = tuple(episode.semantic_teacher_tokens.shape[1:])
                if teacher_shape is None:
                    teacher_shape = candidate
                elif candidate != teacher_shape:
                    raise ValueError(
                        "All semantic teacher grids in a batch must share "
                        f"(K, D), got {teacher_shape} and {candidate}."
                    )
        semantic_teacher_tokens: Tensor | None = None
        semantic_teacher_mask: Tensor | None = None
        if teacher_shape is not None:
            teacher_k, teacher_d = teacher_shape
            semantic_teacher_tokens = torch.zeros(
                batch_size,
                max_seq_len,
                teacher_k,
                teacher_d,
                dtype=next(
                    episode.semantic_teacher_tokens.dtype
                    for episode in episodes
                    if episode.semantic_teacher_tokens is not None
                ),
            )
            semantic_teacher_mask = torch.zeros(
                batch_size, max_seq_len, dtype=torch.bool
            )

        for row, episode in enumerate(episodes):
            if episode.actions.ndim != 1:
                raise NotImplementedError(
                    "OfflineDataSource currently supports discrete action tensors only. "
                    "Use an action discretization wrapper for continuous-control environments."
                )
            if episode.env_id not in self.env_id_to_index:
                raise KeyError(
                    f"Episode env_id {episode.env_id!r} is not in config.env.env_ids "
                    f"{list(self.env_id_to_index)}."
                )
            seq_len = t_seqs[row]
            obs_tokens[row, :seq_len] = episode.obs_tokens[:seq_len]
            actions[row, :seq_len] = episode.actions[:seq_len].to(dtype=torch.long)
            rewards[row, :seq_len] = episode.rewards[:seq_len].to(dtype=torch.float32)
            env_ids[row] = self.env_id_to_index[episode.env_id]
            episode_success[row] = 1.0 if episode.metadata.get("success") else 0.0
            if episode.semantic_teacher_tokens is not None:
                assert semantic_teacher_tokens is not None
                assert semantic_teacher_mask is not None
                semantic_teacher_tokens[row, :seq_len] = (
                    episode.semantic_teacher_tokens[:seq_len]
                )
                semantic_teacher_mask[row, :seq_len] = True

        return SequenceBatch(
            obs_tokens=obs_tokens,
            actions=actions,
            rewards=rewards,
            episode_lengths=episode_lengths,
            env_ids=env_ids,
            episode_success=episode_success,
            semantic_teacher_tokens=semantic_teacher_tokens,
            semantic_teacher_mask=semantic_teacher_mask,
        )


class MixedDataSource:
    """Step-level multi-env sampling without mixing environments inside a batch."""

    def __init__(self, sources: dict[str, OfflineDataSource], weights: dict[str, float]) -> None:
        if not sources:
            raise ValueError("MixedDataSource requires at least one source.")
        missing = set(sources) - set(weights)
        if missing:
            raise ValueError(f"Missing weights for sources: {sorted(missing)}")
        self.sources = sources
        self.env_ids = list(sources)
        self.weights = torch.tensor([weights[env_id] for env_id in self.env_ids], dtype=torch.float32)
        if (self.weights <= 0).any():
            raise ValueError("MixedDataSource weights must be positive.")

    def sample_batch(self, batch_size: int) -> SequenceBatch:
        """Sample a full batch from one environment source."""

        source_index = int(torch.multinomial(self.weights, num_samples=1).item())
        env_id = self.env_ids[source_index]
        return self.sources[env_id].sample_batch(batch_size)


# Backwards-compatible aliases for scripts that still import dataset names.
EpisodeDataset = TokenizedEpisodeDataset
