"""Lightweight online replay buffer for model-based RL.

Stores tokenized episodes collected during real-env interaction.
Provides two services:
  1. sample_batch() → SequenceBatch for WM supervised refresh
  2. sample_beliefs() → start beliefs for imagined rollout

This module depends only on existing data infrastructure (OfflineDataSource,
TokenizedEpisodeDataset, SequenceBatch). It does NOT import rl/ or collectors/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from sembelief_wm.data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from sembelief_wm.data.storage import save_tokenized_episode
from sembelief_wm.data.schema import TokenizedEpisode
from sembelief_wm.types import SequenceBatch


class OnlineReplayBuffer:
    """Stores tokenized episodes and samples batches for WM training or belief grounding.

    Design:
      - Episodes are stored to disk for persistence across restarts.
      - De-duplicates by episode_id.
      - Wraps OfflineDataSource for batch sampling (padding, collation).
    """

    def __init__(self, root: str | Path, *, max_episodes: int | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        tokenized_root = self.root / "tokenized"

        if (tokenized_root / "manifest.jsonl").exists():
            dataset = TokenizedEpisodeDataset.from_directory(tokenized_root)
            self.episodes: list[TokenizedEpisode] = list(dataset.episodes)
        else:
            self.episodes = []

        self._episode_ids: set[str] = {
            str(ep.metadata.get("episode_id"))
            for ep in self.episodes
            if ep.metadata.get("episode_id") is not None
        }
        self._max_episodes = max_episodes
        self._data_source: OfflineDataSource | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.episodes) == 0

    @property
    def size(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(int(ep.episode_length) for ep in self.episodes)

    def add_episode(self, episode_id: str, episode: TokenizedEpisode) -> None:
        """Add a tokenized episode, skipping duplicates."""
        if episode_id in self._episode_ids:
            return
        episode.metadata.setdefault("episode_id", episode_id)
        save_tokenized_episode(
            episode,
            self.root,
            episode_id=episode_id,
            tokenizer="online_replay",
            config_hash=None,
        )
        self.episodes.append(episode)
        self._episode_ids.add(episode_id)
        self._data_source = None  # invalidate cached source

        # Evict oldest if over capacity (memory + disk)
        if self._max_episodes is not None and len(self.episodes) > self._max_episodes:
            evicted = self.episodes[:-self._max_episodes]
            self.episodes = self.episodes[-self._max_episodes:]
            self._episode_ids = {
                str(ep.metadata.get("episode_id"))
                for ep in self.episodes
                if ep.metadata.get("episode_id") is not None
            }
            self._data_source = None
            # Clean up evicted episode files from disk
            tokenized_dir = self.root / "tokenized"
            for ep in evicted:
                eid = ep.metadata.get("episode_id")
                if eid is not None:
                    pt_file = tokenized_dir / f"{eid}.pt"
                    pt_file.unlink(missing_ok=True)
            # Rewrite manifest to reflect only remaining episodes
            self._rewrite_manifest()

    def _rewrite_manifest(self) -> None:
        """Rewrite manifest.jsonl to match current in-memory episodes."""
        manifest_path = self.root / "tokenized" / "manifest.jsonl"
        if not manifest_path.exists():
            return
        # Read existing manifest entries, filter to remaining episode IDs
        import json
        remaining_ids = self._episode_ids
        kept_lines = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                if entry.get("episode_id") in remaining_ids:
                    kept_lines.append(stripped)
        # Atomic rewrite
        tmp_path = manifest_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
        import os
        os.replace(tmp_path, manifest_path)

    def sample_batch(self, batch_size: int, config: object) -> SequenceBatch:
        """Sample a batch of episodes for WM training.

        Args:
            batch_size: Number of episodes to sample.
            config: The global Config object (needed by OfflineDataSource for
                    collation settings like env_ids).
        """
        if self.is_empty:
            raise ValueError("OnlineReplayBuffer is empty.")
        if self._data_source is None:
            self._data_source = OfflineDataSource(
                TokenizedEpisodeDataset(self.episodes), config
            )
        return self._data_source.sample_batch(batch_size)


class MixedDataSampler:
    """Samples batches from a mix of offline and online data sources.

    Handles the offline/online ratio logic that was previously duplicated
    in AlternatingWorldModelUpdater and Phase2Trainer.
    """

    def __init__(
        self,
        offline_source: OfflineDataSource,
        online_buffer: OnlineReplayBuffer | None = None,
        online_ratio: float = 0.0,
    ) -> None:
        self.offline_source = offline_source
        self.online_buffer = online_buffer
        self.online_ratio = online_ratio

    def sample(self, batch_size: int, config: object) -> SequenceBatch:
        """Sample a mixed batch respecting online_ratio."""
        if (
            self.online_buffer is None
            or self.online_buffer.is_empty
            or self.online_ratio <= 0.0
        ):
            return self.offline_source.sample_batch(batch_size)

        if self.online_ratio >= 1.0:
            return self.online_buffer.sample_batch(batch_size, config)

        # For batch_size=1, can't split — randomly pick one source
        if batch_size <= 1:
            import random
            if random.random() < self.online_ratio:
                return self.online_buffer.sample_batch(batch_size, config)
            return self.offline_source.sample_batch(batch_size)

        online_count = max(1, min(batch_size - 1, round(batch_size * self.online_ratio)))
        offline_count = batch_size - online_count

        online_batch = self.online_buffer.sample_batch(online_count, config)
        offline_batch = self.offline_source.sample_batch(offline_count)
        return _concat_sequence_batches(offline_batch, online_batch)


def _pad_sequence_batch(batch: SequenceBatch, target_t: int) -> SequenceBatch:
    """Pad a SequenceBatch to target sequence length."""
    B, T = batch.obs_tokens.shape[0], batch.obs_tokens.shape[1]
    if T >= target_t:
        return batch
    pad_t = target_t - T
    K, D = batch.obs_tokens.shape[2], batch.obs_tokens.shape[3]
    obs_pad = torch.zeros(B, pad_t, K, D, dtype=batch.obs_tokens.dtype, device=batch.obs_tokens.device)
    act_pad = torch.zeros(B, pad_t, *batch.actions.shape[2:], dtype=batch.actions.dtype, device=batch.actions.device)
    rew_pad = torch.zeros(B, pad_t, dtype=batch.rewards.dtype, device=batch.rewards.device)
    return SequenceBatch(
        obs_tokens=torch.cat([batch.obs_tokens, obs_pad], dim=1),
        actions=torch.cat([batch.actions, act_pad], dim=1),
        rewards=torch.cat([batch.rewards, rew_pad], dim=1),
        episode_lengths=batch.episode_lengths,
        env_ids=batch.env_ids,
    )


def _concat_sequence_batches(a: SequenceBatch, b: SequenceBatch) -> SequenceBatch:
    """Concatenate two SequenceBatches, padding to the same sequence length."""
    t_a = a.obs_tokens.shape[1]
    t_b = b.obs_tokens.shape[1]
    if t_a != t_b:
        target = max(t_a, t_b)
        a = _pad_sequence_batch(a, target)
        b = _pad_sequence_batch(b, target)

    env_ids: Tensor | None = None
    if a.env_ids is not None and b.env_ids is not None:
        env_ids = torch.cat([a.env_ids, b.env_ids], dim=0)

    return SequenceBatch(
        obs_tokens=torch.cat([a.obs_tokens, b.obs_tokens], dim=0),
        actions=torch.cat([a.actions, b.actions], dim=0),
        rewards=torch.cat([a.rewards, b.rewards], dim=0),
        episode_lengths=torch.cat([a.episode_lengths, b.episode_lengths], dim=0),
        env_ids=env_ids,
    )
