"""Storage helpers for raw and tokenized SemBelief-WM episodes."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .schema import RawEpisode, TokenizedEpisode, episode_from_dict, episode_to_dict


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a torch payload via temp file + atomic rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)
    os.replace(tmp, target)


def append_jsonl(path: str | Path, entry: dict[str, Any]) -> None:
    """Append one JSONL entry after the corresponding payload is durable."""

    manifest = Path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest.jsonl file."""

    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    entries: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def save_raw_episode(
    episode: RawEpisode,
    root: str | Path,
    *,
    episode_id: str,
    split: str = "train",
) -> Path:
    """Save one raw episode and append its manifest entry."""

    root_path = Path(root)
    episode_path = root_path / "raw" / f"{episode_id}.pt"
    payload = {
        "env_id": episode.env_id,
        "observations": episode.observations,
        "actions": episode.actions,
        "model_actions": episode.model_actions,
        "rewards": episode.rewards,
        "dones": episode.dones,
        "infos": episode.infos,
        "metadata": episode.metadata,
    }
    atomic_torch_save(payload, episode_path)
    append_jsonl(
        root_path / "raw" / "manifest.jsonl",
        {
            "episode_id": episode_id,
            "path": episode_path.name,
            "env_id": episode.env_id,
            "strategy": episode.metadata.get("strategy"),
            "length": episode.length,
            "success": bool(episode.metadata.get("success", False)),
            "seed": episode.metadata.get("seed"),
            "split": split,
        },
    )
    return episode_path


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Filter metadata to only JSON-serializable values for manifest."""

    import numpy as np

    safe: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, np.ndarray):
            continue  # numpy arrays stored in .pt file, not manifest
        elif isinstance(v, (list, tuple)):
            # Keep if all elements are JSON-safe primitives
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
                safe[k] = list(v)
        elif isinstance(v, dict):
            safe[k] = _json_safe_metadata(v)
    return safe


def save_tokenized_episode(
    episode: TokenizedEpisode,
    root: str | Path,
    *,
    episode_id: str,
    tokenizer: str,
    config_hash: str | None = None,
) -> Path:
    """Save one tokenized episode and append its manifest entry."""

    root_path = Path(root)
    episode_path = root_path / "tokenized" / f"{episode_id}.pt"
    atomic_torch_save(episode_to_dict(episode), episode_path)
    append_jsonl(
        root_path / "tokenized" / "manifest.jsonl",
        {
            "episode_id": episode_id,
            "path": episode_path.name,
            "env_id": episode.env_id,
            "tokenizer": tokenizer,
            "token_shape": list(episode.obs_tokens.shape),
            "semantic_teacher_shape": (
                None
                if episode.semantic_teacher_tokens is None
                else list(episode.semantic_teacher_tokens.shape)
            ),
            "length": int(episode.episode_length),
            "split": episode.split,
            "config_hash": config_hash,
            "metadata": _json_safe_metadata(episode.metadata),
        },
    )
    return episode_path


def load_tokenized_episode(path: str | Path) -> TokenizedEpisode:
    """Load one tokenized episode from disk."""

    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return episode_from_dict(data)


def existing_episode_ids(entries: Iterable[dict[str, Any]]) -> set[str]:
    """Return manifest episode ids for resume/skip logic."""

    return {str(entry["episode_id"]) for entry in entries if "episode_id" in entry}
