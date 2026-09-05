"""Fixed, split-safe H2 posterior panels for Critic-only pretraining."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..types import BeliefState


FORMAT = "qwen_vjepa_critic_h2_posterior_cache_v1"


def load_critic_h2_cache(
    path: str | Path,
    *,
    source_checkpoint: str | Path,
) -> dict[str, Any]:
    cache_path = Path(path).resolve()
    source_path = Path(source_checkpoint).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Critic H2 cache is missing: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported Critic H2 cache format: {payload.get('format')}")
    provenance = payload.get("provenance", {})
    if Path(provenance.get("source_checkpoint", "")).resolve() != source_path:
        raise ValueError(
            "Critic H2 cache belongs to a different Reward-Head checkpoint: "
            f"cache={provenance.get('source_checkpoint')}, requested={source_path}"
        )
    source_stat = source_path.stat()
    if (
        int(provenance.get("source_checkpoint_size", -1)) != source_stat.st_size
        or int(provenance.get("source_checkpoint_mtime_ns", -1))
        != source_stat.st_mtime_ns
    ):
        raise ValueError("Critic H2 cache source checkpoint fingerprint changed")
    if provenance.get("official_validation_used_for_training") is not False:
        raise ValueError("Critic H2 cache does not prove held-out split isolation")
    names = payload.get("bucket_names")
    if names != ["initial_h1", "initial_h2", "suffix_h1", "suffix_h2"]:
        raise ValueError(f"unexpected Critic H2 buckets: {names}")
    for split in ("train_buckets", "validation_buckets"):
        buckets = payload.get(split)
        if not isinstance(buckets, dict):
            raise ValueError(f"Critic H2 cache has no {split}")
        for name in names:
            slots = buckets.get(name)
            if not isinstance(slots, torch.Tensor) or slots.ndim != 3:
                raise ValueError(f"invalid {split}/{name} tensor")
            if slots.shape[1:] != (36, 2048) or len(slots) == 0:
                raise ValueError(
                    f"unexpected {split}/{name} shape: {tuple(slots.shape)}"
                )
    return payload


def install_critic_h2_cache(
    *,
    pipeline: Any,
    world_model: Any,
    cache_path: str | Path,
    source_checkpoint: str | Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Install balanced train sampling and immutable official validation slots."""
    cache = load_critic_h2_cache(
        cache_path, source_checkpoint=source_checkpoint
    )
    names: list[str] = cache["bucket_names"]
    train_buckets = [cache["train_buckets"][name] for name in names]
    validation_buckets = [cache["validation_buckets"][name] for name in names]
    dtype = next(world_model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(seed + 3107)
    cursor = 0

    def sample_balanced(batch_size: int) -> BeliefState:
        nonlocal cursor
        if batch_size <= 0:
            raise ValueError("Critic H2 batch size must be positive")
        bucket_count = len(train_buckets)
        counts = [batch_size // bucket_count] * bucket_count
        for offset in range(batch_size % bucket_count):
            counts[(cursor + offset) % bucket_count] += 1
        cursor = (cursor + batch_size % bucket_count) % bucket_count
        parts: list[torch.Tensor] = []
        id_parts: list[torch.Tensor] = []
        for bucket_id, (bucket, count) in enumerate(zip(train_buckets, counts)):
            if count == 0:
                continue
            indices = torch.randint(
                len(bucket), (count,), generator=generator, device="cpu"
            )
            parts.append(bucket[indices])
            id_parts.append(torch.full((count,), bucket_id, dtype=torch.long))
        slots = torch.cat(parts)
        bucket_ids = torch.cat(id_parts)
        permutation = torch.randperm(
            len(slots), generator=generator, device="cpu"
        )
        slots = slots[permutation]
        pipeline._last_sample_bucket_ids = bucket_ids[permutation]
        return BeliefState(slots=slots.to(device=device, dtype=dtype))

    pipeline.sample_beliefs_fn = sample_balanced
    pipeline.counterfactual_h2_validation_slots = torch.cat(validation_buckets)
    pipeline.counterfactual_h2_validation_bucket_ids = torch.cat([
        torch.full((len(bucket),), bucket_id, dtype=torch.long)
        for bucket_id, bucket in enumerate(validation_buckets)
    ])
    pipeline.counterfactual_h2_validation_bucket_names = list(names)
    expected_validation_actions = 4 * sum(len(bucket) for bucket in validation_buckets)
    configured_validation_actions = int(
        pipeline.config.critic_warmup_validation_size
    )
    if configured_validation_actions != expected_validation_actions:
        raise ValueError(
            "Critic validation size must equal four first-action targets per "
            f"cached state: configured={configured_validation_actions}, "
            f"expected={expected_validation_actions}"
        )
    pipeline.critic_h2_cache_metadata = {
        "format": cache["format"],
        "path": str(Path(cache_path).resolve()),
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "train_bucket_sizes": {
            name: len(bucket) for name, bucket in zip(names, train_buckets)
        },
        "validation_bucket_sizes": {
            name: len(bucket) for name, bucket in zip(names, validation_buckets)
        },
        "official_validation_used_for_training": False,
        "validation_action_targets": expected_validation_actions,
    }
    print(
        "Installed split-safe H2 Critic panels: train="
        f"{pipeline.critic_h2_cache_metadata['train_bucket_sizes']} "
        "official_validation="
        f"{pipeline.critic_h2_cache_metadata['validation_bucket_sizes']}",
        flush=True,
    )
    return pipeline.critic_h2_cache_metadata
