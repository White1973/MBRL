"""Versioning for behavior-changing world-model execution semantics."""
from __future__ import annotations

from typing import Any, Mapping


BIDIRECTIONAL_ATTENTION_MASK_VERSION = 2


def world_model_semantics(attention_mode: str) -> dict[str, Any]:
    return {
        "attention_mode": attention_mode,
        "bidirectional_attention_mask_version": (
            BIDIRECTIONAL_ATTENTION_MASK_VERSION
            if attention_mode == "bidirectional"
            else None
        ),
    }


def validate_world_model_semantics(
    checkpoint: Mapping[str, Any],
    *,
    attention_mode: str,
    context: str,
) -> None:
    """Reject checkpoints trained with the action-blind mask implementation."""
    if attention_mode != "bidirectional":
        return
    semantics = checkpoint.get("wm_semantics")
    version = (
        semantics.get("bidirectional_attention_mask_version")
        if isinstance(semantics, Mapping)
        else None
    )
    if version != BIDIRECTIONAL_ATTENTION_MASK_VERSION:
        raise ValueError(
            f"{context} predates the effective bidirectional-attention fix "
            f"(required mask version={BIDIRECTIONAL_ATTENTION_MASK_VERSION}, "
            f"checkpoint version={version!r})."
        )
