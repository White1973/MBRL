"""Visual-token preprocessing utilities for SemBelief-WM.

The world-model training stack consumes precomputed observation tokens with
shape (B, K_vis, D). This module owns the lightweight, testable part of that
offline preprocessing path:

    V-JEPA raw tokens (B, 576, 1408)
        -> token compression (B, 36, 1408)
        -> projection (B, 36, D)

The concrete V-JEPA 2 model wrapper is intentionally left outside this module
until the remote dependency/API surface is verified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..config import Config


VideoLayout = Literal["btchw", "bcthw"]


def duplicate_single_frame_clip(
    frames: Tensor,
    *,
    layout: VideoLayout = "btchw",
) -> Tensor:
    """Duplicate single-frame image tensors into a two-frame pseudo clip.

    Args:
        frames: Image tensor with shape (B, C, H, W).
        layout: Output video layout. "btchw" returns (B, 2, C, H, W);
            "bcthw" returns (B, C, 2, H, W).

    V-JEPA 2 uses a temporal convolution with kernel size 2, so single Sokoban
    frames need this explicit pseudo-clip conversion before model inference.
    """

    if frames.ndim != 4:
        raise ValueError(
            "duplicate_single_frame_clip expects image tensors with shape "
            f"(B, C, H, W), got {tuple(frames.shape)}."
        )
    if layout == "btchw":
        return frames.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    if layout == "bcthw":
        return frames.unsqueeze(2).repeat(1, 1, 2, 1, 1)
    raise ValueError(f"Unsupported video layout: {layout}")


class VJEPATokenCompressor(nn.Module):
    """Compress raw V-JEPA patch tokens along the token axis."""

    def __init__(self, *, input_tokens: int, output_tokens: int) -> None:
        super().__init__()
        if input_tokens <= 0 or output_tokens <= 0:
            raise ValueError("Token counts must be positive.")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.pool = nn.AdaptiveAvgPool1d(output_tokens)

    def forward(self, tokens: Tensor) -> Tensor:
        """Compress tokens from (B, K_raw, D_raw) to (B, K_vis, D_raw)."""

        if tokens.ndim != 3:
            raise ValueError(
                "VJEPATokenCompressor expects tokens with shape (B, K, D), "
                f"got {tuple(tokens.shape)}."
            )
        if tokens.shape[1] != self.input_tokens:
            raise ValueError(
                f"Expected {self.input_tokens} input tokens, got {tokens.shape[1]}."
            )

        channel_first = tokens.transpose(1, 2)
        pooled = self.pool(channel_first)
        return pooled.transpose(1, 2)


class VisualTokenProjector(nn.Module):
    """Project compressed visual tokens into the world-model hidden dimension."""

    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Projection dimensions must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Project tokens from (B, K_vis, D_raw) to (B, K_vis, D)."""

        if tokens.ndim != 3:
            raise ValueError(
                "VisualTokenProjector expects tokens with shape (B, K, D_raw), "
                f"got {tuple(tokens.shape)}."
            )
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected visual token dim {self.input_dim}, got {tokens.shape[-1]}."
            )
        return self.proj(tokens)


class VisualTokenPreprocessor(nn.Module):
    """Offline raw-token preprocessor producing world-model observation tokens."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.raw_tokens = config.encoder.vjepa2_raw_tokens
        self.raw_dim = config.encoder.vjepa2_raw_dim
        self.output_tokens = config.encoder.compressed_tokens
        self.hidden_dim = config.hidden_dim
        self.compressor = VJEPATokenCompressor(
            input_tokens=self.raw_tokens,
            output_tokens=self.output_tokens,
        )
        self.projector = VisualTokenProjector(
            input_dim=self.raw_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, raw_tokens: Tensor) -> Tensor:
        """Convert raw V-JEPA tokens to training-time observation tokens."""

        compressed = self.compressor(raw_tokens)
        return self.projector(compressed)


def save_observation_tokens(tokens: Tensor, path: str | Path) -> None:
    """Save precomputed observation tokens to a torch file."""

    if tokens.ndim not in (3, 4):
        raise ValueError(
            "Observation tokens must have shape (B, K, D) or (B, T, K, D), "
            f"got {tuple(tokens.shape)}."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs_tokens": tokens.detach().cpu()}, path)


def load_observation_tokens(path: str | Path, *, device: torch.device | str | None = None) -> Tensor:
    """Load precomputed observation tokens from a torch file."""

    payload = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    if isinstance(payload, Tensor):
        tokens = payload
    elif isinstance(payload, dict) and "obs_tokens" in payload:
        tokens = payload["obs_tokens"]
    else:
        raise ValueError("Expected a Tensor or a dict containing 'obs_tokens'.")

    if tokens.ndim not in (3, 4):
        raise ValueError(
            "Loaded observation tokens must have shape (B, K, D) or (B, T, K, D), "
            f"got {tuple(tokens.shape)}."
        )
    return tokens
