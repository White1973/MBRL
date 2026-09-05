"""Image observation tokenizer using V-JEPA 2.

Converts RGB frames into observation tokens for world-model training:
    RGB (H, W, 3) uint8
    -> resize to (384, 384)
    -> normalize
    -> V-JEPA 2 ViT-g encoder (frozen)
    -> (576, 1408) raw tokens
    -> AdaptiveAvgPool1d -> (36, 1408) compressed tokens
    -> Linear(1408, D_model) -> (36, D_model) projected tokens

The V-JEPA 2 model is loaded once and kept frozen. All inference is
done in float16/bfloat16 for memory efficiency.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...config import Config
from ...model import VJEPATokenCompressor, VisualTokenProjector
from ..schema import Observation


class ImageTokenizer:
    """Tokenize RGB image observations using V-JEPA 2 + compression + projection.

    The tokenizer owns three components:
    1. V-JEPA 2 encoder (frozen, loaded from HuggingFace)
    2. Token compressor (AdaptiveAvgPool1d: 576 -> 36 tokens)
    3. Token projector (Linear: 1408 -> D_model)

    Components 2-3 are trainable and saved as part of the world model.
    For offline precomputation, their weights must match the training config.
    """

    def __init__(
        self,
        config: Config,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._device = torch.device(device)
        self._dtype = dtype
        self._image_size = 384  # V-JEPA 2 ViT-g input resolution
        self._raw_tokens = config.encoder.vjepa2_raw_tokens  # 576
        self._raw_dim = config.encoder.vjepa2_raw_dim  # 1408
        self._output_tokens = config.encoder.compressed_tokens  # 36
        self._output_dim = config.hidden_dim

        # Load V-JEPA 2 encoder (frozen)
        self._encoder = self._load_vjepa2(config.encoder.vjepa2_model)

        # Compression + projection (trainable, but used in eval mode for precompute)
        self._compressor = VJEPATokenCompressor(
            input_tokens=self._raw_tokens,
            output_tokens=self._output_tokens,
        ).to(self._device, self._dtype)

        self._projector = VisualTokenProjector(
            input_dim=self._raw_dim,
            hidden_dim=self._output_dim,
        ).to(self._device, self._dtype)

        # ImageNet normalization (V-JEPA 2 uses this)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=self._dtype)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=self._dtype)

    def _load_vjepa2(self, model_name: str) -> nn.Module:
        """Load V-JEPA 2 encoder from HuggingFace, frozen."""
        from transformers import VJEPA2Model

        model = VJEPA2Model.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
        ).to(self._device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        # We only need the encoder, not the predictor
        return model

    def _preprocess_frame(self, frame: np.ndarray) -> Tensor:
        """Convert a single RGB frame (H, W, 3) uint8 to model input tensor.

        Returns shape (1, 1, 3, 384, 384) — single batch, single frame.
        """
        # HWC uint8 -> CHW float [0, 1]
        t = torch.from_numpy(frame).permute(2, 0, 1).to(self._device, self._dtype) / 255.0
        # Resize to 384x384
        t = F.interpolate(t.unsqueeze(0), size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
        # Normalize with ImageNet stats
        t = (t - self._mean.view(1, 3, 1, 1)) / self._std.view(1, 3, 1, 1)
        # Add temporal dim: (1, 3, 384, 384) -> (1, 1, 3, 384, 384)
        return t.unsqueeze(1)

    def _preprocess_batch(self, frames: list[np.ndarray]) -> Tensor:
        """Convert batch of RGB frames to model input tensor.

        Returns shape (B, 1, 3, 384, 384).
        """
        tensors = []
        for frame in frames:
            t = torch.from_numpy(frame).permute(2, 0, 1).to(self._device, self._dtype) / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)
            t = (t - self._mean.view(1, 3, 1, 1)) / self._std.view(1, 3, 1, 1)
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)  # (B, 3, 384, 384)
        return batch.unsqueeze(1)  # (B, 1, 3, 384, 384)

    @torch.no_grad()
    def _encode_raw(self, pixel_values: Tensor) -> Tensor:
        """Run V-JEPA 2 encoder. Input (B, 1, 3, 384, 384), output (B, 576, 1408)."""
        output = self._encoder(pixel_values_videos=pixel_values, skip_predictor=True)
        return output.last_hidden_state  # (B, N, D) where N=576, D=1408

    @torch.no_grad()
    def tokenize(self, obs: Observation) -> Tensor:
        """Tokenize a single RGB observation. Returns (K, D_model)."""
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"ImageTokenizer expects np.ndarray, got {type(obs)}")
        pixel_values = self._preprocess_frame(obs)
        raw_tokens = self._encode_raw(pixel_values)  # (1, 576, 1408)
        compressed = self._compressor(raw_tokens)  # (1, 36, 1408)
        projected = self._projector(compressed)  # (1, 36, D_model)
        return projected.squeeze(0).float()  # (36, D_model) in fp32

    @torch.no_grad()
    def batch_tokenize(self, observations: list[Observation]) -> Tensor:
        """Tokenize a batch of RGB observations. Returns (N, K, D_model)."""
        pixel_values = self._preprocess_batch(observations)  # (N, 1, 3, 384, 384)
        raw_tokens = self._encode_raw(pixel_values)  # (N, 576, 1408)
        compressed = self._compressor(raw_tokens)  # (N, 36, 1408)
        projected = self._projector(compressed)  # (N, 36, D_model)
        return projected.float()  # (N, 36, D_model) in fp32

    @torch.no_grad()
    def batch_semantic_teacher_tokens(self, observations: list[Observation]) -> Tensor:
        """Return frozen compressed V-JEPA features without the WM projector.

        This is intentionally a teacher-only path.  In the Qwen-native world
        model recipe these `(K, 1408)` features are stored beside the Qwen
        observation tokens and are never passed to ``posterior_step``.
        Keeping the features before ``VisualTokenProjector`` avoids making a
        learned V-JEPA-to-Qwen projection part of the teacher target.
        """
        pixel_values = self._preprocess_batch(observations)
        raw_tokens = self._encode_raw(pixel_values)
        return self._compressor(raw_tokens).float()

    @torch.no_grad()
    def semantic_teacher_tokenize(self, obs: Observation) -> Tensor:
        """Return one frozen compressed V-JEPA feature grid `(K, D_vjepa)`."""
        return self.batch_semantic_teacher_tokens([obs]).squeeze(0)

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def semantic_teacher_dim(self) -> int:
        return self._raw_dim

    def save_weights(self, path: str) -> None:
        """Save compressor + projector weights for reproducibility."""
        torch.save({
            "compressor": self._compressor.state_dict(),
            "projector": self._projector.state_dict(),
        }, path)

    def load_weights(self, path: str) -> None:
        """Load compressor + projector weights."""
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._compressor.load_state_dict(state["compressor"])
        self._projector.load_state_dict(state["projector"])
