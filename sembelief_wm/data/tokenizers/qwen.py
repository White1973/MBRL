"""Native Qwen2.5-VL visual tokenization for the world model.

The world model consumes the image embeddings produced by Qwen's own vision
tower *after* its visual merger, i.e. in the same hidden space as the Qwen
language backbone.  These are spatially resampled to the configured fixed
grid, never mean pooled and never replaced by learned world-query tokens.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from ...config import Config
from ..schema import Observation


def _resolve_qwen_feature_model(model: torch.nn.Module) -> torch.nn.Module:
    """Find the wrapped model that exposes ``get_image_features``.

    ``QwenTransitionBackbone`` holds a PEFT wrapper in full training, while
    standalone preprocessing owns the base Qwen conditional-generation model.
    Both arrangements are supported without duplicating the visual tower.
    """
    candidates: list[torch.nn.Module] = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base = get_base_model()
            if isinstance(base, torch.nn.Module):
                candidates.append(base)
        except Exception:
            pass
    for candidate in list(candidates):
        for name in ("model", "base_model"):
            child = getattr(candidate, name, None)
            if isinstance(child, torch.nn.Module):
                candidates.append(child)
    for candidate in candidates:
        if callable(getattr(candidate, "get_image_features", None)):
            return candidate
    raise TypeError(
        "Could not locate Qwen2.5-VL get_image_features on the supplied model."
    )


class QwenVisionTokenizer:
    """Extract fixed-grid native Qwen2.5-VL embeddings from RGB observations."""

    def __init__(
        self,
        config: Config,
        *,
        device: str | torch.device = "cpu",
        model: torch.nn.Module | None = None,
    ) -> None:
        self.config = config
        self._device = torch.device(device)
        self._owns_model = model is None
        self._processor = self._load_processor(config.backbone.model_name)
        if model is None:
            model = self._load_model(config)
        self._feature_model = _resolve_qwen_feature_model(model)
        self._visual = getattr(self._feature_model, "visual", None)
        if self._visual is None:
            raise TypeError("Qwen model does not expose a visual tower.")
        self._hidden_dim = self._infer_hidden_dim(self._feature_model)
        if self._hidden_dim != config.hidden_dim:
            raise ValueError(
                "Qwen native visual hidden dimension must match the WM, got "
                f"vision={self._hidden_dim}, WM={config.hidden_dim}."
            )
        side = math.isqrt(config.encoder.compressed_tokens)
        if side * side != config.encoder.compressed_tokens:
            raise ValueError(
                "Qwen spatial resampling requires a square number of output "
                f"tokens, got {config.encoder.compressed_tokens}."
            )
        self._grid_side = side

    @classmethod
    def from_transition_backbone(
        cls,
        config: Config,
        backbone: Any,
        *,
        device: str | torch.device,
    ) -> "QwenVisionTokenizer":
        """Share the WM's loaded Qwen tower for real-environment posterior use."""
        model = getattr(backbone, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("QwenVisionTokenizer requires a QwenTransitionBackbone.")
        return cls(config, device=device, model=model)

    def _load_processor(self, model_name: str) -> Any:
        try:
            # Qwen2.5-VL uses the same image preprocessor contract as Qwen2-VL.
            # Importing AutoProcessor in transformers 4.54 also initializes
            # AutoVideoProcessor, which unnecessarily requires torchvision in
            # this environment. The direct image processor is the native Qwen
            # implementation and returns exactly pixel_values/image_grid_thw.
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                Qwen2VLImageProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Native Qwen tokenization requires the Qwen2-VL image processor."
            ) from exc
        return Qwen2VLImageProcessor.from_pretrained(model_name)

    def _load_model(self, config: Config) -> torch.nn.Module:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Native Qwen tokenization requires Qwen2_5_VLForConditionalGeneration."
            ) from exc
        dtype = (
            torch.bfloat16 if config.training.dtype == "bf16" else torch.float32
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.backbone.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    @staticmethod
    def _infer_hidden_dim(model: torch.nn.Module) -> int:
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None)
        hidden_dim = getattr(text_config, "hidden_size", None)
        if not isinstance(hidden_dim, int):
            hidden_dim = getattr(config, "hidden_size", None)
        if not isinstance(hidden_dim, int):
            raise ValueError("Could not infer Qwen text hidden size.")
        return hidden_dim

    @property
    def output_tokens(self) -> int:
        return self.config.encoder.compressed_tokens

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    @staticmethod
    def _to_pil(observation: Observation) -> Image.Image:
        if not isinstance(observation, np.ndarray):
            raise TypeError(
                "QwenVisionTokenizer expects RGB np.ndarray observations, got "
                f"{type(observation)}."
            )
        if observation.ndim != 3 or observation.shape[-1] != 3:
            raise ValueError(
                "QwenVisionTokenizer expects HWC RGB arrays, got "
                f"{tuple(observation.shape)}."
            )
        if observation.dtype != np.uint8:
            observation = np.clip(observation, 0, 255).astype(np.uint8)
        return Image.fromarray(observation, mode="RGB")

    def _visual_device(self) -> torch.device:
        try:
            return next(self._visual.parameters()).device
        except StopIteration:
            return self._device

    def _resample_spatial_grid(self, features: Tensor, grid_thw: Tensor) -> Tensor:
        """Resample Qwen's row-major image grid to K fixed spatial slots."""
        if grid_thw.numel() != 3:
            raise ValueError(f"Expected one image grid_thw triple, got {grid_thw}.")
        temporal, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
        merge = int(getattr(self._visual, "spatial_merge_size", 1))
        merged_h, merged_w = grid_h // merge, grid_w // merge
        expected_tokens = temporal * merged_h * merged_w
        if features.ndim != 2 or features.shape != (expected_tokens, self._hidden_dim):
            raise ValueError(
                "Unexpected native Qwen image-feature shape: "
                f"features={tuple(features.shape)}, expected="
                f"({expected_tokens}, {self._hidden_dim})."
            )
        if temporal != 1:
            raise ValueError(
                "The WM image tokenizer expects one image per observation; "
                f"received temporal grid {temporal}."
            )
        grid = features.reshape(temporal, merged_h, merged_w, self._hidden_dim)
        grid = grid.permute(0, 3, 1, 2)
        # This is local 2-D grid resampling, not global/mean pooling. Slot
        # (r, c) always corresponds to the same image region after resize.
        # Small rendered boards can yield fewer native Qwen cells than 6x6;
        # interpolate those instead of using an averaging operation to invent
        # a higher-resolution grid.
        if merged_h >= self._grid_side and merged_w >= self._grid_side:
            grid = F.adaptive_avg_pool2d(grid, (self._grid_side, self._grid_side))
        else:
            grid = F.interpolate(
                grid,
                size=(self._grid_side, self._grid_side),
                mode="bilinear",
                align_corners=False,
            )
        return grid.permute(0, 2, 3, 1).reshape(
            self.config.encoder.compressed_tokens, self._hidden_dim
        )

    @torch.no_grad()
    def batch_tokenize(self, observations: list[Observation]) -> Tensor:
        """Return native Qwen visual embeddings with shape `(N, K, D)`."""
        if not observations:
            raise ValueError("batch_tokenize requires at least one observation.")
        encoded = self._processor(
            images=[self._to_pil(observation) for observation in observations],
            return_tensors="pt",
        )
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError(
                "Qwen image processor did not return pixel_values and "
                "image_grid_thw; use a Qwen2.5-VL image processor/checkpoint."
            )
        visual_device = self._visual_device()
        visual_was_training = self._visual.training
        self._visual.eval()
        try:
            feature_parts = self._feature_model.get_image_features(
                pixel_values.to(visual_device),
                image_grid_thw.to(visual_device),
            )
        finally:
            self._visual.train(visual_was_training)
        if isinstance(feature_parts, Tensor):
            # Qwen2.5-VL returns a tuple, but fail safely rather than guessing
            # image boundaries if a different implementation returns a tensor.
            raise TypeError(
                "Qwen get_image_features returned one concatenated tensor; "
                "per-image token boundaries are required for spatial resampling."
            )
        if len(feature_parts) != len(observations):
            raise RuntimeError(
                "Qwen image-feature count does not match observations: "
                f"{len(feature_parts)} vs {len(observations)}."
            )
        tokens = [
            self._resample_spatial_grid(features, grid)
            for features, grid in zip(feature_parts, image_grid_thw, strict=True)
        ]
        return torch.stack(tokens, dim=0).float()

    @torch.no_grad()
    def tokenize(self, observation: Observation) -> Tensor:
        return self.batch_tokenize([observation]).squeeze(0)


class QwenVJEPAObservationTokenizer:
    """Runtime pair: Qwen inputs plus frozen V-JEPA teacher targets."""

    def __init__(self, qwen: QwenVisionTokenizer, vjepa: Any) -> None:
        self.qwen = qwen
        self.vjepa = vjepa
        self.provenance = {
            "input_encoder": "qwen2.5_vl_native",
            "semantic_teacher": "vjepa2_compressed_raw",
        }

    @property
    def output_tokens(self) -> int:
        return self.qwen.output_tokens

    @property
    def output_dim(self) -> int:
        return self.qwen.output_dim

    def tokenize(self, observation: Observation) -> Tensor:
        return self.qwen.tokenize(observation)

    def batch_tokenize(self, observations: list[Observation]) -> Tensor:
        return self.qwen.batch_tokenize(observations)

    def semantic_teacher_tokenize(self, observation: Observation) -> Tensor:
        return self.vjepa.semantic_teacher_tokenize(observation)

    def batch_semantic_teacher_tokens(self, observations: list[Observation]) -> Tensor:
        return self.vjepa.batch_semantic_teacher_tokens(observations)
