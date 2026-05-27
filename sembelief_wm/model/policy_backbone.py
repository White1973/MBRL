"""Qwen backbone wrapper for LLM-based policy.

Adapts the QwenTransitionBackbone to serve as the backbone in LLMActorCritic.
Belief slots (B, K, D) are fed as soft prefix tokens to the Qwen transformer,
and the output hidden states (B, K, D) are returned for pooling by the policy.

Two construction modes:
  - from_config: creates an independent Qwen + LoRA, separate from the WM
  - from_shared: wraps the WM's existing QwenTransitionBackbone

When shared, policy gradients flow through the same LoRA adapters as the WM.
When independent, the policy has its own LoRA parameters.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class QwenPolicyBackbone(nn.Module):
    """Wraps QwenTransitionBackbone for use as LLMActorCritic backbone.

    LLMActorCritic expects: backbone(belief_slots: (B, K, D)) -> (B, K, D)
    QwenTransitionBackbone expects: forward(tokens: (B, S, D)) -> (B, S, D)

    This wrapper bridges the two. It's intentionally thin — the real work
    is done by the underlying Qwen transformer + LoRA.
    """

    def __init__(self, qwen_backbone: nn.Module, *, shared: bool = False) -> None:
        super().__init__()
        # Store as attribute (not submodule) when shared, to avoid
        # double-counting parameters in state_dict / optimizer.
        if shared:
            # Non-owning reference — WM owns the backbone parameters.
            object.__setattr__(self, "_qwen", qwen_backbone)
            self._shared = True
        else:
            # Owning — policy has its own backbone parameters.
            self.qwen = qwen_backbone
            self._shared = False

    @property
    def _backbone(self) -> nn.Module:
        """Access the underlying backbone regardless of ownership mode."""
        if self._shared:
            return self._qwen
        return self.qwen

    @classmethod
    def from_shared(cls, qwen_backbone: nn.Module) -> "QwenPolicyBackbone":
        """Wrap an existing backbone (e.g., from world model).

        The policy and WM share the same transformer + LoRA parameters.
        The shared params are NOT in self.parameters(), so the caller must
        pass backbone.trainable_parameters() as extra_params to PPOUpdater.
        """
        return cls(qwen_backbone, shared=True)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        device_map: str | dict[str, int | str] | None = "auto",
        attn_implementation: str | None = None,
    ) -> "QwenPolicyBackbone":
        """Create an independent Qwen + LoRA backbone for the policy.

        This loads a fresh Qwen model with its own LoRA adapters,
        completely separate from the world model's backbone.

        Args:
            config: sembelief_wm Config object.
            device_map: HF device mapping.
            attn_implementation: attention implementation override.
        """
        from .backbone_qwen import QwenTransitionBackbone

        backbone = QwenTransitionBackbone.from_config(
            config,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        return cls(backbone, shared=False)

    def forward(self, belief_slots: Tensor) -> Tensor:
        """Process belief slots through the Qwen transformer.

        Args:
            belief_slots: (B, K, D) belief slot tokens.

        Returns:
            hidden_states: (B, K, D) transformer output hidden states.
        """
        if belief_slots.ndim != 3:
            raise ValueError(
                f"QwenPolicyBackbone expects belief_slots with shape (B, K, D), "
                f"got {tuple(belief_slots.shape)}."
            )
        return self._backbone(belief_slots)

    @property
    def is_shared(self) -> bool:
        """Whether this backbone shares parameters with another module."""
        return self._shared

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return LoRA parameters for optimizer construction.

        Identifies LoRA params by name (containing 'lora') rather than
        by requires_grad, because world_model.requires_grad_(False) may
        have turned off all grads before this is called.

        Both shared and independent modes return the backbone's LoRA
        params. For shared mode, these are NOT in self.parameters() (because
        the backbone is not registered as a submodule), so the caller must
        pass them as extra_params to PPOUpdater to ensure they get optimized.

        For independent mode, they are already in self.parameters(), but
        returning them here is still safe — PPOUpdater deduplicates by data_ptr.
        """
        return [
            p for n, p in self._backbone.named_parameters()
            if "lora" in n.lower()
        ]
