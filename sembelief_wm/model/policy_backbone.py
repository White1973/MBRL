"""Qwen backbone wrapper for LLM-based policy.

Adapts the QwenTransitionBackbone to serve as the backbone in LLMActorCritic.
Belief slots (B, K, D) are fed as soft prefix tokens to the Qwen transformer,
and the output hidden states (B, K, D) are returned for pooling by the policy.

Construction modes:
  - from_config: creates an independent Qwen + LoRA, separate from the WM
  - from_shared: wraps the WM's existing QwenTransitionBackbone
  - from_shared_adapter: shares one Qwen base but selects an independent LoRA

Normal Phase-2 assembly uses ``from_shared_adapter``: Qwen base weights occupy
memory once, while WM/actor/critic LoRA parameters and optimizers are disjoint.
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

    def __init__(
        self,
        qwen_backbone: nn.Module,
        *,
        shared: bool = False,
        adapter_name: str = "default",
        adapter_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.adapter_name = adapter_name
        self.adapter_trainable = adapter_trainable
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

        The policy and WM share the same transformer + default LoRA parameters.
        The shared params are NOT in self.parameters(), so the caller must
        pass backbone.trainable_parameters() as extra_params to PPOUpdater.
        """
        return cls(
            qwen_backbone,
            shared=True,
            adapter_name="default",
            adapter_trainable=False,
        )

    @classmethod
    def from_shared_adapter(
        cls,
        qwen_backbone: nn.Module,
        *,
        adapter_name: str,
        trainable: bool,
    ) -> "QwenPolicyBackbone":
        """Create a non-owning view of one LoRA in a shared Qwen base."""
        available = getattr(qwen_backbone, "lora_adapter_names", ())
        if adapter_name not in available:
            raise ValueError(
                f"Qwen backbone has no adapter {adapter_name!r}; "
                f"available={tuple(available)}"
            )
        return cls(
            qwen_backbone,
            shared=True,
            adapter_name=adapter_name,
            adapter_trainable=trainable,
        )

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
        return cls(
            backbone,
            shared=False,
            adapter_name="default",
            adapter_trainable=True,
        )

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
        forward_with_adapter = getattr(
            self._backbone,
            "forward_with_adapter",
            None,
        )
        if callable(forward_with_adapter):
            set_trainable = getattr(
                self._backbone,
                "set_lora_adapter_trainable",
                None,
            )
            if callable(set_trainable):
                set_trainable(
                    self.adapter_name,
                    self.adapter_trainable,
                )
            return forward_with_adapter(
                belief_slots,
                adapter_name=self.adapter_name,
            )
        return self._backbone(belief_slots)

    @property
    def is_shared(self) -> bool:
        """Whether this backbone shares parameters with another module."""
        return self._shared

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return this view's selected LoRA parameters for optimization.

        Multi-adapter views delegate to QwenTransitionBackbone so actor and
        critic never return one another's tensors. Non-owning views must pass
        these as PPO ``extra_params`` because the shared Qwen container is
        registered under the world model rather than the policy.
        """
        adapter_parameters = getattr(
            self._backbone,
            "lora_adapter_parameters",
            None,
        )
        if callable(adapter_parameters):
            return list(adapter_parameters(self.adapter_name))
        return [
            p for n, p in self._backbone.named_parameters()
            if "lora" in n.lower()
        ]
