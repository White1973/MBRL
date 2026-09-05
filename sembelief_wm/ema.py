"""EMA teacher target-stabilization for SemBelief-WM.

The EMA teacher maintains a slow-moving copy of the world model's transition
core. The dynamics loss becomes:

    L_dyn = MSE(prior, ema_posterior.detach())

instead of using the online posterior as the dynamics target. This stabilizes
the target branch and reduces co-drift by decoupling the target from rapid
online updates.

EMA is complementary to SIGReg. Optionally it can be combined with a
lightweight variance regularizer to explicitly prevent collapse.

Reference: world-model-ablations.md §2.3
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor

from .config import Config
from .types import BeliefState


class EMATeacher(nn.Module):
    """Exponential moving average of the transition core parameters.

    Usage:
        ema = EMATeacher(config, transition_core)
        # After each optimizer step:
        ema.update(transition_core)
        # To get stabilized posterior targets:
        with torch.no_grad():
            ema_posterior = ema.posterior_step(prev_belief, prev_actions, obs_tokens)
    """

    def __init__(self, config: Config, transition_core: nn.Module) -> None:
        super().__init__()
        self.decay = config.ema.decay
        # Keep a non-owning reference. The transition core is already
        # registered as ``WorldModel.transition``; registering it again below
        # the EMA module duplicates every Qwen key in state_dict/checkpoints.
        object.__setattr__(self, "_online", transition_core)
        self._param_names: list[str] = []
        trainable_params = [
            (name, param)
            for name, param in transition_core.named_parameters()
            if param.requires_grad
        ]
        if not trainable_params:
            raise ValueError("EMATeacher requires at least one trainable parameter.")
        for index, (name, param) in enumerate(trainable_params):
            self._param_names.append(name)
            self.register_buffer(
                f"shadow_{index}",
                param.detach().clone(),
                persistent=True,
            )

    @property
    def online(self) -> nn.Module:
        return object.__getattribute__(self, "_online")

    def _named_online_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, param)
            for name, param in self.online.named_parameters()
            if param.requires_grad
        ]

    def _shadow_buffer(self, index: int) -> Tensor:
        return getattr(self, f"shadow_{index}")

    @torch.no_grad()
    def update(self, transition_core: nn.Module) -> None:
        """Update shadow parameters with EMA of online parameters."""
        online_params = [
            (name, param)
            for name, param in transition_core.named_parameters()
            if param.requires_grad
        ]
        if [name for name, _ in online_params] != self._param_names:
            raise ValueError("EMA parameter set changed after initialization.")
        for index, (_, online_param) in enumerate(online_params):
            shadow = self._shadow_buffer(index)
            shadow.mul_(self.decay).add_(online_param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def reset_to_online(self) -> None:
        """Initialize all shadows from the currently loaded online model."""
        online_params = self._named_online_parameters()
        if [name for name, _ in online_params] != self._param_names:
            raise ValueError("EMA parameter set changed after initialization.")
        for index, (_, online_param) in enumerate(online_params):
            self._shadow_buffer(index).copy_(online_param.detach())

    @contextmanager
    def _use_shadow_parameters(self) -> None:
        online_params = self._named_online_parameters()
        if [name for name, _ in online_params] != self._param_names:
            raise ValueError("EMA parameter set changed after initialization.")
        originals: list[Tensor] = []
        try:
            for index, (_, param) in enumerate(online_params):
                originals.append(param.data)
                param.data = self._shadow_buffer(index)
            yield
        finally:
            for (_, param), original in zip(online_params, originals):
                param.data = original

    @torch.no_grad()
    def posterior_step(
        self,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        observation_tokens: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefState:
        """Compute posterior using the EMA shadow model."""
        with self._use_shadow_parameters():
            return self.online.posterior_step(
                prev_belief=prev_belief,
                prev_actions=prev_actions,
                observation_tokens=observation_tokens,
                env_ids=env_ids,
            )


def ema_variance_loss(beliefs: BeliefState | Tensor, eps: float = 1e-4) -> Tensor:
    """Lightweight variance floor for use with EMA teacher.

    Unlike SIGReg's full EP test, this is just the variance component:
    -log(std + eps) to prevent collapse, with no gradient dead zone at std=0.
    """
    slots = beliefs.slots if isinstance(beliefs, BeliefState) else beliefs
    # Flatten to (N, D)
    flat = slots.reshape(-1, slots.shape[-1]).float()
    std = flat.std(dim=0, correction=0)
    return -torch.log(std.clamp(min=eps)).mean()
