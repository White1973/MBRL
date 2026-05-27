"""SIGReg anti-collapse regularization for SemBelief-WM.

This module implements the Phase 1 anti-collapse term:
- Epps-Pulley characteristic-function test on random projections
- VICReg-style variance floor
- detached running buffer over posterior beliefs

SIGReg always operates on posterior beliefs only and in float32.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import Config
from ..types import BeliefState


@dataclass(frozen=True)
class SIGRegTerms:
    """Structured SIGReg outputs for logging and total-loss composition.

    ep/var are *N-scaled (matching main branch). Unscaled values are
    provided separately for interpretable logging.
    """

    ep: Tensor           # EP loss * N (for loss composition)
    var: Tensor          # variance loss * N (for loss composition)
    ep_unscaled: Tensor  # EP loss before *N (for logging)
    var_unscaled: Tensor # variance loss before *N (for logging)
    mean_std: Tensor
    min_std: Tensor
    num_current: int
    num_total: int


def flatten_posterior_beliefs(beliefs: BeliefState | Tensor) -> Tensor:
    """Flatten posterior beliefs to `(N, D)` for SIGReg statistics."""

    slots = beliefs.slots if isinstance(beliefs, BeliefState) else beliefs

    if slots.ndim == 2:
        return slots
    if slots.ndim == 3:
        return slots.reshape(-1, slots.shape[-1])
    if slots.ndim == 4:
        return slots.reshape(-1, slots.shape[-1])

    raise ValueError(
        "SIGReg expects belief tensors with shape (N, D), (B, K, D), or (B, T, K, D), "
        f"got {tuple(slots.shape)}."
    )


def vicreg_variance_loss(embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """VICReg-style variance floor on flattened embeddings."""

    if embeddings.ndim != 2:
        raise ValueError(
            f"vicreg_variance_loss expects (N, D), got {tuple(embeddings.shape)}."
        )

    std = embeddings.std(dim=0, correction=0)
    loss = torch.relu(1.0 - std).mean()
    return loss, std.mean(), std.min()


def epps_pulley_loss(
    embeddings: Tensor,
    *,
    projections: Tensor,
    quadrature_points: Tensor,
    quadrature_weights: Tensor,
) -> Tensor:
    """Approximate the Epps-Pulley discrepancy on fixed random projections."""

    if embeddings.ndim != 2:
        raise ValueError(
            f"epps_pulley_loss expects (N, D), got {tuple(embeddings.shape)}."
        )
    if projections.ndim != 2:
        raise ValueError(
            f"projections must have shape (D, M), got {tuple(projections.shape)}."
        )
    if quadrature_points.ndim != 1 or quadrature_weights.ndim != 1:
        raise ValueError("Quadrature points and weights must be 1D tensors.")
    if quadrature_points.shape != quadrature_weights.shape:
        raise ValueError(
            "Quadrature points and weights must have identical shapes, got "
            f"{tuple(quadrature_points.shape)} vs {tuple(quadrature_weights.shape)}."
        )
    if embeddings.shape[1] != projections.shape[0]:
        raise ValueError(
            f"Embedding dim {embeddings.shape[1]} does not match projection dim "
            f"{projections.shape[0]}."
        )

    projected = embeddings @ projections
    points = quadrature_points.to(device=embeddings.device, dtype=embeddings.dtype)
    weights = quadrature_weights.to(device=embeddings.device, dtype=embeddings.dtype)

    phases = projected.unsqueeze(-1) * points.view(1, 1, -1)
    empirical_real = torch.cos(phases).mean(dim=0)
    empirical_imag = torch.sin(phases).mean(dim=0)

    target_real = torch.exp(-0.5 * points.pow(2)).view(1, -1)
    squared_error = (empirical_real - target_real).pow(2) + empirical_imag.pow(2)
    kernel_weight = torch.exp(-points.pow(2)).view(1, -1)
    integrand = squared_error * kernel_weight
    return (integrand * weights.view(1, -1)).sum(dim=1).mean()


class RunningBuffer(nn.Module):
    """FIFO buffer of detached posterior-belief samples for SIGReg."""

    def __init__(self, *, capacity: int, hidden_dim: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.register_buffer("_storage", torch.empty(0, hidden_dim, dtype=torch.float32))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # _storage has dynamic size; resize before loading so shapes match.
        key = prefix + "_storage"
        if key in state_dict:
            self._storage = state_dict[key].new_empty(state_dict[key].shape)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __len__(self) -> int:
        return int(self._storage.shape[0])

    def flush(self) -> None:
        self._storage = self._storage.new_empty((0, self.hidden_dim))

    def get_all(self) -> Tensor:
        return self._storage

    def push(self, beliefs: BeliefState | Tensor) -> None:
        if self.capacity <= 0:
            self.flush()
            return

        samples = flatten_posterior_beliefs(beliefs)
        if samples.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"RunningBuffer expected hidden dim {self.hidden_dim}, got {samples.shape[-1]}."
            )

        detached = samples.detach().to(device=self._storage.device, dtype=self._storage.dtype)
        updated = torch.cat([self._storage, detached], dim=0)
        if updated.shape[0] > self.capacity:
            updated = updated[-self.capacity :]
        self._storage = updated


class SIGRegLoss(nn.Module):
    """Compute SIGReg on posterior beliefs with a detached running buffer."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.running_buffer = RunningBuffer(
            capacity=config.sigreg.buffer_size,
            hidden_dim=config.hidden_dim,
        )

        projections = torch.randn(
            config.hidden_dim,
            config.sigreg.num_projections,
            dtype=torch.float32,
        )
        self.register_buffer("projections", F.normalize(projections, dim=0))

        quadrature_points, quadrature_weights = _gauss_legendre_rule(
            order=config.sigreg.num_quadrature,
            interval_radius=config.sigreg.integration_range,
        )
        self.register_buffer("quadrature_points", quadrature_points)
        self.register_buffer("quadrature_weights", quadrature_weights)

    def flush(self) -> None:
        self.running_buffer.flush()

    def forward(
        self,
        posterior_beliefs: BeliefState | Tensor,
        *,
        update_buffer: bool = True,
    ) -> SIGRegTerms:
        """Compute SIGReg on current beliefs only (no history mixing).

        EP and var losses are scaled by N (sample count) to match main
        branch semantics.  The running buffer is used only for monitoring
        (effective_rank etc.) and is NOT mixed into the EP computation.
        """
        current = flatten_posterior_beliefs(posterior_beliefs).to(dtype=torch.float32)
        if current.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"SIGReg expected hidden dim {self.hidden_dim}, got {current.shape[-1]}."
            )

        N = current.shape[0]

        # EP on current beliefs only — no history mixing (fixes main-vs-refactor drift)
        ep_unscaled = epps_pulley_loss(
            current,
            projections=self.projections.to(device=current.device),
            quadrature_points=self.quadrature_points,
            quadrature_weights=self.quadrature_weights,
        )
        var_unscaled, mean_std, min_std = vicreg_variance_loss(current)

        # Scale by N to match main branch (ep_loss * N, var_loss * N)
        ep = ep_unscaled * N
        var = var_unscaled * N

        if update_buffer:
            self.running_buffer.push(current)

        return SIGRegTerms(
            ep=ep,
            var=var,
            ep_unscaled=ep_unscaled,
            var_unscaled=var_unscaled,
            mean_std=mean_std,
            min_std=min_std,
            num_current=N,
            num_total=N,  # no history mixing, so total = current
        )


def _gauss_legendre_rule(order: int, interval_radius: float) -> tuple[Tensor, Tensor]:
    """Return Gauss-Legendre nodes and weights on `[-interval_radius, interval_radius]`."""

    if order <= 0:
        raise ValueError(f"order must be positive, got {order}.")
    if interval_radius <= 0:
        raise ValueError(f"interval_radius must be positive, got {interval_radius}.")

    nodes = [0.0] * order
    weights = [0.0] * order
    num_roots = (order + 1) // 2

    for idx in range(num_roots):
        root = math.cos(math.pi * (idx + 0.75) / (order + 0.5))

        for _ in range(100):
            poly, derivative = _legendre_polynomial_and_derivative(order, root)
            next_root = root - poly / derivative
            if abs(next_root - root) < 1e-14:
                root = next_root
                break
            root = next_root
        else:
            raise RuntimeError("Gauss-Legendre root finder failed to converge.")

        _, derivative = _legendre_polynomial_and_derivative(order, root)
        weight = 2.0 / ((1.0 - root * root) * derivative * derivative)
        mapped_root = interval_radius * root
        mapped_weight = interval_radius * weight

        nodes[idx] = -mapped_root
        nodes[order - 1 - idx] = mapped_root
        weights[idx] = mapped_weight
        weights[order - 1 - idx] = mapped_weight

    return (
        torch.tensor(nodes, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def _legendre_polynomial_and_derivative(order: int, x: float) -> tuple[float, float]:
    """Evaluate Legendre polynomial P_n(x) and its derivative."""

    if order == 0:
        return 1.0, 0.0
    if order == 1:
        return x, 1.0

    p_nm2 = 1.0
    p_nm1 = x
    for degree in range(2, order + 1):
        p_n = ((2 * degree - 1) * x * p_nm1 - (degree - 1) * p_nm2) / degree
        p_nm2, p_nm1 = p_nm1, p_n

    derivative = order * (x * p_nm1 - p_nm2) / (x * x - 1.0)
    return p_nm1, derivative
