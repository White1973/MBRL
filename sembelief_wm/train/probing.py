"""Belief probing for SemBelief-WM evaluation.

Train linear probes from frozen posterior beliefs to structured state labels
(player position, box positions, etc.) to measure whether the belief space
encodes interpretable environment state.

Usage:
    probe = BeliefProbe.from_config(config, label_spec)
    probe.fit(beliefs, labels)
    metrics = probe.evaluate(beliefs, labels)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ProbeSpec:
    """Specification for one probing task."""
    name: str
    label_type: Literal["classification", "regression"]
    num_classes: int = 0  # for classification
    output_dim: int = 0   # for regression (e.g., 2 for (row, col))


@dataclass
class ProbeMetrics:
    """Results from evaluating a probe."""
    name: str
    accuracy: float = 0.0       # classification accuracy (if applicable)
    mse: float = 0.0            # regression MSE (if applicable)
    baseline_acc: float = 0.0   # majority-class baseline (classification)
    baseline_mse: float = 0.0   # mean-prediction baseline (regression)


class LinearProbe(nn.Module):
    """Single linear probe: frozen belief -> label prediction."""

    def __init__(self, input_dim: int, spec: ProbeSpec) -> None:
        super().__init__()
        self.spec = spec
        if spec.label_type == "classification":
            self.head = nn.Linear(input_dim, spec.num_classes)
        else:
            self.head = nn.Linear(input_dim, spec.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class BeliefProbe:
    """Collection of linear probes trained on frozen beliefs.

    Probes are trained independently with a simple SGD loop.
    Beliefs are expected to be pre-extracted and frozen (no gradients flow back).
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_belief_slots: int,
        specs: list[ProbeSpec],
        pool: Literal["mean", "flatten"] = "mean",
        device: torch.device | str = "cpu",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_belief_slots = num_belief_slots
        self.pool = pool
        self.device = torch.device(device)

        if pool == "flatten":
            input_dim = num_belief_slots * hidden_dim
        else:
            input_dim = hidden_dim

        self.probes: dict[str, LinearProbe] = {}
        for spec in specs:
            probe = LinearProbe(input_dim, spec).to(self.device)
            self.probes[spec.name] = probe

    def _pool_beliefs(self, beliefs: Tensor) -> Tensor:
        """Pool belief slots: (N, K, D) -> (N, input_dim)."""
        if beliefs.ndim == 4:
            # (B, T, K, D) -> (B*T, K, D)
            beliefs = beliefs.reshape(-1, beliefs.shape[-2], beliefs.shape[-1])
        if self.pool == "flatten":
            return beliefs.reshape(beliefs.shape[0], -1)
        return beliefs.mean(dim=1)  # (N, D)

    def fit(
        self,
        beliefs: Tensor,
        labels: dict[str, Tensor],
        *,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
    ) -> dict[str, list[float]]:
        """Train all probes on frozen beliefs.

        Args:
            beliefs: (N, K, D) frozen posterior beliefs
            labels: dict mapping probe name -> label tensor
                    classification: (N,) int64
                    regression: (N, output_dim) float

        Returns:
            Training loss history per probe.
        """
        pooled = self._pool_beliefs(beliefs.detach()).to(self.device)
        N = pooled.shape[0]
        history: dict[str, list[float]] = {}

        for name, probe in self.probes.items():
            if name not in labels:
                continue
            target = labels[name].to(self.device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
            losses: list[float] = []
            probe.train()

            for epoch in range(epochs):
                perm = torch.randperm(N, device=self.device)
                epoch_loss = 0.0
                num_batches = 0
                for i in range(0, N, batch_size):
                    idx = perm[i:i + batch_size]
                    x = pooled[idx]
                    y = target[idx]
                    logits = probe(x)

                    if probe.spec.label_type == "classification":
                        loss = F.cross_entropy(logits, y)
                    else:
                        loss = F.mse_loss(logits, y.float())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / max(num_batches, 1)
                losses.append(avg_loss)

            history[name] = losses
        return history

    @torch.no_grad()
    def evaluate(
        self,
        beliefs: Tensor,
        labels: dict[str, Tensor],
    ) -> list[ProbeMetrics]:
        """Evaluate all probes on held-out beliefs."""
        pooled = self._pool_beliefs(beliefs.detach()).to(self.device)
        results: list[ProbeMetrics] = []

        for name, probe in self.probes.items():
            if name not in labels:
                continue
            target = labels[name].to(self.device)
            probe.eval()
            logits = probe(pooled)

            if probe.spec.label_type == "classification":
                preds = logits.argmax(dim=-1)
                acc = (preds == target).float().mean().item()
                # Majority class baseline
                counts = torch.bincount(target, minlength=probe.spec.num_classes)
                baseline_acc = (counts.max().item() / max(target.shape[0], 1))
                results.append(ProbeMetrics(
                    name=name,
                    accuracy=acc,
                    baseline_acc=baseline_acc,
                ))
            else:
                mse = F.mse_loss(logits, target.float()).item()
                # Mean-prediction baseline
                mean_pred = target.float().mean(dim=0, keepdim=True).expand_as(target)
                baseline_mse = F.mse_loss(mean_pred, target.float()).item()
                results.append(ProbeMetrics(
                    name=name,
                    mse=mse,
                    baseline_mse=baseline_mse,
                ))

        return results


def build_sokoban_probes(
    hidden_dim: int,
    num_belief_slots: int,
    grid_size: int = 6,
    device: torch.device | str = "cpu",
) -> BeliefProbe:
    """Build probes for Sokoban structured state labels.

    Probes:
    - player_pos: classification over grid_size^2 positions
    - box_pos: classification over grid_size^2 positions (first box)
    - boxes_on_target: binary classification (any box on target?)
    - is_solvable: binary classification
    """
    specs = [
        ProbeSpec(
            name="player_pos",
            label_type="classification",
            num_classes=grid_size * grid_size,
        ),
        ProbeSpec(
            name="box_pos",
            label_type="classification",
            num_classes=grid_size * grid_size,
        ),
        ProbeSpec(
            name="boxes_on_target",
            label_type="classification",
            num_classes=2,
        ),
        ProbeSpec(
            name="is_solvable",
            label_type="classification",
            num_classes=2,
        ),
    ]
    return BeliefProbe(
        hidden_dim=hidden_dim,
        num_belief_slots=num_belief_slots,
        specs=specs,
        pool="mean",
        device=device,
    )


def extract_sokoban_labels(
    state_labels: list[dict],
    grid_size: int = 6,
) -> dict[str, Tensor]:
    """Extract probe labels from Sokoban state_labels metadata.

    Args:
        state_labels: list of per-step dicts with keys:
            player_pos, box_positions, boxes_on_target, is_solvable
        grid_size: grid dimension for position encoding

    Returns:
        Dict of label tensors suitable for BeliefProbe.fit/evaluate.
    """
    player_pos_labels = []
    box_pos_labels = []
    boxes_on_target_labels = []
    is_solvable_labels = []

    for sl in state_labels:
        # Player position -> flat grid index
        pr, pc = sl["player_pos"]
        player_pos_labels.append(pr * grid_size + pc)

        # First box position -> flat grid index
        if sl["box_positions"]:
            br, bc = sl["box_positions"][0]
            box_pos_labels.append(br * grid_size + bc)
        else:
            box_pos_labels.append(0)

        # Any box on target?
        bot = sl.get("boxes_on_target", [])
        boxes_on_target_labels.append(1 if any(bot) else 0)

        # Is solvable?
        is_solvable_labels.append(1 if sl.get("is_solvable", False) else 0)

    return {
        "player_pos": torch.tensor(player_pos_labels, dtype=torch.long),
        "box_pos": torch.tensor(box_pos_labels, dtype=torch.long),
        "boxes_on_target": torch.tensor(boxes_on_target_labels, dtype=torch.long),
        "is_solvable": torch.tensor(is_solvable_labels, dtype=torch.long),
    }
