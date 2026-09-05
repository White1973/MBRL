"""Read-only, episode-grouped validation for a trained Le-WM scalar Critic."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    if len(target) == 0:
        raise ValueError("frozen Critic validation received no transitions")
    target_variance = target.var(unbiased=False)
    residual_variance = (target - prediction).var(unbiased=False)
    explained_variance = (
        0.0 if float(target_variance) < 1e-8
        else float(1.0 - residual_variance / target_variance)
    )
    mse = float((prediction - target).pow(2).mean())
    # The training replay mean is intentionally unavailable to this independent
    # evaluator.  The test-set mean is the optimal constant on the test set and
    # therefore gives the Critic a conservative, leakage-free comparator.
    baseline_mse = float((target - target.mean()).pow(2).mean())
    improvement = 1.0 - mse / max(baseline_mse, 1e-12)
    return {
        "explained_variance": explained_variance,
        "mse": mse,
        "constant_baseline_mse": baseline_mse,
        "mse_improvement": improvement,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(unbiased=False)),
        "transitions": float(len(target)),
    }


@torch.no_grad()
def evaluate_episode_trajectories(
    *, policy: Any, episodes: list[Any], gamma: float, reward_scale: float,
) -> tuple[dict[str, float], list[Tensor], list[Tensor]]:
    """Evaluate returns without flattening away the episode grouping."""
    episode_predictions: list[Tensor] = []
    episode_targets: list[Tensor] = []
    for episode in episodes:
        trajectory = episode.info.get("_policy_trajectory")
        if not trajectory:
            continue
        rewards = trajectory["rewards"].float() * reward_scale
        target = torch.empty_like(rewards)
        running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            target[index] = running
        states = trajectory["states"].to(next(policy.parameters()).device)
        actions = trajectory["actions"].to(states.device)
        prediction = policy.evaluate_values(states, actions).float()
        episode_predictions.append(prediction.detach().cpu())
        episode_targets.append(target.detach().cpu())
    if not episode_targets:
        raise RuntimeError("fixed-level collection produced no policy trajectories")
    metrics = _metrics(
        torch.cat(episode_predictions), torch.cat(episode_targets)
    )
    metrics["episodes"] = float(len(episode_targets))
    return metrics, episode_predictions, episode_targets


def episode_bootstrap_intervals(
    episode_predictions: list[Tensor], episode_targets: list[Tensor], *,
    repeats: int, seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap whole episodes so within-trajectory transitions stay grouped."""
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    if len(episode_predictions) != len(episode_targets):
        raise ValueError("prediction/target episode counts differ")
    count = len(episode_targets)
    generator = torch.Generator().manual_seed(seed)
    tracked = {"explained_variance": [], "mse_improvement": [], "mse": []}
    for _ in range(repeats):
        indices = torch.randint(0, count, (count,), generator=generator).tolist()
        sample = _metrics(
            torch.cat([episode_predictions[index] for index in indices]),
            torch.cat([episode_targets[index] for index in indices]),
        )
        for name in tracked:
            tracked[name].append(sample[name])
    intervals: dict[str, dict[str, float]] = {}
    for name, values in tracked.items():
        ordered = torch.tensor(values, dtype=torch.float64).sort().values
        lower_index = max(0, int(0.025 * repeats) - 1)
        upper_index = min(repeats - 1, int(0.975 * repeats))
        intervals[name] = {
            "lower_95": float(ordered[lower_index]),
            "upper_95": float(ordered[upper_index]),
        }
    return intervals


def grouped_bootstrap_intervals(
    grouped_predictions: list[list[Tensor]],
    grouped_targets: list[list[Tensor]],
    *, repeats: int, seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap independent levels while retaining their rollout replicates."""
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    if len(grouped_predictions) != len(grouped_targets):
        raise ValueError("prediction/target group counts differ")
    if not grouped_targets or any(not group for group in grouped_targets):
        raise ValueError("bootstrap groups must be non-empty")
    for predictions, targets in zip(
        grouped_predictions, grouped_targets, strict=True
    ):
        if len(predictions) != len(targets):
            raise ValueError("prediction/target replicate counts differ")
    count = len(grouped_targets)
    generator = torch.Generator().manual_seed(seed)
    tracked = {"explained_variance": [], "mse_improvement": [], "mse": []}
    for _ in range(repeats):
        group_indices = torch.randint(
            0, count, (count,), generator=generator
        ).tolist()
        predictions = [
            episode
            for index in group_indices
            for episode in grouped_predictions[index]
        ]
        targets = [
            episode
            for index in group_indices
            for episode in grouped_targets[index]
        ]
        sample = _metrics(torch.cat(predictions), torch.cat(targets))
        for name in tracked:
            tracked[name].append(sample[name])
    intervals: dict[str, dict[str, float]] = {}
    for name, values in tracked.items():
        ordered = torch.tensor(values, dtype=torch.float64).sort().values
        lower_index = max(0, int(0.025 * repeats) - 1)
        upper_index = min(repeats - 1, int(0.975 * repeats))
        intervals[name] = {
            "lower_95": float(ordered[lower_index]),
            "upper_95": float(ordered[upper_index]),
        }
    return intervals
