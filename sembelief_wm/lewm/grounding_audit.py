"""Read-only audit of return learnability and posterior/prior value semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..types import BeliefState


def _quantiles(values: Tensor) -> dict[str, float]:
    values = values.float().reshape(-1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _icc(matrix: Tensor) -> dict[str, float]:
    """One-way random-effects ICC for [independent levels, replicates]."""
    matrix = matrix.float()
    levels, replicates = matrix.shape
    if levels < 2 or replicates < 2:
        raise ValueError("ICC requires at least two levels and two replicates")
    level_mean = matrix.mean(1)
    grand_mean = matrix.mean()
    ms_between = replicates * (level_mean - grand_mean).square().sum() / (levels - 1)
    ms_within = (matrix - level_mean[:, None]).square().sum() / (
        levels * (replicates - 1)
    )
    denominator = ms_between + (replicates - 1) * ms_within
    raw_icc = float((ms_between - ms_within) / denominator.clamp_min(1e-12))
    signal_variance = float(((ms_between - ms_within) / replicates).clamp_min(0))
    noise_variance = float(ms_within)
    return {
        "icc_single_rollout": raw_icc,
        "signal_variance": signal_variance,
        "within_level_noise_variance": noise_variance,
        "predictable_variance_fraction": (
            signal_variance / max(signal_variance + noise_variance, 1e-12)
        ),
        "between_level_mean_variance": float(level_mean.var(unbiased=False)),
    }


def _icc_interval(matrix: Tensor, *, repeats: int, seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(repeats):
        indices = torch.randint(
            0, len(matrix), (len(matrix),), generator=generator
        )
        values.append(_icc(matrix[indices])["icc_single_rollout"])
    ordered = torch.tensor(values).sort().values
    return {
        "lower_95": float(ordered[max(0, int(0.025 * repeats) - 1)]),
        "upper_95": float(ordered[min(repeats - 1, int(0.975 * repeats))]),
    }


def _pearson(first: Tensor, second: Tensor) -> float:
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum() / denominator.clamp_min(1e-12))


def _ev(prediction: Tensor, target: Tensor) -> float:
    target = target.float()
    prediction = prediction.float()
    variance = target.var(unbiased=False)
    if float(variance) < 1e-12:
        return 0.0
    return float(1.0 - (target - prediction).var(unbiased=False) / variance)


def _value_report(actual: Tensor, imagined: Tensor, target: Tensor) -> dict[str, Any]:
    actual = actual.float(); imagined = imagined.float(); target = target.float()
    difference = imagined - actual
    rmse = float(difference.square().mean().sqrt())
    return {
        "samples": len(actual),
        "actual_posterior_value": _quantiles(actual),
        "imagined_prior_value": _quantiles(imagined),
        "value_difference": _quantiles(difference),
        "value_mae": float(difference.abs().mean()),
        "value_rmse": rmse,
        "value_nrmse_by_actual_std": rmse / max(float(actual.std(unbiased=False)), 1e-12),
        "value_pearson": _pearson(actual, imagined),
        "value_semantic_ev": _ev(imagined, actual),
        "actual_posterior_return_ev": _ev(actual, target),
        "imagined_prior_return_ev": _ev(imagined, target),
        "return_ev_degradation": _ev(actual, target) - _ev(imagined, target),
        "real_return": _quantiles(target),
    }


@torch.no_grad()
def run_grounding_audit(
    *, pipeline: Any, evaluator: Any, output_path: str | Path,
    level_count: int = 64, rollout_repeats: int = 8,
    bootstrap_repeats: int = 2000, seed: int = 20261018,
    reward_scale: float = 0.1, batch_size: int = 64,
) -> dict[str, Any]:
    if min(
        level_count, rollout_repeats, bootstrap_repeats, batch_size
    ) <= 0:
        raise ValueError("grounding audit sizes must be positive")
    levels = list(getattr(evaluator, "eval_levels", None) or [])
    if len(levels) < level_count:
        raise RuntimeError(
            f"grounding audit requires {level_count} fixed levels, got {len(levels)}"
        )
    selected = levels[:level_count]
    expanded = selected * rollout_repeats
    collector = evaluator.collector
    policy = pipeline.policy
    world_model = pipeline.world_model
    # Tensor version counters detect in-place parameter mutation without
    # cloning two multi-billion-parameter module trees onto CPU.
    before_policy = {
        name: value._version for name, value in policy.named_parameters()
    }
    before_wm = {
        name: value._version for name, value in world_model.named_parameters()
    }
    old = (
        collector.config.capture_policy_trajectory,
        collector.config.deterministic,
        collector.config.exploration_epsilon,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        collector.config.capture_policy_trajectory = True
        collector.config.deterministic = False
        collector.config.exploration_epsilon = 0.0
        collection = collector.collect(
            len(expanded), levels=expanded, collect_tokenized=False,
            device=collector.device, dtype=collector.dtype,
        )
    finally:
        (
            collector.config.capture_policy_trajectory,
            collector.config.deterministic,
            collector.config.exploration_epsilon,
        ) = old

    if len(collection.episodes) != level_count * rollout_repeats:
        raise RuntimeError("grounding audit episode count mismatch")
    initial_returns = torch.empty(level_count, rollout_repeats)
    successes = torch.empty(level_count, rollout_repeats)
    trajectories = []
    targets_by_episode = []
    for index, episode in enumerate(collection.episodes):
        trajectory = episode.info.get("_policy_trajectory")
        required = ("states", "next_states", "actions", "rewards")
        if not trajectory or any(name not in trajectory for name in required):
            raise RuntimeError("grounding audit real trajectory is incomplete")
        rewards = trajectory["rewards"].float() * reward_scale
        targets = torch.empty_like(rewards)
        running = torch.zeros((), dtype=rewards.dtype)
        for step in range(len(rewards) - 1, -1, -1):
            running = rewards[step] + float(pipeline.config.gamma) * running
            targets[step] = running
        repeat = index // level_count
        level = index % level_count
        initial_returns[level, repeat] = targets[0]
        successes[level, repeat] = float(episode.success)
        trajectories.append(trajectory)
        targets_by_episode.append(targets)

    return_icc = _icc(initial_returns)
    return_icc["interval_95"] = _icc_interval(
        initial_returns, repeats=bootstrap_repeats, seed=seed + 1
    )
    success_icc = _icc(successes)
    success_icc["interval_95"] = _icc_interval(
        successes, repeats=bootstrap_repeats, seed=seed + 2
    )

    device = next(policy.parameters()).device
    dtype = next(world_model.parameters()).dtype
    horizon_reports: dict[str, Any] = {}
    for horizon in (1, 2, 3):
        actual_values: list[Tensor] = []
        imagined_values: list[Tensor] = []
        return_targets: list[Tensor] = []
        latent_rms: list[Tensor] = []
        starts: list[Tensor] = []
        endpoints: list[Tensor] = []
        action_sequences: list[Tensor] = []
        local_targets: list[Tensor] = []

        def flush() -> None:
            if not starts:
                return
            current = torch.stack(starts).to(device=device, dtype=dtype)
            actual = torch.stack(endpoints).to(device=device, dtype=dtype)
            actions = torch.stack(action_sequences).long().to(device)
            imagined = BeliefState(current)
            for step in range(horizon):
                imagined = world_model.prior_step(imagined, actions[:, step])
            zeros = torch.zeros(len(current), device=device, dtype=torch.long)
            actual_values.append(
                policy.evaluate_values(actual, zeros).float().cpu()
            )
            imagined_values.append(
                policy.evaluate_values(imagined.slots, zeros).float().cpu()
            )
            return_targets.append(torch.stack(local_targets).float())
            latent_rms.append(
                (imagined.slots.float() - actual.float())
                .square().flatten(1).mean(1).sqrt().cpu()
            )
            starts.clear(); endpoints.clear(); action_sequences.clear(); local_targets.clear()

        for trajectory, targets in zip(
            trajectories, targets_by_episode, strict=True
        ):
            length = len(trajectory["actions"])
            for start in range(max(0, length - horizon)):
                starts.append(trajectory["states"][start])
                endpoints.append(trajectory["states"][start + horizon])
                action_sequences.append(
                    trajectory["actions"][start:start + horizon]
                )
                local_targets.append(targets[start + horizon])
                if len(starts) >= batch_size:
                    flush()
        flush()
        actual_tensor = torch.cat(actual_values)
        imagined_tensor = torch.cat(imagined_values)
        target_tensor = torch.cat(return_targets)
        latent_tensor = torch.cat(latent_rms)
        horizon_reports[f"h{horizon}"] = {
            **_value_report(actual_tensor, imagined_tensor, target_tensor),
            "latent_prior_vs_actual_posterior_rms": _quantiles(latent_tensor),
        }
        print(
            f"Le-WM grounding audit H{horizon}: "
            f"samples={len(actual_tensor)} "
            f"value_pearson={horizon_reports[f'h{horizon}']['value_pearson']:.4f} "
            f"actual_return_ev={horizon_reports[f'h{horizon}']['actual_posterior_return_ev']:.4f} "
            f"imagined_return_ev={horizon_reports[f'h{horizon}']['imagined_prior_return_ev']:.4f}",
            flush=True,
        )

    changed_policy = [
        name for name, value in policy.named_parameters()
        if value._version != before_policy[name]
    ]
    changed_wm = [
        name for name, value in world_model.named_parameters()
        if value._version != before_wm[name]
    ]
    if changed_policy or changed_wm:
        raise RuntimeError(
            "grounding audit mutated parameters: "
            f"policy={changed_policy[:8]}, wm={changed_wm[:8]}"
        )
    levels_payload = json.dumps(
        selected, sort_keys=True, separators=(",", ":")
    ).encode()
    report = {
        "format": "lewm_return_prior_grounding_audit_v1",
        "config": {
            "levels": level_count,
            "rollout_repeats": rollout_repeats,
            "episodes": len(collection.episodes),
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
            "reward_scale": reward_scale,
            "levels_sha256": hashlib.sha256(levels_payload).hexdigest(),
        },
        "parameter_mutation": {
            "policy_changed_tensors": len(changed_policy),
            "world_model_changed_tensors": len(changed_wm),
            "check": "torch_tensor_version_counter",
        },
        "return_target": {
            "initial_return": _quantiles(initial_returns),
            "initial_return_icc": return_icc,
            "per_level_mean_return": _quantiles(initial_returns.mean(1)),
            "success_rate": float(successes.mean()),
            "levels_with_any_success": float(successes.any(1).float().mean()),
            "success_icc": success_icc,
        },
        "posterior_prior": horizon_reports,
        "interpretation_contract": {
            "return_noise_dominated": (
                "initial-return ICC is near zero or its interval includes zero"
            ),
            "posterior_value_failure": (
                "actual_posterior_return_ev is below the value threshold"
            ),
            "prior_value_semantics_failure": (
                "actual posterior EV is useful but imagined prior EV degrades, "
                "or actual/imaged value agreement is weak"
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Le-WM grounding audit report: {output}", flush=True)
    return report
