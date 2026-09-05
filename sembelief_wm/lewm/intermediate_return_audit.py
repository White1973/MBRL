"""Read-only conditional-return audit from exact real intermediate states.

Each audit state fixes the real Sokoban board, the frozen posterior belief and
the elapsed environment time.  Repeated continuations change only the frozen
Actor's categorical sampling RNG.  This separates irreducible single-rollout
noise from a failure of the Critic to predict conditional mean return.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..types import BeliefState
from .grounding_audit import _ev, _icc, _icc_interval, _pearson, _quantiles


def _prediction_report(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    residual = target - prediction
    baseline_mse = float((target - target.mean()).square().mean())
    mse = float(residual.square().mean())
    return {
        "samples": len(target),
        "explained_variance": _ev(prediction, target),
        "pearson": _pearson(prediction, target),
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": float(residual.abs().mean()),
        "constant_baseline_mse": baseline_mse,
        "mse_improvement_over_constant": (
            1.0 - mse / max(baseline_mse, 1e-12)
        ),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
    }


def _bootstrap_prediction_interval(
    prediction: Tensor, target: Tensor, *, repeats: int, seed: int,
) -> dict[str, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    metrics: dict[str, list[float]] = {
        "explained_variance": [],
        "pearson": [],
        "mse_improvement_over_constant": [],
    }
    for _ in range(repeats):
        indices = torch.randint(0, len(target), (len(target),), generator=generator)
        report = _prediction_report(prediction[indices], target[indices])
        for name in metrics:
            metrics[name].append(report[name])
    result: dict[str, dict[str, float]] = {}
    for name, values in metrics.items():
        ordered = torch.tensor(values).sort().values
        result[name] = {
            "lower_95": float(ordered[max(0, int(0.025 * repeats) - 1)]),
            "upper_95": float(ordered[min(repeats - 1, int(0.975 * repeats))]),
        }
    return result


def _select_steps(length: int, states_per_episode: int) -> list[tuple[int, str]]:
    """Choose spread-out non-initial, non-terminal decision states."""
    candidates = list(range(1, length))
    if not candidates:
        return []
    count = min(states_per_episode, len(candidates))
    if count == 1:
        positions = [len(candidates) // 2]
    else:
        positions = [
            round(index * (len(candidates) - 1) / (count - 1))
            for index in range(count)
        ]
    labels = (
        ["middle"] if count == 1 else
        ["early", "late"] if count == 2 else
        ["early", *(["middle"] * (count - 2)), "late"]
    )
    return [(candidates[position], labels[index]) for index, position in enumerate(positions)]


def _set_elapsed_steps(env: Any, elapsed_steps: int) -> None:
    """Restore both wrapper and gym-sokoban horizon counters."""
    if not hasattr(env, "_step_count"):
        raise RuntimeError("intermediate audit env has no wrapper step counter")
    raw_env = getattr(env, "_env", None)
    if raw_env is None or not hasattr(raw_env, "num_env_steps"):
        raise RuntimeError("intermediate audit env has no raw step counter")
    env._step_count = elapsed_steps
    raw_env.num_env_steps = elapsed_steps


@torch.no_grad()
def run_intermediate_return_audit(
    *,
    pipeline: Any,
    evaluator: Any,
    output_path: str | Path,
    level_count: int = 32,
    states_per_episode: int = 3,
    continuation_repeats: int = 12,
    bootstrap_repeats: int = 2000,
    seed: int = 20261118,
    reward_scale: float = 0.1,
    critic_batch_size: int = 64,
    continuation_batch_size: int = 8,
    progress_every: int = 8,
) -> dict[str, Any]:
    sizes = (
        level_count, states_per_episode, continuation_repeats,
        bootstrap_repeats, critic_batch_size, continuation_batch_size,
    )
    if min(sizes) <= 0:
        raise ValueError("intermediate return audit sizes must be positive")
    levels = list(getattr(evaluator, "eval_levels", None) or [])
    if len(levels) < level_count:
        raise RuntimeError(
            f"intermediate audit requires {level_count} fixed levels, got {len(levels)}"
        )

    collector = evaluator.collector
    policy = pipeline.policy
    world_model = pipeline.world_model
    selected_levels = levels[:level_count]
    before_policy = {name: value._version for name, value in policy.named_parameters()}
    before_wm = {name: value._version for name, value in world_model.named_parameters()}

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
            level_count,
            levels=selected_levels,
            collect_tokenized=False,
            device=collector.device,
            dtype=collector.dtype,
        )
    finally:
        (
            collector.config.capture_policy_trajectory,
            collector.config.deterministic,
            collector.config.exploration_epsilon,
        ) = old

    records: list[dict[str, Any]] = []
    latent_states: list[Tensor] = []
    for level_index, episode in enumerate(collection.episodes):
        trajectory = episode.info.get("_policy_trajectory")
        required = ("states", "board_states", "room_fixed")
        if not trajectory or any(name not in trajectory for name in required):
            raise RuntimeError("intermediate audit trajectory lacks latent/board state")
        length = len(trajectory["states"])
        for elapsed_steps, bucket in _select_steps(length, states_per_episode):
            latent_states.append(trajectory["states"][elapsed_steps].clone())
            records.append({
                "level_index": level_index,
                "elapsed_steps": elapsed_steps,
                "remaining_steps": collector.config.max_steps - elapsed_steps,
                "trajectory_length": length,
                "bucket": bucket,
                "board_state": trajectory["board_states"][elapsed_steps].clone(),
                "room_fixed": trajectory["room_fixed"].clone(),
            })
    if len(records) < 2:
        raise RuntimeError("intermediate audit found fewer than two usable states")

    device = collector.device
    dtype = collector.dtype
    critic_parts: list[Tensor] = []
    for start in range(0, len(latent_states), critic_batch_size):
        states = torch.stack(latent_states[start:start + critic_batch_size]).to(
            device=device, dtype=dtype
        )
        zeros = torch.zeros(len(states), device=device, dtype=torch.long)
        critic_parts.append(policy.evaluate_values(states, zeros).float().cpu())
    critic_values = torch.cat(critic_parts)

    returns = torch.empty(len(records), continuation_repeats)
    successes = torch.empty(len(records), continuation_repeats)
    gamma = float(pipeline.config.gamma)
    for state_index, (record, latent_cpu) in enumerate(zip(records, latent_states, strict=True)):
        for repeat_start in range(0, continuation_repeats, continuation_batch_size):
            batch_repeats = min(
                continuation_batch_size, continuation_repeats - repeat_start
            )
            batch_seed = seed + 1_000_003 + state_index * 10_007 + repeat_start
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            envs = [
                collector.env_factory(batch_seed + local_index)
                for local_index in range(batch_repeats)
            ]
            try:
                for env in envs:
                    env.reset_with_room(record["board_state"], record["room_fixed"])
                    _set_elapsed_steps(env, int(record["elapsed_steps"]))
                belief = BeliefState(
                    latent_cpu.unsqueeze(0).expand(batch_repeats, -1, -1)
                    .clone().to(device=device, dtype=dtype)
                )
                active_ids = list(range(batch_repeats))
                batch_returns = torch.zeros(batch_repeats)
                batch_successes = torch.zeros(batch_repeats)
                local_step = 0
                while active_ids and local_step < int(record["remaining_steps"]):
                    env_ids = collector.env_id_tensor.expand(len(active_ids))
                    action, _log_prob, _entropy, _value = policy.act(
                        belief.slots,
                        env_ids=env_ids,
                        deterministic=False,
                    )
                    survivor_positions: list[int] = []
                    survivor_actions: list[Tensor] = []
                    survivor_tokens: list[Tensor] = []
                    survivor_ids: list[int] = []
                    for position, local_id in enumerate(active_ids):
                        action_item = action[position:position + 1]
                        obs, reward, done, info = envs[local_id].step(
                            collector.model_to_env_action(int(action_item.item()))
                        )
                        batch_returns[local_id] += (
                            (gamma ** local_step) * float(reward) * reward_scale
                        )
                        if bool(info.get("success", False)):
                            batch_successes[local_id] = 1.0
                        if not done:
                            survivor_positions.append(position)
                            survivor_actions.append(action_item)
                            survivor_tokens.append(collector.tokenizer.tokenize(obs))
                            survivor_ids.append(local_id)
                    local_step += 1
                    if survivor_ids:
                        survivor_index = torch.tensor(
                            survivor_positions, device=device, dtype=torch.long
                        )
                        survivor_belief = BeliefState(
                            belief.slots.index_select(0, survivor_index)
                        )
                        survivor_action = torch.cat(survivor_actions).to(device)
                        observation_tokens = torch.stack(survivor_tokens).to(device, dtype)
                        survivor_env_ids = collector.env_id_tensor.expand(len(survivor_ids))
                        belief = world_model.posterior_step(
                            prev_belief=survivor_belief,
                            prev_actions=(
                                survivor_action + collector.wm_action_id_offset
                            ),
                            observation_tokens=observation_tokens,
                            env_ids=survivor_env_ids,
                        )
                    active_ids = survivor_ids
                repeat_end = repeat_start + batch_repeats
                returns[state_index, repeat_start:repeat_end] = batch_returns
                successes[state_index, repeat_start:repeat_end] = batch_successes
            finally:
                for env in envs:
                    close_fn = getattr(env, "close", None)
                    if callable(close_fn):
                        close_fn()
        if progress_every > 0 and (
            (state_index + 1) % progress_every == 0 or state_index + 1 == len(records)
        ):
            print(
                "  intermediate continuation progress: "
                f"{state_index + 1}/{len(records)} states "
                f"({(state_index + 1) * continuation_repeats} continuations)",
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
            "intermediate audit mutated parameters: "
            f"policy={changed_policy[:8]}, wm={changed_wm[:8]}"
        )

    conditional_means = returns.mean(1)
    within_std = returns.std(1, unbiased=False)
    return_icc = _icc(returns)
    return_icc["interval_95"] = _icc_interval(
        returns, repeats=bootstrap_repeats, seed=seed + 11
    )
    signal = return_icc["signal_variance"]
    noise = return_icc["within_level_noise_variance"]
    return_icc["conditional_mean_reliability"] = (
        signal / max(signal + noise / continuation_repeats, 1e-12)
    )
    mean_report = _prediction_report(critic_values, conditional_means)
    mean_report["bootstrap_interval_95"] = _bootstrap_prediction_interval(
        critic_values, conditional_means, repeats=bootstrap_repeats, seed=seed + 12
    )
    first_report = _prediction_report(critic_values, returns[:, 0])
    all_report = _prediction_report(
        critic_values[:, None].expand_as(returns), returns
    )

    state_rows = []
    for index, record in enumerate(records):
        state_rows.append({
            "state_index": index,
            "level_index": record["level_index"],
            "elapsed_steps": record["elapsed_steps"],
            "remaining_steps": record["remaining_steps"],
            "trajectory_length": record["trajectory_length"],
            "bucket": record["bucket"],
            "critic_value": float(critic_values[index]),
            "conditional_mean_return": float(conditional_means[index]),
            "within_state_return_std": float(within_std[index]),
            "conditional_success_probability": float(successes[index].mean()),
        })

    levels_payload = json.dumps(
        selected_levels, sort_keys=True, separators=(",", ":")
    ).encode()
    report = {
        "format": "lewm_exact_intermediate_conditional_return_audit_v1",
        "config": {
            "levels": level_count,
            "states_per_episode_requested": states_per_episode,
            "states": len(records),
            "continuation_repeats": continuation_repeats,
            "continuation_batch_size": continuation_batch_size,
            "continuations": len(records) * continuation_repeats,
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
            "gamma": gamma,
            "reward_scale": reward_scale,
            "max_episode_steps": collector.config.max_steps,
            "levels_sha256": hashlib.sha256(levels_payload).hexdigest(),
        },
        "conditioning_contract": {
            "fixed_per_state": [
                "real room_state", "room_fixed", "posterior latent",
                "elapsed environment steps", "remaining horizon",
            ],
            "varied_per_repeat": "frozen Actor categorical sampling RNG only",
            "posterior_recomputed_at_start": False,
            "continuation_posterior_updates": True,
        },
        "parameter_mutation": {
            "policy_changed_tensors": len(changed_policy),
            "world_model_changed_tensors": len(changed_wm),
            "check": "torch_tensor_version_counter",
        },
        "return_learnability": {
            "single_continuation_return": _quantiles(returns),
            "conditional_mean_return": _quantiles(conditional_means),
            "within_state_return_std": _quantiles(within_std),
            "icc": return_icc,
            "success_probability": _quantiles(successes.mean(1)),
        },
        "critic_grounding": {
            "critic_value": _quantiles(critic_values),
            "vs_conditional_mean_return": mean_report,
            "vs_first_single_continuation": first_report,
            "vs_all_single_continuations": all_report,
        },
        "state_results": state_rows,
        "interpretation_contract": {
            "label_noise_problem": (
                "single-rollout ICC is weak but conditional-mean reliability rises "
                "with repeats"
            ),
            "critic_or_latent_grounding_failure": (
                "conditional means are reliable but Critic-vs-conditional-mean EV/MSE "
                "improvement remains weak"
            ),
            "single_rollout_evaluation_failure": (
                "Critic agrees with conditional means but appears weak against one rollout"
            ),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "Le-WM intermediate audit: "
        f"states={len(records)} repeats={continuation_repeats} "
        f"ICC={return_icc['icc_single_rollout']:.4f} "
        f"mean_reliability={return_icc['conditional_mean_reliability']:.4f} "
        f"critic_mean_EV={mean_report['explained_variance']:.4f} "
        f"critic_mean_gain={mean_report['mse_improvement_over_constant']:.2%}",
        flush=True,
    )
    print(f"Le-WM intermediate audit report: {output}", flush=True)
    return report
