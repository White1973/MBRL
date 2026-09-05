"""Read-only current-policy action grounding from exact real intermediates.

For each fixed board/posterior/time state, force every first action and compare
the deployed H-step imagined PPO objective with full real continuation return.
No proxy Q head is trained: this audits the exact Actor/WM/Reward/Critic stack
that would supply PPO advantages.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..rl.imagined_real_action_ranking_audit import ranking_metrics
from ..types import BeliefState
from .intermediate_return_audit import _select_steps, _set_elapsed_steps


def _metric_intervals(
    imagined_q: Tensor,
    real_q: Tensor,
    actor_probabilities: Tensor,
    group_ids: Tensor,
    *,
    repeats: int,
    seed: int,
    tie_epsilon: float,
    advantage_margin: float,
) -> dict[str, dict[str, float]]:
    """Bootstrap independent levels while retaining their state clusters."""
    groups = torch.unique(group_ids).tolist()
    generator = torch.Generator().manual_seed(seed)
    tracked: dict[str, list[float]] = defaultdict(list)
    names = (
        "top1_accuracy", "pairwise_accuracy", "advantage_sign_accuracy",
        "harmful_positive_rate", "mean_regret",
    )
    for _ in range(repeats):
        sampled = torch.randint(0, len(groups), (len(groups),), generator=generator)
        indices = torch.cat([
            torch.nonzero(group_ids == groups[index], as_tuple=False).flatten()
            for index in sampled.tolist()
        ])
        values = ranking_metrics(
            imagined_q[indices], real_q[indices], actor_probabilities[indices],
            tie_epsilon=tie_epsilon, advantage_margin=advantage_margin,
        )
        for name in names:
            value = values[name]
            if value is not None:
                tracked[name].append(float(value))
    intervals: dict[str, dict[str, float]] = {}
    for name in names:
        values = tracked.get(name, [])
        if not values:
            continue
        ordered = torch.tensor(values).sort().values
        intervals[name] = {
            "lower_95": float(ordered[max(0, int(0.025 * len(ordered)) - 1)]),
            "upper_95": float(ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]),
        }
    return intervals


@torch.no_grad()
def _imagined_action_values(
    *,
    pipeline: Any,
    policy: Any,
    latent_states: list[Tensor],
    repeats: int,
    repeat_batch_size: int,
    seed: int,
    progress_every: int,
) -> tuple[Tensor, Tensor, Tensor]:
    imagined = getattr(pipeline.imagine_fn, "__self__", None)
    required = (
        "dynamics_step", "predict_reward", "reward_transform", "config",
    )
    if imagined is None or any(not hasattr(imagined, name) for name in required):
        raise RuntimeError("action gate requires the production ImaginedCollector")
    horizon = int(imagined.config.horizon)
    gamma = float(pipeline.config.gamma)
    device = next(policy.parameters()).device
    dtype = next(pipeline.world_model.parameters()).dtype
    values = torch.empty(len(latent_states), 4)
    h1_probabilities = torch.empty_like(values)
    endpoint_probabilities = torch.empty_like(values)

    for state_index, latent_cpu in enumerate(latent_states):
        for action_id in range(4):
            repeat_values: list[Tensor] = []
            repeat_h1: list[Tensor] = []
            repeat_endpoint: list[Tensor] = []
            for repeat_start in range(0, repeats, repeat_batch_size):
                count = min(repeat_batch_size, repeats - repeat_start)
                # Reuse the same continuation RNG stream across forced actions.
                local_seed = seed + state_index * 10007 + repeat_start
                torch.manual_seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)
                belief = BeliefState(
                    latent_cpu.unsqueeze(0).expand(count, -1, -1)
                    .clone().to(device=device, dtype=dtype)
                )
                total = torch.zeros(count, device=device)
                finished = torch.zeros(count, device=device, dtype=torch.bool)
                endpoint_probability = torch.zeros(count, device=device)
                for step in range(horizon):
                    if step == 0:
                        action = torch.full(
                            (count,), action_id, device=device, dtype=torch.long
                        )
                    else:
                        action = policy.act(belief.slots, deterministic=False)[0]
                    next_belief = imagined.dynamics_step(belief, action)
                    logits = imagined.predict_reward(next_belief)
                    reward = imagined.reward_transform(
                        logits, step_index=step, horizon=horizon
                    ).reshape(-1).float()
                    probability = torch.sigmoid(logits).reshape(-1).float()
                    if step == 0:
                        repeat_h1.append(probability.cpu())
                    endpoint_probability = probability
                    if imagined.config.termination_mode == "predicted_success":
                        success = (
                            probability >= float(imagined.config.success_threshold)
                        ) & ~finished
                    else:
                        success = torch.zeros_like(finished)
                    reward = torch.where(~finished, reward, torch.zeros_like(reward))
                    total += (gamma ** step) * reward
                    finished |= success
                    belief = next_belief
                if imagined.config.bootstrap_with_value:
                    bootstrap_fn = getattr(policy, "bootstrap_value", None)
                    if callable(bootstrap_fn):
                        bootstrap = bootstrap_fn(belief.slots).reshape(-1).float()
                    else:
                        bootstrap = policy.act(
                            belief.slots, deterministic=False
                        )[3].reshape(-1).float()
                    bootstrap = torch.where(
                        finished, torch.zeros_like(bootstrap), bootstrap
                    )
                    total += (gamma ** horizon) * bootstrap
                repeat_values.append(total.cpu())
                repeat_endpoint.append(endpoint_probability.cpu())
            values[state_index, action_id] = torch.cat(repeat_values).mean()
            h1_probabilities[state_index, action_id] = torch.cat(repeat_h1).mean()
            endpoint_probabilities[state_index, action_id] = torch.cat(
                repeat_endpoint
            ).mean()
        if progress_every > 0 and (
            (state_index + 1) % progress_every == 0
            or state_index + 1 == len(latent_states)
        ):
            print(
                "  imagined action grounding progress: "
                f"{state_index + 1}/{len(latent_states)} states",
                flush=True,
            )
    return values, h1_probabilities, endpoint_probabilities


@torch.no_grad()
def _real_action_values(
    *,
    pipeline: Any,
    evaluator: Any,
    records: list[dict[str, Any]],
    latent_states: list[Tensor],
    repeats: int,
    repeat_batch_size: int,
    reward_scale: float,
    seed: int,
    progress_every: int,
) -> tuple[Tensor, Tensor]:
    collector = evaluator.collector
    policy = pipeline.policy
    world_model = pipeline.world_model
    gamma = float(pipeline.config.gamma)
    device = collector.device
    dtype = collector.dtype
    values = torch.empty(len(records), 4)
    success_probabilities = torch.empty_like(values)

    for state_index, (record, latent_cpu) in enumerate(
        zip(records, latent_states, strict=True)
    ):
        for action_id in range(4):
            action_returns: list[Tensor] = []
            action_successes: list[Tensor] = []
            for repeat_start in range(0, repeats, repeat_batch_size):
                count = min(repeat_batch_size, repeats - repeat_start)
                local_seed = seed + state_index * 10007 + repeat_start
                torch.manual_seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)
                envs = [
                    collector.env_factory(local_seed + local_index)
                    for local_index in range(count)
                ]
                try:
                    for env in envs:
                        env.reset_with_room(record["board_state"], record["room_fixed"])
                        _set_elapsed_steps(env, int(record["elapsed_steps"]))
                    belief = BeliefState(
                        latent_cpu.unsqueeze(0).expand(count, -1, -1)
                        .clone().to(device=device, dtype=dtype)
                    )
                    active_ids = list(range(count))
                    batch_returns = torch.zeros(count)
                    batch_successes = torch.zeros(count)
                    local_step = 0
                    while active_ids and local_step < int(record["remaining_steps"]):
                        if local_step == 0:
                            action = torch.full(
                                (len(active_ids),), action_id,
                                device=device, dtype=torch.long,
                            )
                        else:
                            env_ids = collector.env_id_tensor.expand(len(active_ids))
                            action = policy.act(
                                belief.slots, env_ids=env_ids, deterministic=False
                            )[0]
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
                            indices = torch.tensor(
                                survivor_positions, device=device, dtype=torch.long
                            )
                            survivor_belief = BeliefState(
                                belief.slots.index_select(0, indices)
                            )
                            survivor_action = torch.cat(survivor_actions).to(device)
                            observation_tokens = torch.stack(survivor_tokens).to(
                                device, dtype
                            )
                            env_ids = collector.env_id_tensor.expand(len(survivor_ids))
                            belief = world_model.posterior_step(
                                prev_belief=survivor_belief,
                                prev_actions=(
                                    survivor_action + collector.wm_action_id_offset
                                ),
                                observation_tokens=observation_tokens,
                                env_ids=env_ids,
                            )
                        active_ids = survivor_ids
                    action_returns.append(batch_returns)
                    action_successes.append(batch_successes)
                finally:
                    for env in envs:
                        close_fn = getattr(env, "close", None)
                        if callable(close_fn):
                            close_fn()
            values[state_index, action_id] = torch.cat(action_returns).mean()
            success_probabilities[state_index, action_id] = torch.cat(
                action_successes
            ).mean()
        if progress_every > 0 and (
            (state_index + 1) % progress_every == 0
            or state_index + 1 == len(records)
        ):
            print(
                "  real action grounding progress: "
                f"{state_index + 1}/{len(records)} states",
                flush=True,
            )
    return values, success_probabilities


@torch.no_grad()
def run_action_grounding_audit(
    *,
    pipeline: Any,
    evaluator: Any,
    output_path: str | Path,
    level_count: int = 32,
    states_per_episode: int = 3,
    real_repeats: int = 8,
    imagined_repeats: int = 16,
    real_batch_size: int = 8,
    imagined_batch_size: int = 8,
    bootstrap_repeats: int = 2000,
    seed: int = 20261218,
    reward_scale: float = 0.1,
    tie_epsilon: float = 0.01,
    advantage_margin: float = 0.01,
    progress_every: int = 4,
) -> dict[str, Any]:
    sizes = (
        level_count, states_per_episode, real_repeats, imagined_repeats,
        real_batch_size, imagined_batch_size, bootstrap_repeats,
    )
    if min(sizes) <= 0:
        raise ValueError("action grounding audit sizes must be positive")
    levels = list(getattr(evaluator, "eval_levels", None) or [])
    if len(levels) < level_count:
        raise RuntimeError(
            f"action grounding requires {level_count} levels, got {len(levels)}"
        )
    selected_levels = levels[:level_count]
    collector = evaluator.collector
    policy = pipeline.policy
    world_model = pipeline.world_model
    imagined = getattr(pipeline.imagine_fn, "__self__", None)
    if imagined is None:
        raise RuntimeError("pipeline imagine_fn is not a bound collector")
    if int(imagined.config.horizon) != 3:
        raise RuntimeError("released action gate requires production H3")
    if imagined.config.termination_mode != "predicted_success":
        raise RuntimeError("released action gate requires predicted_success termination")
    if not bool(imagined.config.bootstrap_with_value):
        raise RuntimeError("released action gate requires production value bootstrap")
    if getattr(imagined, "relative_action_value", None) is not None:
        raise RuntimeError("released action gate requires relative shaping disabled")
    policy.set_deterministic_forward_mode()

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
            level_count, levels=selected_levels, collect_tokenized=False,
            device=collector.device, dtype=collector.dtype,
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
            raise RuntimeError("action grounding trajectory lacks latent/board state")
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
        raise RuntimeError("action grounding found fewer than two usable states")

    device = collector.device
    dtype = collector.dtype
    actor_logit_parts: list[Tensor] = []
    for start in range(0, len(latent_states), imagined_batch_size):
        state_tensor = torch.stack(
            latent_states[start:start + imagined_batch_size]
        ).to(device=device, dtype=dtype)
        actor_logit_parts.append(policy.actor_logits(state_tensor).float().cpu())
    actor_logits = torch.cat(actor_logit_parts)
    actor_probabilities = actor_logits.softmax(-1)
    imagined_q, h1_probability, endpoint_probability = _imagined_action_values(
        pipeline=pipeline, policy=policy, latent_states=latent_states,
        repeats=imagined_repeats, repeat_batch_size=imagined_batch_size,
        seed=seed + 100_000, progress_every=progress_every,
    )
    real_q, real_success = _real_action_values(
        pipeline=pipeline, evaluator=evaluator, records=records,
        latent_states=latent_states, repeats=real_repeats,
        repeat_batch_size=real_batch_size, reward_scale=reward_scale,
        seed=seed + 200_000, progress_every=progress_every,
    )
    group_ids = torch.tensor([record["level_index"] for record in records])
    metrics = ranking_metrics(
        imagined_q, real_q, actor_probabilities,
        tie_epsilon=tie_epsilon, advantage_margin=advantage_margin,
    )
    actor_metrics = ranking_metrics(
        actor_logits, real_q, actor_probabilities,
        tie_epsilon=tie_epsilon, advantage_margin=advantage_margin,
    )
    intervals = _metric_intervals(
        imagined_q, real_q, actor_probabilities, group_ids,
        repeats=bootstrap_repeats, seed=seed + 300_000,
        tie_epsilon=tie_epsilon, advantage_margin=advantage_margin,
    )

    reasons: list[str] = []
    if metrics["comparable_pairs"] < max(32, len(records) // 2):
        reasons.append("too_few_real_comparable_pairs")
    if metrics["pairwise_accuracy"] < 0.60:
        reasons.append("pairwise_accuracy_below_0.60")
    if intervals["pairwise_accuracy"]["lower_95"] <= 0.50:
        reasons.append("pairwise_accuracy_lower_95_not_above_chance")
    sign_accuracy = metrics["advantage_sign_accuracy"]
    if sign_accuracy is None or sign_accuracy < 0.60:
        reasons.append("advantage_sign_accuracy_below_0.60")
    harmful = metrics["harmful_positive_rate"]
    if harmful is None or harmful > 0.10:
        reasons.append("harmful_positive_rate_above_0.10")
    if metrics["pairwise_accuracy"] + 0.02 < actor_metrics["pairwise_accuracy"]:
        reasons.append("imagined_pairwise_worse_than_actor_baseline")
    if metrics["mean_regret"] > actor_metrics["mean_regret"] + 0.01:
        reasons.append("imagined_regret_worse_than_actor_baseline")

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
            "action grounding mutated parameters: "
            f"policy={changed_policy[:8]}, wm={changed_wm[:8]}"
        )

    state_rows = [{
        "state_index": index,
        "level_index": record["level_index"],
        "elapsed_steps": record["elapsed_steps"],
        "remaining_steps": record["remaining_steps"],
        "bucket": record["bucket"],
        "imagined_q": imagined_q[index].tolist(),
        "real_q": real_q[index].tolist(),
        "real_success_probability": real_success[index].tolist(),
        "actor_probabilities": actor_probabilities[index].tolist(),
    } for index, record in enumerate(records)]
    levels_payload = json.dumps(
        selected_levels, sort_keys=True, separators=(",", ":")
    ).encode()
    report = {
        "format": "lewm_exact_intermediate_action_grounding_v1",
        "config": {
            "levels": level_count,
            "states": len(records),
            "states_per_episode_requested": states_per_episode,
            "real_repeats": real_repeats,
            "imagined_repeats": imagined_repeats,
            "bootstrap_repeats": bootstrap_repeats,
            "seed": seed,
            "gamma": float(pipeline.config.gamma),
            "real_reward_scale": reward_scale,
            "tie_epsilon": tie_epsilon,
            "advantage_margin": advantage_margin,
            "levels_sha256": hashlib.sha256(levels_payload).hexdigest(),
        },
        "deployment_contract": {
            "horizon": int(imagined.config.horizon),
            "termination_mode": imagined.config.termination_mode,
            "success_threshold": float(imagined.config.success_threshold),
            "value_bootstrap": bool(imagined.config.bootstrap_with_value),
            "reward_mapping": "per_transition_success_conservative",
            "reward_scale": reward_scale,
            "relative_action_value": False,
            "forced_first_action": True,
            "continuation_policy": "current frozen Actor, stochastic",
            "real_target": "full remaining real-environment discounted return",
        },
        "parameter_mutation": {
            "policy_changed_tensors": len(changed_policy),
            "world_model_changed_tensors": len(changed_wm),
            "check": "torch_tensor_version_counter",
        },
        "action_grounding": metrics,
        "level_cluster_bootstrap_interval_95": intervals,
        "actor_logits_baseline": actor_metrics,
        "gate": {"passed": not reasons, "reasons": reasons},
        "diagnostics": {
            "h1_success_probability": h1_probability.tolist(),
            "h3_endpoint_success_probability": endpoint_probability.tolist(),
        },
        "state_results": state_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save({
        "imagined_q": imagined_q,
        "real_q": real_q,
        "real_success_probability": real_success,
        "actor_logits": actor_logits,
        "actor_probabilities": actor_probabilities,
        "group_ids": group_ids,
    }, output.with_suffix(".details.pt"))
    print(
        "Le-WM current-policy action gate: "
        f"passed={not reasons} states={len(records)} "
        f"pairwise={metrics['pairwise_accuracy']:.4f} "
        f"pairwise_L95={intervals['pairwise_accuracy']['lower_95']:.4f} "
        f"sign={metrics['advantage_sign_accuracy']} "
        f"harmful={metrics['harmful_positive_rate']} "
        f"actor_pairwise={actor_metrics['pairwise_accuracy']:.4f}",
        flush=True,
    )
    if reasons:
        print("  gate reasons: " + ", ".join(reasons), flush=True)
    print(f"Le-WM action grounding report: {output}", flush=True)
    return report
