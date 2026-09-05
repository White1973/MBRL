"""Guarded, reversible Actor-only PPO probe for Le-WM.

The Critic remains the real-return scalar V learned by REAL_CRITIC_PROBE.  It
may score/bootstrap imagined trajectories, but neither it nor the frozen World
Model is optimized here.  Acceptance is decided on the same fixed real layouts
before and after the short PPO run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from ..rl.gae import trajectory_to_ppo_batch
from ..rl.trajectory import PPOBatch
from .config import LeWMStage


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _clone(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _max_delta(
    parameters: list[torch.nn.Parameter], before: list[torch.Tensor],
) -> float:
    return max(
        (
            float((parameter.detach().cpu() - initial).abs().max())
            for parameter, initial in zip(parameters, before, strict=True)
        ),
        default=0.0,
    )


def _concat_batches(batches: list[PPOBatch]) -> PPOBatch:
    if not batches:
        raise RuntimeError("Actor probe produced no imagined PPO batch")
    return PPOBatch(**{
        name: torch.cat([getattr(batch, name) for batch in batches])
        for name in PPOBatch.__dataclass_fields__
    })


def _bootstrap_paired(
    before: list[float], after: list[float], *, repeats_per_level: int,
    bootstraps: int, seed: int,
) -> dict[str, float]:
    delta = torch.tensor(after, dtype=torch.float64) - torch.tensor(
        before, dtype=torch.float64
    )
    if len(delta) % repeats_per_level != 0:
        raise RuntimeError("paired evaluation does not form complete level groups")
    level_count = len(delta) // repeats_per_level
    # Collection order is [all levels] repeated R times.
    grouped = delta.reshape(repeats_per_level, level_count).mean(dim=0)
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(bootstraps):
        index = torch.randint(
            0, level_count, (level_count,), generator=generator
        )
        samples.append(grouped[index].mean())
    distribution = torch.stack(samples).sort().values
    return {
        "point": float(grouped.mean()),
        "lower_95": float(distribution[int(0.025 * (bootstraps - 1))]),
        "upper_95": float(distribution[int(0.975 * (bootstraps - 1))]),
    }


def _evaluate_real(
    pipeline: Any, *, levels: list[dict], repeats: int, seed: int,
    deterministic: bool,
) -> tuple[dict[str, float], list[float], list[float]]:
    collector = pipeline.real_collector
    selected = levels * repeats
    cfg = collector.config
    old = (cfg.capture_policy_trajectory, cfg.deterministic, cfg.exploration_epsilon)
    cuda_devices: list[int] = []
    if torch.cuda.is_available():
        cuda_devices = [
            collector.device.index
            if collector.device.index is not None else torch.cuda.current_device()
        ]
    try:
        cfg.capture_policy_trajectory = False
        cfg.deterministic = deterministic
        cfg.exploration_epsilon = 0.0
        # Baseline evaluation must not consume the PPO rollout RNG stream.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            result = collector.collect(
                len(selected), levels=selected, collect_tokenized=False,
                device=collector.device, dtype=collector.dtype,
            )
    finally:
        cfg.capture_policy_trajectory, cfg.deterministic, cfg.exploration_epsilon = old
    successes = [float(episode.success) for episode in result.episodes]
    returns = [float(episode.reward) for episode in result.episodes]
    return result.metrics, successes, returns


@torch.no_grad()
def _actor_panel(pipeline: Any, *, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(pipeline.policy.parameters()).device
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        beliefs = pipeline.sample_beliefs_fn(count)
        states = getattr(beliefs, "slots", beliefs).detach()
        logits = pipeline.policy.actor_logits(states).float().detach()
    return states, logits


@torch.no_grad()
def _panel_kl(policy: Any, states: torch.Tensor, old_logits: torch.Tensor) -> dict[str, float]:
    new_logits = policy.actor_logits(states).float()
    old_log_prob = old_logits.log_softmax(dim=-1)
    new_log_prob = new_logits.log_softmax(dim=-1)
    old_probability = old_log_prob.exp()
    kl = (old_probability * (old_log_prob - new_log_prob)).sum(dim=-1)
    return {
        "mean": float(kl.mean()),
        "max": float(kl.max()),
        "argmax_changed_fraction": float(
            (old_logits.argmax(-1) != new_logits.argmax(-1)).float().mean()
        ),
    }


def _ppo_metrics(metrics: Any) -> dict[str, float]:
    return {
        field: float(getattr(metrics, field))
        for field in metrics.__dataclass_fields__
    }


def run_guarded_actor_probe(
    *, pipeline: Any, checkpoint_dir: Path | None, source_update: int,
) -> None:
    if checkpoint_dir is None:
        raise RuntimeError("Actor PPO probe requires checkpoint_dir")
    repository = Path(__file__).resolve().parents[2]
    root = checkpoint_dir.resolve()
    expected_parent = (repository / "checkpoints").resolve()
    if root.parent != expected_parent or not root.name.startswith("lewm_"):
        raise RuntimeError(
            "Actor PPO output must be a direct <repo>/checkpoints/lewm_* directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    if pipeline.real_collector is None or pipeline.evaluator is None:
        raise RuntimeError("Actor PPO probe requires fixed real evaluation")
    if pipeline.real_collector.policy is not pipeline.policy:
        raise RuntimeError("Actor PPO and real evaluator must share Actor identity")

    cfg = pipeline.config
    updates = int(cfg.total_updates)
    if updates <= 0 or updates > 10:
        raise ValueError("guarded Actor probe allows 1..10 PPO updates")
    if cfg.ppo.epochs > 2:
        raise ValueError("guarded Actor probe allows at most two PPO epochs")
    target_kl = pipeline.ppo_updater.config.target_kl
    if target_kl is None or target_kl > 0.01:
        raise ValueError("guarded Actor probe requires target_kl <= 0.01")
    actor_lr = float(pipeline.ppo_updater.config.lr)
    if actor_lr > 1e-5:
        raise ValueError("guarded Actor probe requires actor_lr <= 1e-5")

    levels = list(getattr(pipeline.evaluator, "eval_levels", None) or [])
    level_count = min(len(levels), _env_int("LEWM_ACTOR_PROBE_EVAL_LEVELS", 512))
    eval_repeats = _env_int("LEWM_ACTOR_PROBE_EVAL_REPEATS", 1)
    bootstraps = _env_int("LEWM_ACTOR_PROBE_BOOTSTRAPS", 2000)
    seed = _env_int("LEWM_ACTOR_PROBE_SEED", 20261318)
    train_seed = _env_int(
        "LEWM_ACTOR_PROBE_TRAIN_SEED", seed + 1_000_003
    )
    deterministic = os.environ.get("LEWM_ACTOR_PROBE_DETERMINISTIC", "0") == "1"
    if level_count < 32 or eval_repeats <= 0 or bootstraps < 100:
        raise ValueError("Actor probe needs >=32 levels, positive repeats, and >=100 bootstraps")
    selected_levels = levels[:level_count]

    actor_parameters = list(pipeline.ppo_updater._actor_trainable)
    critic_parameters = list(pipeline.ppo_updater._critic_trainable)
    actor_before = _clone(actor_parameters)
    critic_before = _clone(critic_parameters)
    optimizer_before = copy.deepcopy(pipeline.ppo_updater.state_dict())
    world_versions = [parameter._version for parameter in pipeline.world_model.parameters()]
    panel_states, panel_logits = _actor_panel(
        pipeline, count=_env_int("LEWM_ACTOR_PROBE_KL_STATES", 128), seed=seed + 1
    )

    print(
        f"Le-WM guarded Actor probe: baseline real evaluation on "
        f"{level_count} fixed levels x {eval_repeats}", flush=True,
    )
    before_metrics, before_success, before_returns = _evaluate_real(
        pipeline, levels=selected_levels, repeats=eval_repeats, seed=seed,
        deterministic=deterministic,
    )

    # Evaluation runs in fork_rng and therefore leaves the caller RNG intact.
    # Reset explicitly here so independent PPO branches can share an identical
    # source checkpoint and evaluation stream while sampling different
    # posterior starts/actions for imagined training.
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    print(
        f"Le-WM guarded Actor probe: imagined-training seed={train_seed}",
        flush=True,
    )

    update_reports: list[dict[str, Any]] = []
    for local_update in range(1, updates + 1):
        batches = []
        reward_values = []
        valid_steps = 0
        for _ in range(cfg.rollouts_per_update):
            beliefs = pipeline.sample_beliefs_fn(cfg.rollout_batch_size)
            trajectory = pipeline.imagine_fn(beliefs)
            batch = trajectory_to_ppo_batch(
                trajectory, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
            )
            batches.append(batch)
            valid = (
                trajectory.mask.bool()
                if trajectory.mask is not None
                else torch.ones_like(trajectory.rewards, dtype=torch.bool)
            )
            reward_values.append(trajectory.rewards[valid].float())
            valid_steps += int(valid.sum())
        ppo_batch = _concat_batches(batches)
        metrics = pipeline.ppo_updater.update(
            ppo_batch, critic_only=False, actor_enabled=True,
            critic_enabled=False,
        )
        report = {
            "local_update": local_update,
            "samples": len(ppo_batch.actions),
            "valid_steps": valid_steps,
            "imagined_reward_mean": float(torch.cat(reward_values).mean()),
            **_ppo_metrics(metrics),
        }
        update_reports.append(report)
        print(
            f"[Le-WM Actor probe {local_update}/{updates}] "
            f"samples={len(ppo_batch.actions)} entropy={metrics.post_update_entropy:.4f} "
            f"post_kl={metrics.post_update_kl_divergence:.6f} "
            f"coverage={metrics.sample_coverage:.2%}", flush=True,
        )

    print("Le-WM guarded Actor probe: paired post-PPO real evaluation", flush=True)
    after_metrics, after_success, after_returns = _evaluate_real(
        pipeline, levels=selected_levels, repeats=eval_repeats, seed=seed,
        deterministic=deterministic,
    )
    success_interval = _bootstrap_paired(
        before_success, after_success, repeats_per_level=eval_repeats,
        bootstraps=bootstraps, seed=seed + 2,
    )
    return_interval = _bootstrap_paired(
        before_returns, after_returns, repeats_per_level=eval_repeats,
        bootstraps=bootstraps, seed=seed + 3,
    )
    actor_delta = _max_delta(actor_parameters, actor_before)
    critic_delta = _max_delta(critic_parameters, critic_before)
    world_mutated = any(
        parameter._version != version
        for parameter, version in zip(
            pipeline.world_model.parameters(), world_versions, strict=True
        )
    )
    cumulative_kl = _panel_kl(pipeline.policy, panel_states, panel_logits)

    reasons = []
    max_post_kl = max(report["post_update_kl_divergence"] for report in update_reports)
    min_entropy = min(report["post_update_entropy"] for report in update_reports)
    if actor_delta == 0.0:
        reasons.append("actor_did_not_update")
    if critic_delta != 0.0:
        reasons.append("critic_mutated")
    if world_mutated:
        reasons.append("world_model_mutated")
    if max_post_kl > _env_float("LEWM_ACTOR_PROBE_MAX_POST_KL", target_kl):
        reasons.append("per_update_kl_exceeded")
    if cumulative_kl["mean"] > _env_float(
        "LEWM_ACTOR_PROBE_MAX_CUMULATIVE_KL", target_kl
    ):
        reasons.append("cumulative_kl_exceeded")
    if min_entropy < _env_float("LEWM_ACTOR_PROBE_MIN_ENTROPY", 0.30):
        reasons.append("entropy_below_floor")
    if success_interval["point"] < -_env_float(
        "LEWM_ACTOR_PROBE_MAX_SUCCESS_DROP", 0.005
    ):
        reasons.append("real_success_regressed")
    if return_interval["point"] < -_env_float(
        "LEWM_ACTOR_PROBE_MAX_RETURN_DROP", 0.01
    ):
        reasons.append("real_return_regressed")

    accepted = not reasons
    # "Improved" is an evidence claim, not merely a positive point estimate.
    # The first five-update probe exposed why this distinction matters: its
    # return delta was +0.00039 but the paired 95% interval spanned roughly
    # [-0.13, +0.13].  Keep such safe runs, but label them inconclusive.
    improved = accepted and (
        success_interval["lower_95"] > 0.0
        or (
            success_interval["point"] >= 0.0
            and return_interval["lower_95"] > 0.0
        )
    )
    decision = (
        "accepted_improved" if improved else
        "accepted_inconclusive" if accepted else
        "rolled_back"
    )

    report = {
        "format": "lewm_guarded_actor_probe_v1",
        "source_checkpoint": pipeline._loaded_checkpoint_metadata,
        "configuration": {
            "updates": updates,
            "actor_lr": actor_lr,
            "critic_updates": 0,
            "ppo_epochs": cfg.ppo.epochs,
            "target_kl": target_kl,
            "rollout_batch_size": cfg.rollout_batch_size,
            "rollouts_per_update": cfg.rollouts_per_update,
            "rollout_horizon": cfg.rollout_horizon,
            "eval_levels": level_count,
            "eval_repeats": eval_repeats,
            "eval_deterministic": deterministic,
            "seed": seed,
            "imagined_training_seed": train_seed,
        },
        "before_real": before_metrics,
        "after_real": after_metrics,
        "paired_intervals": {
            "success_rate_delta": success_interval,
            "return_delta": return_interval,
        },
        "ppo_updates": update_reports,
        "invariants": {
            "actor_parameter_max_delta": actor_delta,
            "critic_parameter_max_delta": critic_delta,
            "world_model_mutated": world_mutated,
            "posterior_actor_panel_kl": cumulative_kl,
        },
        "decision": decision,
        "accepted": accepted,
        "rollback_reasons": reasons,
    }

    if not accepted:
        with torch.no_grad():
            for parameter, initial in zip(
                actor_parameters, actor_before, strict=True
            ):
                parameter.copy_(initial.to(parameter.device, parameter.dtype))
        pipeline.ppo_updater.load_state_dict(optimizer_before)
        report["rollback_actor_parameter_max_delta"] = _max_delta(
            actor_parameters, actor_before
        )
    else:
        from ..model.checkpoint_semantics import world_model_semantics

        backbone = getattr(
            getattr(pipeline.world_model, "transition", None), "backbone", None
        )
        checkpoint = {
            "format": "lewm_orchestration_v1",
            "stage": LeWMStage.ACTOR_PPO.value,
            "update": source_update + updates,
            "source_update": source_update,
            "policy": pipeline.policy.state_dict(),
            "ppo_optimizer": pipeline.ppo_updater.state_dict(),
            "lewm_gate": dict(
                pipeline._loaded_checkpoint_metadata.get("probe_gate", {})
            ),
            "actor_probe": {
                "accepted": True,
                "decision": decision,
                "report": "actor_probe.json",
            },
            "wm_semantics": world_model_semantics(
                getattr(backbone, "attention_mode", "unknown")
            ),
        }
        torch.save(checkpoint, root / "latest.pt")

    levels_payload = json.dumps(
        selected_levels, sort_keys=True, separators=(",", ":")
    ).encode()
    report["levels_sha256"] = hashlib.sha256(levels_payload).hexdigest()
    output_path = root / "actor_probe.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Le-WM guarded Actor probe {decision}: "
        f"success_delta={success_interval['point']:+.4f}, "
        f"return_delta={return_interval['point']:+.4f}, "
        f"cumulative_KL={cumulative_kl['mean']:.6f}; report={output_path}",
        flush=True,
    )
