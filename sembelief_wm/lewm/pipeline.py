"""Independent Le-WM training state machine.

Only stable primitives are shared: the policy, PPOUpdater's critic regression,
RealCollector and PPOBatch.  No generic MBRLPipeline training branch or
environment-controlled gate is called here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .config import LeWMOrchestrationConfig, LeWMStage
from .core_contract import verify_locked_shared_core
from .data import LeWMCriticBatch, real_collection_to_critic_batch


_CRITIC_PREFIXES = (
    "q_head.", "ordered_value_head.", "value_head.",
    "value_net.", "value_readout.",
)
_LOCKED_IMAGE_TOKENIZER_SHA256 = (
    "271b2a6a1d829ae86a032bf7892c8cb3b47dd36435664d13636425e65ab72db1"
)


def _subset(batch: LeWMCriticBatch, index: torch.Tensor) -> LeWMCriticBatch:
    return LeWMCriticBatch(**{
        name: getattr(batch, name)[index]
        for name in LeWMCriticBatch.__dataclass_fields__
    })


def _append(
    current: LeWMCriticBatch | None, incoming: LeWMCriticBatch | None, *, capacity: int,
    keep_oldest: bool = False,
) -> LeWMCriticBatch | None:
    if incoming is None:
        return current
    if current is None:
        combined = incoming
    else:
        combined = LeWMCriticBatch(**{
            name: torch.cat([getattr(current, name), getattr(incoming, name)])
            for name in LeWMCriticBatch.__dataclass_fields__
        })
    count = len(combined.actions)
    if count <= capacity:
        return combined
    selected = (
        torch.arange(capacity, device=combined.actions.device)
        if keep_oldest else
        torch.arange(count - capacity, count, device=combined.actions.device)
    )
    return _subset(combined, selected)


def _explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    variance = target.float().var(unbiased=False)
    if float(variance) < 1e-8:
        return 0.0
    residual = (target.float() - prediction.float()).var(unbiased=False)
    return float(1.0 - residual / variance)


class LeWMOrchestrator:
    """Fail-closed Le-WM orchestration with an explicit stage contract."""

    def __init__(
        self, *, config: Any, imagine_fn: Any, sample_beliefs_fn: Any,
        ppo_updater: Any, policy: Any, evaluator: Any = None,
        wm_refresher: Any = None, wm_refresh_sample_fn: Any = None,
        world_model: Any = None, real_collector: Any = None,
        logger: Any = None, offline_behavior_cloner: Any = None,
        behavior_sample_fn: Any = None,
    ) -> None:
        del offline_behavior_cloner, behavior_sample_fn
        self.config = config
        self.imagine_fn = imagine_fn  # exposed only for read-only audits
        self.sample_beliefs_fn = sample_beliefs_fn
        self.ppo_updater = ppo_updater
        self.policy = policy
        self.evaluator = evaluator
        self.wm_refresher = wm_refresher
        self.wm_refresh_sample_fn = wm_refresh_sample_fn
        self.world_model = world_model
        self.real_collector = real_collector
        self.logger = logger
        self.online_actor_warmup_fn = None
        self.shared_core_contract = verify_locked_shared_core()
        self.lewm = LeWMOrchestrationConfig.from_environment()
        self._train_replay: LeWMCriticBatch | None = None
        self._validation: LeWMCriticBatch | None = None
        self._ev_ema: float | None = None
        self._gate_streak = 0
        self._source_update = 0
        self._loaded_checkpoint_metadata: dict[str, Any] = {}

        if self.wm_refresher is not None:
            raise RuntimeError(
                "Le-WM orchestration currently requires frozen_wm; "
                "alternating WM is not an approved stage"
            )
        if getattr(policy, "critic_source", None) == "qwen_slotwise_q":
            raise RuntimeError("Le-WM orchestration forbids imagined-target Q Critic")

    def _critic_module(self) -> Any:
        for name in ("ordered_value_head", "value_head", "value_net", "q_head"):
            module = getattr(self.policy, name, None)
            if module is not None:
                return module
        raise RuntimeError("Le-WM checkpoint contract found no Critic module")

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore Actor strictly and keep a freshly initialized Critic."""
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        from ..model.checkpoint_semantics import validate_world_model_semantics

        backbone = getattr(
            getattr(self.world_model, "transition", None), "backbone", None
        )
        validate_world_model_semantics(
            payload,
            attention_mode=getattr(backbone, "attention_mode", "unknown"),
            context=f"Le-WM source checkpoint {path}",
        )
        policy_state = payload.get("policy")
        if not isinstance(policy_state, dict):
            raise RuntimeError("Le-WM source checkpoint has no policy state_dict")
        critic = self._critic_module()
        fresh_critic = {
            key: value.detach().cpu().clone()
            for key, value in critic.state_dict().items()
        }
        has_behavior = any(
            key.startswith("behavior_slotwise_head.") for key in policy_state
        )
        if has_behavior and getattr(self.policy, "behavior_slotwise_head", None) is None:
            self.policy.capture_behavior_reference()
        if self.lewm.stage in {
            LeWMStage.REAL_CRITIC_VALIDATE, LeWMStage.ACTOR_PPO,
        }:
            if payload.get("format") != "lewm_orchestration_v1":
                raise RuntimeError(
                    "frozen validation/Actor PPO requires a Le-WM v1 checkpoint"
                )
            source_stage = payload.get("stage")
            allowed_source = source_stage == LeWMStage.REAL_CRITIC_PROBE.value
            if source_stage == LeWMStage.ACTOR_PPO.value:
                allowed_source = bool(
                    payload.get("actor_probe", {}).get("accepted", False)
                )
            if not allowed_source:
                raise RuntimeError(
                    "frozen validation/Actor PPO requires a passed real-Critic "
                    "checkpoint or an accepted Actor-probe checkpoint"
                )
            gate = payload.get("lewm_gate", {})
            if not bool(gate.get("passed", False)):
                raise RuntimeError(
                    "frozen validation/Actor PPO refuses a checkpoint whose "
                    "real-Critic gate is not preserved"
                )
            self.policy.load_state_dict(policy_state, strict=True)
            if os.environ.get("DISCARD_RESUME_OPTIMIZER", "0") != "1":
                raise RuntimeError(
                    "frozen validation/Actor PPO requires "
                    "DISCARD_RESUME_OPTIMIZER=1"
                )
            if os.environ.get("KEEP_ASSEMBLED_WORLD_MODEL", "0") != "1":
                raise RuntimeError(
                    "frozen validation/Actor PPO requires "
                    "KEEP_ASSEMBLED_WORLD_MODEL=1"
                )
            self._source_update = int(payload.get("source_update", 0))
            self._loaded_checkpoint_metadata = {
                "path": str(checkpoint_path.resolve()),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "update": int(payload.get("update", 0)),
                "source_update": self._source_update,
                "probe_gate": gate,
                "stage": source_stage,
            }
            print(
                "Le-WM checkpoint: Actor and real-return Critic restored "
                f"strictly for {self.lewm.stage.value}.", flush=True,
            )
            return int(payload.get("update", 0))
        actor_state = {
            key: value for key, value in policy_state.items()
            if not key.startswith(_CRITIC_PREFIXES)
        }
        incompatible = self.policy.load_state_dict(actor_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        noncritic_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(_CRITIC_PREFIXES)
        ]
        if unexpected or noncritic_missing:
            raise RuntimeError(
                "Le-WM strict Actor restore failed: "
                f"unexpected={unexpected}, missing={noncritic_missing}"
            )
        critic.load_state_dict(fresh_critic, strict=True)
        if os.environ.get("KEEP_ASSEMBLED_WORLD_MODEL", "0") != "1":
            raise RuntimeError(
                "Le-WM orchestration requires KEEP_ASSEMBLED_WORLD_MODEL=1; "
                "a PPO checkpoint may not replace the selected frozen WM"
            )
        if os.environ.get("DISCARD_RESUME_OPTIMIZER", "0") != "1":
            raise RuntimeError(
                "Le-WM real Critic probe requires DISCARD_RESUME_OPTIMIZER=1"
            )
        self._source_update = int(payload.get("update", 0))
        print(
            "Le-WM checkpoint: Actor restored strictly; frozen WM retained; "
            "ordered scalar Critic freshly initialized.", flush=True,
        )
        return self._source_update

    def _run_frozen_critic_validation(self, root: Path | None) -> None:
        from .validation import (
            evaluate_episode_trajectories,
            grouped_bootstrap_intervals,
        )

        if root is None:
            raise RuntimeError("frozen Critic validation requires checkpoint_dir")
        if self.real_collector is None or self.evaluator is None:
            raise RuntimeError("frozen Critic validation requires the real evaluator")
        levels = list(getattr(self.evaluator, "eval_levels", None) or [])
        if not levels:
            raise RuntimeError("frozen Critic validation requires fixed eval levels")
        level_count = min(
            len(levels), int(os.environ.get("LEWM_FROZEN_VALIDATION_LEVELS", "256"))
        )
        rollout_repeats = int(os.environ.get(
            "LEWM_FROZEN_VALIDATION_ROLLOUT_REPEATS", "1"
        ))
        bootstrap_repeats = int(os.environ.get(
            "LEWM_FROZEN_VALIDATION_BOOTSTRAPS", "1000"
        ))
        seed = int(os.environ.get("LEWM_FROZEN_VALIDATION_SEED", "20260818"))
        if level_count <= 0 or rollout_repeats <= 0:
            raise ValueError("frozen validation level and rollout counts must be positive")
        selected_levels = levels[:level_count] * rollout_repeats
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.policy.named_parameters()
        }
        cfg = self.real_collector.config
        old = (cfg.capture_policy_trajectory, cfg.deterministic, cfg.exploration_epsilon)
        deterministic_actions = (
            os.environ.get("LEWM_FROZEN_VALIDATION_DETERMINISTIC", "0") == "1"
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            cfg.capture_policy_trajectory = True
            cfg.deterministic = deterministic_actions
            cfg.exploration_epsilon = 0.0
            result = self.real_collector.collect(
                len(selected_levels), levels=selected_levels,
                collect_tokenized=False, device=self.real_collector.device,
                dtype=self.real_collector.dtype,
            )
        finally:
            (
                cfg.capture_policy_trajectory,
                cfg.deterministic,
                cfg.exploration_epsilon,
            ) = old
        metrics, episode_predictions, episode_targets = evaluate_episode_trajectories(
            policy=self.policy, episodes=result.episodes,
            gamma=float(self.config.gamma), reward_scale=self.lewm.reward_scale,
        )
        expected_episodes = level_count * rollout_repeats
        if len(episode_targets) != expected_episodes:
            raise RuntimeError(
                "fixed-level validation lost policy trajectories: "
                f"expected={expected_episodes}, actual={len(episode_targets)}"
            )
        grouped_predictions = [
            [
                episode_predictions[repeat * level_count + level]
                for repeat in range(rollout_repeats)
            ]
            for level in range(level_count)
        ]
        grouped_targets = [
            [
                episode_targets[repeat * level_count + level]
                for repeat in range(rollout_repeats)
            ]
            for level in range(level_count)
        ]
        intervals = grouped_bootstrap_intervals(
            grouped_predictions, grouped_targets,
            repeats=bootstrap_repeats, seed=seed + 1,
        )
        parameter_delta = max(
            float((parameter.detach().cpu() - before[name]).abs().max())
            for name, parameter in self.policy.named_parameters()
        )
        if parameter_delta != 0.0:
            raise RuntimeError("frozen Critic validation mutated policy parameters")
        point_passed = (
            metrics["explained_variance"] >= self.lewm.ev_threshold
            and metrics["mse_improvement"] >= self.lewm.mse_improvement_threshold
        )
        confidence_passed = (
            intervals["explained_variance"]["lower_95"] >= self.lewm.ev_threshold
            and intervals["mse_improvement"]["lower_95"]
            >= self.lewm.mse_improvement_threshold
        )
        levels_payload = json.dumps(
            levels[:level_count], sort_keys=True, separators=(",", ":")
        ).encode()
        report = {
            "format": "lewm_frozen_critic_validation_v2",
            "checkpoint": self._loaded_checkpoint_metadata,
            "levels": {
                "count": level_count,
                "rollout_repeats": rollout_repeats,
                "sha256": hashlib.sha256(levels_payload).hexdigest(),
            },
            "policy": {
                "deterministic_actions": deterministic_actions,
                "parameter_max_delta": parameter_delta,
            },
            "bootstrap": {
                "repeats": bootstrap_repeats,
                "seed": seed + 1,
                "unit": "level",
                "groups": level_count,
            },
            "metrics": metrics,
            "intervals": intervals,
            "thresholds": {
                "explained_variance": self.lewm.ev_threshold,
                "mse_improvement": self.lewm.mse_improvement_threshold,
            },
            "point_gate_passed": point_passed,
            "confidence_gate_passed": confidence_passed,
        }
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "frozen_critic_validation.json"
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            "Le-WM FROZEN CRITIC VALIDATION "
            f"{'PASSED' if confidence_passed else 'FAILED'}: "
            f"EV={metrics['explained_variance']:.4f} "
            f"(95% lower={intervals['explained_variance']['lower_95']:.4f}), "
            f"MSE gain={metrics['mse_improvement']:.2%} "
            f"(95% lower={intervals['mse_improvement']['lower_95']:.2%}); "
            f"report={output_path}", flush=True,
        )
        if (
            not confidence_passed
            and os.environ.get("LEWM_FROZEN_VALIDATION_REQUIRE_PASS", "1") == "1"
        ):
            raise RuntimeError("independent frozen Critic confidence gate failed")

    def _collect_real_split(self, update: int) -> None:
        if self.real_collector is None:
            raise RuntimeError("REAL_CRITIC_PROBE requires RealCollector")
        cfg = self.real_collector.config
        old_capture = cfg.capture_policy_trajectory
        old_deterministic = cfg.deterministic
        old_epsilon = cfg.exploration_epsilon
        try:
            cfg.capture_policy_trajectory = True
            cfg.deterministic = False
            cfg.exploration_epsilon = 0.0
            result = self.real_collector.collect(
                self.lewm.collect_episodes,
                update_id=3_000_000 + update,
                collect_tokenized=False,
                device=self.real_collector.device,
                dtype=self.real_collector.dtype,
            )
        finally:
            cfg.capture_policy_trajectory = old_capture
            cfg.deterministic = old_deterministic
            cfg.exploration_epsilon = old_epsilon
        validation_episodes, training_episodes = [], []
        for index, episode in enumerate(result.episodes):
            (validation_episodes if index % 5 == 0 else training_episodes).append(
                episode
            )
        device = next(self.policy.parameters()).device
        train = real_collection_to_critic_batch(
            SimpleNamespace(episodes=training_episodes),
            gamma=float(self.config.gamma), device=device,
            reward_scale=self.lewm.reward_scale,
        )
        validation = real_collection_to_critic_batch(
            SimpleNamespace(episodes=validation_episodes),
            gamma=float(self.config.gamma), device=device,
            reward_scale=self.lewm.reward_scale,
        )
        self._train_replay = _append(
            self._train_replay, train, capacity=self.lewm.replay_capacity
        )
        self._validation = _append(
            self._validation, validation,
            capacity=self.lewm.validation_capacity, keep_oldest=True,
        )

    def _critic_updates(self, update: int) -> dict[str, float]:
        if self._train_replay is None:
            raise RuntimeError("real collection produced no train trajectories")
        count = min(self.lewm.train_samples, len(self._train_replay.actions))
        losses, evs = [], []
        for step in range(self.lewm.critic_updates_per_collection):
            generator = torch.Generator(
                device=self._train_replay.actions.device
            ).manual_seed(20260817 + update * 101 + step)
            index = torch.randperm(
                len(self._train_replay.actions), generator=generator,
                device=self._train_replay.actions.device,
            )[:count]
            metrics = self.ppo_updater.update(
                _subset(self._train_replay, index),
                critic_only=True, actor_enabled=False, critic_enabled=True,
            )
            losses.append(float(metrics.value_loss))
            evs.append(float(metrics.explained_variance))
        return {
            "lewm/critic_train_samples": float(count),
            "lewm/critic_train_loss": sum(losses) / len(losses),
            "lewm/critic_train_ev": sum(evs) / len(evs),
        }

    @torch.no_grad()
    def _heldout_metrics(self) -> dict[str, float]:
        if self._validation is None or len(self._validation.actions) < 16:
            return {
                "lewm/heldout_samples": 0.0,
                "lewm/gate_passed": 0.0,
            }
        prediction = self.policy.evaluate_values(
            self._validation.states, self._validation.actions
        ).float()
        target = self._validation.returns.float()
        mse = float((prediction - target).pow(2).mean())
        baseline = self._train_replay.returns.float().mean().expand_as(target)
        baseline_mse = float((baseline - target).pow(2).mean())
        improvement = 1.0 - mse / max(baseline_mse, 1e-12)
        ev = _explained_variance(prediction, target)
        self._ev_ema = (
            ev if self._ev_ema is None else
            (1.0 - self.lewm.ev_ema_alpha) * self._ev_ema
            + self.lewm.ev_ema_alpha * ev
        )
        passed = (
            self._ev_ema >= self.lewm.ev_threshold
            and improvement >= self.lewm.mse_improvement_threshold
        )
        self._gate_streak = self._gate_streak + 1 if passed else 0
        return {
            "lewm/heldout_samples": float(len(target)),
            "lewm/heldout_ev": ev,
            "lewm/heldout_ev_ema": self._ev_ema,
            "lewm/heldout_mse": mse,
            "lewm/constant_baseline_mse": baseline_mse,
            "lewm/mse_improvement": improvement,
            "lewm/target_mean": float(target.mean()),
            "lewm/target_std": float(target.std(unbiased=False)),
            "lewm/prediction_mean": float(prediction.mean()),
            "lewm/prediction_std": float(prediction.std(unbiased=False)),
            "lewm/gate_streak": float(self._gate_streak),
            "lewm/gate_passed": float(
                self._gate_streak >= self.lewm.gate_patience
            ),
            "ppo/actor_update": 0.0,
        }

    def _save_checkpoint(self, path: Path, update: int) -> None:
        from ..model.checkpoint_semantics import world_model_semantics

        backbone = getattr(
            getattr(self.world_model, "transition", None), "backbone", None
        )
        torch.save({
            "format": "lewm_orchestration_v1",
            "stage": self.lewm.stage.value,
            "update": update,
            "source_update": self._source_update,
            "policy": self.policy.state_dict(),
            "ppo_optimizer": self.ppo_updater.state_dict(),
            "lewm_gate": {
                "ev_ema": self._ev_ema,
                "streak": self._gate_streak,
                "passed": self._gate_streak >= self.lewm.gate_patience,
            },
            "wm_semantics": world_model_semantics(
                getattr(backbone, "attention_mode", "unknown")
            ),
        }, path)
        print(f"  Le-WM checkpoint saved: {path} (update {update})", flush=True)

    def train(
        self, *, checkpoint_dir: str | Path | None = None,
        start_update: int = 0,
    ) -> None:
        if self.lewm.stage is LeWMStage.REAL_CRITIC_VALIDATE:
            self._run_frozen_critic_validation(
                Path(checkpoint_dir) if checkpoint_dir is not None else None
            )
            return
        if self.lewm.stage is LeWMStage.ACTOR_PPO:
            from .actor_probe import run_guarded_actor_probe

            run_guarded_actor_probe(
                pipeline=self,
                checkpoint_dir=(
                    Path(checkpoint_dir) if checkpoint_dir is not None else None
                ),
                source_update=start_update,
            )
            return
        if self.lewm.stage is not LeWMStage.REAL_CRITIC_PROBE:
            raise RuntimeError(f"unreleased Le-WM stage: {self.lewm.stage.value}")
        if self.real_collector is None:
            raise RuntimeError("Le-WM real Critic probe was not wired to real env")
        if self.real_collector.policy is not self.policy:
            raise RuntimeError("Le-WM collector and orchestration do not share Actor identity")
        provenance = getattr(self.real_collector.tokenizer, "provenance", {})
        if provenance.get("sha256") != _LOCKED_IMAGE_TOKENIZER_SHA256:
            raise RuntimeError(
                "Le-WM tokenizer coordinate-system contract changed: "
                f"expected={_LOCKED_IMAGE_TOKENIZER_SHA256}, "
                f"actual={provenance.get('sha256')}"
            )
        root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if root is not None:
            repository = Path(__file__).resolve().parents[2]
            lewm_checkpoint_root = (repository / "checkpoints").resolve()
            resolved = root.resolve()
            if (
                resolved.parent != lewm_checkpoint_root
                or not resolved.name.startswith("lewm_")
            ):
                raise RuntimeError(
                    "Le-WM output isolation requires a direct "
                    "<repo>/checkpoints/lewm_* directory; got "
                    f"{resolved}"
                )
            root.mkdir(parents=True, exist_ok=True)
        actor_parameters = [
            parameter.detach().cpu().clone()
            for parameter in self.policy.actor_parameters()
        ]
        last_update = start_update
        for update in range(start_update + 1, self.config.total_updates + 1):
            self.policy.set_deterministic_forward_mode()
            self._collect_real_split(update)
            metrics = {
                "update": float(update),
                "progress": float(update) / max(1, self.config.total_updates),
                "lewm/train_replay_size": float(len(self._train_replay.actions)),
                "lewm/validation_size": float(len(self._validation.actions)),
                "lewm/imagined_target_updates": 0.0,
                "lewm/actor_updates": 0.0,
            }
            metrics.update(self._critic_updates(update))
            metrics.update(self._heldout_metrics())
            actor_delta = max(
                float((parameter.detach().cpu() - before).abs().max())
                for parameter, before in zip(
                    self.policy.actor_parameters(), actor_parameters
                )
            )
            metrics["lewm/actor_parameter_max_delta"] = actor_delta
            if actor_delta != 0.0:
                raise RuntimeError("Le-WM REAL_CRITIC_PROBE mutated Actor parameters")
            if self.logger is not None:
                self.logger.log_scalars(update, metrics)
            print(
                f"[Le-WM real Critic {update}/{self.config.total_updates}] "
                f"heldout_ev={metrics.get('lewm/heldout_ev', float('nan')):.4f} "
                f"ema={metrics.get('lewm/heldout_ev_ema', float('nan')):.4f} "
                f"mse_gain={metrics.get('lewm/mse_improvement', float('nan')):.2%} "
                f"streak={self._gate_streak}/{self.lewm.gate_patience}",
                flush=True,
            )
            last_update = update
            if root is not None and (
                update % self.config.checkpoint_every == 0
                or self._gate_streak >= self.lewm.gate_patience
            ):
                self._save_checkpoint(root / "latest.pt", update)
            if self._gate_streak >= self.lewm.gate_patience:
                print(
                    "Le-WM REAL_CRITIC_PROBE PASSED. Actor PPO remains locked; "
                    "this checkpoint is Critic evidence only.", flush=True,
                )
                break
        if root is not None and last_update > start_update:
            self._save_checkpoint(root / "latest.pt", last_update)
