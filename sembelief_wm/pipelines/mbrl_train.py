"""Model-based RL training pipeline — orchestration only, no algorithm logic.

This module wires together:
  - collectors/imagined.py  → trajectory production
  - rl/gae.py               → advantage estimation
  - rl/ppo.py               → policy optimization
  - collectors/belief_sampler.py → start-belief sampling
  - trainers/wm_refresher.py    → world model refresh (optional)

It does NOT contain PPO loss formulas, GAE math, dynamics forward passes,
or world-model training details. Those live in their respective modules.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from ..rl.trajectory import Trajectory, PPOBatch
from ..rl.gae import trajectory_to_ppo_batch
from ..rl.ppo import PPOUpdater, PPOConfig, PPOMetrics
from ..rl.counterfactual_h2_ppo import (
    counterfactual_h2_ppo_batch,
    ordered_h2_action_sequences,
)


def _four_action_ranking_metrics(
    prediction: Tensor, target: Tensor,
) -> dict[str, float]:
    """Tie-aware ranking metrics for complete ordered four-action groups."""
    if prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("prediction must have shape (groups,4)")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    predicted_best = prediction.argmax(dim=1)
    target_max = target.max(dim=1, keepdim=True).values
    target_best = torch.isclose(target, target_max, atol=1e-6, rtol=1e-5)
    top1 = target_best.gather(1, predicted_best[:, None]).float().mean()
    correct = torch.zeros((), device=target.device)
    comparable = torch.zeros((), device=target.device)
    for left in range(4):
        for right in range(left + 1, 4):
            target_delta = target[:, left] - target[:, right]
            informative = target_delta.abs() > 1e-6
            if bool(informative.any()):
                prediction_delta = prediction[:, left] - prediction[:, right]
                correct += (
                    prediction_delta[informative].sign()
                    == target_delta[informative].sign()
                ).float().sum()
                comparable += informative.float().sum()
    margin = (
        prediction.topk(2, dim=1).values[:, 0]
        - prediction.topk(2, dim=1).values[:, 1]
    ).mean()
    flat_target = target.flatten()
    flat_prediction = prediction.flatten()
    variance = flat_target.var(unbiased=False)
    ev = torch.zeros((), device=target.device) if float(variance) < 1e-8 else (
        1.0 - (flat_target - flat_prediction).var(unbiased=False) / variance
    )
    centered_target = target - target.mean(dim=1, keepdim=True)
    centered_prediction = prediction - prediction.mean(dim=1, keepdim=True)
    centered_variance = centered_target.flatten().var(unbiased=False)
    centered_ev = (
        torch.zeros((), device=target.device)
        if float(centered_variance) < 1e-8
        else 1.0
        - (
            centered_target.flatten() - centered_prediction.flatten()
        ).var(unbiased=False)
        / centered_variance
    )
    return {
        "top1_accuracy": float(top1),
        "pairwise_accuracy": float(correct / comparable.clamp_min(1.0)),
        "informative_pairs": float(comparable),
        "q_margin": float(margin),
        "explained_variance": float(ev),
        "centered_explained_variance": float(centered_ev),
    }


# ---------------------------------------------------------------------------
# Protocols — what the pipeline needs from the outside world
# ---------------------------------------------------------------------------

class Logger(Protocol):
    """Minimal logging contract."""
    def log_scalars(self, step: int, metrics: dict[str, float]) -> None: ...


class Evaluator(Protocol):
    """Real-environment evaluation."""
    def evaluate(self, num_episodes: int) -> dict[str, float]: ...


class WorldModelRefresherProtocol(Protocol):
    """What the pipeline expects from a WM refresher."""
    def refresh(self, sample_fn: Callable[[int], Any]) -> Any: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class OnlineCollector(Protocol):
    """Collects real episodes for replay / WM refresh.

    Returns metrics dict (success_rate, avg_return, etc.).
    Implementations may also write to a replay buffer internally.
    """
    def collect_and_store(
        self, num_episodes: int, update_id: int
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Training loop schedule — no algorithm hyperparameters here."""
    total_updates: int = 200
    eval_every: int = 20
    eval_episodes: int = 128
    eval_at_start: bool = False
    checkpoint_every: int = 50

    # Imagined rollout
    rollout_batch_size: int = 256
    rollout_horizon: int = 8
    rollouts_per_update: int = 1
    debug_rollout_progress: bool = bool(
        int(os.environ.get("DEBUG_ROLLOUT_PROGRESS", "0"))
    )

    # PPO (forwarded to PPOUpdater)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # World model refresh schedule
    wm_refresh_every: int = 0        # 0 = frozen, >0 = alternating
    wm_refresh_steps: int = 5        # supervised steps per refresh

    # Real data collection schedule (for online replay)
    collect_every: int = 0           # 0 = no online collection
    collect_episodes: int = 8

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Bootstrap
    use_value_bootstrap: bool = False  # False = zero bootstrap (finite horizon)
    critic_warmup_min_updates: int = 0
    critic_warmup_ev_threshold: float = 0.2
    critic_warmup_ev_patience: int = 3
    critic_warmup_validation_fraction: float = 0.2
    critic_warmup_validation_size: int = 256
    critic_warmup_replay_capacity: int = 4096
    critic_warmup_train_samples: int = 512
    critic_warmup_ev_ema_alpha: float = 0.2
    critic_warmup_mse_improvement: float = 0.05
    # Diagnostic classifier threshold. Conservative reward runs set this to
    # the checkpoint-calibrated confidence floor instead of hard-coding 0.5.
    reward_success_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MBRLPipeline:
    """Orchestrates model-based RL training.

    All algorithm logic is delegated to injected components:
      - imagine_fn: produces Trajectory from start beliefs
      - sample_beliefs_fn: samples starting beliefs from data
      - ppo_updater: updates policy from PPOBatch
      - evaluator: runs real-env eval
      - wm_refresher: refreshes world model (optional)
      - real_collector: collects real episodes (optional)
    """

    def __init__(
        self,
        *,
        config: PipelineConfig,
        # Core components (required)
        imagine_fn: Callable[[Any], Trajectory],
        sample_beliefs_fn: Callable[[int], Any],
        ppo_updater: PPOUpdater,
        policy: nn.Module,
        # Optional components
        evaluator: Evaluator | None = None,
        wm_refresher: Any | None = None,  # WorldModelRefresher
        wm_refresh_sample_fn: Callable[[int], Any] | None = None,
        world_model: nn.Module | None = None,  # for checkpoint save/load
        real_collector: OnlineCollector | None = None,
        logger: Logger | None = None,
        offline_behavior_cloner: Any | None = None,
        behavior_sample_fn: Callable[[int], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        self.config = config
        self.imagine_fn = imagine_fn
        self.sample_beliefs_fn = sample_beliefs_fn
        self.ppo_updater = ppo_updater
        self.policy = policy
        self.evaluator = evaluator
        self.wm_refresher = wm_refresher
        self.wm_refresh_sample_fn = wm_refresh_sample_fn
        self.world_model = world_model
        self.real_collector = real_collector
        self.logger = logger
        self.offline_behavior_cloner = offline_behavior_cloner
        self.behavior_sample_fn = behavior_sample_fn
        self.best_eval_success_rate = float("-inf")
        self.best_eval_update = -1
        self._last_evaluated_actor_update = -1
        self._critic_warmup_complete = config.critic_warmup_min_updates <= 0
        self._critic_warmup_ev_streak = 0
        self._critic_warmup_ev_ema: float | None = None
        self._critic_bucket_ema: dict[str, dict[str, float]] = {}
        self._critic_candidate_saved = False
        self._critic_stabilization_lr_applied = False
        self._critic_warmup_updates = 0
        self._critic_warmup_replay: PPOBatch | None = None
        self._critic_warmup_replay_bucket_ids: Tensor | None = None
        self._critic_warmup_validation: PPOBatch | None = None
        self._last_critic_warmup_metrics: dict[str, float] = {}
        self._actor_ppo_updates = 0
        self._last_wm_refresh_actor_update = -1
        self._last_checkpointed_actor_update = -1
        # Real-return Critic evidence is valid only within one frozen latent
        # representation generation.  Training replay may grow inside that
        # generation; validation is filled once and then remains immutable.
        self._real_critic_train_replay: PPOBatch | None = None
        self._real_critic_validation: PPOBatch | None = None
        self._real_critic_cache_generation = 0
        self._real_critic_anchor_failure_streak = 0
        self._grounded_agreement_blocked_streak = 0
        self._rollout_critic_ev_ema: float | None = None
        self._rollout_critic_ev_streak = 0
        self._rollout_critic_ready = False
        self._rollout_critic_released_once = False
        self._rollout_critic_post_release_failures = 0
        self._grounded_training_safety_snapshot: dict | None = None
        self._actor_rejection_streak = 0
        # FIFO imagined replay used only by the P0 group-aware joint Critic.
        # Targets are long-horizon bootstrapped at collection time. It is
        # invalidated whenever the latent representation changes.
        self._joint_h2_replay: PPOBatch | None = None
        self._joint_target_critic: list[Tensor] | None = None
        self._fixed_rollout_critic_panels: list[PPOBatch] | None = None
        self._actor_real_probe_states: Tensor | None = None
        self._actor_real_probe_reference_logits: Tensor | None = None

    @staticmethod
    def _cpu_ppo_batch(batch: PPOBatch | None) -> PPOBatch | None:
        if batch is None:
            return None
        return PPOBatch(**{
            name: getattr(batch, name).detach().cpu()
            for name in PPOBatch.__dataclass_fields__
        })

    @staticmethod
    def _device_ppo_batch(batch: PPOBatch, device: torch.device) -> PPOBatch:
        return PPOBatch(**{
            name: getattr(batch, name).to(device=device)
            for name in PPOBatch.__dataclass_fields__
        })

    @staticmethod
    def _append_ppo_batch(
        first: PPOBatch | None,
        second: PPOBatch | None,
        *,
        capacity: int,
        keep_oldest: bool = False,
    ) -> PPOBatch | None:
        """Append PPO data with a deterministic capacity policy.

        ``keep_oldest`` is used for held-out validation: after filling, the
        exact same latent transitions and targets are retained.  Training
        replay keeps the newest samples instead.
        """
        if second is None or len(second.actions) == 0:
            return first
        if capacity <= 0:
            raise ValueError("PPO replay capacity must be positive")
        if first is None:
            combined = second
        else:
            combined = PPOBatch(**{
                name: torch.cat([getattr(first, name), getattr(second, name)])
                for name in PPOBatch.__dataclass_fields__
            })
        count = len(combined.actions)
        if count <= capacity:
            return combined
        index = slice(0, capacity) if keep_oldest else slice(count - capacity, count)
        return PPOBatch(**{
            name: getattr(combined, name)[index]
            for name in PPOBatch.__dataclass_fields__
        })

    def _append_fresh_h2_replay(
        self,
        fresh_batch: PPOBatch,
        *,
        capacity: int,
    ) -> None:
        """Append the complete on-policy rollout to the Critic replay.

        Actor-only gates may subsequently filter a separate PPO batch by
        advantage agreement or Q margin.  Applying those masks here would
        train the Critic on a selected subset while evaluating its rollout EV
        on the complete fresh distribution.
        """
        self._joint_h2_replay = self._append_ppo_batch(
            self._joint_h2_replay,
            fresh_batch,
            capacity=capacity,
        )

    def _initialize_fixed_rollout_critic_panels(self) -> None:
        """Lock rollout validation targets without perturbing training RNG."""
        if os.environ.get("FIXED_ROLLOUT_CRITIC_GATE", "0") != "1":
            return
        if self._fixed_rollout_critic_panels is not None:
            return
        if (
            os.environ.get("FIXED_ROLLOUT_GATE_REQUIRE_ACTOR0", "1") == "1"
            and self._actor_ppo_updates != 0
        ):
            raise RuntimeError(
                "fixed rollout Critic panels must be created before Actor PPO"
            )
        panel_count = int(os.environ.get(
            "FIXED_ROLLOUT_GATE_PANELS", "12"
        ))
        batch_size = int(os.environ.get(
            "FIXED_ROLLOUT_GATE_BATCH_SIZE", "128"
        ))
        seed = int(os.environ.get(
            "FIXED_ROLLOUT_GATE_SEED", "20260815"
        ))
        if panel_count < 3 or batch_size <= 0:
            raise ValueError(
                "fixed rollout gate needs at least 3 positive-size panels"
            )
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        panels: list[PPOBatch] = []
        self._stabilize_policy_forward()
        try:
            for panel_index in range(panel_count):
                panel_seed = seed + panel_index
                torch.manual_seed(panel_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(panel_seed)
                if os.environ.get(
                    "FIXED_ROLLOUT_GATE_USE_REAL_POSTERIOR", "0"
                ) == "1":
                    states = self._actor_real_probe_states
                    if states is None:
                        raise RuntimeError(
                            "fixed rollout gate is waiting for the held-out "
                            "real posterior probe"
                        )
                    from ..types import BeliefState
                    generator = torch.Generator(
                        device=states.device
                    ).manual_seed(panel_seed)
                    index = torch.randint(
                        len(states), (batch_size,), generator=generator,
                        device=states.device,
                    )
                    beliefs = BeliefState(slots=states[index])
                else:
                    beliefs = self.sample_beliefs_fn(batch_size)
                trajectory = self.imagine_fn(beliefs)
                panels.append(trajectory_to_ppo_batch(
                    trajectory,
                    gamma=self.config.gamma,
                    gae_lambda=1.0,
                ))
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self._fixed_rollout_critic_panels = panels
        print(
            "FIXED_ROLLOUT_CRITIC_GATE locked "
            f"panels={panel_count} samples="
            f"{sum(len(panel.actions) for panel in panels)} seed={seed}",
            flush=True,
        )

    @torch.no_grad()
    def _fixed_rollout_critic_metrics(self) -> dict[str, float]:
        panels = self._fixed_rollout_critic_panels
        if not panels:
            raise RuntimeError("fixed rollout Critic panels are unavailable")
        evs = []
        mses = []
        for panel in panels:
            predictions = []
            for begin in range(0, len(panel.actions), 64):
                predictions.append(self.policy.evaluate_values(
                    panel.states[begin:begin + 64],
                    panel.actions[begin:begin + 64],
                ).float())
            prediction = torch.cat(predictions)
            target = panel.returns.float()
            variance = target.var(unbiased=False)
            evs.append(0.0 if float(variance) < 1e-8 else float(
                1.0 - (target - prediction).var(unbiased=False) / variance
            ))
            mses.append(float((target - prediction).pow(2).mean()))
        ev_tensor = torch.tensor(evs, dtype=torch.float64)
        mean = float(ev_tensor.mean())
        std = float(ev_tensor.std(unbiased=False))
        z_value = float(os.environ.get("FIXED_ROLLOUT_GATE_Z", "1.96"))
        if z_value < 0.0:
            raise ValueError("FIXED_ROLLOUT_GATE_Z must be non-negative")
        lcb = mean - z_value * std / math.sqrt(len(evs))
        threshold = float(os.environ.get("ROLLOUT_CRITIC_EV_GATE", "0.10"))
        return {
            "critic_rollout_gate/fixed_panel_ev_mean": mean,
            "critic_rollout_gate/fixed_panel_ev_std": std,
            "critic_rollout_gate/fixed_panel_ev_min": min(evs),
            "critic_rollout_gate/fixed_panel_ev_max": max(evs),
            "critic_rollout_gate/fixed_panel_ev_lcb": lcb,
            "critic_rollout_gate/fixed_panel_mse_mean": sum(mses) / len(mses),
            "critic_rollout_gate/fixed_panel_pass_fraction": sum(
                value > threshold for value in evs
            ) / len(evs),
            "critic_rollout_gate/fixed_panel_count": float(len(evs)),
            "_fixed_rollout_gate_statistic": lcb,
        }

    @torch.no_grad()
    def _capture_actor_real_posterior_probe(self, episodes: list[Any]) -> None:
        """Lock real posterior states while Actor is still at update zero."""
        if os.environ.get("REAL_POSTERIOR_ACTOR_GATE", "0") != "1":
            return
        if self._actor_real_probe_states is not None:
            return
        if self._actor_ppo_updates != 0:
            raise RuntimeError("real posterior Actor probe was captured too late")
        states = []
        initial_only = (
            os.environ.get("REAL_POSTERIOR_ACTOR_INITIAL_ONLY", "0") == "1"
        )
        for episode in episodes:
            trajectory = episode.info.get("_policy_trajectory")
            if trajectory and len(trajectory["states"]):
                states.append(
                    trajectory["states"][:1]
                    if initial_only else trajectory["states"]
                )
        if not states:
            raise RuntimeError(
                "REAL_POSTERIOR_ACTOR_GATE found no captured policy states"
            )
        self._lock_actor_real_posterior_probe_states(
            torch.cat(states), initial_only=initial_only
        )

    @torch.no_grad()
    def _lock_actor_real_posterior_probe_states(
        self, all_states: Tensor, *, initial_only: bool = False
    ) -> None:
        """Lock an explicit fixed posterior panel before the first Actor update."""
        if self._actor_real_probe_states is not None:
            return
        if self._actor_ppo_updates != 0:
            raise RuntimeError("real posterior Actor probe was captured too late")
        capacity = int(os.environ.get(
            "REAL_POSTERIOR_ACTOR_PROBE_SIZE", "512"
        ))
        if capacity <= 0:
            raise ValueError("REAL_POSTERIOR_ACTOR_PROBE_SIZE must be positive")
        if len(all_states) > capacity:
            index = torch.linspace(
                0, len(all_states) - 1, capacity
            ).round().long()
            all_states = all_states[index]
        device = next(self.policy.parameters()).device
        states_device = all_states.to(device=device)
        logits_fn = getattr(self.policy, "actor_logits", None)
        if not callable(logits_fn):
            raise RuntimeError("real posterior Actor gate requires actor_logits()")
        logits = torch.cat([
            logits_fn(states_device[begin:begin + 64]).float()
            for begin in range(0, len(states_device), 64)
        ])
        counts = torch.bincount(logits.argmax(-1), minlength=4)
        max_fraction = float(counts.max() / max(1, len(logits)))
        max_allowed = float(os.environ.get(
            "REAL_POSTERIOR_ACTOR_MAX_ARGMAX_FRACTION", "0.80"
        ))
        min_actions = int(os.environ.get(
            "REAL_POSTERIOR_ACTOR_MIN_ARGMAX_ACTIONS", "2"
        ))
        if max_fraction > max_allowed or int((counts > 0).sum()) < min_actions:
            raise RuntimeError(
                "Actor0 already fails the real posterior deployment gate: "
                f"counts={counts.tolist()}, max_fraction={max_fraction:.4f}"
            )
        self._actor_real_probe_states = states_device
        self._actor_real_probe_reference_logits = logits.detach().clone()
        print(
            "REAL_POSTERIOR_ACTOR_GATE locked "
            f"states={len(states_device)} counts={counts.tolist()} "
            f"initial_only={int(initial_only)}",
            flush=True,
        )

    @torch.no_grad()
    def _actor_real_posterior_metrics(self) -> dict[str, float]:
        states = self._actor_real_probe_states
        reference = self._actor_real_probe_reference_logits
        if states is None or reference is None:
            raise RuntimeError("real posterior Actor probe is unavailable")
        logits_fn = getattr(self.policy, "actor_logits")
        logits = torch.cat([
            logits_fn(states[begin:begin + 64]).float()
            for begin in range(0, len(states), 64)
        ])
        probabilities = logits.softmax(-1)
        reference_probabilities = reference.softmax(-1)
        actions = logits.argmax(-1)
        counts = torch.bincount(actions, minlength=4)
        fractions = counts.float() / max(1, len(actions))
        delta = logits - reference
        max_allowed = float(os.environ.get(
            "REAL_POSTERIOR_ACTOR_MAX_ARGMAX_FRACTION", "0.80"
        ))
        min_actions = int(os.environ.get(
            "REAL_POSTERIOR_ACTOR_MIN_ARGMAX_ACTIONS", "2"
        ))
        degenerate = (
            float(fractions.max()) > max_allowed
            or int((counts > 0).sum()) < min_actions
        )
        metrics = {
            "real_posterior_actor/probe_states": float(len(states)),
            "real_posterior_actor/argmax_max_action_fraction": float(
                fractions.max()
            ),
            "real_posterior_actor/num_argmax_actions": float(
                (counts > 0).sum()
            ),
            "real_posterior_actor/action_0_fraction": float(fractions[0]),
            "real_posterior_actor/action_1_fraction": float(fractions[1]),
            "real_posterior_actor/action_2_fraction": float(fractions[2]),
            "real_posterior_actor/action_3_fraction": float(fractions[3]),
            "real_posterior_actor/argmax_flip_rate": float(
                (actions != reference.argmax(-1)).float().mean()
            ),
            "real_posterior_actor/kl_from_start": float((
                reference_probabilities
                * (
                    reference_probabilities.clamp_min(1e-8).log()
                    - probabilities.clamp_min(1e-8).log()
                )
            ).sum(-1).mean()),
            "real_posterior_actor/logit_delta_state_std": float(
                delta.std(0).mean()
            ),
            "real_posterior_actor/degenerate": float(degenerate),
        }
        if degenerate:
            metrics["_actor_update_rejected"] = 1.0
        return metrics

    def _invalidate_real_critic_latent_cache(self) -> None:
        """Drop evidence encoded by an older accepted WM representation."""
        self._real_critic_train_replay = None
        self._real_critic_validation = None
        self._pending_real_critic_batch = None
        self._real_critic_anchor_ev_ema = None
        self._real_critic_anchor_ev_streak = 0
        self._real_critic_anchor_failure_streak = 0
        self._real_critic_anchor_ready = False
        # A changed latent representation also invalidates EV measured on
        # rollouts encoded by the previous representation.  Actor must earn a
        # fresh three-check rollout gate before it can resume.
        self._rollout_critic_ev_ema = None
        self._rollout_critic_ev_streak = 0
        self._rollout_critic_ready = False
        self._rollout_critic_released_once = False
        self._rollout_critic_post_release_failures = 0
        self._grounded_training_safety_snapshot = None
        self._joint_h2_replay = None
        self._joint_target_critic = None
        self._fixed_rollout_critic_panels = None
        self._actor_real_probe_states = None
        self._actor_real_probe_reference_logits = None
        self._real_critic_cache_generation += 1

    def _ensure_joint_target_critic(self) -> None:
        """Lazily snapshot the complete trainable Critic on its device."""
        parameters = list(self.ppo_updater._critic_trainable)
        if self._joint_target_critic is None:
            self._joint_target_critic = [
                parameter.detach().clone() for parameter in parameters
            ]
        if len(self._joint_target_critic) != len(parameters):
            raise RuntimeError("joint target Critic parameter set changed")

    @torch.no_grad()
    def _swap_joint_target_critic(self) -> None:
        """Swap online and EMA-target Critic values without a second model."""
        self._ensure_joint_target_critic()
        assert self._joint_target_critic is not None
        for parameter, target in zip(
            self.ppo_updater._critic_trainable,
            self._joint_target_critic,
        ):
            online = parameter.detach().clone()
            parameter.copy_(target)
            target.copy_(online)

    @torch.no_grad()
    def _update_joint_target_critic(self) -> float:
        """Polyak-update the frozen bootstrap Critic and return RMS lag."""
        self._ensure_joint_target_critic()
        assert self._joint_target_critic is not None
        tau = float(os.environ.get("JOINT_TARGET_CRITIC_TAU", "0.01"))
        if not 0.0 < tau <= 1.0:
            raise ValueError("JOINT_TARGET_CRITIC_TAU must be in (0, 1]")
        squared = count = 0.0
        for parameter, target in zip(
            self.ppo_updater._critic_trainable,
            self._joint_target_critic,
        ):
            target.mul_(1.0 - tau).add_(parameter.detach(), alpha=tau)
            delta = parameter.detach().float() - target.float()
            squared += float(delta.square().sum())
            count += float(delta.numel())
        return math.sqrt(squared / max(count, 1.0))

    def _update_rollout_critic_gate(self, rollout_ev: float) -> dict[str, float]:
        """Strict initial release followed by failure-count hysteresis."""
        alpha = float(os.environ.get("ROLLOUT_CRITIC_EV_EMA_ALPHA", "0.2"))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ROLLOUT_CRITIC_EV_EMA_ALPHA must be in (0, 1]")
        threshold = float(os.environ.get("ROLLOUT_CRITIC_EV_GATE", "0.10"))
        patience = int(os.environ.get("ROLLOUT_CRITIC_EV_PATIENCE", "3"))
        if patience <= 0:
            raise ValueError("ROLLOUT_CRITIC_EV_PATIENCE must be positive")
        post_release_patience = int(os.environ.get(
            "ROLLOUT_CRITIC_POST_RELEASE_PATIENCE", str(patience)
        ))
        if post_release_patience <= 0:
            raise ValueError(
                "ROLLOUT_CRITIC_POST_RELEASE_PATIENCE must be positive"
            )

        previous = self._rollout_critic_ev_ema
        rollout_ema = (
            rollout_ev
            if previous is None
            else (1.0 - alpha) * previous + alpha * rollout_ev
        )
        # Requiring the current check as well as the EMA prevents a stale high
        # average from hiding an immediate distribution-level Critic failure.
        passed = rollout_ev > threshold and rollout_ema > threshold
        self._rollout_critic_ev_ema = rollout_ema
        just_released = False
        rollback_required = False
        hysteresis_enabled = (
            os.environ.get("GROUNDED_GATE_HYSTERESIS", "0") == "1"
        )
        if not hysteresis_enabled:
            self._rollout_critic_ev_streak = (
                self._rollout_critic_ev_streak + 1 if passed else 0
            )
            self._rollout_critic_ready = (
                self._rollout_critic_ev_streak >= patience
            )
        elif not self._rollout_critic_released_once:
            self._rollout_critic_ev_streak = (
                self._rollout_critic_ev_streak + 1 if passed else 0
            )
            self._rollout_critic_ready = (
                self._rollout_critic_ev_streak >= patience
            )
            if self._rollout_critic_ready:
                self._rollout_critic_released_once = True
                self._rollout_critic_post_release_failures = 0
                just_released = True
        else:
            self._rollout_critic_ev_streak = (
                self._rollout_critic_ev_streak + 1 if passed else 0
            )
            self._rollout_critic_post_release_failures = (
                0 if passed
                else self._rollout_critic_post_release_failures + 1
            )
            rollback_required = (
                self._rollout_critic_post_release_failures
                >= post_release_patience
            )
            self._rollout_critic_ready = not rollback_required
            if rollback_required:
                self._rollout_critic_released_once = False
        return {
            "critic_rollout_gate/heldout_mc_ev": float(rollout_ev),
            "critic_rollout_gate/heldout_mc_ev_ema": float(rollout_ema),
            "critic_rollout_gate/pass": float(passed),
            "critic_rollout_gate/streak": float(self._rollout_critic_ev_streak),
            "critic_rollout_gate/ready": float(self._rollout_critic_ready),
            "critic_rollout_gate/released_once": float(
                getattr(self, "_rollout_critic_released_once", False)
            ),
            "critic_rollout_gate/post_release_failures": float(
                getattr(
                    self, "_rollout_critic_post_release_failures", 0
                )
            ),
            "critic_rollout_gate/hysteresis_enabled": float(
                hysteresis_enabled
            ),
            "_critic_rollout_gate_just_released": float(just_released),
            "_critic_rollout_gate_rollback_required": float(
                rollback_required
            ),
        }

    def _snapshot_grounded_training_safety(self) -> None:
        correction_fn = getattr(self.policy, "actor_logit_correction", None)
        self._grounded_training_safety_snapshot = {
            "ppo": self.ppo_updater.snapshot_training_transaction(),
            "actor_updates": int(self._actor_ppo_updates),
            "actor_logit_correction": (
                correction_fn() if callable(correction_fn) else None
            ),
            "joint_target_critic": (
                None if self._joint_target_critic is None else [
                    value.detach().clone()
                    for value in self._joint_target_critic
                ]
            ),
        }

    def _restore_grounded_training_safety(self) -> None:
        snapshot = self._grounded_training_safety_snapshot
        if snapshot is None:
            raise RuntimeError("grounded safety rollback has no snapshot")
        self.ppo_updater.restore_training_transaction(snapshot["ppo"])
        self._actor_ppo_updates = int(snapshot["actor_updates"])
        correction = snapshot["actor_logit_correction"]
        restore_correction = getattr(
            self.policy, "restore_actor_logit_correction", None
        )
        if correction is not None and callable(restore_correction):
            restore_correction(correction)
        target = snapshot["joint_target_critic"]
        self._joint_target_critic = (
            None if target is None else [value.detach().clone() for value in target]
        )

    def _snapshot_actor_update_transaction(self) -> dict:
        correction_fn = getattr(self.policy, "actor_logit_correction", None)
        include_critic = (
            os.environ.get("COUNTERFACTUAL_ACTOR_TRANSACTION_GATE", "0") == "1"
        )
        return {
            "ppo": (
                self.ppo_updater.snapshot_training_transaction()
                if include_critic
                else self.ppo_updater.snapshot_actor_transaction()
            ),
            "include_critic": include_critic,
            "actor_logit_correction": (
                correction_fn() if callable(correction_fn) else None
            ),
        }

    def _restore_actor_update_transaction(self, snapshot: dict) -> None:
        if snapshot.get("include_critic", False):
            self.ppo_updater.restore_training_transaction(snapshot["ppo"])
        else:
            self.ppo_updater.restore_actor_transaction(snapshot["ppo"])
        correction = snapshot["actor_logit_correction"]
        restore_correction = getattr(
            self.policy, "restore_actor_logit_correction", None
        )
        if correction is not None and callable(restore_correction):
            restore_correction(correction)

    @torch.no_grad()
    def _counterfactual_actor_validation_metrics(
        self,
    ) -> tuple[dict[str, float], bool, float]:
        """Evaluate the immutable, level-disjoint four-action H1/H2 panel."""
        validation = self._critic_warmup_validation
        bucket_ids_by_action = getattr(
            self, "_counterfactual_validation_action_bucket_ids", None
        )
        bucket_names = getattr(
            self, "counterfactual_h2_validation_bucket_names", None
        )
        if validation is None or bucket_ids_by_action is None or not bucket_names:
            raise RuntimeError("counterfactual Actor validation panel is unavailable")
        grouped_states = validation.states.reshape(
            -1, 4, *validation.states.shape[1:]
        )[:, 0]
        logits = torch.cat([
            self.policy.actor_logits(grouped_states[begin:begin + 64]).float()
            for begin in range(0, len(grouped_states), 64)
        ])
        targets = validation.returns.float().view(-1, 4)
        group_bucket_ids = bucket_ids_by_action.view(-1, 4)[:, 0]
        top1_gate = float(os.environ.get(
            "COUNTERFACTUAL_ACTOR_TOP1_GATE", "0.60"
        ))
        pairwise_gate = float(os.environ.get(
            "COUNTERFACTUAL_ACTOR_PAIRWISE_GATE", "0.60"
        ))
        result: dict[str, float] = {}
        all_passed = True
        scores = []
        for bucket_id, bucket_name in enumerate(bucket_names):
            selected = group_bucket_ids == bucket_id
            bucket = _four_action_ranking_metrics(
                logits[selected], targets[selected]
            )
            passed = (
                bucket["top1_accuracy"] >= top1_gate
                and bucket["pairwise_accuracy"] >= pairwise_gate
                and bucket["informative_pairs"] > 0.0
            )
            all_passed = all_passed and passed
            scores.append(
                0.5 * (
                    bucket["top1_accuracy"] + bucket["pairwise_accuracy"]
                )
            )
            prefix = f"actor_validation/bucket_{bucket_name}"
            result.update({f"{prefix}/{key}": value for key, value in bucket.items()})
            result[f"{prefix}/passed"] = float(passed)
        score = float(sum(scores) / max(1, len(scores)))
        result["actor_validation/all_buckets_passed"] = float(all_passed)
        result["actor_validation/mean_ranking_score"] = score
        return result, all_passed, score

    @torch.no_grad()
    def _recenter_counterfactual_replay(self, batch: PPOBatch) -> PPOBatch:
        """Refresh policy-centering before reusing immutable four-action Q targets."""
        count = len(batch.actions)
        if count == 0 or count % 4:
            raise RuntimeError("counterfactual replay must contain complete action groups")
        actions = batch.actions.long().view(-1, 4)
        expected = torch.arange(4, device=actions.device).expand_as(actions)
        if not torch.equal(actions, expected):
            raise RuntimeError("counterfactual replay action groups are not ordered 0..3")
        states = batch.states.view(-1, 4, *batch.states.shape[1:])[:, 0]
        logits = torch.cat([
            self.policy.actor_logits(states[begin:begin + 64]).float()
            for begin in range(0, len(states), 64)
        ])
        probabilities = logits.softmax(-1)
        q_targets = batch.returns.float().view(-1, 4)
        centered = q_targets - (probabilities * q_targets).sum(-1, keepdim=True)
        advantages = 4.0 * probabilities * centered
        old_log_probs = logits.log_softmax(-1).flatten()
        return PPOBatch(
            states=batch.states,
            actions=batch.actions,
            advantages=advantages.flatten(),
            returns=batch.returns,
            old_log_probs=old_log_probs,
            old_values=batch.old_values,
        )

    @staticmethod
    def _concatenate_ppo_batches(
        first: PPOBatch | None,
        second: PPOBatch,
    ) -> PPOBatch:
        """Concatenate PPO batches in warmup and released-Actor stages."""
        if first is None:
            return second
        return PPOBatch(**{
            name: torch.cat([getattr(first, name), getattr(second, name)])
            for name in PPOBatch.__dataclass_fields__
        })

    @staticmethod
    def validate_joint_critic_grounding_contract(
        *, eval_episodes: int
    ) -> dict[str, float]:
        """Fail fast if a staged run would replace rather than add grounding."""
        if os.environ.get("REQUIRE_JOINT_CRITIC_GROUNDING", "0") != "1":
            return {}
        required_flags = (
            "REAL_RETURN_CRITIC_ANCHOR",
            "REAL_CRITIC_COUNTERFACTUAL_FIRST_ACTION",
            "REAL_CRITIC_FIXED_EVAL_LEVELS",
            "REAL_CRITIC_FIRST_TRANSITION_ONLY",
            "REQUIRE_GROUNDED_ACTOR_GATE",
            "REQUIRE_ROLLOUT_CRITIC_EV_GATE",
            "IMAGINED_CRITIC_UPDATE",
            "GROUP_AWARE_JOINT_CRITIC",
            "VALUE_BOOTSTRAP",
            "JOINT_TARGET_CRITIC",
            "JOINT_CRITIC_PCGRAD",
        )
        disabled = [
            name for name in required_flags
            if os.environ.get(name, "0") != "1"
        ]
        if disabled:
            raise RuntimeError(
                "joint Critic grounding contract requires these flags=1: "
                + ", ".join(disabled)
            )
        calibration_updates = int(os.environ.get(
            "REAL_CRITIC_CALIBRATION_UPDATES", "1"
        ))
        maximum_calibration_updates = int(os.environ.get(
            "JOINT_CRITIC_MAX_REAL_CALIBRATION_UPDATES", "2"
        ))
        if not 1 <= calibration_updates <= maximum_calibration_updates:
            raise RuntimeError(
                "joint Critic grounding requires limited real calibration: "
                f"REAL_CRITIC_CALIBRATION_UPDATES={calibration_updates}, "
                f"allowed=1..{maximum_calibration_updates}"
            )
        fixed_level_offset = int(os.environ.get(
            "REAL_CRITIC_FIXED_LEVEL_OFFSET", "0"
        ))
        if fixed_level_offset < eval_episodes:
            raise RuntimeError(
                "fixed Critic grounding levels overlap rapid evaluation: "
                f"offset={fixed_level_offset}, eval_episodes={eval_episodes}"
            )
        fixed_level_limit = int(os.environ.get(
            "REAL_CRITIC_FIXED_LEVEL_LIMIT", "32"
        ))
        if fixed_level_limit < 20:
            raise RuntimeError(
                "joint Critic grounding needs at least 20 fixed initial levels"
            )
        real_fraction = float(os.environ.get(
            "JOINT_CRITIC_REAL_FRACTION", "0.25"
        ))
        if not 0.0 < real_fraction < 0.5:
            raise RuntimeError(
                "JOINT_CRITIC_REAL_FRACTION must be in (0, 0.5)"
            )
        if int(os.environ.get("JOINT_H2_REPLAY_CAPACITY", "4096")) < 512:
            raise RuntimeError(
                "JOINT_H2_REPLAY_CAPACITY must retain at least 512 samples"
            )
        return {
            "critic_joint_rehearsal/contract_valid": 1.0,
            "critic_joint_rehearsal/real_calibration_updates": float(
                calibration_updates
            ),
            "critic_joint_rehearsal/fixed_level_offset": float(
                fixed_level_offset
            ),
            "critic_joint_rehearsal/fixed_level_limit": float(
                fixed_level_limit
            ),
            "critic_joint_rehearsal/group_aware": 1.0,
            "critic_joint_rehearsal/value_bootstrap": 1.0,
            "critic_joint_rehearsal/real_fraction": real_fraction,
        }

    def train(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        start_update: int = 0,
    ) -> None:
        """Main training loop."""
        cfg = self.config
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)

        # Formal Actor training is a separate release stage. Fail before an
        # expensive real-environment baseline if the launcher forgot to load
        # the released Critic checkpoint or its immutable H1/H2 audit panel.
        if os.environ.get("REQUIRE_RELEASED_CRITIC_AT_START", "0") == "1":
            missing_release_state: list[str] = []
            if not self._critic_warmup_complete:
                missing_release_state.append("critic_warmup_complete")
            if self._critic_warmup_validation is None:
                missing_release_state.append("critic_warmup_validation")
            if getattr(
                self, "_counterfactual_validation_action_bucket_ids", None
            ) is None:
                missing_release_state.append("validation_action_bucket_ids")
            if not getattr(
                self, "counterfactual_h2_validation_bucket_names", None
            ):
                missing_release_state.append("validation_bucket_names")
            if missing_release_state:
                raise RuntimeError(
                    "formal PPO requires the released Stage-3 Critic state; "
                    "missing " + ", ".join(missing_release_state)
                )
            print(
                "Formal PPO release contract verified: Critic Gate is "
                "complete and the fixed H1/H2 Actor validation panel is "
                "restored.",
                flush=True,
            )

        initialization_metrics: dict[str, float] = {}
        joint_contract_metrics = self.validate_joint_critic_grounding_contract(
            eval_episodes=cfg.eval_episodes
        )
        if joint_contract_metrics:
            initialization_metrics.update(joint_contract_metrics)
            print(
                "JOINT_CRITIC_GROUNDING contract verified: fresh imagined "
                "Critic rehearsal + limited fixed-real calibration; Actor "
                "remains gated.",
                flush=True,
            )
        self._stabilize_policy_forward()
        initial_eval_success_rate: float | None = None
        if (
            start_update == 0
            and self.offline_behavior_cloner is not None
            and os.environ.get("SKIP_OFFLINE_BC_ON_RESUME", "0") != "1"
        ):
            print("=== Offline latent behavior cloning ===", flush=True)
            initialization_metrics.update(self.offline_behavior_cloner.fit())
            if checkpoint_root is not None:
                bc_init_path = checkpoint_root / "bc_init.pt"
                self._save_checkpoint(bc_init_path, 0)
                print(
                    f"  Offline BC initialization saved: {bc_init_path}",
                    flush=True,
                )

            online_warmup = getattr(self, "online_actor_warmup_fn", None)
            if callable(online_warmup):
                print(
                    "=== Solver-supervised online-rendered Actor warm-up ===",
                    flush=True,
                )
                initialization_metrics.update(online_warmup())
                capture = getattr(self.policy, "capture_behavior_reference", None)
                if callable(capture):
                    capture()
                if checkpoint_root is not None:
                    warmup_path = checkpoint_root / "online_actor_init.pt"
                    self._save_checkpoint(warmup_path, 0)
                    print(
                        f"  Online-rendered Actor initialization saved: {warmup_path}",
                        flush=True,
                    )

            pre_ppo_gate_failures: list[str] = []
            # Hard deployment gate on the exact fixed-eval reset observations.
            # Offline validation alone cannot detect a policy that generalizes
            # to only two of Sokoban's four actions on real initial layouts.
            if os.environ.get("PRE_PPO_ACTOR_GATE", "0") == "1":
                initial_evaluator = getattr(self.evaluator, "evaluate_initial_policy", None)
                if not callable(initial_evaluator):
                    raise RuntimeError(
                        "PRE_PPO_ACTOR_GATE requires a fixed-level evaluator "
                        "with evaluate_initial_policy()"
                    )
                gate_metrics = initial_evaluator(
                    batch_size=int(os.environ.get("PRE_PPO_ACTOR_GATE_BATCH_SIZE", "32"))
                )
                initialization_metrics.update(gate_metrics)
                covered = int(gate_metrics["online_actor_gate/num_predicted_actions"])
                max_fraction = gate_metrics["online_actor_gate/max_action_fraction"]
                min_covered = int(os.environ.get("PRE_PPO_ACTOR_GATE_MIN_ACTIONS", "4"))
                max_allowed = float(os.environ.get(
                    "PRE_PPO_ACTOR_GATE_MAX_ACTION_FRACTION", "0.60"
                ))
                print(f"PRE_PPO_ACTOR_GATE {gate_metrics}", flush=True)
                if covered < min_covered or max_fraction > max_allowed:
                    pre_ppo_gate_failures.append(
                        "PRE_PPO_ACTOR_GATE rejected BC actor: "
                        f"covered={covered} (required>={min_covered}), "
                        f"max_action_fraction={max_fraction:.4f} "
                        f"(required<={max_allowed:.4f})."
                    )

            if os.environ.get("PRE_PPO_REWARD_ACTION_GATE", "0") == "1":
                from ..rl.counterfactual_action_audit import (
                    run_same_start_four_action_audit,
                )
                if checkpoint_root is None:
                    raise RuntimeError(
                        "PRE_PPO_REWARD_ACTION_GATE requires checkpoint_dir"
                    )
                audit = run_same_start_four_action_audit(
                    pipeline=self,
                    policy=self.policy,
                    world_model=self.world_model,
                    output_path=checkpoint_root / "pre_ppo_h3_action_audit.json",
                    num_samples=int(os.environ.get(
                        "PRE_PPO_REWARD_ACTION_GATE_SAMPLES", "512"
                    )),
                    batch_size=int(os.environ.get(
                        "PRE_PPO_REWARD_ACTION_GATE_BATCH_SIZE", "8"
                    )),
                    seed=int(os.environ.get(
                        "PRE_PPO_REWARD_ACTION_GATE_SEED", "20260802"
                    )),
                )
                h3_rank = float(audit["counterfactual"]["h3_bc"][
                    "reference_is_best_rate"
                ])
                injection = getattr(
                    self.world_model, "_reward_head_injection", {}
                )
                solver_ranking = injection.get("ranking_test", {})
                if injection.get("solver_ranking_repair"):
                    # The legacy audit uses one logged demonstration action and
                    # incorrectly marks alternative optimal Sokoban actions as
                    # negatives. A solver-ranked checkpoint is gated on its
                    # episode-disjoint optimal-action-set test instead.
                    h3_rank = float(solver_ranking.get("top1", -1.0))
                    initialization_metrics[
                        "reward_action_gate/legacy_logged_expert_top1_rate"
                    ] = float(audit["counterfactual"]["h3_bc"][
                        "reference_is_best_rate"
                    ])
                min_h3_rank = float(os.environ.get(
                    "PRE_PPO_REWARD_ACTION_GATE_MIN_EXPERT_TOP1",
                    "0.55" if injection.get("solver_ranking_repair") else "0.35"
                ))
                initialization_metrics[
                    "reward_action_gate/h3_expert_top1_rate"
                ] = h3_rank
                print(
                    "PRE_PPO_REWARD_ACTION_GATE "
                    f"h3_expert_top1_rate={h3_rank:.4f} "
                    f"required>={min_h3_rank:.4f}",
                    flush=True,
                )
                if h3_rank < min_h3_rank:
                    pre_ppo_gate_failures.append(
                        "PRE_PPO_REWARD_ACTION_GATE rejected Reward Head: "
                        f"expert first action is H3 top-1 only {h3_rank:.4f} "
                        f"(required>={min_h3_rank:.4f}; random baseline=0.25)."
                    )
            if pre_ppo_gate_failures:
                raise RuntimeError(
                    "\n".join(pre_ppo_gate_failures)
                    + "\nPre-PPO quality gates failed; no PPO rollout was run."
                )

        # A pre-update baseline is essential for separating genuine policy
        # improvement from variance in the first periodic evaluation. On
        # resume, log it at ``start_update`` so branch experiments establish
        # the exact shared checkpoint performance before taking another step.
        if (
            cfg.eval_at_start
            and self.evaluator is not None
        ):
            self._stabilize_policy_forward()
            t_eval = time.time()
            baseline_metrics: dict[str, float] = {
                "update": float(start_update),
                "progress": float(start_update) / float(cfg.total_updates),
                # W&B uses this as the custom step for eval/* metrics.  A
                # fresh Actor branch therefore always starts its evaluation
                # curve at x=0 even when the source checkpoint's global
                # update is non-zero (for example update 220).
                "eval/actor_ppo_update": float(self._actor_ppo_updates),
            }
            baseline_metrics.update(initialization_metrics)
            baseline_metrics.update(self.evaluator.evaluate(cfg.eval_episodes))
            baseline_metrics["timing/eval_sec"] = time.time() - t_eval
            if self.logger is not None:
                self.logger.log_scalars(start_update, baseline_metrics)
            sr = baseline_metrics.get("eval/success_rate", -1.0)
            if math.isfinite(float(sr)):
                initial_eval_success_rate = float(sr)
            if checkpoint_root is not None:
                self._maybe_save_best_checkpoint(
                    checkpoint_root, start_update, baseline_metrics
                )
            print(
                (
                    "[actor_ppo_update 0] "
                    if os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1"
                    else f"[update {start_update}/{cfg.total_updates}] "
                )
                +
                f"eval baseline sr={sr:.2f}",
                flush=True,
            )
        elif initialization_metrics and self.logger is not None:
            self.logger.log_scalars(0, initialization_metrics)

        # Explicit isolated BC gate: fit Actor, evaluate the fixed real set,
        # save update-0, and stop before collecting any imagined PPO rollout.
        if os.environ.get("BC_ONLY", "0") == "1":
            print("BC_ONLY=1: stopped after fixed pre-PPO evaluation.", flush=True)
            return

        last_completed_update = start_update
        eval_early_stop_patience = int(
            os.environ.get("EVAL_EARLY_STOP_PATIENCE", "0")
        )
        eval_early_stop_drop = float(
            os.environ.get("EVAL_EARLY_STOP_DROP", "0.0")
        )
        eval_early_stop_failures = 0
        flat_sr_patience = int(os.environ.get("EVAL_FLAT_SR_PATIENCE", "0"))
        flat_sr_tolerance = float(os.environ.get("EVAL_FLAT_SR_TOLERANCE", "1e-6"))
        flat_sr_baseline: float | None = None
        flat_sr_failures = 0
        flat_sr_changed = False
        critic_diagnostic_only = (
            os.environ.get("REAL_CRITIC_DIAGNOSTIC_ONLY", "0") == "1"
        )
        critic_diagnostic_updates = 0
        critic_pretrain_budget_exhausted = False
        p0_critic_only = os.environ.get("P0_CRITIC_ONLY", "0") == "1"
        if (
            os.environ.get("REQUIRE_ROLLOUT_CRITIC_EV_GATE", "0") == "1"
            and os.environ.get(
                "FIXED_ROLLOUT_GATE_USE_REAL_POSTERIOR", "0"
            ) != "1"
        ):
            self._initialize_fixed_rollout_critic_panels()
        initial_replay_sampler = getattr(self, "replay_sampler", None)
        if (
            os.environ.get("FORMAL_UNIFIED_PPO", "0") == "1"
            and initial_replay_sampler is not None
            and hasattr(initial_replay_sampler, "collection_complete")
        ):
            expected_online = min(
                self._actor_ppo_updates, initial_replay_sampler.online_target
            )
            align_visibility = getattr(
                initial_replay_sampler, "align_online_visibility", None
            )
            if callable(align_visibility):
                align_visibility(
                    self._actor_ppo_updates,
                    require_available=True,
                )
            if initial_replay_sampler.online_size != expected_online:
                raise RuntimeError(
                    "formal PPO replay does not match its Actor checkpoint: "
                    f"actor_update={self._actor_ppo_updates}, "
                    f"online={initial_replay_sampler.online_size}, "
                    f"expected={expected_online}. Start Actor0 with a fresh "
                    "ONLINE_REPLAY_ROOT, or resume PPO with its matching pool."
                )
        for update in range(start_update, cfg.total_updates):
            t0 = time.time()
            current_update = update + 1
            metrics: dict[str, float] = {
                "update": float(current_update),
                "progress": float(current_update) / float(cfg.total_updates),
            }
            replay_sampler = getattr(self, "replay_sampler", None)
            unified_replay = bool(
                replay_sampler is not None
                and hasattr(replay_sampler, "collection_complete")
            )
            if unified_replay:
                metrics.update({
                    "buffer/offline_episodes": float(
                        len(replay_sampler.offline_pool)
                    ),
                    "buffer/online_episodes": float(replay_sampler.online_size),
                    "buffer/online_target": float(replay_sampler.online_target),
                    "buffer/online_fraction": float(
                        replay_sampler.expected_online_fraction
                    ),
                    "buffer/frozen": float(replay_sampler.collection_complete),
                })
            if os.environ.get("CRITIC_PRETRAIN_ONLY", "0") == "1":
                metrics["critic_pretrain/step"] = float(
                    current_update - start_update
                )

            # --- 1. Collect real episodes (periodic) ---
            if (
                self._should_collect(current_update)
                and not (unified_replay and replay_sampler.collection_complete)
            ):
                self._stabilize_policy_forward()
                t_collect = time.time()
                assert self.real_collector is not None
                fixed_counterfactual = (
                    os.environ.get("REAL_RETURN_CRITIC_ANCHOR", "0") == "1"
                    and os.environ.get(
                        "REAL_CRITIC_COUNTERFACTUAL_FIRST_ACTION", "0"
                    ) == "1"
                    and os.environ.get(
                        "REAL_CRITIC_FIXED_EVAL_LEVELS", "0"
                    ) == "1"
                )
                fixed_cache_ready = (
                    fixed_counterfactual
                    and self._real_critic_train_replay is not None
                    and self._real_critic_validation is not None
                )
                if fixed_counterfactual:
                    # Fixed counterfactual trajectories are collected below;
                    # do not also pay for unrelated random online episodes.
                    from types import SimpleNamespace
                    collect_result = SimpleNamespace(metrics={}, episodes=[])
                else:
                    collect_result = self.real_collector.collect_and_store(
                        cfg.collect_episodes, current_update
                    )
                # Extract metrics from result
                if hasattr(collect_result, 'metrics'):
                    metrics.update({
                        f"collect/{k}": v
                        for k, v in collect_result.metrics.items()
                    })
                elif isinstance(collect_result, dict):
                    metrics.update({
                        f"collect/{k}": v
                        for k, v in collect_result.items()
                    })
                metrics["timing/collect_sec"] = time.time() - t_collect
                if unified_replay:
                    metrics.update({
                        "buffer/online_episodes": float(replay_sampler.online_size),
                        "buffer/online_fraction": float(
                            replay_sampler.expected_online_fraction
                        ),
                        "buffer/frozen": float(
                            replay_sampler.collection_complete
                        ),
                    })
                if os.environ.get("REAL_RETURN_CRITIC_ANCHOR", "0") == "1":
                    from ..rl.real_return_anchor import collection_to_ppo_batch
                    from types import SimpleNamespace
                    anchor_collect_result = collect_result
                    if os.environ.get(
                        "REAL_CRITIC_COUNTERFACTUAL_FIRST_ACTION", "0"
                    ) == "1":
                        counterfactual_episodes = int(os.environ.get(
                            "REAL_CRITIC_COUNTERFACTUAL_EPISODES", "16"
                        ))
                        if counterfactual_episodes <= 0:
                            raise ValueError(
                                "REAL_CRITIC_COUNTERFACTUAL_EPISODES must be positive"
                            )
                        # Complete groups of four share the exact initial level
                        # and differ only in their forced first action.
                        group_count = max(1, math.ceil(counterfactual_episodes / 4))
                        forced_actions = [
                            action for _ in range(group_count) for action in range(4)
                        ]
                        collector_config = self.real_collector.config
                        old_deterministic = collector_config.deterministic
                        old_epsilon = collector_config.exploration_epsilon
                        old_capture = collector_config.capture_policy_trajectory
                        if not fixed_cache_ready:
                            collect_kwargs: dict[str, Any] = {}
                            if fixed_counterfactual:
                                levels = getattr(self.evaluator, "eval_levels", None)
                                if not levels:
                                    raise RuntimeError(
                                        "REAL_CRITIC_FIXED_EVAL_LEVELS requires "
                                        "a fixed-level evaluator"
                                    )
                                limit = int(os.environ.get(
                                    "REAL_CRITIC_FIXED_LEVEL_LIMIT", "32"
                                ))
                                offset = int(os.environ.get(
                                    "REAL_CRITIC_FIXED_LEVEL_OFFSET", "0"
                                ))
                                if offset < 0:
                                    raise ValueError(
                                        "REAL_CRITIC_FIXED_LEVEL_OFFSET must be non-negative"
                                    )
                                fixed_levels = list(levels[offset:offset + limit])
                                if len(fixed_levels) < 5:
                                    raise RuntimeError(
                                        "fixed Critic grounding needs at least 5 levels; "
                                        f"available={len(fixed_levels)}, offset={offset}, "
                                        f"limit={limit}"
                                    )
                                group_count = len(fixed_levels)
                                forced_actions = [
                                    action for _ in fixed_levels for action in range(4)
                                ]
                                collect_kwargs["levels"] = [
                                    level for level in fixed_levels for _ in range(4)
                                ]
                            else:
                                base_seed = (
                                    self.real_collector.config.seed_offset
                                    + 10_000_000 + current_update * group_count
                                )
                                collect_kwargs["seeds"] = [
                                    base_seed + group
                                    for group in range(group_count)
                                    for _ in range(4)
                                ]
                            try:
                                collector_config.deterministic = True
                                collector_config.exploration_epsilon = 0.0
                                collector_config.capture_policy_trajectory = True
                                anchor_collect_result = self.real_collector.collect(
                                    group_count * 4,
                                    update_id=2_000_000 + current_update,
                                    collect_tokenized=False,
                                    device=self.real_collector.device,
                                    dtype=self.real_collector.dtype,
                                    forced_first_actions=forced_actions,
                                    **collect_kwargs,
                                )
                            finally:
                                collector_config.deterministic = old_deterministic
                                collector_config.exploration_epsilon = old_epsilon
                                collector_config.capture_policy_trajectory = old_capture
                            metrics.update({
                                f"critic_counterfactual_collect/{key}": float(value)
                                for key, value in anchor_collect_result.metrics.items()
                            })
                        else:
                            metrics["critic_counterfactual_collect/fixed_cache_reused"] = 1.0
                            anchor_collect_result = None
                    if anchor_collect_result is not None:
                        validation_episodes = [
                            episode for index, episode in enumerate(
                                anchor_collect_result.episodes
                            )
                            if ((index // 4) % 5 == 0)
                        ]
                        self._capture_actor_real_posterior_probe(
                            validation_episodes
                        )
                        training_episodes = [
                            episode for index, episode in enumerate(
                                anchor_collect_result.episodes
                            )
                            if ((index // 4) % 5 != 0)
                        ]
                        current_training_batch = collection_to_ppo_batch(
                            SimpleNamespace(episodes=training_episodes),
                            gamma=cfg.gamma,
                            device=next(self.policy.parameters()).device,
                            reward_scale=float(os.environ.get(
                                "REAL_RETURN_REWARD_SCALE", "0.1"
                            )),
                            first_transition_only=(
                                os.environ.get(
                                    "REAL_CRITIC_FIRST_TRANSITION_ONLY", "0"
                                ) == "1"
                            ),
                        )
                        validation_batch = collection_to_ppo_batch(
                            SimpleNamespace(episodes=validation_episodes),
                            gamma=cfg.gamma,
                            device=next(self.policy.parameters()).device,
                            reward_scale=float(os.environ.get(
                                "REAL_RETURN_REWARD_SCALE", "0.1"
                            )),
                            first_transition_only=(
                                os.environ.get(
                                    "REAL_CRITIC_FIRST_TRANSITION_ONLY", "0"
                                ) == "1"
                            ),
                        )
                        train_capacity = int(os.environ.get(
                            "REAL_CRITIC_REPLAY_CAPACITY", "2048"
                        ))
                        validation_capacity = int(os.environ.get(
                            "REAL_CRITIC_VALIDATION_SIZE", "256"
                        ))
                        self._real_critic_train_replay = self._append_ppo_batch(
                            self._real_critic_train_replay,
                            current_training_batch,
                            capacity=train_capacity,
                        )
                        self._real_critic_validation = self._append_ppo_batch(
                            self._real_critic_validation,
                            validation_batch,
                            capacity=validation_capacity,
                            keep_oldest=True,
                        )
                    # Sampling happens after the imagined stage so the entire
                    # same-generation replay, rather than only the newest 16
                    # episodes, remains available for repeated calibration.
                    self._pending_real_critic_batch = (
                        self._real_critic_train_replay
                    )
                    metrics.update({
                        "critic_real_anchor/cache_generation": float(
                            self._real_critic_cache_generation
                        ),
                        "critic_real_anchor/fixed_level_offset": float(
                            int(os.environ.get(
                                "REAL_CRITIC_FIXED_LEVEL_OFFSET", "0"
                            ))
                        ),
                        "critic_real_anchor/train_replay_size": float(
                            0 if self._real_critic_train_replay is None
                            else len(self._real_critic_train_replay.actions)
                        ),
                        "critic_real_anchor/fixed_validation_size": float(
                            0 if self._real_critic_validation is None
                            else len(self._real_critic_validation.actions)
                        ),
                    })
                actor_every = int(os.environ.get("REAL_ACTOR_PPO_EVERY", "0"))
                grounded_actor_ready = (
                    bool(getattr(self, "_real_critic_anchor_ready", False))
                    and bool(getattr(
                        self, "_reward_prior_calibration_feasible", False
                    ))
                )
                require_grounded_gate = (
                    os.environ.get("REQUIRE_GROUNDED_ACTOR_GATE", "0") == "1"
                )
                if (
                    actor_every > 0
                    and self._critic_warmup_complete
                    and current_update % actor_every == 0
                    and (not require_grounded_gate or grounded_actor_ready)
                ):
                    collector_config = self.real_collector.config
                    old_epsilon = collector_config.exploration_epsilon
                    old_capture = collector_config.capture_policy_trajectory
                    try:
                        collector_config.exploration_epsilon = 0.0
                        collector_config.capture_policy_trajectory = True
                        actor_result = self.real_collector.collect(
                            int(os.environ.get("REAL_ACTOR_PPO_EPISODES", "4")),
                            update_id=1_000_000 + current_update,
                            collect_tokenized=False,
                            device=self.real_collector.device,
                            dtype=self.real_collector.dtype,
                        )
                    finally:
                        collector_config.exploration_epsilon = old_epsilon
                        collector_config.capture_policy_trajectory = old_capture
                    from ..rl.real_return_anchor import collection_to_ppo_batch
                    self._pending_real_actor_batch = collection_to_ppo_batch(
                        actor_result, gamma=cfg.gamma,
                        device=next(self.policy.parameters()).device,
                        reward_scale=float(os.environ.get(
                            "REAL_RETURN_REWARD_SCALE", "0.1"
                        )),
                    )
                    metrics.update({
                        f"real_actor/{key}": float(value)
                        for key, value in actor_result.metrics.items()
                    })

            # --- 2. World model refresh (periodic) ---
            if self._should_refresh_wm(current_update):
                if os.environ.get("FORMAL_UNIFIED_PPO", "0") == "1":
                    self._last_wm_refresh_actor_update = (
                        self._actor_ppo_updates
                    )
                t_refresh = time.time()
                assert self.wm_refresher is not None
                assert self.wm_refresh_sample_fn is not None, (
                    "wm_refresh_sample_fn required when wm_refresher is set"
                )
                refresh_result = self.wm_refresher.refresh(self.wm_refresh_sample_fn)
                # RefreshMetrics dataclass — extract fields
                if hasattr(refresh_result, "avg_total_loss"):
                    metrics["wm_refresh/dynamics_loss"] = refresh_result.avg_dynamics_loss
                    metrics["wm_refresh/reward_loss"] = refresh_result.avg_reward_loss
                    metrics["wm_refresh/total_loss"] = refresh_result.avg_total_loss
                    metrics["wm_refresh/num_steps"] = float(refresh_result.num_steps)
                    metrics["wm_refresh/refresh_step"] = float(
                        getattr(refresh_result, "refresh_step", 0)
                    )
                    refresh_diagnostics = getattr(
                        refresh_result, "diagnostics", {}
                    )
                    metrics.update({
                        f"wm_refresh/{key}": float(value)
                        for key, value in refresh_diagnostics.items()
                    })
                elif isinstance(refresh_result, dict):
                    metrics.update(
                        {f"wm_refresh/{k}": v for k, v in refresh_result.items()}
                    )
                metrics["timing/wm_refresh_sec"] = time.time() - t_refresh
                # A guarded accepted refresh changes the latent coordinate
                # system.  Cached latent tensors from the previous generation
                # are no longer a valid train/held-out comparison.
                refresh_diagnostics = getattr(refresh_result, "diagnostics", {})
                if (
                    os.environ.get("REAL_RETURN_CRITIC_ANCHOR", "0") == "1"
                    and float(refresh_diagnostics.get("guard/accepted", 0.0)) == 1.0
                ):
                    self._invalidate_real_critic_latent_cache()
                    metrics["critic_real_anchor/cache_invalidated_after_wm"] = 1.0
                    metrics["critic_real_anchor/cache_generation"] = float(
                        self._real_critic_cache_generation
                    )

            # --- 3. Imagined rollout ---
            # Phase1Trainer switches the shared WM/Qwen container to train()
            # during an alternating refresh. Restore deterministic policy
            # forwards before collecting old-policy probabilities. Eval mode
            # leaves all actor/critic gradients enabled for PPO below.
            self._stabilize_policy_forward()
            t_rollout = time.time()
            trajectories = []
            multicontinuation_batches: list[PPOBatch] = []
            counterfactual_h2_batches: list[PPOBatch] = []
            counterfactual_h2_bucket_ids: list[Tensor] = []
            counterfactual_h2_metric_parts: list[dict[str, float]] = []
            continuation_variances: list[Tensor] = []
            continuation_action_disagreements: list[Tensor] = []
            exact_counterfactual_h2 = (
                os.environ.get("COUNTERFACTUAL_H2_PPO", "0") == "1"
            )
            if exact_counterfactual_h2:
                if cfg.rollout_horizon != 2:
                    raise RuntimeError(
                        "COUNTERFACTUAL_H2_PPO requires rollout_horizon=2"
                    )
                if cfg.use_value_bootstrap:
                    raise RuntimeError(
                        "COUNTERFACTUAL_H2_PPO requires finite-H2 zero bootstrap"
                    )
                if cfg.rollout_batch_size % 4 != 0:
                    raise RuntimeError(
                        "COUNTERFACTUAL_H2_PPO requires rollout_batch_size "
                        "divisible by four"
                    )
            critic_target_replicas = (
                int(os.environ.get("CRITIC_TARGET_CONTINUATIONS", "1"))
                if not self._critic_warmup_complete
                and not exact_counterfactual_h2 else 1
            )
            if critic_target_replicas <= 0:
                raise ValueError("CRITIC_TARGET_CONTINUATIONS must be positive")
            use_joint_target = (
                os.environ.get("GROUP_AWARE_JOINT_CRITIC", "0") == "1"
                and os.environ.get("JOINT_TARGET_CRITIC", "0") == "1"
            )
            if use_joint_target:
                # Actor parameters are disjoint. Swap only Critic-owned
                # tensors so rollout rewards/actions use the current policy,
                # while all Q baselines and the H2 leaf bootstrap come from a
                # slowly moving frozen target for this entire collection.
                self._swap_joint_target_critic()
            if getattr(cfg, "debug_rollout_progress", False):
                print(
                    f"[update {current_update}/{cfg.total_updates}] "
                    f"starting imagined rollout chunks={cfg.rollouts_per_update} "
                    f"batch={cfg.rollout_batch_size}",
                    flush=True,
                )
            rollout_collector = getattr(self.imagine_fn, "__self__", None)
            rollout_config = getattr(rollout_collector, "config", None)
            configured_bootstrap = (
                getattr(rollout_config, "bootstrap_with_value", None)
                if rollout_config is not None else None
            )
            stable_warmup_targets = bool(
                not self._critic_warmup_complete
                and configured_bootstrap
                and os.environ.get(
                    "CRITIC_WARMUP_ZERO_BOOTSTRAP", "1"
                ) == "1"
            )
            if stable_warmup_targets:
                rollout_config.bootstrap_with_value = False
            metrics["rollout/value_bootstrap_enabled"] = float(
                bool(configured_bootstrap) and not stable_warmup_targets
            )
            metrics["critic_warmup/stable_zero_bootstrap_targets"] = float(
                stable_warmup_targets
            )
            if (
                exact_counterfactual_h2
                and not self._critic_warmup_complete
                and self._critic_warmup_validation is None
                and hasattr(self, "counterfactual_h2_validation_slots")
            ):
                validation_slots_cpu = self.counterfactual_h2_validation_slots
                prototype = self.sample_beliefs_fn(1)
                prototype_slots = getattr(prototype, "slots", None)
                if prototype_slots is None:
                    raise TypeError(
                        "level-disjoint H2 validation requires BeliefState samples"
                    )
                validation_slots = validation_slots_cpu.to(
                    device=prototype_slots.device, dtype=prototype_slots.dtype
                )
                validation_starts = type(prototype)(
                    slots=validation_slots.repeat_interleave(16, dim=0)
                )
                validation_sequences = ordered_h2_action_sequences(
                    len(validation_slots), device=validation_slots.device
                )
                validation_trajectory = self.imagine_fn(
                    validation_starts,
                    forced_action_sequences=validation_sequences,
                )
                validation_batch, validation_metrics = (
                    counterfactual_h2_ppo_batch(
                        validation_trajectory, gamma=cfg.gamma
                    )
                )
                self._critic_warmup_validation = validation_batch
                validation_bucket_ids = (
                    self.counterfactual_h2_validation_bucket_ids.to(
                        validation_batch.actions.device
                    )
                )
                self._counterfactual_validation_action_bucket_ids = (
                    validation_bucket_ids.repeat_interleave(4)
                )
                metrics.update({
                    "counterfactual_validation/level_disjoint": 1.0,
                    "counterfactual_validation/states": float(
                        len(validation_slots)
                    ),
                    "counterfactual_validation/sequences": validation_metrics[
                        "counterfactual_h2/sequences"
                    ],
                })
            try:
                for chunk_idx in range(cfg.rollouts_per_update):
                    t_chunk = time.time()
                    counterfactual_warmup = (
                        not self._critic_warmup_complete
                        and not exact_counterfactual_h2
                        and os.environ.get(
                            "CRITIC_WARMUP_COUNTERFACTUAL_ACTIONS", "0"
                        ) == "1"
                    )
                    if exact_counterfactual_h2:
                        # Keep the number of unique starts comparable to the
                        # legacy four-action warm-up while enumerating all 16
                        # (a0,a1) sequences for each start.
                        base_beliefs = self.sample_beliefs_fn(
                            cfg.rollout_batch_size // 4
                        )
                        base_slots = getattr(base_beliefs, "slots", None)
                        if base_slots is None:
                            raise TypeError(
                                "counterfactual H2 PPO requires a belief "
                                "object with a slots tensor"
                            )
                        start_beliefs = type(base_beliefs)(
                            slots=base_slots.repeat_interleave(16, dim=0)
                        )
                        forced_sequences = ordered_h2_action_sequences(
                            len(base_slots), device=base_slots.device
                        )
                        primary = self.imagine_fn(
                            start_beliefs,
                            forced_action_sequences=forced_sequences,
                        )
                        exact_batch, exact_metrics = (
                            counterfactual_h2_ppo_batch(
                                primary, gamma=cfg.gamma
                            )
                        )
                        counterfactual_h2_batches.append(exact_batch)
                        sampled_bucket_ids = getattr(
                            self, "_last_sample_bucket_ids", None
                        )
                        if sampled_bucket_ids is not None:
                            action_bucket_ids = sampled_bucket_ids.to(
                                exact_batch.actions.device
                            ).repeat_interleave(4)
                            if len(action_bucket_ids) != len(exact_batch.actions):
                                raise RuntimeError(
                                    "exact-H2 bucket IDs lost four-action alignment"
                                )
                            counterfactual_h2_bucket_ids.append(action_bucket_ids)
                        counterfactual_h2_metric_parts.append(exact_metrics)
                    elif counterfactual_warmup:
                        if cfg.rollout_batch_size % 4 != 0:
                            raise ValueError(
                                "counterfactual Critic warm-up requires "
                                "rollout_batch_size divisible by four"
                            )
                        base_beliefs = self.sample_beliefs_fn(
                            cfg.rollout_batch_size // 4
                        )
                        base_slots = getattr(base_beliefs, "slots", None)
                        if base_slots is None:
                            raise TypeError(
                                "counterfactual Critic warm-up requires a "
                                "belief object with a slots tensor"
                            )
                        start_beliefs = type(base_beliefs)(
                            slots=base_slots.repeat_interleave(4, dim=0)
                        )
                        forced_first_actions = torch.arange(
                            4, device=base_slots.device, dtype=torch.long
                        ).repeat(len(base_slots))
                        primary = self.imagine_fn(
                            start_beliefs,
                            forced_first_actions=forced_first_actions,
                        )
                    else:
                        start_beliefs = self.sample_beliefs_fn(
                            cfg.rollout_batch_size
                        )
                        primary = self.imagine_fn(start_beliefs)
                    trajectories.append(primary)
                    if critic_target_replicas > 1:
                        replicas = [primary]
                        for _ in range(1, critic_target_replicas):
                            # The dynamics is functional: every call starts
                            # from the same posterior belief and conditions on
                            # the primary a0. Only later policy actions are
                            # resampled, making the mean a valid Q(s0,a0)
                            # target rather than an average across actions.
                            replicas.append(self.imagine_fn(
                                start_beliefs,
                                forced_first_actions=primary.actions[:, 0],
                            ))
                        replica_returns = []
                        for replica in replicas[1:]:
                            if not torch.equal(
                                primary.states[:, 0], replica.states[:, 0]
                            ):
                                raise RuntimeError(
                                    "Multi-continuation Critic target requires "
                                    "bit-identical initial latent states"
                                )
                        first_actions = torch.stack([
                            replica.actions[:, 0] for replica in replicas
                        ])
                        continuation_action_disagreements.append(
                            (first_actions != first_actions[:1]).any(dim=0).float()
                        )
                        discounts = torch.pow(
                            torch.as_tensor(
                                cfg.gamma,
                                device=primary.rewards.device,
                                dtype=primary.rewards.dtype,
                            ),
                            torch.arange(
                                primary.horizon,
                                device=primary.rewards.device,
                                dtype=primary.rewards.dtype,
                            ),
                        )
                        for replica in replicas:
                            mask = (
                                replica.mask
                                if replica.mask is not None
                                else torch.ones_like(replica.rewards)
                            )
                            replica_returns.append(
                                (replica.rewards * mask * discounts).sum(dim=1)
                            )
                        returns_by_replica = torch.stack(replica_returns, dim=0)
                        mean_return = returns_by_replica.mean(dim=0)
                        continuation_variances.append(
                            returns_by_replica.var(dim=0, unbiased=False)
                        )
                        initial_value = primary.values[:, 0]
                        multicontinuation_batches.append(PPOBatch(
                            states=primary.states[:, 0],
                            actions=primary.actions[:, 0],
                            advantages=mean_return - initial_value,
                            returns=mean_return,
                            old_log_probs=primary.log_probs[:, 0],
                            old_values=initial_value,
                        ))
                    if getattr(cfg, "debug_rollout_progress", False):
                        print(
                            f"[update {current_update}/{cfg.total_updates}] "
                            f"rollout chunk {chunk_idx + 1}/{cfg.rollouts_per_update} "
                            f"done in {time.time() - t_chunk:.1f}s",
                            flush=True,
                        )
            finally:
                if stable_warmup_targets:
                    rollout_config.bootstrap_with_value = configured_bootstrap
                if use_joint_target:
                    self._swap_joint_target_critic()
            trajectory = self._concat_trajectories(trajectories)
            metrics["timing/rollout_sec"] = time.time() - t_rollout
            metrics["rollout/chunks_per_update"] = float(
                cfg.rollouts_per_update
            )
            metrics["critic_target/continuations"] = float(
                4 if exact_counterfactual_h2 else critic_target_replicas
            )
            metrics["counterfactual_h2/enabled"] = float(
                exact_counterfactual_h2
            )
            if counterfactual_h2_metric_parts:
                for name in (
                    "counterfactual_h2/base_states",
                    "counterfactual_h2/sequences",
                ):
                    metrics[name] = sum(part[name] for part in counterfactual_h2_metric_parts)
                metrics["counterfactual_h2/action_coverage"] = min(
                    part["counterfactual_h2/action_coverage"]
                    for part in counterfactual_h2_metric_parts
                )
                metrics[
                    "counterfactual_h2/weighted_advantage_zero_error"
                ] = max(
                    part["counterfactual_h2/weighted_advantage_zero_error"]
                    for part in counterfactual_h2_metric_parts
                )
                for name in (
                    "counterfactual_h2/raw_advantage_abs_mean",
                    "counterfactual_h2/q_margin_mean",
                ):
                    metrics[name] = sum(
                        part[name] for part in counterfactual_h2_metric_parts
                    ) / len(counterfactual_h2_metric_parts)
            if continuation_variances:
                all_continuation_variances = torch.cat(continuation_variances)
                metrics["critic_target/continuation_variance_mean"] = float(
                    all_continuation_variances.float().mean()
                )
                metrics["critic_target/mean_target_noise_variance"] = float(
                    all_continuation_variances.float().mean()
                    / critic_target_replicas
                )
                metrics["critic_target/first_action_disagreement_fraction"] = float(
                    torch.cat(continuation_action_disagreements).mean()
                )
            metrics["critic_joint_rehearsal/target_critic"] = float(
                use_joint_target
            )

            # Trajectory diagnostics
            metrics.update(
                self._trajectory_metrics(
                    trajectory,
                    success_threshold=cfg.reward_success_threshold,
                )
            )

            # --- 4. PPO update ---
            self._stabilize_policy_forward()
            t_ppo = time.time()
            critic_only = not self._critic_warmup_complete
            if critic_only:
                self._critic_warmup_updates += 1
            metrics["critic_warmup/updates"] = float(
                self._critic_warmup_updates
            )
            require_grounded_gate = (
                os.environ.get("REQUIRE_GROUNDED_ACTOR_GATE", "0") == "1"
            )
            reward_prior_ready = bool(getattr(
                self, "_reward_prior_calibration_feasible", False
            ))
            real_critic_ready = bool(getattr(
                self, "_real_critic_anchor_ready", False
            ))
            grounded_actor_ready = reward_prior_ready and real_critic_ready
            actor_update_allowed = (
                not critic_only
                and not critic_diagnostic_only
                and (not require_grounded_gate or grounded_actor_ready)
            )
            critic_pretrain_stage = (
                os.environ.get("CRITIC_PRETRAIN_ONLY", "0") == "1"
            )
            if critic_pretrain_stage:
                # This run produces the release checkpoint; Actor PPO belongs
                # to a separate run whose counter starts at zero.
                actor_update_allowed = False
            metrics.update({
                "actor_grounding_gate/required": float(require_grounded_gate),
                "actor_grounding_gate/reward_prior_ready": float(reward_prior_ready),
                "actor_grounding_gate/real_critic_ready": float(real_critic_ready),
                "actor_grounding_gate/actor_update_allowed": float(
                    actor_update_allowed
                ),
            })
            gate_just_passed = False
            critic_candidate_just_qualified = False
            joint_critic_grounding = (
                os.environ.get("REQUIRE_JOINT_CRITIC_GROUNDING", "0") == "1"
            )
            if exact_counterfactual_h2:
                if not counterfactual_h2_batches:
                    raise RuntimeError(
                        "counterfactual H2 PPO produced no exact batches"
                    )
                # This is the crucial Q-to-Actor bridge: returns are exact
                # action-conditioned H2 Q targets, while advantages are
                # policy-centered counterfactual advantages. No Q value is
                # passed through scalar-V GAE.
                ppo_batch = PPOBatch(**{
                    name: torch.cat([
                        getattr(batch, name)
                        for batch in counterfactual_h2_batches
                    ])
                    for name in PPOBatch.__dataclass_fields__
                })
                ppo_bucket_ids = (
                    torch.cat(counterfactual_h2_bucket_ids)
                    if counterfactual_h2_bucket_ids else None
                )
            else:
                ppo_bucket_ids = None
                ppo_batch = trajectory_to_ppo_batch(
                    trajectory,
                    gamma=cfg.gamma,
                    # Joint grounding trains on the same finite-horizon MC target
                    # used by the fresh-rollout EV gate.  This avoids optimizing a
                    # lambda-return while gating on a different target.
                    gae_lambda=(
                        1.0 if critic_only or joint_critic_grounding
                        else cfg.gae_lambda
                    ),
                )
            if critic_only and multicontinuation_batches and not exact_counterfactual_h2:
                # Only s0 is shared across replicas. Later states have already
                # diverged and must never have their realised returns averaged.
                ppo_batch = PPOBatch(**{
                    name: torch.cat([
                        getattr(batch, name)
                        for batch in multicontinuation_batches
                    ])
                    for name in PPOBatch.__dataclass_fields__
                })
                metrics["critic_target/unique_initial_states"] = float(
                    len(ppo_batch.actions)
                )
            # Immutable source for fresh imagined Critic rehearsal. Actor
            # grounding below is intentionally allowed to replace/filter
            # ``ppo_batch`` but must never change this complete rollout batch.
            fresh_critic_batch = ppo_batch
            if os.environ.get("REQUIRE_ROLLOUT_CRITIC_EV_GATE", "0") == "1":
                # This gate uses fresh, never-trained-on imagined rollouts and
                # finite-horizon Monte Carlo targets. It detects a Critic that
                # fits fixed real starts but fails on the Actor's actual batch.
                rollout_gate_batch = trajectory_to_ppo_batch(
                    trajectory, gamma=cfg.gamma, gae_lambda=1.0
                )
                with torch.no_grad():
                    rollout_prediction = self.policy.evaluate_values(
                        rollout_gate_batch.states, rollout_gate_batch.actions
                    ).float()
                    rollout_target = rollout_gate_batch.returns.float()
                    rollout_variance = rollout_target.var(unbiased=False)
                    rollout_ev = 0.0 if float(rollout_variance) < 1e-8 else float(
                        1.0 - (rollout_target - rollout_prediction).var(unbiased=False)
                        / rollout_variance
                    )
                gate_statistic = rollout_ev
                fixed_gate_metrics: dict[str, float] = {}
                if os.environ.get(
                    "FIXED_ROLLOUT_CRITIC_GATE", "0"
                ) == "1":
                    self._initialize_fixed_rollout_critic_panels()
                    fixed_gate_metrics = self._fixed_rollout_critic_metrics()
                    gate_statistic = fixed_gate_metrics.pop(
                        "_fixed_rollout_gate_statistic"
                    )
                rollout_gate_metrics = self._update_rollout_critic_gate(
                    gate_statistic
                )
                if fixed_gate_metrics:
                    rollout_gate_metrics.update(fixed_gate_metrics)
                    rollout_gate_metrics.update({
                        "critic_rollout_gate/random_monitor_ev": float(
                            rollout_ev
                        ),
                        # Preserve the historical key as the fixed-panel mean;
                        # pass/fail is based on the separately logged LCB.
                        "critic_rollout_gate/heldout_mc_ev": float(
                            fixed_gate_metrics[
                                "critic_rollout_gate/fixed_panel_ev_mean"
                            ]
                        ),
                        "critic_rollout_gate/gate_statistic_lcb": float(
                            gate_statistic
                        ),
                    })
                just_released = bool(rollout_gate_metrics.pop(
                    "_critic_rollout_gate_just_released", 0.0
                ))
                rollback_required = bool(rollout_gate_metrics.pop(
                    "_critic_rollout_gate_rollback_required", 0.0
                ))
                metrics.update(rollout_gate_metrics)
                if just_released:
                    self._snapshot_grounded_training_safety()
                    metrics[
                        "critic_rollout_gate/safety_snapshot_created"
                    ] = 1.0
                if rollback_required:
                    self._restore_grounded_training_safety()
                    metrics.update({
                        "critic_rollout_gate/safety_rollback": 1.0,
                        "ppo/actor_update": float(self._actor_ppo_updates),
                    })
                    if checkpoint_root is not None:
                        self._save_checkpoint(
                            checkpoint_root / "safety_rollback.pt",
                            last_completed_update,
                        )
                    if self.logger is not None:
                        self.logger.log_scalars(current_update, metrics)
                    raise RuntimeError(
                        "fresh-rollout Critic failed "
                        f"{self._rollout_critic_post_release_failures} "
                        "consecutive post-release checks; restored the "
                        "Actor/Critic/optimizer safety snapshot"
                    )
                grounded_actor_ready = (
                    reward_prior_ready and real_critic_ready
                    and self._rollout_critic_ready
                )
                actor_update_allowed = (
                    not critic_only and not critic_diagnostic_only
                    and (not require_grounded_gate or grounded_actor_ready)
                )
                metrics["actor_grounding_gate/rollout_critic_ready"] = float(
                    self._rollout_critic_ready
                )
                metrics["actor_grounding_gate/actor_update_allowed"] = float(
                    actor_update_allowed
                )
            # Preserve the action-conditioned advantage evidence used by the
            # Actor update. A shared action bias can make deterministic argmax
            # collapse even when policy entropy remains nearly maximal.
            with torch.no_grad():
                action_ids = ppo_batch.actions.long()
                action_counts = torch.bincount(action_ids, minlength=4)
                action_adv_sum = torch.zeros(4, device=ppo_batch.advantages.device)
                action_adv_abs = torch.zeros_like(action_adv_sum)
                action_adv_sum.scatter_add_(0, action_ids, ppo_batch.advantages.float())
                action_adv_abs.scatter_add_(0, action_ids, ppo_batch.advantages.float().abs())
                action_den = action_counts.float().clamp_min(1.0)
                action_adv_mean = action_adv_sum / action_den
                metrics.update({
                    "ppo/action_0_count": float(action_counts[0]),
                    "ppo/action_1_count": float(action_counts[1]),
                    "ppo/action_2_count": float(action_counts[2]),
                    "ppo/action_3_count": float(action_counts[3]),
                    "ppo/action_advantage_mean_spread": float(action_adv_mean.max() - action_adv_mean.min()),
                    "ppo/action_advantage_mean_0": float(action_adv_mean[0]),
                    "ppo/action_advantage_mean_1": float(action_adv_mean[1]),
                    "ppo/action_advantage_mean_2": float(action_adv_mean[2]),
                    "ppo/action_advantage_mean_3": float(action_adv_mean[3]),
                    "ppo/action_advantage_abs_sum_0": float(action_adv_abs[0]),
                    "ppo/action_advantage_abs_sum_1": float(action_adv_abs[1]),
                    "ppo/action_advantage_abs_sum_2": float(action_adv_abs[2]),
                    "ppo/action_advantage_abs_sum_3": float(action_adv_abs[3]),
                })
            agreement_blocked = False
            if (
                actor_update_allowed
                and os.environ.get("GROUNDED_ADVANTAGE_AGREEMENT", "0") == "1"
            ):
                # The real-return calibrated Critic acts only as a direction
                # verifier. It supplies no expert action label: both inputs
                # and Q values remain functions of latent belief and action.
                with torch.no_grad():
                    grounded_q = torch.stack([
                        self.policy.evaluate_values(
                            ppo_batch.states,
                            torch.full_like(ppo_batch.actions, action_id),
                        ) for action_id in range(4)
                    ], dim=-1).float()
                    selected_q = grounded_q.gather(
                        1, ppo_batch.actions.long()[:, None]
                    ).squeeze(1)
                    grounded_advantage = selected_q - grounded_q.mean(dim=1)
                    imagined_advantage = ppo_batch.advantages.float()
                    margin = float(os.environ.get(
                        "GROUNDED_ADVANTAGE_MIN_MARGIN", "0.01"
                    ))
                    informative = grounded_advantage.abs() >= margin
                    same_direction = (
                        imagined_advantage.sign() == grounded_advantage.sign()
                    )
                    keep = informative & same_direction
                    conflict = informative & ~same_direction
                total_count = len(keep)
                kept_count = int(keep.sum())
                metrics.update({
                    "grounded_agreement/kept_fraction": float(
                        keep.float().mean()
                    ),
                    "grounded_agreement/conflict_fraction": float(
                        conflict.float().mean()
                    ),
                    "grounded_agreement/uninformative_fraction": float(
                        (~informative).float().mean()
                    ),
                    "grounded_agreement/effective_samples": float(kept_count),
                    "grounded_agreement/original_samples": float(total_count),
                })
                minimum_samples = int(os.environ.get(
                    "GROUNDED_ADVANTAGE_MIN_SAMPLES", "32"
                ))
                minimum_fraction = float(os.environ.get(
                    "GROUNDED_ADVANTAGE_MIN_FRACTION", "0.20"
                ))
                agreement_blocked = (
                    kept_count < minimum_samples
                    or kept_count < minimum_fraction * total_count
                )
                if agreement_blocked:
                    self._grounded_agreement_blocked_streak += 1
                else:
                    self._grounded_agreement_blocked_streak = 0
                metrics["grounded_agreement/update_blocked"] = float(
                    agreement_blocked
                )
                metrics["grounded_agreement/consecutive_blocked_updates"] = float(
                    self._grounded_agreement_blocked_streak
                )
                if not agreement_blocked:
                    ppo_batch = PPOBatch(**{
                        name: getattr(ppo_batch, name)[keep]
                        for name in PPOBatch.__dataclass_fields__
                    })
                else:
                    actor_update_allowed = False
                    metrics["actor_grounding_gate/actor_update_allowed"] = 0.0
                    fallback_after = int(os.environ.get(
                        "REAL_ACTOR_FALLBACK_AFTER_BLOCKS", "0"
                    ))
                    if (
                        fallback_after > 0
                        and self._grounded_agreement_blocked_streak
                        >= fallback_after
                    ):
                        # Collect a fresh trajectory from the current Actor.
                        # No expert action or explicit state is supplied; the
                        # environment contributes only actual rewards and the
                        # observations are encoded back into latent beliefs.
                        if self.real_collector is None:
                            raise RuntimeError(
                                "real Actor fallback requires RealCollector"
                            )
                        collector_config = self.real_collector.config
                        old_deterministic = collector_config.deterministic
                        old_epsilon = collector_config.exploration_epsilon
                        old_capture = collector_config.capture_policy_trajectory
                        try:
                            collector_config.deterministic = False
                            collector_config.exploration_epsilon = 0.0
                            collector_config.capture_policy_trajectory = True
                            fallback_result = self.real_collector.collect(
                                int(os.environ.get(
                                    "REAL_ACTOR_FALLBACK_EPISODES", "8"
                                )),
                                update_id=3_000_000 + current_update,
                                collect_tokenized=False,
                                device=self.real_collector.device,
                                dtype=self.real_collector.dtype,
                            )
                        finally:
                            collector_config.deterministic = old_deterministic
                            collector_config.exploration_epsilon = old_epsilon
                            collector_config.capture_policy_trajectory = old_capture
                        from ..rl.real_return_anchor import collection_to_ppo_batch
                        fallback_batch = collection_to_ppo_batch(
                            fallback_result,
                            gamma=cfg.gamma,
                            device=next(self.policy.parameters()).device,
                            reward_scale=float(os.environ.get(
                                "REAL_RETURN_REWARD_SCALE", "0.1"
                            )),
                        )
                        if fallback_batch is not None and len(
                            fallback_batch.actions
                        ) >= int(os.environ.get(
                            "REAL_ACTOR_FALLBACK_MIN_TRANSITIONS", "32"
                        )):
                            ppo_batch = fallback_batch
                            actor_update_allowed = True
                            agreement_blocked = False
                            self._grounded_agreement_blocked_streak = 0
                            metrics.update({
                                "grounded_agreement/fallback_active": 1.0,
                                "grounded_agreement/fallback_episodes": float(
                                    len(fallback_result.episodes)
                                ),
                                "grounded_agreement/fallback_samples": float(
                                    len(fallback_batch.actions)
                                ),
                                "grounded_agreement/consecutive_blocked_updates": 0.0,
                                "actor_grounding_gate/actor_update_allowed": 1.0,
                            })
                            metrics.update({
                                f"real_actor_fallback/{key}": float(value)
                                for key, value in fallback_result.metrics.items()
                            })
                        else:
                            metrics["grounded_agreement/fallback_active"] = 0.0
            min_margin = float(os.environ.get(
                "IMAGINED_ADVANTAGE_MIN_Q_MARGIN", "0"
            ))
            if actor_update_allowed and min_margin > 0.0:
                with torch.no_grad():
                    q_values = torch.stack([
                        self.policy.evaluate_values(
                            ppo_batch.states,
                            torch.full_like(ppo_batch.actions, action_id),
                        ) for action_id in range(4)
                    ], dim=-1).float()
                    top2 = q_values.topk(2, dim=-1).values
                    confident = (top2[:, 0] - top2[:, 1]) >= min_margin
                metrics["ppo/confident_advantage_fraction"] = float(
                    confident.float().mean()
                )
                if int(confident.sum()) >= 32:
                    ppo_batch = PPOBatch(**{
                        name: getattr(ppo_batch, name)[confident]
                        for name in PPOBatch.__dataclass_fields__
                    })
            if (
                actor_update_allowed
                and os.environ.get("IMAGINED_ACTION_BALANCE", "0") == "1"
            ):
                counts = torch.bincount(
                    ppo_batch.actions.long(), minlength=4
                ).float().clamp_min(1.0)
                inverse = (counts.sum() / (4.0 * counts)).clamp(0.5, 2.0)
                ppo_batch = PPOBatch(**{
                    **{
                        name: getattr(ppo_batch, name)
                        for name in PPOBatch.__dataclass_fields__
                    },
                    "advantages": ppo_batch.advantages * inverse[
                        ppo_batch.actions.long()
                    ].to(ppo_batch.advantages.dtype),
                })
                metrics["ppo/action_balance_max_weight"] = float(inverse.max())
            real_actor_batch = getattr(self, "_pending_real_actor_batch", None)
            if actor_update_allowed and real_actor_batch is not None:
                from ..rl.real_return_anchor import normalize_advantages
                ratio = float(os.environ.get("REAL_ACTOR_PPO_RATIO", "0.10"))
                if not 0.0 < ratio < 0.5:
                    raise ValueError("REAL_ACTOR_PPO_RATIO must be in (0, 0.5)")
                real_count = min(
                    len(real_actor_batch.actions),
                    max(1, round(len(ppo_batch.actions) * ratio / (1.0 - ratio))),
                )
                generator = torch.Generator(
                    device=real_actor_batch.actions.device
                ).manual_seed(20260831 + current_update)
                index = torch.randperm(
                    len(real_actor_batch.actions), generator=generator,
                    device=real_actor_batch.actions.device,
                )[:real_count]
                real_actor_batch = PPOBatch(**{
                    name: getattr(real_actor_batch, name)[index]
                    for name in PPOBatch.__dataclass_fields__
                })
                ppo_batch = normalize_advantages(ppo_batch)
                real_actor_batch = normalize_advantages(real_actor_batch)
                ppo_batch = PPOBatch(**{
                    name: torch.cat([
                        getattr(ppo_batch, name), getattr(real_actor_batch, name)
                    ]) for name in PPOBatch.__dataclass_fields__
                })
                metrics["real_actor/ppo_samples"] = float(real_count)
                metrics["real_actor/ppo_fraction"] = float(
                    real_count / len(ppo_batch.actions)
                )
                self._pending_real_actor_batch = None

            if p0_critic_only:
                actor_update_allowed = False
                metrics["actor_grounding_gate/actor_update_allowed"] = 0.0
                metrics["p0_critic_validation/actor_frozen"] = 1.0

            metrics["ppo/batch_size"] = float(ppo_batch.states.shape[0])
            behavior_batch = None
            if actor_update_allowed and cfg.ppo.behavior_bc_coef > 0.0:
                if self.behavior_sample_fn is None:
                    raise RuntimeError(
                        "expert rehearsal requires a posterior behavior sampler"
                    )
                behavior_batch = self.behavior_sample_fn(
                    cfg.ppo.behavior_bc_batch_size
                )
            critic_validation_batch = None
            update_batch = ppo_batch
            if critic_only:
                fraction = cfg.critic_warmup_validation_fraction
                if not 0.0 < fraction < 0.5:
                    raise ValueError(
                        "critic_warmup_validation_fraction must be in (0, 0.5)"
                    )
                def subset(batch: PPOBatch, index: slice) -> PPOBatch:
                    return PPOBatch(**{
                        name: getattr(batch, name)[index]
                        for name in PPOBatch.__dataclass_fields__
                    })

                count = ppo_batch.actions.shape[0]
                current_validation = (
                    0 if self._critic_warmup_validation is None
                    else self._critic_warmup_validation.actions.shape[0]
                )
                needed = max(0, cfg.critic_warmup_validation_size - current_validation)
                validation_count = min(
                    needed, max(1, int(count * fraction)) if needed else 0
                )
                if (
                    validation_count
                    and os.environ.get(
                        "CRITIC_WARMUP_COUNTERFACTUAL_ACTIONS", "0"
                    ) == "1"
                ):
                    if needed % 4 != 0:
                        raise RuntimeError(
                            "counterfactual Critic validation size must be "
                            "divisible by four"
                        )
                    validation_count = min(
                        needed,
                        max(4, (validation_count // 4) * 4),
                    )
                if validation_count:
                    self._critic_warmup_validation = self._concatenate_ppo_batches(
                        self._critic_warmup_validation,
                        subset(ppo_batch, slice(0, validation_count)),
                    )
                train_part = subset(ppo_batch, slice(validation_count, count))
                train_bucket_ids = (
                    None if ppo_bucket_ids is None
                    else ppo_bucket_ids[validation_count:count]
                )
                self._critic_warmup_replay = self._concatenate_ppo_batches(
                    self._critic_warmup_replay, train_part
                )
                if train_bucket_ids is not None:
                    self._critic_warmup_replay_bucket_ids = (
                        train_bucket_ids
                        if self._critic_warmup_replay_bucket_ids is None
                        else torch.cat([
                            self._critic_warmup_replay_bucket_ids,
                            train_bucket_ids,
                        ])
                    )
                replay_count = self._critic_warmup_replay.actions.shape[0]
                if replay_count > cfg.critic_warmup_replay_capacity:
                    self._critic_warmup_replay = subset(
                        self._critic_warmup_replay,
                        slice(replay_count - cfg.critic_warmup_replay_capacity, replay_count),
                    )
                    if self._critic_warmup_replay_bucket_ids is not None:
                        self._critic_warmup_replay_bucket_ids = (
                            self._critic_warmup_replay_bucket_ids[-cfg.critic_warmup_replay_capacity:]
                        )
                    replay_count = cfg.critic_warmup_replay_capacity
                sample_count = min(cfg.critic_warmup_train_samples, replay_count)
                generator = torch.Generator(
                    device=self._critic_warmup_replay.actions.device
                ).manual_seed(20260820 + current_update)
                replay_returns = self._critic_warmup_replay.returns.float()
                # Value regression must preserve the on-policy outcome prior.
                # Sampling 50/50 by the sign of the realised return changes
                # E[G|s] and therefore trains a systematically biased V_pi.
                if exact_counterfactual_h2:
                    if replay_count % 4 != 0 or sample_count % 4 != 0:
                        raise RuntimeError(
                            "exact H2 Critic replay must contain complete "
                            "four-action groups"
                        )
                    group_count = replay_count // 4
                    selected_groups = torch.randperm(
                        group_count,
                        generator=generator,
                        device=self._critic_warmup_replay.actions.device,
                    )[:sample_count // 4]
                    indices = (
                        selected_groups[:, None] * 4
                        + torch.arange(
                            4,
                            device=self._critic_warmup_replay.actions.device,
                        )[None, :]
                    ).reshape(-1)
                else:
                    indices = torch.randperm(
                        replay_count,
                        generator=generator,
                        device=self._critic_warmup_replay.actions.device,
                    )[:sample_count]
                update_batch = PPOBatch(**{
                    name: getattr(self._critic_warmup_replay, name)[indices]
                    for name in PPOBatch.__dataclass_fields__
                })
                update_bucket_ids = (
                    None if self._critic_warmup_replay_bucket_ids is None
                    else self._critic_warmup_replay_bucket_ids[indices]
                )
                critic_validation_batch = self._critic_warmup_validation
                metrics["critic_warmup/replay_size"] = float(replay_count)
                metrics["critic_warmup/validation_size"] = float(
                    critic_validation_batch.actions.shape[0]
                )
                metrics["critic_warmup/replay_positive_fraction"] = float(
                    (replay_returns > 0.0).float().mean()
                )
                metrics["critic_warmup/train_positive_fraction"] = float(
                    (update_batch.returns.float() > 0.0).float().mean()
                )
            # Actor gating must not erase fresh imagined Critic rehearsal.
            imagined_critic_enabled = (
                os.environ.get("IMAGINED_CRITIC_UPDATE", "1") == "1"
            )
            block_imagined_update = not (
                actor_update_allowed or imagined_critic_enabled
            )
            main_update_batch = update_batch
            if block_imagined_update:
                main_update_batch = PPOBatch(**{
                    name: getattr(update_batch, name)[:0]
                    for name in PPOBatch.__dataclass_fields__
                })
            fixed_replay_fraction = float(os.environ.get(
                "COUNTERFACTUAL_FIXED_REPLAY_FRACTION", "0"
            ))
            fixed_replay_samples = 0
            if fixed_replay_fraction < 0.0 or fixed_replay_fraction >= 1.0:
                raise ValueError(
                    "COUNTERFACTUAL_FIXED_REPLAY_FRACTION must be in [0,1)"
                )
            if (
                actor_update_allowed
                and exact_counterfactual_h2
                and fixed_replay_fraction > 0.0
                and self._critic_warmup_replay is not None
            ):
                fresh_groups = len(main_update_batch.actions) // 4
                available_groups = len(self._critic_warmup_replay.actions) // 4
                wanted_groups = max(
                    1,
                    round(
                        fresh_groups * fixed_replay_fraction
                        / (1.0 - fixed_replay_fraction)
                    ),
                )
                chosen_groups = min(available_groups, wanted_groups)
                generator = torch.Generator(
                    device=self._critic_warmup_replay.actions.device
                ).manual_seed(20261001 + current_update)
                group_indices = torch.randperm(
                    available_groups,
                    generator=generator,
                    device=self._critic_warmup_replay.actions.device,
                )[:chosen_groups]
                indices = (
                    group_indices[:, None] * 4
                    + torch.arange(
                        4, device=group_indices.device
                    )[None, :]
                ).reshape(-1)
                fixed_batch = PPOBatch(**{
                    name: getattr(self._critic_warmup_replay, name)[indices]
                    for name in PPOBatch.__dataclass_fields__
                })
                fixed_batch = self._recenter_counterfactual_replay(fixed_batch)
                main_update_batch = self._concatenate_ppo_batches(
                    main_update_batch, fixed_batch
                )
                fixed_replay_samples = len(fixed_batch.actions)
            metrics["actor_grounding_gate/imagined_update_blocked"] = float(
                block_imagined_update
            )
            metrics.update({
                "critic_joint_rehearsal/enabled": float(
                    joint_critic_grounding
                ),
                "critic_joint_rehearsal/imagined_critic_enabled": float(
                    imagined_critic_enabled
                ),
                "critic_joint_rehearsal/fresh_mc_samples": float(
                    len(fresh_critic_batch.actions)
                    if imagined_critic_enabled else 0
                ),
                "critic_joint_rehearsal/actor_batch_samples": float(
                    len(update_batch.actions) if actor_update_allowed else 0
                ),
                "counterfactual_fixed_replay/fraction_requested": float(
                    fixed_replay_fraction
                ),
                "counterfactual_fixed_replay/samples": float(
                    fixed_replay_samples
                ),
                "counterfactual_fixed_replay/fraction_actual": float(
                    fixed_replay_samples / max(1, len(main_update_batch.actions))
                ),
            })
            group_joint_update = (
                joint_critic_grounding
                and os.environ.get("GROUP_AWARE_JOINT_CRITIC", "0") == "1"
            )
            actor_transaction_snapshot = None
            actor_validation_before_score = None
            actor_validation_before_gate = None
            if (
                actor_update_allowed
                and os.environ.get("TRANSACTIONAL_ACTOR_GATE", "0") == "1"
            ):
                actor_transaction_snapshot = (
                    self._snapshot_actor_update_transaction()
                )
                if (
                    exact_counterfactual_h2
                    and os.environ.get(
                        "COUNTERFACTUAL_ACTOR_TRANSACTION_GATE", "0"
                    ) == "1"
                ):
                    _, actor_validation_before_gate, actor_validation_before_score = (
                        self._counterfactual_actor_validation_metrics()
                    )
            joint_result = None
            real_critic_batch = getattr(self, "_pending_real_critic_batch", None)
            if group_joint_update:
                if not cfg.use_value_bootstrap:
                    raise RuntimeError(
                        "GROUP_AWARE_JOINT_CRITIC requires --value-bootstrap; "
                        "zero-bootstrap H2 and full real returns are different "
                        "value functions"
                    )
                if not imagined_critic_enabled:
                    raise RuntimeError(
                        "GROUP_AWARE_JOINT_CRITIC requires IMAGINED_CRITIC_UPDATE=1"
                    )
                if real_critic_batch is None:
                    raise RuntimeError(
                        "group-aware joint Critic requires fixed-real replay "
                        "before its first update"
                    )
                h2_capacity = int(os.environ.get(
                    "JOINT_H2_REPLAY_CAPACITY", "4096"
                ))
                self._append_fresh_h2_replay(
                    fresh_critic_batch,
                    capacity=h2_capacity,
                )
                assert self._joint_h2_replay is not None
                metrics.update({
                    "critic_joint_rehearsal/group_aware": 1.0,
                    "critic_joint_rehearsal/single_optimizer_path": 1.0,
                    "critic_joint_rehearsal/value_bootstrap": 1.0,
                    "critic_joint_rehearsal/h2_replay_size": float(
                        len(self._joint_h2_replay.actions)
                    ),
                })
                # Actor PPO remains on the current on-policy batch. Critic is
                # explicitly disabled here and receives exactly one separate
                # *joint* optimization path below.
                actor_batch = main_update_batch
                if not actor_update_allowed:
                    actor_batch = PPOBatch(**{
                        name: getattr(main_update_batch, name)[:0]
                        for name in PPOBatch.__dataclass_fields__
                    })
                ppo_metrics = self.ppo_updater.update(
                    actor_batch,
                    behavior_batch=(
                        behavior_batch if actor_update_allowed else None
                    ),
                    critic_only=False,
                    actor_enabled=actor_update_allowed,
                    critic_enabled=False,
                )
                joint_result = self.ppo_updater.update_joint_critic(
                    self._joint_h2_replay,
                    real_critic_batch,
                    train_samples=int(os.environ.get(
                        "JOINT_CRITIC_TRAIN_SAMPLES", "512"
                    )),
                    real_fraction=float(os.environ.get(
                        "JOINT_CRITIC_REAL_FRACTION", "0.25"
                    )),
                    ranking_coef=float(os.environ.get(
                        "JOINT_CRITIC_RANKING_COEF", "0.05"
                    )),
                    ranking_temperature=float(os.environ.get(
                        "JOINT_CRITIC_RANKING_TEMPERATURE", "0.05"
                    )),
                    project_conflicting_gradients=(
                        os.environ.get("JOINT_CRITIC_PCGRAD", "1") == "1"
                    ),
                )
                if use_joint_target:
                    target_lag = self._update_joint_target_critic()
                    metrics.update({
                        "critic_joint_rehearsal/target_tau": float(
                            os.environ.get("JOINT_TARGET_CRITIC_TAU", "0.01")
                        ),
                        "critic_joint_rehearsal/target_online_rms_lag": (
                            target_lag
                        ),
                    })
            else:
                ppo_metrics = self.ppo_updater.update(
                    main_update_batch,
                    behavior_batch=(
                        behavior_batch if actor_update_allowed else None
                    ),
                    critic_only=critic_only,
                    actor_enabled=actor_update_allowed,
                    critic_enabled=imagined_critic_enabled,
                    critic_bucket_ids=(
                        update_bucket_ids
                        if critic_only and exact_counterfactual_h2 else None
                    ),
                )
            metrics.update(self._ppo_metrics_to_dict(ppo_metrics))
            if joint_result is not None:
                metrics.update({
                    "ppo/value_loss": joint_result.loss,
                    "ppo/explained_variance": (
                        joint_result.h2_explained_variance
                    ),
                    "ppo/critic_grad_norm": joint_result.grad_norm,
                    "ppo/critic_num_minibatches": joint_result.minibatches,
                    "critic_joint_rehearsal/joint_loss": joint_result.loss,
                    "critic_joint_rehearsal/h2_mse": joint_result.h2_mse,
                    "critic_joint_rehearsal/real_mse": joint_result.real_mse,
                    "critic_joint_rehearsal/ranking_loss": (
                        joint_result.ranking_loss
                    ),
                    "critic_joint_rehearsal/h2_train_ev": (
                        joint_result.h2_explained_variance
                    ),
                    "critic_joint_rehearsal/real_train_ev": (
                        joint_result.real_explained_variance
                    ),
                    "critic_joint_rehearsal/h2_samples": (
                        joint_result.h2_samples
                    ),
                    "critic_joint_rehearsal/fixed_real_samples": (
                        joint_result.real_samples
                    ),
                    "critic_joint_rehearsal/fixed_real_groups": (
                        joint_result.real_groups
                    ),
                    "critic_joint_rehearsal/gradient_cosine": (
                        joint_result.gradient_cosine
                    ),
                    "critic_joint_rehearsal/gradient_conflict_fraction": (
                        joint_result.gradient_conflict_fraction
                    ),
                })
            if real_critic_batch is not None:
                # Legacy mode calibrates real returns in separate steps. The
                # P0 group-aware path has already consumed real groups in the
                # same backward pass as H2 and must not step again here.
                calibration_updates = int(os.environ.get(
                    "REAL_CRITIC_CALIBRATION_UPDATES", "5"
                ))
                train_samples = int(os.environ.get(
                    "REAL_CRITIC_TRAIN_SAMPLES", "512"
                ))
                if calibration_updates <= 0 or train_samples <= 0:
                    raise ValueError(
                        "REAL_CRITIC_CALIBRATION_UPDATES and "
                        "REAL_CRITIC_TRAIN_SAMPLES must be positive"
                    )
                if joint_result is not None:
                    # The real source was already optimized together with H2
                    # in the same backward path. A second real-only Adam step
                    # would recreate the P0 overwrite bug.
                    calibration_updates = 0
                    anchor_count = int(joint_result.real_samples)
                    anchor_losses = [joint_result.real_mse]
                    anchor_evs = [joint_result.real_explained_variance]
                else:
                    anchor_count = min(
                        len(real_critic_batch.actions), train_samples
                    )
                    anchor_losses: list[float] = []
                    anchor_evs: list[float] = []
                    for calibration_index in range(calibration_updates):
                        generator = torch.Generator(
                            device=real_critic_batch.actions.device
                        ).manual_seed(
                            20260901
                            + 1009 * self._real_critic_cache_generation
                            + 31 * current_update
                            + calibration_index
                        )
                        index = torch.randperm(
                            len(real_critic_batch.actions),
                            generator=generator,
                            device=real_critic_batch.actions.device,
                        )[:anchor_count]
                        anchor = PPOBatch(**{
                            name: getattr(real_critic_batch, name)[index]
                            for name in PPOBatch.__dataclass_fields__
                        })
                        anchor_metrics = self.ppo_updater.update(
                            anchor, critic_only=False, actor_enabled=False
                        )
                        anchor_losses.append(float(anchor_metrics.value_loss))
                        anchor_evs.append(float(
                            anchor_metrics.explained_variance
                        ))
                metrics["critic_real_anchor/samples"] = float(anchor_count)
                metrics["critic_joint_rehearsal/fixed_real_samples"] = float(
                    anchor_count
                )
                metrics["critic_real_anchor/calibration_updates"] = float(
                    calibration_updates
                )
                metrics["critic_real_anchor/value_loss"] = float(
                    sum(anchor_losses) / len(anchor_losses)
                )
                metrics["critic_real_anchor/explained_variance"] = float(
                    sum(anchor_evs) / len(anchor_evs)
                )
                validation = getattr(self, "_real_critic_validation", None)
                if validation is not None and len(validation.actions) >= 16:
                    with torch.no_grad():
                        prediction = self.policy.evaluate_values(
                            validation.states, validation.actions
                        ).float()
                        target = validation.returns.float()
                        variance = target.var(unbiased=False)
                        real_ev = 0.0 if float(variance) < 1e-8 else float(
                            1.0 - (target - prediction).var(unbiased=False) / variance
                        )
                        real_mse = float((target - prediction).pow(2).mean())
                    metrics["critic_real_anchor/heldout_ev"] = real_ev
                    metrics["critic_real_anchor/heldout_mse"] = real_mse
                    metrics["critic_real_anchor/heldout_samples"] = float(
                        len(validation.actions)
                    )
                    if (
                        os.environ.get(
                            "REAL_CRITIC_FIRST_TRANSITION_ONLY", "0"
                        ) == "1"
                        and len(validation.actions) % 4 == 0
                    ):
                        grouped_actions = validation.actions.long().view(-1, 4)
                        expected_actions = torch.arange(
                            4, device=grouped_actions.device
                        ).expand_as(grouped_actions)
                        if not torch.equal(grouped_actions, expected_actions):
                            raise RuntimeError(
                                "counterfactual validation lost complete ordered "
                                "four-action level groups"
                            )
                        grouped_prediction = prediction.view(-1, 4)
                        grouped_target = target.view(-1, 4)
                        predicted_best = grouped_prediction.argmax(dim=1)
                        target_max = grouped_target.max(dim=1, keepdim=True).values
                        target_best_mask = torch.isclose(
                            grouped_target, target_max, atol=1e-6, rtol=1e-5
                        )
                        top1_correct = target_best_mask.gather(
                            1, predicted_best[:, None]
                        ).float().mean()
                        pairwise_correct = torch.zeros(
                            (), device=grouped_target.device
                        )
                        pairwise_count = torch.zeros(
                            (), device=grouped_target.device
                        )
                        for left in range(4):
                            for right in range(left + 1, 4):
                                target_delta = (
                                    grouped_target[:, left]
                                    - grouped_target[:, right]
                                )
                                informative = target_delta.abs() > 1e-6
                                if bool(informative.any()):
                                    predicted_delta = (
                                        grouped_prediction[:, left]
                                        - grouped_prediction[:, right]
                                    )
                                    pairwise_correct += (
                                        predicted_delta[informative].sign()
                                        == target_delta[informative].sign()
                                    ).float().sum()
                                    pairwise_count += informative.float().sum()
                        q_margin = (
                            grouped_prediction.topk(2, dim=1).values[:, 0]
                            - grouped_prediction.topk(2, dim=1).values[:, 1]
                        ).mean()
                        metrics.update({
                            "critic_real_anchor/heldout_level_groups": float(
                                len(grouped_target)
                            ),
                            "critic_real_anchor/heldout_top1_accuracy": float(
                                top1_correct
                            ),
                            "critic_real_anchor/heldout_pairwise_accuracy": float(
                                pairwise_correct / pairwise_count.clamp_min(1.0)
                            ),
                            "critic_real_anchor/heldout_informative_pairs": float(
                                pairwise_count
                            ),
                            "critic_real_anchor/heldout_q_margin": float(q_margin),
                        })
                    alpha = float(os.environ.get(
                        "REAL_CRITIC_EV_EMA_ALPHA", "0.2"
                    ))
                    previous_ev_ema = getattr(
                        self, "_real_critic_anchor_ev_ema", None
                    )
                    ev_ema = (
                        real_ev if previous_ev_ema is None
                        else (1.0 - alpha) * previous_ev_ema + alpha * real_ev
                    )
                    threshold = float(os.environ.get(
                        "REAL_CRITIC_EV_GATE", "0.10"
                    ))
                    require_ranking_gate = (
                        os.environ.get("REAL_CRITIC_REQUIRE_RANKING_GATE", "0")
                        == "1"
                    )
                    top1_accuracy = metrics.get(
                        "critic_real_anchor/heldout_top1_accuracy"
                    )
                    pairwise_accuracy = metrics.get(
                        "critic_real_anchor/heldout_pairwise_accuracy"
                    )
                    q_margin = metrics.get(
                        "critic_real_anchor/heldout_q_margin"
                    )
                    ranking_passed = True
                    if require_ranking_gate:
                        if (
                            top1_accuracy is None
                            or pairwise_accuracy is None
                            or q_margin is None
                        ):
                            ranking_passed = False
                        else:
                            ranking_passed = (
                                top1_accuracy >= float(os.environ.get(
                                    "REAL_CRITIC_TOP1_GATE", "0.60"
                                ))
                                and pairwise_accuracy >= float(os.environ.get(
                                    "REAL_CRITIC_PAIRWISE_GATE", "0.60"
                                ))
                                and q_margin >= float(os.environ.get(
                                    "REAL_CRITIC_Q_MARGIN_GATE", "0.01"
                                ))
                            )
                    patience = int(os.environ.get(
                        "REAL_CRITIC_EV_GATE_PATIENCE", "3"
                    ))
                    streak = int(getattr(
                        self, "_real_critic_anchor_ev_streak", 0
                    ))
                    joint_gate_passed = ev_ema >= threshold and ranking_passed
                    streak = streak + 1 if joint_gate_passed else 0
                    failure_streak = int(getattr(
                        self, "_real_critic_anchor_failure_streak", 0
                    ))
                    failure_streak = 0 if joint_gate_passed else failure_streak + 1
                    was_ready = bool(getattr(
                        self, "_real_critic_anchor_ready", False
                    ))
                    self._real_critic_anchor_ev_ema = ev_ema
                    self._real_critic_anchor_ev_streak = streak
                    self._real_critic_anchor_failure_streak = failure_streak
                    # Avoid reacting to one noisy held-out check, but do not
                    # permanently latch readiness either. Two consecutive
                    # joint failures pause Actor; recovery again needs the
                    # configured consecutive-pass patience.
                    self._real_critic_anchor_ready = (
                        (was_ready and failure_streak < 2)
                        or streak >= patience
                    )
                    metrics.update({
                        "critic_real_anchor/heldout_ev_ema": float(ev_ema),
                        "critic_real_anchor/ranking_gate_required": float(
                            require_ranking_gate
                        ),
                        "critic_real_anchor/ranking_gate_passed": float(
                            ranking_passed
                        ),
                        "critic_real_anchor/joint_gate_passed": float(
                            joint_gate_passed
                        ),
                        "critic_real_anchor/ev_gate_streak": float(streak),
                        "critic_real_anchor/joint_gate_failure_streak": float(
                            failure_streak
                        ),
                        "critic_real_anchor/ready": float(getattr(
                            self, "_real_critic_anchor_ready", False
                        )),
                    })
                self._pending_real_critic_batch = None
            critic_diagnostic_stop = False
            if critic_diagnostic_only and not critic_only:
                critic_diagnostic_updates += 1
                diagnostic_budget = int(os.environ.get(
                    "REAL_CRITIC_DIAGNOSTIC_MAX_UPDATES", "50"
                ))
                if diagnostic_budget <= 0:
                    raise ValueError(
                        "REAL_CRITIC_DIAGNOSTIC_MAX_UPDATES must be positive"
                    )
                diagnostic_passed = bool(getattr(
                    self, "_real_critic_anchor_ready", False
                ))
                critic_diagnostic_stop = (
                    diagnostic_passed
                    or critic_diagnostic_updates >= diagnostic_budget
                )
                metrics.update({
                    "critic_diagnostic/active": 1.0,
                    "critic_diagnostic/update": float(
                        critic_diagnostic_updates
                    ),
                    "critic_diagnostic/budget": float(diagnostic_budget),
                    "critic_diagnostic/passed": float(diagnostic_passed),
                    "critic_diagnostic/exhausted": float(
                        not diagnostic_passed
                        and critic_diagnostic_updates >= diagnostic_budget
                    ),
                })
            if actor_update_allowed:
                self._actor_ppo_updates += 1
            metrics["ppo/actor_update"] = float(self._actor_ppo_updates)
            counterfactual_actor_rejected = False
            if (
                actor_update_allowed
                and exact_counterfactual_h2
                and self._critic_warmup_validation is not None
                and hasattr(
                    self, "_counterfactual_validation_action_bucket_ids"
                )
                and self._actor_ppo_updates % int(os.environ.get(
                    "COUNTERFACTUAL_ACTOR_VALIDATION_EVERY", "1"
                )) == 0
            ):
                validation_metrics, actor_bucket_gate, validation_score = (
                    self._counterfactual_actor_validation_metrics()
                )
                metrics.update(validation_metrics)
                self._counterfactual_actor_validation_passed = actor_bucket_gate
                if (
                    os.environ.get(
                        "COUNTERFACTUAL_ACTOR_TRANSACTION_GATE", "0"
                    ) == "1"
                ):
                    if (
                        actor_validation_before_score is None
                        or actor_validation_before_gate is None
                    ):
                        raise RuntimeError(
                            "counterfactual Actor transaction has no pre-update score"
                        )
                    tolerance = float(os.environ.get(
                        "COUNTERFACTUAL_ACTOR_SCORE_DROP_TOLERANCE", "0"
                    ))
                    material_score_drop = (
                        validation_score + tolerance < actor_validation_before_score
                    )
                    # Once every held-out bucket has passed its absolute
                    # quality floor, never accept a candidate that loses that
                    # status. Within the passing region, tolerate ordinary
                    # minibatch variation and roll back only a material drop.
                    lost_absolute_gate = bool(
                        actor_validation_before_gate and not actor_bucket_gate
                    )
                    counterfactual_actor_rejected = bool(
                        lost_absolute_gate or material_score_drop
                    )
                    metrics.update({
                        "actor_validation/pre_update_mean_ranking_score": float(
                            actor_validation_before_score
                        ),
                        "actor_validation/pre_update_all_buckets_passed": float(
                            actor_validation_before_gate
                        ),
                        "actor_validation/score_drop_tolerance": tolerance,
                        "actor_validation/material_score_drop": float(
                            material_score_drop
                        ),
                        "actor_validation/lost_absolute_gate": float(
                            lost_absolute_gate
                        ),
                        "actor_validation/non_regression_passed": float(
                            not counterfactual_actor_rejected
                        ),
                    })
            actor_diagnostics = getattr(
                self, "actor_function_diagnostics_fn", None
            )
            actor_rejected = counterfactual_actor_rejected
            if actor_update_allowed and callable(actor_diagnostics):
                actor_gate_metrics = actor_diagnostics(
                    self._actor_ppo_updates
                )
                actor_rejected = actor_rejected or bool(actor_gate_metrics.pop(
                    "_actor_update_rejected", 0.0
                ))
                metrics.update(actor_gate_metrics)
                if os.environ.get(
                    "REAL_POSTERIOR_ACTOR_GATE", "0"
                ) == "1":
                    deployment_metrics = self._actor_real_posterior_metrics()
                    actor_rejected = actor_rejected or bool(
                        deployment_metrics.pop("_actor_update_rejected", 0.0)
                    )
                    metrics.update(deployment_metrics)
            # Counterfactual non-regression is a complete transaction gate in
            # its own right.  It must roll back even when no optional external
            # Actor diagnostic callback has been installed.
            if actor_update_allowed and actor_rejected:
                if actor_transaction_snapshot is None:
                    raise RuntimeError(
                        "Actor gate requested rollback without a "
                        "transaction snapshot"
                    )
                self._restore_actor_update_transaction(
                    actor_transaction_snapshot
                )
                self._actor_ppo_updates -= 1
                self._actor_rejection_streak += 1
                metrics.update({
                    "ppo/actor_update": float(self._actor_ppo_updates),
                    "actor_transaction/rejected": 1.0,
                    "actor_transaction/rejection_streak": float(
                        self._actor_rejection_streak
                    ),
                })
                rejection_patience = int(os.environ.get(
                    "ACTOR_TRANSACTION_REJECT_PATIENCE", "3"
                ))
                if self._actor_rejection_streak >= rejection_patience:
                    raise RuntimeError(
                        "Actor candidate violated the counterfactual safety "
                        "Gate "
                        f"{self._actor_rejection_streak} times; each update "
                        "was rolled back including Adam state"
                    )
            elif actor_update_allowed and actor_transaction_snapshot is not None:
                self._actor_rejection_streak = 0
                metrics.update({
                    "actor_transaction/rejected": 0.0,
                    "actor_transaction/rejection_streak": 0.0,
                })
            metrics["critic_warmup/active"] = float(critic_only)
            if unified_replay and os.environ.get(
                "FORMAL_UNIFIED_PPO", "0"
            ) == "1":
                # Grow only after an Actor transaction has been accepted.  This
                # makes the persisted pool exactly match the formal PPO clock:
                # Actor update 0 -> 0 online, update N -> min(N, target).
                desired_online = min(
                    self._actor_ppo_updates, replay_sampler.online_target
                )
                align_visibility = getattr(
                    replay_sampler, "align_online_visibility", None
                )
                visible_before = int(replay_sampler.online_size)
                stored_before = int(getattr(
                    replay_sampler,
                    "stored_online_size",
                    replay_sampler.online_size,
                ))
                if callable(align_visibility):
                    align_visibility(
                        self._actor_ppo_updates,
                        require_available=False,
                    )
                reused_existing = (
                    replay_sampler.online_size >= desired_online
                    and stored_before >= desired_online
                    and visible_before < desired_online
                )
                if replay_sampler.online_size < desired_online:
                    if cfg.collect_episodes != 1:
                        raise RuntimeError(
                            "formal unified PPO requires collect_episodes=1"
                        )
                    if self.real_collector is None:
                        raise RuntimeError(
                            "formal unified PPO requires an online collector"
                        )
                    t_collect = time.time()
                    collect_result = self.real_collector.collect_and_store(
                        1, self._actor_ppo_updates
                    )
                    if hasattr(collect_result, "metrics"):
                        metrics.update({
                            f"collect/{key}": value
                            for key, value in collect_result.metrics.items()
                        })
                    metrics["timing/collect_sec"] = time.time() - t_collect
                    if callable(align_visibility):
                        align_visibility(
                            self._actor_ppo_updates,
                            require_available=True,
                        )
                metrics.update({
                    "buffer/online_episodes": float(replay_sampler.online_size),
                    "buffer/online_fraction": float(
                        replay_sampler.expected_online_fraction
                    ),
                    "buffer/frozen": float(replay_sampler.collection_complete),
                    "buffer/schedule_actor_update": float(
                        self._actor_ppo_updates
                    ),
                    "buffer/stored_online_episodes": float(getattr(
                        replay_sampler,
                        "stored_online_size",
                        replay_sampler.online_size,
                    )),
                    "buffer/reused_ahead_episode": float(reused_existing),
                })
                if replay_sampler.online_size != desired_online:
                    raise RuntimeError(
                        "formal unified replay fell behind the Actor clock: "
                        f"actor_update={self._actor_ppo_updates}, "
                        f"online={replay_sampler.online_size}, "
                        f"expected={desired_online}"
                    )
            if unified_replay:
                sampled = (
                    replay_sampler.last_online_count
                    + replay_sampler.last_offline_count
                )
                metrics.update({
                    "batch/online_samples": float(
                        replay_sampler.last_online_count
                    ),
                    "batch/offline_samples": float(
                        replay_sampler.last_offline_count
                    ),
                    "batch/realized_online_fraction": float(
                        replay_sampler.last_online_count / max(1, sampled)
                    ),
                })
            if critic_validation_batch is not None:
                with torch.no_grad():
                    prediction_parts = []
                    for begin in range(0, len(critic_validation_batch.actions), 64):
                        prediction_parts.append(self.policy.evaluate_values(
                            critic_validation_batch.states[begin:begin + 64],
                            critic_validation_batch.actions[begin:begin + 64],
                        ).float())
                    predictions = torch.cat(prediction_parts)
                    targets = critic_validation_batch.returns.float()
                    target_variance = targets.var(unbiased=False)
                    validation_ev = 0.0 if float(target_variance) < 1e-8 else float(
                        1.0
                        - (targets - predictions).var(unbiased=False)
                        / target_variance
                    )
                    validation_mse = float((targets - predictions).pow(2).mean())
                    target_mean = float(targets.mean())
                    prediction_mean = float(predictions.mean())
                    target_std = float(targets.std(unbiased=False))
                    prediction_std = float(predictions.std(unbiased=False))
                    prediction_bias = prediction_mean - target_mean
                    positive = targets > 0.0
                    nonpositive = ~positive
                    ranking_passed = True
                    if os.environ.get(
                        "CRITIC_WARMUP_REQUIRE_RANKING", "0"
                    ) == "1":
                        if len(targets) % 4 != 0:
                            raise RuntimeError(
                                "counterfactual Critic held-out panel lost "
                                "complete four-action groups"
                            )
                        grouped_actions = (
                            critic_validation_batch.actions.long().view(-1, 4)
                        )
                        expected_actions = torch.arange(
                            4, device=grouped_actions.device
                        ).expand_as(grouped_actions)
                        if not torch.equal(grouped_actions, expected_actions):
                            raise RuntimeError(
                                "counterfactual Critic held-out panel lost "
                                "ordered actions [0,1,2,3]"
                            )
                        grouped_prediction = predictions.view(-1, 4)
                        grouped_target = targets.view(-1, 4)
                        predicted_best = grouped_prediction.argmax(dim=1)
                        target_max = grouped_target.max(
                            dim=1, keepdim=True
                        ).values
                        target_best_mask = torch.isclose(
                            grouped_target, target_max, atol=1e-6, rtol=1e-5
                        )
                        top1_accuracy = float(target_best_mask.gather(
                            1, predicted_best[:, None]
                        ).float().mean())
                        pairwise_correct = torch.zeros(
                            (), device=grouped_target.device
                        )
                        pairwise_count = torch.zeros(
                            (), device=grouped_target.device
                        )
                        for left in range(4):
                            for right in range(left + 1, 4):
                                target_delta = (
                                    grouped_target[:, left]
                                    - grouped_target[:, right]
                                )
                                informative = target_delta.abs() > 1e-6
                                if bool(informative.any()):
                                    prediction_delta = (
                                        grouped_prediction[:, left]
                                        - grouped_prediction[:, right]
                                    )
                                    pairwise_correct += (
                                        prediction_delta[informative].sign()
                                        == target_delta[informative].sign()
                                    ).float().sum()
                                    pairwise_count += informative.float().sum()
                        pairwise_accuracy = float(
                            pairwise_correct / pairwise_count.clamp_min(1.0)
                        )
                        q_margin = float(
                            grouped_prediction.topk(2, dim=1).values.diff(
                                dim=1
                            ).neg().mean()
                        )
                        ranking_passed = (
                            top1_accuracy >= float(os.environ.get(
                                "CRITIC_WARMUP_TOP1_GATE", "0.60"
                            ))
                            and pairwise_accuracy >= float(os.environ.get(
                                "CRITIC_WARMUP_PAIRWISE_GATE", "0.60"
                            ))
                            and q_margin >= float(os.environ.get(
                                "CRITIC_WARMUP_Q_MARGIN_GATE", "0.001"
                            ))
                        )
                        metrics.update({
                            "critic_warmup/heldout_level_groups": float(
                                len(grouped_target)
                            ),
                            "critic_warmup/heldout_top1_accuracy": top1_accuracy,
                            "critic_warmup/heldout_pairwise_accuracy": (
                                pairwise_accuracy
                            ),
                            "critic_warmup/heldout_informative_pairs": float(
                                pairwise_count
                            ),
                            "critic_warmup/heldout_q_margin": q_margin,
                            "critic_warmup/ranking_gate_passed": float(
                                ranking_passed
                            ),
                        })
                        bucket_ids_by_action = getattr(
                            self,
                            "_counterfactual_validation_action_bucket_ids",
                            None,
                        )
                        bucket_names = getattr(
                            self,
                            "counterfactual_h2_validation_bucket_names",
                            None,
                        )
                        if bucket_ids_by_action is not None and bucket_names:
                            group_bucket_ids = bucket_ids_by_action.view(-1, 4)[:, 0]
                            stable_bucket_passes = 0
                            raw_bucket_passes = 0
                            minimum_bucket_passes = int(os.environ.get(
                                "CRITIC_RELEASE_MIN_BUCKETS_PASSED",
                                str(len(bucket_names)),
                            ))
                            if not 1 <= minimum_bucket_passes <= len(bucket_names):
                                raise ValueError(
                                    "CRITIC_RELEASE_MIN_BUCKETS_PASSED must be "
                                    f"in [1,{len(bucket_names)}]"
                                )
                            bucket_ema_alpha = float(os.environ.get(
                                "CRITIC_BUCKET_EMA_ALPHA", "0.20"
                            ))
                            if not 0.0 < bucket_ema_alpha <= 1.0:
                                raise ValueError(
                                    "CRITIC_BUCKET_EMA_ALPHA must be in (0,1]"
                                )
                            top1_gate = float(os.environ.get(
                                "CRITIC_WARMUP_TOP1_GATE", "0.60"
                            ))
                            pairwise_gate = float(os.environ.get(
                                "CRITIC_WARMUP_PAIRWISE_GATE", "0.60"
                            ))
                            margin_gate = float(os.environ.get(
                                "CRITIC_WARMUP_Q_MARGIN_GATE", "0.001"
                            ))
                            ev_gate = float(os.environ.get(
                                "CRITIC_WARMUP_EV_THRESHOLD", "0.10"
                            ))
                            # A PPO release only needs a useful, non-degenerate
                            # Critic; it need not satisfy the stricter, uniform
                            # probe threshold in every small validation bucket.
                            # Keep the legacy thresholds as defaults and allow
                            # this protocol to set explicit per-bucket floors.
                            release_bucket_ev_gate = float(os.environ.get(
                                "CRITIC_RELEASE_BUCKET_EV_GATE", str(ev_gate)
                            ))
                            release_require_bucket_ev = os.environ.get(
                                "CRITIC_RELEASE_REQUIRE_BUCKET_EV", "1"
                            ) == "1"
                            release_bucket_pairwise_gate = float(os.environ.get(
                                "CRITIC_RELEASE_BUCKET_PAIRWISE_GATE",
                                str(pairwise_gate),
                            ))
                            release_bucket_margin_gate = float(os.environ.get(
                                "CRITIC_RELEASE_BUCKET_Q_MARGIN_GATE",
                                str(margin_gate),
                            ))
                            release_default_top1_gate = float(os.environ.get(
                                "CRITIC_RELEASE_BUCKET_TOP1_GATE",
                                str(top1_gate),
                            ))
                            release_overall_top1_gate = float(os.environ.get(
                                "CRITIC_RELEASE_OVERALL_TOP1_GATE",
                                str(top1_gate),
                            ))
                            release_overall_pairwise_gate = float(os.environ.get(
                                "CRITIC_RELEASE_OVERALL_PAIRWISE_GATE",
                                str(pairwise_gate),
                            ))
                            release_overall_margin_gate = float(os.environ.get(
                                "CRITIC_RELEASE_OVERALL_Q_MARGIN_GATE",
                                str(margin_gate),
                            ))
                            ranking_passed = (
                                top1_accuracy >= release_overall_top1_gate
                                and pairwise_accuracy
                                >= release_overall_pairwise_gate
                                and q_margin >= release_overall_margin_gate
                            )
                            metrics.update({
                                "critic_warmup/release_overall_top1_gate": (
                                    release_overall_top1_gate
                                ),
                                "critic_warmup/release_overall_pairwise_gate": (
                                    release_overall_pairwise_gate
                                ),
                                "critic_warmup/release_bucket_ev_gate": (
                                    release_bucket_ev_gate
                                ),
                            })
                            for bucket_id, bucket_name in enumerate(bucket_names):
                                selected = group_bucket_ids == bucket_id
                                if not bool(selected.any()):
                                    raise RuntimeError(
                                        f"empty level-disjoint validation bucket: {bucket_name}"
                                    )
                                bucket_result = _four_action_ranking_metrics(
                                    grouped_prediction[selected],
                                    grouped_target[selected],
                                )
                                bucket_top1_gate = float(os.environ.get(
                                    "CRITIC_RELEASE_"
                                    f"{bucket_name.upper()}_TOP1_GATE",
                                    str(release_default_top1_gate),
                                ))
                                raw_passed = (
                                    (
                                        not release_require_bucket_ev
                                        or bucket_result[
                                            "centered_explained_variance"
                                        ] >= release_bucket_ev_gate
                                    )
                                    and bucket_result["top1_accuracy"]
                                    >= bucket_top1_gate
                                    and bucket_result["pairwise_accuracy"]
                                    >= release_bucket_pairwise_gate
                                    and bucket_result["informative_pairs"] > 0.0
                                    and bucket_result["q_margin"]
                                    >= release_bucket_margin_gate
                                )
                                raw_bucket_passes += int(raw_passed)
                                tracked_keys = (
                                    "centered_explained_variance",
                                    "top1_accuracy",
                                    "pairwise_accuracy",
                                    "q_margin",
                                )
                                bucket_ema = self._critic_bucket_ema.setdefault(
                                    bucket_name, {}
                                )
                                for key in tracked_keys:
                                    current_value = bucket_result[key]
                                    previous_value = bucket_ema.get(key)
                                    bucket_ema[key] = (
                                        current_value
                                        if previous_value is None
                                        else bucket_ema_alpha * current_value
                                        + (1.0 - bucket_ema_alpha) * previous_value
                                    )
                                stable_passed = (
                                    (
                                        not release_require_bucket_ev
                                        or bucket_ema[
                                            "centered_explained_variance"
                                        ] >= release_bucket_ev_gate
                                    )
                                    and bucket_ema["top1_accuracy"]
                                    >= bucket_top1_gate
                                    and bucket_ema["pairwise_accuracy"]
                                    >= release_bucket_pairwise_gate
                                    and bucket_result["informative_pairs"] > 0.0
                                    and bucket_ema["q_margin"]
                                    >= release_bucket_margin_gate
                                )
                                stable_bucket_passes += int(stable_passed)
                                prefix = f"critic_warmup/bucket_{bucket_name}"
                                metrics.update({
                                    f"{prefix}/{key}": value
                                    for key, value in bucket_result.items()
                                })
                                metrics[f"{prefix}/raw_passed"] = float(raw_passed)
                                metrics[f"{prefix}/passed"] = float(stable_passed)
                                metrics[f"{prefix}/release_top1_gate"] = (
                                    bucket_top1_gate
                                )
                                metrics[f"{prefix}/release_ev_required"] = float(
                                    release_require_bucket_ev
                                )
                                for key, value in bucket_ema.items():
                                    metrics[f"{prefix}/{key}_ema"] = value
                            raw_bucket_gate_passed = (
                                raw_bucket_passes >= minimum_bucket_passes
                            )
                            bucket_gate_passed = (
                                stable_bucket_passes >= minimum_bucket_passes
                            )
                            metrics.update({
                                "critic_warmup/release_min_buckets_passed": float(
                                    minimum_bucket_passes
                                ),
                                "critic_warmup/raw_buckets_passed": float(
                                    raw_bucket_passes
                                ),
                                "critic_warmup/stable_buckets_passed": float(
                                    stable_bucket_passes
                                ),
                            })
                            if (
                                raw_bucket_gate_passed
                                and not self._critic_candidate_saved
                            ):
                                self._critic_candidate_saved = True
                                critic_candidate_just_qualified = True
                                if not self._critic_stabilization_lr_applied:
                                    lr_factor = float(os.environ.get(
                                        "CRITIC_STABILIZATION_LR_FACTOR", "0.25"
                                    ))
                                    if not 0.0 < lr_factor <= 1.0:
                                        raise ValueError(
                                            "CRITIC_STABILIZATION_LR_FACTOR must be in (0,1]"
                                        )
                                    for group in self.ppo_updater.optimizer.param_groups:
                                        if group.get("group_name") == "critic":
                                            group["lr"] *= lr_factor
                                            metrics[
                                                "critic_warmup/stabilization_lr"
                                            ] = float(group["lr"])
                                    self._critic_stabilization_lr_applied = True
                            ranking_passed = ranking_passed and bucket_gate_passed
                            metrics[
                                "critic_warmup/level_disjoint_bucket_gate_passed"
                            ] = float(bucket_gate_passed)
                            metrics[
                                "critic_warmup/ranking_gate_passed"
                            ] = float(ranking_passed)

                    def subset_mse(mask: Tensor) -> float:
                        if not bool(mask.any()):
                            return float("nan")
                        return float(
                            (targets[mask] - predictions[mask]).pow(2).mean()
                        )

                    replay = self._critic_warmup_replay
                    global_mean = replay.returns.float().mean()
                    if getattr(self.policy, "critic_source", None) == "qwen_slotwise_q":
                        action_means = []
                        for action_id in range(4):
                            selected = replay.actions.long() == action_id
                            action_means.append(
                                replay.returns[selected].float().mean()
                                if bool(selected.any()) else global_mean
                            )
                        baseline = torch.stack(action_means)[
                            critic_validation_batch.actions.long()
                        ]
                    else:
                        # A scalar V_pi(s) has no action input; compare it with
                        # the strongest matching constant baseline, not an
                        # action-conditioned oracle unavailable to the model.
                        baseline = torch.full_like(targets, global_mean)
                    baseline_mse = float((targets - baseline).pow(2).mean())
                    mse_improvement = 1.0 - validation_mse / max(baseline_mse, 1e-8)
                alpha = cfg.critic_warmup_ev_ema_alpha
                self._critic_warmup_ev_ema = (
                    validation_ev if self._critic_warmup_ev_ema is None
                    else alpha * validation_ev
                    + (1.0 - alpha) * self._critic_warmup_ev_ema
                )
                metrics["critic_warmup/heldout_ev"] = validation_ev
                metrics["critic_warmup/heldout_ev_ema"] = self._critic_warmup_ev_ema
                metrics["critic_warmup/heldout_mse"] = validation_mse
                metrics["critic_warmup/baseline_mse"] = baseline_mse
                metrics["critic_warmup/mse_improvement"] = mse_improvement
                metrics["critic_warmup/heldout_target_mean"] = target_mean
                metrics["critic_warmup/heldout_prediction_mean"] = prediction_mean
                metrics["critic_warmup/heldout_prediction_bias"] = prediction_bias
                metrics["critic_warmup/heldout_target_std"] = target_std
                metrics["critic_warmup/heldout_prediction_std"] = prediction_std
                metrics["critic_warmup/heldout_positive_fraction"] = float(
                    positive.float().mean()
                )
                metrics["critic_warmup/heldout_positive_mse"] = subset_mse(positive)
                metrics["critic_warmup/heldout_nonpositive_mse"] = subset_mse(
                    nonpositive
                )
                validation_full = (
                    len(critic_validation_batch.actions)
                    >= cfg.critic_warmup_validation_size
                )
                passed = (
                    validation_full
                    and self._critic_warmup_ev_ema
                    >= cfg.critic_warmup_ev_threshold
                    and mse_improvement >= cfg.critic_warmup_mse_improvement
                    and ranking_passed
                )
                self._critic_warmup_ev_streak = (
                    self._critic_warmup_ev_streak + 1 if passed else 0
                )
                if (
                    self._critic_warmup_updates
                    >= cfg.critic_warmup_min_updates
                    and self._critic_warmup_ev_streak
                    >= cfg.critic_warmup_ev_patience
                ):
                    self._critic_warmup_complete = True
                    gate_just_passed = True
                    print(
                        "Critic warm-up gate passed: "
                        f"held-out EV EMA={self._critic_warmup_ev_ema:.4f}, "
                        f"MSE improvement={mse_improvement:.1%}, "
                        f"streak={self._critic_warmup_ev_streak}; "
                        "Actor PPO and WM refresh enable next update.",
                        flush=True,
                    )
                recalibration_remaining = getattr(
                    self, "_critic_recalibration_remaining", None
                )
                if recalibration_remaining is not None:
                    recalibration_remaining -= 1
                    self._critic_recalibration_remaining = recalibration_remaining
                    metrics["critic_recalibration/remaining"] = float(
                        max(0, recalibration_remaining)
                    )
                    if self._critic_warmup_complete:
                        self._critic_recalibration_remaining = None
                        self._critic_recalibration_rollback_fn = None
                        metrics["critic_recalibration/accepted"] = 1.0
                    elif recalibration_remaining <= 0:
                        rollback = getattr(
                            self, "_critic_recalibration_rollback_fn", None
                        )
                        if not callable(rollback):
                            raise RuntimeError(
                                "Critic recalibration expired without rollback state"
                            )
                        rollback()
                        self._critic_warmup_complete = True
                        self._critic_warmup_ev_streak = 0
                        self._critic_recalibration_remaining = None
                        self._critic_recalibration_rollback_fn = None
                        gate_just_passed = True
                        metrics["critic_warmup/complete"] = 1.0
                        metrics["critic_recalibration/accepted"] = 0.0
                        metrics["critic_recalibration/wm_rolled_back"] = 1.0
                        print(
                            "Critic incremental recalibration exhausted its "
                            "budget; rolled back the candidate WM and restored "
                            "the pre-refresh Critic/optimizer. Actor PPO enables "
                            "next update.",
                            flush=True,
                        )
                metrics["critic_warmup/ev_streak"] = float(
                    self._critic_warmup_ev_streak
                )
                metrics["critic_warmup/complete"] = float(
                    self._critic_warmup_complete
                )
                max_warmup_updates = int(os.environ.get(
                    "CRITIC_WARMUP_MAX_UPDATES", "0"
                ))
                if (
                    max_warmup_updates > 0
                    and self._critic_warmup_updates >= max_warmup_updates
                    and not self._critic_warmup_complete
                ):
                    # Stop only after this update has been logged and saved.
                    # That makes latest.pt an exact recovery point instead of
                    # silently losing the final checkpoint interval.
                    critic_pretrain_budget_exhausted = True
                    metrics["critic_pretrain/budget_exhausted"] = 1.0
            metrics["timing/ppo_sec"] = time.time() - t_ppo

            # --- 5. Real-env evaluation (periodic) ---
            if (
                (
                    gate_just_passed
                    and not critic_diagnostic_only
                    and not critic_pretrain_stage
                )
                or self._should_eval(current_update)
            ):
                self._stabilize_policy_forward()
                t_eval = time.time()
                if self.evaluator is not None:
                    eval_metrics = self.evaluator.evaluate(cfg.eval_episodes)
                    metrics.update(eval_metrics)
                metrics["timing/eval_sec"] = time.time() - t_eval
                metrics["eval/actor_ppo_update"] = float(
                    self._actor_ppo_updates
                )
                metrics["eval/gate_baseline"] = float(gate_just_passed)
                self._last_evaluated_actor_update = self._actor_ppo_updates

            # --- Logging ---
            elapsed = time.time() - t0
            metrics["timing/total_sec"] = elapsed

            eval_success_rate = metrics.get("eval/success_rate")
            if eval_success_rate is not None and math.isfinite(float(eval_success_rate)):
                current_sr = float(eval_success_rate)
                if bool(metrics.get("eval/gate_baseline", 0.0)):
                    flat_sr_baseline = current_sr
                    flat_sr_failures = 0
                elif (
                    flat_sr_patience > 0
                    and flat_sr_baseline is not None
                    and self._actor_ppo_updates > 0
                    and not flat_sr_changed
                ):
                    if abs(current_sr - flat_sr_baseline) <= flat_sr_tolerance:
                        flat_sr_failures += 1
                    else:
                        flat_sr_changed = True
                        flat_sr_failures = 0
                    metrics["eval/flat_sr_failures"] = float(flat_sr_failures)
                    metrics["eval/sr_changed_from_actor_start"] = float(flat_sr_changed)
            previous_best_eval_success_rate = self.best_eval_success_rate
            if checkpoint_root is not None and eval_success_rate is not None:
                self._maybe_save_best_checkpoint(
                    checkpoint_root, current_update, metrics
                )
            should_stop_after_logging = False
            critic_pretrain_complete = bool(
                critic_pretrain_stage
                and self._critic_warmup_complete
            )
            metrics["critic_pretrain/complete"] = float(
                critic_pretrain_complete
            )
            p0_critic_passed = (
                p0_critic_only
                and bool(getattr(self, "_real_critic_anchor_ready", False))
                and bool(self._rollout_critic_ready)
            )
            metrics["p0_critic_validation/passed"] = float(
                p0_critic_passed
            )
            if p0_critic_passed:
                should_stop_after_logging = True
            if critic_pretrain_complete:
                should_stop_after_logging = True
            if critic_pretrain_budget_exhausted:
                should_stop_after_logging = True
            if critic_diagnostic_stop:
                should_stop_after_logging = True
            actor_update_limit = int(os.environ.get("ACTOR_UPDATE_LIMIT", "0"))
            actor_limit_reached = (
                actor_update_limit > 0
                and self._actor_ppo_updates >= actor_update_limit
            )
            if actor_limit_reached:
                should_stop_after_logging = True
            if (
                eval_early_stop_patience > 0
                and eval_success_rate is not None
                and math.isfinite(float(eval_success_rate))
                and math.isfinite(previous_best_eval_success_rate)
            ):
                if (
                    float(eval_success_rate)
                    < previous_best_eval_success_rate - eval_early_stop_drop
                ):
                    eval_early_stop_failures += 1
                else:
                    eval_early_stop_failures = 0
                metrics["eval/early_stop_failures"] = float(
                    eval_early_stop_failures
                )
                if eval_early_stop_failures >= eval_early_stop_patience:
                    should_stop_after_logging = True

            self._last_critic_warmup_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if key.startswith("critic_warmup/")
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            }
            if self.logger is not None:
                self.logger.log_scalars(current_update, metrics)

            if flat_sr_patience > 0 and flat_sr_failures >= flat_sr_patience:
                raise RuntimeError(
                    "Actor PPO produced an unchanged real-env success rate "
                    f"({flat_sr_baseline:.6f}) for {flat_sr_failures} consecutive "
                    "post-Actor evaluations; stopping instead of reporting a "
                    "misleading flat training curve."
                )

            # Console summary
            rew = metrics.get("rollout/reward_mean", 0)
            ploss = metrics.get("ppo/policy_loss", 0)
            vloss = metrics.get("ppo/value_loss", 0)
            sr = metrics.get("eval/success_rate", -1)
            sr_str = f" | eval_sr={sr:.2f}" if sr >= 0 else ""
            critic_str = ""
            if critic_pretrain_stage:
                critic_str = (
                    f" | critic={self._critic_warmup_updates}/"
                    f"{int(os.environ.get('CRITIC_WARMUP_MAX_UPDATES', '0'))}"
                    f" ev_ema={metrics.get('critic_warmup/heldout_ev_ema', float('nan')):.4f}"
                    f" top1={metrics.get('critic_warmup/heldout_top1_accuracy', float('nan')):.4f}"
                    f" pairwise={metrics.get('critic_warmup/heldout_pairwise_accuracy', float('nan')):.4f}"
                    f" buckets={metrics.get('critic_warmup/stable_buckets_passed', 0):.0f}/4"
                    f" streak={self._critic_warmup_ev_streak}"
                )
            print(
                (
                    f"[actor_ppo_update {self._actor_ppo_updates}"
                    f"/{actor_update_limit}] "
                    if os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1"
                    else f"[update {current_update}/{cfg.total_updates}] "
                )
                +
                f"rollout={metrics.get('timing/rollout_sec', 0):.1f}s "
                f"ppo={metrics.get('timing/ppo_sec', 0):.1f}s "
                f"total={elapsed:.1f}s "
                f"rew={rew:.4f} ploss={ploss:.4f} vloss={vloss:.4f}"
                f"{sr_str}{critic_str}",
                flush=True,
            )

            # --- Checkpoint ---
            if checkpoint_root is not None and self._should_checkpoint(current_update):
                self._save_checkpoint(
                    checkpoint_root / "latest.pt", current_update
                )
            if (
                checkpoint_root is not None
                and critic_candidate_just_qualified
                and os.environ.get("CRITIC_SAVE_CANDIDATE", "1") == "1"
            ):
                self._save_checkpoint(
                    checkpoint_root / "critic_candidate.pt", current_update
                )
                print(
                    "  Saved critic_candidate.pt on the first raw four-bucket "
                    "pass; Critic LR reduced for EMA stabilization.",
                    flush=True,
                )
            if checkpoint_root is not None and critic_pretrain_complete:
                self._save_checkpoint(
                    checkpoint_root / "latest.pt",
                    current_update,
                )
            last_completed_update = current_update
            if should_stop_after_logging:
                if critic_pretrain_complete:
                    print(
                        "Critic-only pretraining complete: validation gate "
                        "passed, Actor PPO remains 0. "
                        "Saved latest.pt for release finalization.",
                        flush=True,
                    )
                elif p0_critic_passed:
                    print(
                        "P0 Critic validation PASSED with Actor frozen: "
                        f"real_ready={getattr(self, '_real_critic_anchor_ready', False)}, "
                        f"rollout_ready={self._rollout_critic_ready}, "
                        f"rollout_EV_EMA={self._rollout_critic_ev_ema}.",
                        flush=True,
                    )
                elif critic_diagnostic_stop:
                    passed = bool(getattr(
                        self, "_real_critic_anchor_ready", False
                    ))
                    print(
                        "Real Critic fixed-WM diagnostic "
                        f"{'PASSED' if passed else 'EXHAUSTED'} after "
                        f"{critic_diagnostic_updates} diagnostic updates; "
                        f"held-out EV EMA="
                        f"{getattr(self, '_real_critic_anchor_ev_ema', None)}.",
                        flush=True,
                    )
                elif critic_pretrain_budget_exhausted:
                    print(
                        "Critic-only warmup budget exhausted without release; "
                        "saved latest.pt with Actor PPO still at 0. Resume with "
                        "a larger CRITIC_WARMUP_MAX_UPDATES/total_updates.",
                        flush=True,
                    )
                elif actor_limit_reached:
                    print(
                        f"[update {current_update}/{cfg.total_updates}] "
                        f"completed requested Actor PPO updates: "
                        f"{self._actor_ppo_updates}/{actor_update_limit}",
                        flush=True,
                    )
                else:
                    print(
                        f"[update {current_update}/{cfg.total_updates}] "
                        f"early stop: eval_sr={float(eval_success_rate):.4f}, "
                        f"best_sr={self.best_eval_success_rate:.4f}, "
                        f"drop>{eval_early_stop_drop:.4f} for "
                        f"{eval_early_stop_failures} evals",
                        flush=True,
                    )
                break

        # Periodic checkpointing must not leave ``latest.pt`` behind the
        # policy that was actually evaluated at the end of training.
        if (
            checkpoint_root is not None
            and last_completed_update > start_update
            and getattr(self, "_critic_recalibration_remaining", None) is None
            and not self._should_checkpoint(last_completed_update)
        ):
            self._save_checkpoint(
                checkpoint_root / "latest.pt", last_completed_update
            )
        if critic_pretrain_budget_exhausted and not self._critic_warmup_complete:
            raise RuntimeError(
                "Critic warm-up exhausted its explicit budget before the "
                "release Gate passed: "
                f"updates={self._critic_warmup_updates}, "
                f"EV_EMA={self._critic_warmup_ev_ema}, "
                f"streak={self._critic_warmup_ev_streak}. "
                "latest.pt was saved; resume with a larger MAX_UPDATES."
            )
        actor_update_limit = int(os.environ.get("ACTOR_UPDATE_LIMIT", "0"))
        if (
            os.environ.get("CRITIC_PRETRAIN_ONLY", "0") != "1"
            and
            os.environ.get("REQUIRE_ACTOR_UPDATE_LIMIT_REACHED", "0") == "1"
            and actor_update_limit > 0
            and self._actor_ppo_updates < actor_update_limit
        ):
            raise RuntimeError(
                "training budget exhausted before the grounded Critic gates "
                "released/completed Actor PPO: "
                f"actor_updates={self._actor_ppo_updates}/{actor_update_limit}, "
                f"real_critic_ready={bool(getattr(self, '_real_critic_anchor_ready', False))}, "
                f"rollout_critic_ready={self._rollout_critic_ready}, "
                f"rollout_ev_ema={self._rollout_critic_ev_ema}, "
                f"rollout_ev_streak={self._rollout_critic_ev_streak}."
            )
        required_sr_levels = int(os.environ.get(
            "REQUIRE_ACTOR_SR_IMPROVEMENT_LEVELS", "0"
        ))
        if (
            required_sr_levels > 0
            and os.environ.get("CRITIC_PRETRAIN_ONLY", "0") != "1"
        ):
            if initial_eval_success_rate is None:
                raise RuntimeError(
                    "Actor SR improvement gate requires an initial real-env "
                    "evaluation; do not set SKIP_BASELINE_EVAL=1"
                )
            required_delta = required_sr_levels / max(cfg.eval_episodes, 1)
            required_sr = initial_eval_success_rate + required_delta
            if self.best_eval_success_rate + 1e-12 < required_sr:
                raise RuntimeError(
                    "Actor PPO failed the fixed-level SR improvement gate: "
                    f"initial_sr={initial_eval_success_rate:.6f}, "
                    f"best_sr={self.best_eval_success_rate:.6f}, "
                    f"required_sr={required_sr:.6f} "
                    f"(+{required_sr_levels}/{cfg.eval_episodes} levels)."
                )
            print(
                "Actor SR improvement gate PASSED: "
                f"initial_sr={initial_eval_success_rate:.6f}, "
                f"best_sr={self.best_eval_success_rate:.6f}, "
                f"required_delta={required_sr_levels}/{cfg.eval_episodes}.",
                flush=True,
            )
        if p0_critic_only and not (
            bool(getattr(self, "_real_critic_anchor_ready", False))
            and bool(self._rollout_critic_ready)
        ):
            raise RuntimeError(
                "P0 Critic validation budget exhausted without solving all "
                "P0 gates: "
                f"real_critic_ready={bool(getattr(self, '_real_critic_anchor_ready', False))}, "
                f"rollout_critic_ready={self._rollout_critic_ready}, "
                f"rollout_ev_ema={self._rollout_critic_ev_ema}, "
                f"rollout_ev_streak={self._rollout_critic_ev_streak}."
            )
        minimum_actor_updates = int(os.environ.get(
            "MIN_ACTOR_PPO_UPDATES", "0"
        ))
        if self._actor_ppo_updates < minimum_actor_updates:
            raise RuntimeError(
                "Actor PPO completion gate failed: "
                f"actor_updates={self._actor_ppo_updates}, "
                f"required>={minimum_actor_updates}."
            )
        if (
            os.environ.get("CRITIC_PRETRAIN_ONLY", "0") != "1"
            and
            os.environ.get(
                "REQUIRE_COUNTERFACTUAL_ACTOR_VALIDATION", "0"
            ) == "1"
            and not bool(getattr(
                self, "_counterfactual_actor_validation_passed", False
            ))
        ):
            raise RuntimeError(
                "Actor completed PPO but failed one or more level-disjoint "
                "initial/suffix H1/H2 ranking gates"
            )

    def _stabilize_policy_forward(self) -> None:
        """Use deterministic module mode while preserving trainable params."""
        stabilize = getattr(
            self.policy,
            "set_deterministic_forward_mode",
            None,
        )
        if callable(stabilize):
            stabilize()

    # -----------------------------------------------------------------------
    # Schedule helpers
    # -----------------------------------------------------------------------

    def _should_eval(self, update: int) -> bool:
        del update
        actor_update = self._actor_ppo_updates
        evaluate_first_actor_update = (
            os.environ.get("EVAL_AT_ACTOR_UPDATE_ONE", "0") == "1"
            and actor_update == 1
        )
        return (
            self._critic_warmup_complete
            and actor_update > 0
            and self.config.eval_every > 0
            and (
                evaluate_first_actor_update
                or actor_update % self.config.eval_every == 0
            )
            and actor_update != self._last_evaluated_actor_update
        )

    def _should_collect(self, update: int) -> bool:
        if os.environ.get("CRITIC_PRETRAIN_ONLY", "0") == "1":
            # The fixed H1/H2 posterior cache supplies Critic training starts.
            # Online replay belongs exclusively to formal PPO and must begin
            # empty on its own Actor-update clock.
            return False
        replay_sampler = getattr(self, "replay_sampler", None)
        if (
            os.environ.get("FORMAL_UNIFIED_PPO", "0") == "1"
            and replay_sampler is not None
            and hasattr(replay_sampler, "collection_complete")
        ):
            # Formal unified collection happens after an accepted Actor
            # transaction, so update zero begins with an actually empty pool.
            return False
        return (
            self.config.collect_every > 0
            and update % self.config.collect_every == 0
            and self.real_collector is not None
        )

    def _should_refresh_wm(self, update: int) -> bool:
        if os.environ.get("FORMAL_UNIFIED_PPO", "0") == "1":
            return (
                self._critic_warmup_complete
                and self._actor_ppo_updates > 0
                and self.config.wm_refresh_every > 0
                and self._actor_ppo_updates % self.config.wm_refresh_every == 0
                and self._actor_ppo_updates
                != self._last_wm_refresh_actor_update
                and self.wm_refresher is not None
            )
        return (
            self._critic_warmup_complete
            and
            self.config.wm_refresh_every > 0
            and update % self.config.wm_refresh_every == 0
            and self.wm_refresher is not None
        )

    def _should_checkpoint(self, update: int) -> bool:
        # The bounded recalibration rollback closure intentionally holds an
        # exact in-memory snapshot.  Do not emit a checkpoint half-way through
        # that transaction: after either acceptance or rollback normal
        # checkpointing resumes with a self-consistent WM/Critic/optimizer.
        recalibrating = getattr(
            self, "_critic_recalibration_remaining", None
        ) is not None
        if os.environ.get("FORMAL_UNIFIED_PPO", "0") == "1":
            should_save = (
                not recalibrating
                and self.config.checkpoint_every > 0
                and self._actor_ppo_updates > 0
                and self._actor_ppo_updates % self.config.checkpoint_every == 0
                and self._actor_ppo_updates
                != self._last_checkpointed_actor_update
            )
            if should_save:
                self._last_checkpointed_actor_update = self._actor_ppo_updates
            return should_save
        return (
            not recalibrating
            and self.config.checkpoint_every > 0
            and update % self.config.checkpoint_every == 0
        )

    # -----------------------------------------------------------------------
    # Metrics extraction (no algorithm logic, just reading fields)
    # -----------------------------------------------------------------------

    @staticmethod
    def _concat_trajectories(trajectories: list[Trajectory]) -> Trajectory:
        """Concatenate detached rollout chunks along their batch dimension."""
        if not trajectories:
            raise ValueError("At least one trajectory is required per PPO update.")
        if len(trajectories) == 1:
            return trajectories[0]

        def optional_cat(name: str) -> Tensor | None:
            values = [getattr(trajectory, name) for trajectory in trajectories]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Cannot concatenate mixed None/tensor trajectory field {name}."
                )
            return torch.cat(values, dim=0)  # type: ignore[arg-type]

        return Trajectory(
            states=torch.cat([t.states for t in trajectories], dim=0),
            actions=torch.cat([t.actions for t in trajectories], dim=0),
            rewards=torch.cat([t.rewards for t in trajectories], dim=0),
            dones=torch.cat([t.dones for t in trajectories], dim=0),
            log_probs=torch.cat([t.log_probs for t in trajectories], dim=0),
            values=torch.cat([t.values for t in trajectories], dim=0),
            mask=optional_cat("mask"),
            reward_logits=optional_cat("reward_logits"),
            base_rewards=optional_cat("base_rewards"),
            shaping_rewards=optional_cat("shaping_rewards"),
            relative_score_gap=optional_cat("relative_score_gap"),
            relative_top1_top2_margin=optional_cat("relative_top1_top2_margin"),
            relative_selected_rank=optional_cat("relative_selected_rank"),
            relative_selected_is_top1=optional_cat("relative_selected_is_top1"),
        )

    @staticmethod
    def _trajectory_metrics(
        traj: Trajectory,
        success_threshold: float = 0.5,
    ) -> dict[str, float]:
        valid_mask = (
            traj.mask.bool()
            if traj.mask is not None
            else torch.ones_like(traj.rewards, dtype=torch.bool)
        )
        valid_rewards = traj.rewards[valid_mask]
        valid_count = int(valid_mask.sum().item())
        total_count = valid_mask.numel()
        if valid_count == 0:
            reward_mean = reward_std = reward_nonzero_rate = positive_rate = 0.0
        else:
            reward_mean = float(valid_rewards.mean().item())
            reward_std = float(valid_rewards.std(unbiased=False).item())
            reward_nonzero_rate = float((valid_rewards != 0).float().mean().item())
            positive_rate = float((valid_rewards > 0).float().mean().item())

        valid_dones = traj.dones.bool() & valid_mask
        if traj.reward_logits is not None:
            reward_probabilities = torch.sigmoid(traj.reward_logits)
            valid_probabilities = reward_probabilities[valid_mask]
            endpoint_indices = valid_mask.sum(dim=1).clamp_min(1) - 1
            endpoint_probabilities = reward_probabilities.gather(
                1, endpoint_indices.long().unsqueeze(1)
            ).squeeze(1)
            # For fixed-horizon rollouts this is the endpoint classifier's
            # diagnostic prediction; it does not truncate the trajectory.
            predicted_success = endpoint_probabilities >= success_threshold
            probability_metrics = {
                "rollout/reward_probability_mean": float(
                    valid_probabilities.mean().item()
                ),
                "rollout/endpoint_success_probability_mean": float(
                    endpoint_probabilities.mean().item()
                ),
                "rollout/success_threshold": float(success_threshold),
            }
        else:
            predicted_success = valid_dones.any(dim=1)
            probability_metrics = {}
        valid_lengths = valid_mask.sum(dim=1).float()
        result = {
            # Primary reward metrics describe only transitions PPO receives.
            "rollout/reward_mean": reward_mean,
            "rollout/reward_std": reward_std,
            "rollout/reward_nonzero_rate": reward_nonzero_rate,
            "rollout/positive_reward_rate": positive_rate,
            "rollout/valid_steps": float(valid_count),
            "rollout/valid_fraction": float(valid_count / max(total_count, 1)),
            "rollout/mean_valid_horizon": float(valid_lengths.mean().item()),
            "rollout/predicted_success_rate": float(
                predicted_success.float().mean().item()
            ),
            "rollout/first_step_done_rate": float(
                valid_dones[:, 0].float().mean().item()
            ),
            # Raw padded mean remains available to compare against older logs.
            "rollout/raw_reward_mean": float(traj.rewards.mean().item()),
        }
        result.update(probability_metrics)
        if traj.shaping_rewards is not None and traj.base_rewards is not None:
            shaping = traj.shaping_rewards[valid_mask]
            terminal = traj.base_rewards[valid_mask]
            shaping_abs = shaping.abs().mean()
            terminal_abs = terminal.abs().mean()
            result.update({
                "rollout/relative_bonus_mean": float(shaping.mean().item()),
                "rollout/relative_bonus_std": float(
                    shaping.std(unbiased=False).item()
                ),
                "rollout/relative_bonus_abs_mean": float(shaping_abs.item()),
                "rollout/terminal_reward_abs_mean": float(terminal_abs.item()),
                "rollout/shaping_terminal_abs_ratio": float(
                    (shaping_abs / terminal_abs.clamp_min(1e-8)).item()
                ),
            })
        relative_fields = {
            "relative_score_gap": "rollout/relative_four_action_score_gap_mean",
            "relative_top1_top2_margin": "rollout/relative_top1_top2_margin_mean",
            "relative_selected_rank": "rollout/relative_selected_action_rank_mean",
            "relative_selected_is_top1": "rollout/relative_selected_is_top1_rate",
        }
        for field, metric_name in relative_fields.items():
            values = getattr(traj, field)
            if values is not None:
                result[metric_name] = float(values[valid_mask].float().mean().item())
        return result

    @staticmethod
    def _ppo_metrics_to_dict(m: PPOMetrics) -> dict[str, float]:
        return {
            "ppo/policy_loss": m.policy_loss,
            "ppo/value_loss": m.value_loss,
            "ppo/entropy": m.entropy,
            "ppo/post_update_entropy": m.post_update_entropy,
            "ppo/entropy_deficit": m.entropy_deficit,
            "ppo/entropy_floor_active_fraction": m.entropy_floor_active_fraction,
            "ppo/target_kl_early_stop": m.target_kl_early_stop,
            "ppo/behavior_kl": m.behavior_kl,
            "ppo/behavior_bc_loss": m.behavior_bc_loss,
            "ppo/behavior_bc_accuracy": m.behavior_bc_accuracy,
            "ppo/num_minibatches": m.num_minibatches,
            "ppo/sample_coverage": m.sample_coverage,
            "ppo/rollout_reference_logprob_mean_abs": (
                m.rollout_reference_logprob_mean_abs
            ),
            "ppo/rollout_reference_logprob_max_abs": (
                m.rollout_reference_logprob_max_abs
            ),
            "ppo/old_log_probs_recomputed": m.old_log_probs_recomputed,
            "ppo/initial_clip_fraction": m.initial_clip_fraction,
            "ppo/initial_kl_divergence": m.initial_kl_divergence,
            "ppo/attempted_minibatches": m.attempted_minibatches,
            "ppo/max_minibatch_kl": m.max_minibatch_kl,
            "ppo/last_minibatch_kl": m.last_minibatch_kl,
            "ppo/rejected_minibatch_kl": m.rejected_minibatch_kl,
            "ppo/post_update_kl_divergence": (
                m.post_update_kl_divergence
            ),
            "ppo/post_update_clip_fraction": (
                m.post_update_clip_fraction
            ),
            "ppo/critic_num_minibatches": m.critic_num_minibatches,
            "ppo/critic_sample_coverage": m.critic_sample_coverage,
            "ppo/actor_grad_norm": m.actor_grad_norm,
            "ppo/critic_grad_norm": m.critic_grad_norm,
            "ppo/clip_fraction": m.clip_fraction,
            "ppo/kl_divergence": m.kl_divergence,
            "ppo/explained_variance": m.explained_variance,
        }

    # -----------------------------------------------------------------------
    # Checkpoint (pipeline-level, delegates to components)
    # -----------------------------------------------------------------------

    def _maybe_save_best_checkpoint(
        self,
        checkpoint_root: Path,
        update: int,
        metrics: dict[str, float],
    ) -> None:
        success_rate = float(metrics.get("eval/success_rate", -1.0))
        if not math.isfinite(success_rate) or success_rate < 0.0:
            return
        if success_rate <= self.best_eval_success_rate:
            return
        self.best_eval_success_rate = success_rate
        self.best_eval_update = update
        self._save_checkpoint(checkpoint_root / "best.pt", update)
        print(
            "  New best real-env success rate: "
            f"{success_rate:.4f} at update {update}",
            flush=True,
        )

    def _save_checkpoint(self, path: Path, update: int) -> None:
        from ..model.checkpoint_semantics import world_model_semantics
        backbone = getattr(
            getattr(self.world_model, "transition", None), "backbone", None
        )
        checkpoint: dict[str, Any] = {
            "update": update,
            "policy": self.policy.state_dict(),
            "ppo_optimizer": self.ppo_updater.state_dict(),
            "best_eval_success_rate": self.best_eval_success_rate,
            "best_eval_update": self.best_eval_update,
            "last_evaluated_actor_update": (
                self._last_evaluated_actor_update
            ),
            "critic_warmup_complete": self._critic_warmup_complete,
            "critic_warmup_ev_streak": self._critic_warmup_ev_streak,
            "critic_warmup_ev_ema": self._critic_warmup_ev_ema,
            "critic_bucket_ema": self._critic_bucket_ema,
            "critic_candidate_saved": self._critic_candidate_saved,
            "critic_stabilization_lr_applied": (
                self._critic_stabilization_lr_applied
            ),
            "critic_warmup_updates": self._critic_warmup_updates,
            "critic_warmup_release_metrics": dict(
                self._last_critic_warmup_metrics
            ),
            "critic_h2_cache_metadata": dict(
                getattr(self, "critic_h2_cache_metadata", {})
            ),
            "critic_reward_confidence_floor": float(os.environ.get(
                "CRITIC_REWARD_CONFIDENCE_FLOOR", "nan"
            )),
            "critic_target_semantics": os.environ.get(
                "CRITIC_TARGET_SEMANTICS", "unspecified"
            ),
            # A release gate is meaningful only when resume uses the same
            # held-out evidence.  These tensors are small relative to the
            # model checkpoint and are stored on CPU to keep serialization
            # device-independent.
            "critic_warmup_panel_version": 2,
            "critic_warmup_validation": self._cpu_ppo_batch(
                self._critic_warmup_validation
            ),
            "critic_warmup_replay": self._cpu_ppo_batch(
                self._critic_warmup_replay
            ),
            "critic_warmup_replay_bucket_ids": (
                None
                if self._critic_warmup_replay_bucket_ids is None
                else self._critic_warmup_replay_bucket_ids.detach().cpu()
            ),
            "critic_warmup_validation_bucket_ids": (
                None
                if getattr(
                    self, "_counterfactual_validation_action_bucket_ids", None
                ) is None
                else self._counterfactual_validation_action_bucket_ids.detach().cpu()
            ),
            "critic_warmup_validation_bucket_names": list(
                getattr(
                    self, "counterfactual_h2_validation_bucket_names", None
                ) or []
            ),
            "actor_ppo_updates": self._actor_ppo_updates,
            "last_wm_refresh_actor_update": (
                self._last_wm_refresh_actor_update
            ),
            "last_checkpointed_actor_update": (
                self._last_checkpointed_actor_update
            ),
            "training_stage": (
                "critic_pretrain"
                if os.environ.get("CRITIC_PRETRAIN_ONLY", "0") == "1"
                else "ppo"
            ),
            "metric_step_axis": (
                "actor_ppo_update"
                if os.environ.get("WANDB_ACTOR_STEP_AXIS", "0") == "1"
                else "global_pipeline_update"
            ),
            "ppo_protocol": os.environ.get(
                "PPO_PROTOCOL", "unspecified"
            ),
            "ppo_source_checkpoint": os.environ.get(
                "PPO_SOURCE_CHECKPOINT", "unspecified"
            ),
            "optimizer_group_lrs": {
                str(group.get("group_name", index)): float(group["lr"])
                for index, group in enumerate(
                    self.ppo_updater.optimizer.param_groups
                )
            },
            "real_critic_anchor_ev_ema": getattr(
                self, "_real_critic_anchor_ev_ema", None
            ),
            "real_critic_anchor_ev_streak": int(getattr(
                self, "_real_critic_anchor_ev_streak", 0
            )),
            "real_critic_anchor_failure_streak": int(getattr(
                self, "_real_critic_anchor_failure_streak", 0
            )),
            "grounded_agreement_blocked_streak": int(getattr(
                self, "_grounded_agreement_blocked_streak", 0
            )),
            "real_critic_anchor_ready": bool(getattr(
                self, "_real_critic_anchor_ready", False
            )),
            "rollout_critic_ev_ema": self._rollout_critic_ev_ema,
            "rollout_critic_ev_streak": self._rollout_critic_ev_streak,
            "rollout_critic_ready": self._rollout_critic_ready,
            "rollout_critic_released_once": (
                self._rollout_critic_released_once
            ),
            "rollout_critic_post_release_failures": (
                self._rollout_critic_post_release_failures
            ),
            "actor_logit_correction": (
                self.policy.actor_logit_correction()
                if callable(getattr(
                    self.policy, "actor_logit_correction", None
                )) else None
            ),
            "reward_prior_calibration_feasible": bool(getattr(
                self, "_reward_prior_calibration_feasible", False
            )),
            "real_critic_cache_generation": int(getattr(
                self, "_real_critic_cache_generation", 0
            )),
            "wm_semantics": world_model_semantics(
                getattr(backbone, "attention_mode", "unknown")
            ),
        }
        replay_sampler = getattr(self, "replay_sampler", None)
        if replay_sampler is not None and hasattr(
            replay_sampler, "collection_complete"
        ):
            checkpoint["unified_replay"] = {
                "offline_episodes": len(replay_sampler.offline_pool),
                "online_episodes": replay_sampler.online_size,
                "online_target": replay_sampler.online_target,
                "online_fraction": replay_sampler.expected_online_fraction,
                "frozen": replay_sampler.collection_complete,
                "seed": replay_sampler.seed,
            }
        if self.wm_refresher is not None:
            checkpoint["wm_refresher"] = self.wm_refresher.state_dict()
        # Save world model weights for alternating mode resume
        if self.world_model is not None:
            checkpoint["world_model"] = self.world_model.state_dict()
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path} (update {update})", flush=True)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the update number to resume from."""
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        reset_critic = os.environ.get("RESET_RESUME_CRITIC", "0") == "1"
        initial_critic_state = None
        initial_critic_module = None
        if reset_critic:
            q_head = getattr(self.policy, "q_head", None)
            ordered_value_head = getattr(
                self.policy, "ordered_value_head", None
            )
            value_head = getattr(self.policy, "value_head", None)
            value_net = getattr(self.policy, "value_net", None)
            initial_critic_module = (
                q_head or ordered_value_head or value_head or value_net
            )
            if initial_critic_module is None:
                raise RuntimeError(
                    "RESET_RESUME_CRITIC=1 requires a resettable Critic head"
                )
            initial_critic_state = {
                key: value.detach().cpu().clone()
                for key, value in initial_critic_module.state_dict().items()
            }
        from ..model.checkpoint_semantics import validate_world_model_semantics
        backbone = getattr(
            getattr(self.world_model, "transition", None), "backbone", None
        )
        validate_world_model_semantics(
            ckpt,
            attention_mode=getattr(backbone, "attention_mode", "unknown"),
            context=f"Phase-2 checkpoint {path}",
        )
        policy_state = ckpt["policy"]
        # Online-rendered Actor warm-up captures a frozen slotwise behavior
        # reference after pipeline assembly.  On resume, initialization is
        # intentionally skipped, so reconstruct that optional submodule from
        # checkpoint structure before strict state-dict loading.  Silently
        # using strict=False here would discard the reference and change the
        # resumed policy/rehearsal semantics.
        has_behavior_slotwise_state = any(
            key.startswith("behavior_slotwise_head.")
            for key in policy_state
        )
        if (
            has_behavior_slotwise_state
            and getattr(self.policy, "behavior_slotwise_head", None) is None
        ):
            capture = getattr(self.policy, "capture_behavior_reference", None)
            if not callable(capture):
                raise RuntimeError(
                    "Checkpoint contains behavior_slotwise_head tensors, but "
                    "the assembled policy cannot reconstruct that module"
                )
            capture()
            if getattr(self.policy, "behavior_slotwise_head", None) is None:
                raise RuntimeError(
                    "Policy behavior-reference reconstruction did not create "
                    "behavior_slotwise_head"
                )
            print(
                "Resume: reconstructed frozen behavior_slotwise_head before "
                "strict checkpoint loading.",
                flush=True,
            )
        if reset_critic:
            # A reward-semantics experiment may intentionally replace Q(s,a)
            # with a scalar V_pi(s). Restore every compatible Actor/behaviour
            # tensor strictly, while explicitly excluding both the old and new
            # Critic heads. This is safer than a blanket strict=False load,
            # which could silently lose Actor state.
            critic_prefixes = (
                "q_head.", "ordered_value_head.", "value_head.",
                "value_net.", "value_readout."
            )
            actor_state = {
                key: value for key, value in policy_state.items()
                if not key.startswith(critic_prefixes)
            }
            incompatible = self.policy.load_state_dict(actor_state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            noncritic_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith(critic_prefixes)
            ]
            if unexpected or noncritic_missing:
                raise RuntimeError(
                    "Cross-architecture Critic reset failed strict Actor restore: "
                    f"unexpected={unexpected}, noncritic_missing={noncritic_missing}"
                )
        else:
            self.policy.load_state_dict(policy_state)
        if initial_critic_state is not None:
            assert initial_critic_module is not None
            initial_critic_module.load_state_dict(initial_critic_state, strict=True)
            print(
                "Resume: Actor restored strictly; Critic head freshly reset "
                f"for source={getattr(self.policy, 'critic_source', 'unknown')} ",
                flush=True,
            )
        discard_resume_optimizer = (
            os.environ.get("DISCARD_RESUME_OPTIMIZER", "0") == "1"
        )
        if discard_resume_optimizer:
            print(
                "Resume: policy restored; PPO optimizer discarded by "
                "DISCARD_RESUME_OPTIMIZER=1.",
                flush=True,
            )
        else:
            self.ppo_updater.load_state_dict(ckpt["ppo_optimizer"])
            # A released Critic checkpoint intentionally carries the Actor
            # group's warmup-time LR even though that group never stepped.
            # A formal PPO branch may choose a safer Actor LR explicitly;
            # make that override opt-in so ordinary same-run resume continues
            # to preserve scheduler-adjusted optimizer values.
            actor_lr_override = os.environ.get(
                "OVERRIDE_RESUME_ACTOR_LR"
            )
            if actor_lr_override is not None:
                actor_lr = float(actor_lr_override)
                if actor_lr <= 0.0:
                    raise ValueError(
                        "OVERRIDE_RESUME_ACTOR_LR must be positive"
                    )
                actor_groups = [
                    group for group in self.ppo_updater.optimizer.param_groups
                    if group.get("group_name") == "actor"
                ]
                if len(actor_groups) != 1:
                    raise RuntimeError(
                        "OVERRIDE_RESUME_ACTOR_LR requires exactly one named "
                        f"Actor optimizer group, found {len(actor_groups)}"
                    )
                actor_groups[0]["lr"] = actor_lr
                print(
                    "Resume: explicitly set formal PPO Actor LR to "
                    f"{actor_lr:.3e}.",
                    flush=True,
                )
        if self.wm_refresher is not None and "wm_refresher" in ckpt:
            refresher_state = ckpt["wm_refresher"]
            if discard_resume_optimizer:
                # The selected runtime prior can have a different parameter
                # set from the historical checkpoint (for example v2 -> v3
                # adds physics_projection.{weight,bias}). Preserve harmless
                # counters/calibration metadata, but never load Adam moments
                # whose parameter group belongs to the old architecture.
                refresher_state = dict(refresher_state)
                refresher_state.pop("optimizer", None)
                print(
                    "Resume: WM refresher metadata restored; optimizer "
                    "discarded by DISCARD_RESUME_OPTIMIZER=1.",
                    flush=True,
                )
            self.wm_refresher.load_state_dict(refresher_state)
        # Restore world model weights (critical for alternating mode)
        if (
            self.world_model is not None
            and "world_model" in ckpt
            and os.environ.get("KEEP_ASSEMBLED_WORLD_MODEL", "0") != "1"
        ):
            self.world_model.load_state_dict(ckpt["world_model"])
        elif self.world_model is not None and "world_model" in ckpt:
            print(
                "Resume: retained the freshly assembled World Model/Reward "
                "Head instead of restoring the stale PPO checkpoint copy.",
                flush=True,
            )
        self.best_eval_success_rate = float(
            ckpt.get("best_eval_success_rate", float("-inf"))
        )
        self.best_eval_update = int(ckpt.get("best_eval_update", -1))
        self._last_evaluated_actor_update = int(
            ckpt.get("last_evaluated_actor_update", -1)
        )
        self._critic_warmup_complete = bool(
            ckpt.get("critic_warmup_complete", self._critic_warmup_complete)
        )
        self._critic_warmup_ev_streak = int(
            ckpt.get("critic_warmup_ev_streak", 0)
        )
        self._critic_warmup_ev_ema = ckpt.get("critic_warmup_ev_ema")
        self._critic_bucket_ema = ckpt.get("critic_bucket_ema", {})
        self._critic_candidate_saved = bool(
            ckpt.get("critic_candidate_saved", False)
        )
        self._critic_stabilization_lr_applied = bool(
            ckpt.get("critic_stabilization_lr_applied", False)
        )
        if (
            os.environ.get("CRITIC_RELEASE_RESUME", "0") == "1"
            and not self._critic_stabilization_lr_applied
        ):
            lr_factor = float(os.environ.get(
                "CRITIC_STABILIZATION_LR_FACTOR", "0.25"
            ))
            if not 0.0 < lr_factor <= 1.0:
                raise ValueError(
                    "CRITIC_STABILIZATION_LR_FACTOR must be in (0,1]"
                )
            for group in self.ppo_updater.optimizer.param_groups:
                if group.get("group_name") == "critic":
                    group["lr"] *= lr_factor
                    print(
                        "Resume: reduced Critic LR for balanced-bucket "
                        f"stabilization to {group['lr']:.3e}.",
                        flush=True,
                    )
            self._critic_stabilization_lr_applied = True
        self._critic_warmup_updates = int(
            ckpt.get("critic_warmup_updates", 0)
        )
        saved_validation = ckpt.get("critic_warmup_validation")
        saved_replay = ckpt.get("critic_warmup_replay")
        saved_replay_bucket_ids = ckpt.get(
            "critic_warmup_replay_bucket_ids"
        )
        if saved_validation is not None or saved_replay is not None:
            if saved_validation is None or saved_replay is None:
                raise RuntimeError(
                    "Critic warm-up checkpoint contains an incomplete fixed "
                    "panel; validation and replay must be saved together"
                )
            device = next(self.policy.parameters()).device
            self._critic_warmup_validation = self._device_ppo_batch(
                saved_validation, device
            )
            self._critic_warmup_replay = self._device_ppo_batch(
                saved_replay, device
            )
            self._critic_warmup_replay_bucket_ids = (
                None
                if saved_replay_bucket_ids is None
                else saved_replay_bucket_ids.to(device=device)
            )
            saved_validation_bucket_ids = ckpt.get(
                "critic_warmup_validation_bucket_ids"
            )
            saved_validation_bucket_names = ckpt.get(
                "critic_warmup_validation_bucket_names"
            )
            if saved_validation_bucket_ids is None:
                # Version-1 fixed-panel checkpoints did not persist validation
                # bucket metadata. Rebuild it from the deterministic assembled
                # H1/H2 panel instead of silently disabling bucket gates.
                assembled_ids = getattr(
                    self, "counterfactual_h2_validation_bucket_ids", None
                )
                assembled_names = getattr(
                    self, "counterfactual_h2_validation_bucket_names", None
                )
                if assembled_ids is not None and assembled_names:
                    saved_validation_bucket_ids = (
                        assembled_ids.repeat_interleave(4)
                    )
                    saved_validation_bucket_names = list(assembled_names)
            if saved_validation_bucket_ids is not None:
                if len(saved_validation_bucket_ids) != len(
                    self._critic_warmup_validation.actions
                ):
                    raise RuntimeError(
                        "restored Critic validation bucket IDs do not align "
                        "with the fixed validation panel"
                    )
                self._counterfactual_validation_action_bucket_ids = (
                    saved_validation_bucket_ids.to(device=device)
                )
                if not saved_validation_bucket_names:
                    raise RuntimeError(
                        "restored Critic validation bucket IDs lack names"
                    )
                self.counterfactual_h2_validation_bucket_names = list(
                    saved_validation_bucket_names
                )
            print(
                "Resume: restored fixed Critic validation/replay panel "
                f"(validation={len(self._critic_warmup_validation.actions)}, "
                f"replay={len(self._critic_warmup_replay.actions)}).",
                flush=True,
            )
        self._actor_ppo_updates = int(ckpt.get("actor_ppo_updates", 0))
        self._last_wm_refresh_actor_update = int(
            ckpt.get("last_wm_refresh_actor_update", -1)
        )
        self._last_checkpointed_actor_update = int(
            ckpt.get("last_checkpointed_actor_update", -1)
        )
        self._real_critic_anchor_ev_ema = ckpt.get(
            "real_critic_anchor_ev_ema"
        )
        self._real_critic_anchor_ev_streak = int(ckpt.get(
            "real_critic_anchor_ev_streak", 0
        ))
        self._real_critic_anchor_failure_streak = int(ckpt.get(
            "real_critic_anchor_failure_streak", 0
        ))
        self._grounded_agreement_blocked_streak = int(ckpt.get(
            "grounded_agreement_blocked_streak", 0
        ))
        self._real_critic_anchor_ready = bool(ckpt.get(
            "real_critic_anchor_ready", False
        ))
        self._rollout_critic_ev_ema = ckpt.get("rollout_critic_ev_ema")
        self._rollout_critic_ev_streak = int(ckpt.get(
            "rollout_critic_ev_streak", 0
        ))
        self._rollout_critic_ready = bool(ckpt.get(
            "rollout_critic_ready", False
        ))
        self._rollout_critic_released_once = bool(ckpt.get(
            "rollout_critic_released_once", self._rollout_critic_ready
        ))
        self._rollout_critic_post_release_failures = int(ckpt.get(
            "rollout_critic_post_release_failures", 0
        ))
        saved_correction = ckpt.get("actor_logit_correction")
        restore_correction = getattr(
            self.policy, "restore_actor_logit_correction", None
        )
        if saved_correction is not None and callable(restore_correction):
            restore_correction(saved_correction)
        self._real_critic_cache_generation = int(ckpt.get(
            "real_critic_cache_generation", 0
        ))
        if reset_critic:
            self._critic_warmup_complete = False
            self._critic_warmup_ev_streak = 0
            self._critic_warmup_ev_ema = None
            self._critic_bucket_ema = {}
            self._critic_candidate_saved = False
            self._critic_stabilization_lr_applied = False
            self._critic_warmup_updates = 0
            self._actor_ppo_updates = 0
            self._real_critic_anchor_ev_ema = None
            self._real_critic_anchor_ev_streak = 0
            self._real_critic_anchor_failure_streak = 0
            self._real_critic_anchor_ready = False
            self._rollout_critic_ev_ema = None
            self._rollout_critic_ev_streak = 0
            self._rollout_critic_ready = False
            self._rollout_critic_released_once = False
            self._rollout_critic_post_release_failures = 0
        if (
            "reward_prior_calibration_feasible" in ckpt
            and os.environ.get("REWARD_CALIBRATE_AT_START", "0") != "1"
        ):
            self._reward_prior_calibration_feasible = bool(
                ckpt["reward_prior_calibration_feasible"]
            )
        elif os.environ.get("REWARD_CALIBRATE_AT_START", "0") == "1":
            print(
                "Resume: retained fresh Reward calibration instead of stale "
                "checkpoint gate state.",
                flush=True,
            )
        if (
            not self._critic_warmup_complete
            and self._critic_warmup_validation is None
        ):
            # Legacy checkpoints lack the supporting panel. Never carry their
            # EMA into a newly generated validation set: that silently mixes
            # two different release exams.
            self._critic_warmup_ev_streak = 0
            self._critic_warmup_ev_ema = None
            self._critic_bucket_ema = {}
            self._critic_candidate_saved = False
            print(
                "Resume: legacy checkpoint has no fixed Critic panel; reset "
                "all gate EMA/candidate state before rebuilding it.",
                flush=True,
            )
        if os.environ.get("REAL_RETURN_CRITIC_ANCHOR", "0") == "1":
            # Latent replay tensors are deliberately not checkpointed.  A
            # resumed run must rebuild fixed train/validation evidence rather
            # than trusting a gate whose supporting data are absent.
            self._invalidate_real_critic_latent_cache()
        return int(ckpt.get("update", 0))
