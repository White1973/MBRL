"""World model refresh trainer for alternating model-based RL.

Performs supervised world model updates interleaved with PPO policy updates.
Uses the existing Phase1Trainer.train_one_step() as the optimization backend
but exposes a clean, decoupled interface.

Design:
  - Does NOT own the world model — receives it by reference.
  - Does NOT know about PPO, policy, or collectors.
  - Only responsibility: given a data sampler, run N supervised WM updates.
  - Manages its own optimizer/scheduler state for checkpoint/resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from sembelief_wm.config import Config, CurriculumConfig
from sembelief_wm.types import SequenceBatch


@dataclass
class WMRefreshConfig:
    """Configuration for world model refresh cycles."""

    refresh_every: int = 10
    """Refresh WM every N PPO updates. 0 = never refresh (frozen)."""

    updates_per_refresh: int = 5
    """Number of supervised gradient steps per refresh cycle."""

    batch_size: int = 8
    """Batch size (episodes) per supervised step."""

    lr: float = 1e-4
    """Learning rate for WM refresh optimizer."""

    base_lr_factor: float = 0.01
    """Learning-rate multiplier for non-LoRA WM parameters."""

    weight_decay: float = 0.01
    """Weight decay for the independent WM refresh optimizer."""

    warmup_steps: int = 100
    """Number of WM optimizer steps used for linear warmup."""

    grad_clip: float = 1.0
    """Max gradient norm."""

    horizon: int = 8
    """Fixed supervised BPTT horizon used by every refresh step."""

    reward_pos_weight: float | None = None
    reward_loss_coef: float | None = None
    freeze_reward_head: bool = False
    validation_batches: int = 0

    open_dynamics_coef: float = 0.25
    prior_reward_coef: float = 0.5
    open_loop_horizon: int = 4
    open_dynamics_decay: float = 0.9
    prior_reward_decay: float = 1.0
    delta_cosine_coef: float = 0.0
    inverse_action_coef: float = 0.0
    inverse_action_mode: str = "joint"
    inverse_action_lr: float | None = None


def make_phase1_refresh_config(config: Config) -> Config:
    """Build the Phase-1 trainer view used only for Phase-2 WM refresh.

    The world model itself is shared, but its refresh optimizer, scheduler,
    gradient clipping and curriculum must be controlled by ``phase2.wm_refresh``
    rather than by the original Phase-1 training schedule.  Returning a new
    dataclass tree also avoids mutating the config used by the rest of Phase 2.
    """
    refresh = config.phase2.wm_refresh
    if refresh.horizon <= 0:
        raise ValueError("wm_refresh.horizon must be positive")
    if refresh.warmup_steps < 0:
        raise ValueError("wm_refresh.warmup_steps must be non-negative")
    if refresh.grad_clip <= 0.0:
        raise ValueError("wm_refresh.grad_clip must be positive")
    if refresh.reward_pos_weight is not None and refresh.reward_pos_weight <= 0.0:
        raise ValueError("wm_refresh.reward_pos_weight must be positive")
    if refresh.reward_loss_coef is not None and refresh.reward_loss_coef < 0.0:
        raise ValueError("wm_refresh.reward_loss_coef must be non-negative")
    if refresh.validation_batches < 0:
        raise ValueError("wm_refresh.validation_batches must be non-negative")
    if refresh.open_dynamics_coef < 0.0 or refresh.prior_reward_coef < 0.0:
        raise ValueError("WM open-loop loss coefficients must be non-negative")
    if refresh.open_loop_horizon < 0:
        raise ValueError("wm_refresh.open_loop_horizon must be non-negative")
    if not 0.0 < refresh.open_dynamics_decay <= 1.0:
        raise ValueError("wm_refresh.open_dynamics_decay must be in (0, 1]")
    if not 0.0 < refresh.prior_reward_decay <= 1.0:
        raise ValueError("wm_refresh.prior_reward_decay must be in (0, 1]")
    if refresh.delta_cosine_coef < 0.0 or refresh.inverse_action_coef < 0.0:
        raise ValueError("WM action-grounding coefficients must be non-negative")
    if refresh.inverse_action_mode not in {"joint", "prior_frozen"}:
        raise ValueError("wm_refresh.inverse_action_mode must be joint or prior_frozen")
    if refresh.inverse_action_lr is not None and refresh.inverse_action_lr <= 0.0:
        raise ValueError("wm_refresh.inverse_action_lr must be positive")

    training = replace(
        config.training,
        episodes_per_step=refresh.batch_size,
        lr=refresh.lr,
        base_lr_factor=refresh.base_lr_factor,
        weight_decay=refresh.weight_decay,
        warmup_steps=refresh.warmup_steps,
        grad_clip=refresh.grad_clip,
        lambda_reward=(
            config.training.lambda_reward
            if refresh.reward_loss_coef is None
            else refresh.reward_loss_coef
        ),
        open_dynamics_coef=refresh.open_dynamics_coef,
        prior_reward_coef=refresh.prior_reward_coef,
        open_loop_horizon=refresh.open_loop_horizon,
        open_dynamics_decay=refresh.open_dynamics_decay,
        prior_reward_decay=refresh.prior_reward_decay,
        delta_cosine_coef=refresh.delta_cosine_coef,
        inverse_action_coef=refresh.inverse_action_coef,
        inverse_action_mode=refresh.inverse_action_mode,
        inverse_action_lr=refresh.inverse_action_lr,
    )
    curriculum = CurriculumConfig(
        horizons=[refresh.horizon],
        switch_steps=[0],
        horizon_decay=config.curriculum.horizon_decay,
    )
    reward = (
        config.reward
        if refresh.reward_pos_weight is None
        else replace(config.reward, pos_weight=refresh.reward_pos_weight)
    )
    return replace(
        config,
        training=training,
        curriculum=curriculum,
        reward=reward,
    )


@dataclass
class RefreshMetrics:
    """Metrics from one refresh cycle."""

    num_steps: int
    avg_dynamics_loss: float
    avg_reward_loss: float
    avg_total_loss: float
    refresh_step: int
    diagnostics: dict[str, float] = field(default_factory=dict)


class WorldModelRefresher:
    """Runs supervised WM updates using Phase1Trainer as backend.

    This is a thin wrapper that:
      1. Toggles requires_grad on the world model
      2. Calls Phase1Trainer.train_one_step() for N steps
      3. Restores eval mode when done

    The Phase1Trainer is created externally and passed in — this class
    does NOT construct optimizers or schedulers. It just drives the
    existing training infrastructure.
    """

    def __init__(
        self,
        phase1_trainer: Any,
        config: WMRefreshConfig,
        validation_fn: Callable[[int], dict[str, float]] | None = None,
    ) -> None:
        """
        Args:
            phase1_trainer: An instance of sembelief_wm.train.trainer.Phase1Trainer.
                           We use Any to avoid importing it (keeps this module
                           decoupled from the full training infrastructure).
            config: Refresh configuration.
        """
        self._trainer = phase1_trainer
        self._config = config
        self._validation_fn = validation_fn
        self._refresh_step = 0
        world_model = phase1_trainer.world_model
        named_parameters = getattr(world_model, "named_parameters", None)
        self._supervised_param_names: tuple[str, ...] | None = None
        if callable(named_parameters):
            # Capture the Phase-1 trainability mask before the Phase-2 assembly
            # freezes the WM.  In a PEFT model this contains LoRA parameters and
            # the small WM heads, while the Qwen base weights remain excluded.
            self._supervised_param_names = tuple(
                name
                for name, param in named_parameters()
                if param.requires_grad
                and not (
                    config.freeze_reward_head
                    and name.startswith("reward_head.")
                )
            )
        self._validation_baseline = (
            validation_fn(0) if validation_fn is not None else {}
        )
        if validation_fn is not None:
            world_model.requires_grad_(False)
            world_model.eval()

    @property
    def refresh_step(self) -> int:
        return self._refresh_step

    def evaluate_validation(self) -> dict[str, float]:
        """Evaluate the current shared WM on the fixed held-out replay.

        This is intentionally independent of ``refresh()`` so semantic probes
        can evaluate a resumed checkpoint without taking an optimizer step.
        """
        if self._validation_fn is None:
            return {}
        wm = self._trainer.world_model
        ownership = {
            name: parameter.requires_grad
            for name, parameter in wm.named_parameters()
        }
        try:
            metrics = self._validation_fn(self._refresh_step)
        finally:
            current = dict(wm.named_parameters())
            for name, requires_grad in ownership.items():
                current[name].requires_grad_(requires_grad)
            wm.eval()
        return {
            (key[4:] if key.startswith("val/") else key): float(value)
            for key, value in metrics.items()
        }

    def should_refresh(self, ppo_update: int) -> bool:
        """Whether to refresh WM at this PPO update step."""
        if self._config.refresh_every <= 0:
            return False
        return ppo_update > 0 and ppo_update % self._config.refresh_every == 0

    def refresh(
        self,
        sample_fn: callable,
    ) -> RefreshMetrics:
        """Run one refresh cycle of supervised WM updates.

        Args:
            sample_fn: Callable that returns a SequenceBatch when called
                       with (batch_size: int). This decouples the refresher
                       from the specific data source (offline, online, mixed).

        Returns:
            RefreshMetrics with averaged losses over the cycle.
        """
        wm = self._trainer.world_model
        outside_refresh_ownership = {
            name: parameter.requires_grad
            for name, parameter in wm.named_parameters()
        }

        # Restore the original Phase-1 trainability mask.  A blanket
        # ``requires_grad_(True)`` would also unfreeze the PEFT/Qwen base model
        # and turn an intended LoRA refresh into full-model fine-tuning.
        wm.requires_grad_(False)
        if self._supervised_param_names is None:
            # Compatibility path for lightweight protocol fakes.
            wm.requires_grad_(True)
        else:
            current_parameters = dict(wm.named_parameters())
            missing = [
                name for name in self._supervised_param_names
                if name not in current_parameters
            ]
            if missing:
                raise RuntimeError(
                    "World-model parameter set changed after refresher "
                    f"initialization; missing {missing[:3]}"
                )
            for name in self._supervised_param_names:
                current_parameters[name].requires_grad_(True)
        try:
            totals: dict[str, float] = {}
            for _ in range(self._config.updates_per_refresh):
                batch: SequenceBatch = sample_fn(self._config.batch_size)
                step_metrics = self._trainer.train_one_step(
                    global_step=self._refresh_step,
                    batch=batch,
                )
                metrics_dict = step_metrics.as_dict()
                self._refresh_step += 1
                for key, value in metrics_dict.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
        finally:
            # The PEFT container also owns PPO's actor/critic LoRA adapters.
            # A blanket freeze here would silently disable PPO after the first
            # alternating refresh. Restore the exact pre-refresh ownership;
            # Phase-1/WM parameters were already frozen by pipeline assembly,
            # while PPO-owned LoRAs remain trainable.
            current_parameters = dict(wm.named_parameters())
            for name, requires_grad in outside_refresh_ownership.items():
                current_parameters[name].requires_grad_(requires_grad)
            wm.eval()

        n = max(1, self._config.updates_per_refresh)

        def avg(key: str) -> float:
            return totals.get(key, 0.0) / n

        positive_count = totals.get("metric/reward_positive_count", 0.0)
        valid_count = totals.get("metric/reward_valid_count", 0.0)
        tp_post = totals.get("metric/reward_true_positive_count_post", 0.0)
        tp_pri = totals.get("metric/reward_true_positive_count_pri", 0.0)
        predicted_post = totals.get(
            "metric/reward_predicted_positive_count_post", 0.0
        )
        predicted_pri = totals.get(
            "metric/reward_predicted_positive_count_pri", 0.0
        )
        tpr_post = tp_post / max(positive_count, 1.0)
        tpr_pri = tp_pri / max(positive_count, 1.0)
        precision_post = tp_post / max(predicted_post, 1.0)
        precision_pri = tp_pri / max(predicted_pri, 1.0)

        def f1(precision: float, recall: float) -> float:
            return 0.0 if precision + recall == 0.0 else (
                2.0 * precision * recall / (precision + recall)
            )

        diagnostics = {
            # Reward-head classification quality. Counts are summed over every
            # supervised minibatch in this refresh; rates are therefore
            # micro-averaged and remain meaningful with sparse positives.
            "reward/tpr_post": tpr_post,
            "reward/tpr_prior": tpr_pri,
            "reward/precision_post": precision_post,
            "reward/precision_prior": precision_pri,
            "reward/f1_post": f1(precision_post, tpr_post),
            "reward/f1_prior": f1(precision_pri, tpr_pri),
            "reward/accuracy_post": avg("metric/reward_acc_post"),
            "reward/accuracy_prior": avg("metric/reward_acc_pri"),
            "reward/auroc_post": avg("metric/reward_auroc_post"),
            "reward/auroc_prior": avg("metric/reward_auroc_pri"),
            "reward/brier_post": avg("metric/reward_brier_post"),
            "reward/brier_prior": avg("metric/reward_brier_pri"),
            "reward/positive_rate": (
                positive_count / max(valid_count, 1.0)
            ),
            "reward/positive_count": positive_count,
            "reward/valid_count": valid_count,
            "reward/true_positive_count_post": tp_post,
            "reward/true_positive_count_prior": tp_pri,
            "reward/predicted_positive_count_post": predicted_post,
            "reward/predicted_positive_count_prior": predicted_pri,
            "reward/logit_mean_post": avg("metric/reward_logit_mean_post"),
            "reward/logit_mean_prior": avg("metric/reward_logit_mean_pri"),
            "reward/logit_gap": avg("metric/reward_logit_gap"),
            # Dynamics and representation diagnostics make it possible to see
            # whether online refresh improves reward prediction by collapsing
            # or degrading the latent transition model.
            "dynamics/cosine_similarity": avg("metric/cosine_sim"),
            "dynamics/posterior_prior_l2": avg("metric/posterior_prior_l2"),
            "open_loop/dynamics_loss": avg("loss/open_dynamics"),
            "open_loop/prior_reward_loss": avg("loss/open_prior_reward"),
            "open_loop/horizon": avg("open_loop/horizon"),
            "open_loop/dynamics_valid_count": avg(
                "open_loop/dynamics_valid_count"
            ),
            "open_loop/reward_valid_count": avg(
                "open_loop/reward_valid_count"
            ),
            "open_loop/prior_reward_accuracy": avg(
                "open_loop/prior_reward_accuracy"
            ),
            "open_loop/prior_reward_positive_rate": avg(
                "open_loop/prior_reward_positive_rate"
            ),
            "belief/effective_rank": avg("metric/effective_rank"),
            "belief/std_mean": avg("metric/belief_std_mean"),
            "belief/std_min": avg("metric/belief_std_min"),
            "optimization/grad_norm": avg("metric/grad_norm"),
            "optimization/lr": float(
                self._trainer.scheduler.get_last_lr()[0]
            ),
            "optimization/grad_clip": float(self._config.grad_clip),
            "curriculum/horizon": avg("curriculum/horizon"),
        }
        if self._validation_fn is not None:
            validation_ownership = {
                name: parameter.requires_grad
                for name, parameter in wm.named_parameters()
            }
            try:
                validation_metrics = self._validation_fn(self._refresh_step)
            finally:
                current_parameters = dict(wm.named_parameters())
                for name, requires_grad in validation_ownership.items():
                    current_parameters[name].requires_grad_(requires_grad)
            for key, value in self._validation_baseline.items():
                normalized_key = key[4:] if key.startswith("val/") else key
                diagnostics[f"validation_baseline/{normalized_key}"] = float(value)
            for key, value in validation_metrics.items():
                normalized_key = key[4:] if key.startswith("val/") else key
                diagnostics[f"validation/{normalized_key}"] = float(value)
                baseline_value = self._validation_baseline.get(key)
                if baseline_value is not None:
                    diagnostics[f"validation_delta/{normalized_key}"] = (
                        float(value) - float(baseline_value)
                    )
            # Phase1Trainer.evaluate() restores train mode for Phase 1. During
            # PPO imagination the shared WM must remain frozen and in eval mode.
            wm.eval()
        return RefreshMetrics(
            num_steps=n,
            avg_dynamics_loss=avg("loss/dynamics"),
            avg_reward_loss=avg("loss/reward"),
            avg_total_loss=avg("loss/total"),
            refresh_step=self._refresh_step,
            diagnostics=diagnostics,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "refresh_step": self._refresh_step,
            "optimizer": self._trainer.optimizer.state_dict(),
            "scheduler": self._trainer.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._refresh_step = int(state_dict.get("refresh_step", 0))
        opt_state = state_dict.get("optimizer")
        if opt_state is not None:
            self._trainer.optimizer.load_state_dict(opt_state)
        sched_state = state_dict.get("scheduler")
        if sched_state is not None:
            self._trainer.scheduler.load_state_dict(sched_state)
