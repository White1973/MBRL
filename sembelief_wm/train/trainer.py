"""Minimal Phase 1 trainer for SemBelief-WM."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from ..config import Config
from .curriculum import CurriculumManager
from .losses import covariance_loss
from .sigreg import SIGRegTerms
from ..types import BeliefState, SequenceBatch
from .windowing import num_windows, slice_training_window
from ..model.world_model import Phase1Outputs, WorldModel


class DataSource(Protocol):
    """Episode-batch source used by the trainer."""

    def sample_batch(self, batch_size: int) -> SequenceBatch:
        """Return one batch of episode sequences."""


class Logger(Protocol):
    """Minimal scalar logger interface."""

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        """Log scalar metrics for one optimization step."""


@dataclass(frozen=True)
class TrainStepMetrics:
    """Aggregated metrics for one optimizer step."""

    total: float
    dynamics: float
    reward: float
    open_dynamics: float
    prior_reward: float
    sigreg_ep: float
    sigreg_var: float
    horizon: int
    num_windows: int
    # --- diagnostic metrics ---
    cosine_sim: float = 0.0
    posterior_prior_l2: float = 0.0
    grad_norm: float = 0.0
    effective_rank: float = 0.0
    belief_std_mean: float = 0.0
    belief_std_min: float = 0.0
    belief_norm_mean: float = 0.0
    posterior_reward_acc: float = 0.0
    prior_reward_acc: float = 0.0
    posterior_reward_recall: float = 0.0
    prior_reward_recall: float = 0.0
    posterior_reward_precision: float = 0.0
    prior_reward_precision: float = 0.0
    posterior_reward_f1: float = 0.0
    prior_reward_f1: float = 0.0
    posterior_reward_auroc: float = 0.0
    prior_reward_auroc: float = 0.0
    posterior_reward_brier: float = 0.0
    prior_reward_brier: float = 0.0
    valid_transitions: float = 0.0
    # --- P0 recovery monitoring ---
    sigreg_num_samples: float = 0.0
    sigreg_ep_unscaled: float = 0.0
    sigreg_var_unscaled: float = 0.0
    sigreg_cov_unscaled: float = 0.0
    posterior_valid_count: float = 0.0
    # --- Reward diagnostics ---
    reward_positive_rate: float = 0.0
    reward_logit_mean_post: float = 0.0
    reward_logit_mean_pri: float = 0.0
    reward_logit_gap: float = 0.0
    reward_post_minus_pri_on_positive: float = 0.0
    reward_valid_count: float = 0.0
    reward_positive_count: float = 0.0
    posterior_reward_true_positive_count: float = 0.0
    prior_reward_true_positive_count: float = 0.0
    posterior_reward_predicted_positive_count: float = 0.0
    prior_reward_predicted_positive_count: float = 0.0
    open_loop_horizon: float = 0.0
    open_dynamics_valid_count: float = 0.0
    open_reward_valid_count: float = 0.0
    open_prior_reward_acc: float = 0.0
    open_prior_reward_positive_rate: float = 0.0
    delta_cosine_loss: float = 0.0
    inverse_action_loss: float = 0.0
    inverse_action_acc_prior: float = 0.0
    inverse_action_acc_post: float = 0.0
    action_aux_valid_count: float = 0.0
    observation_anchor_loss: float = 0.0
    observation_delta_anchor_loss: float = 0.0
    observation_anchor_valid_count: float = 0.0
    observation_delta_anchor_valid_count: float = 0.0
    vjepa_teacher_prior_loss: float = 0.0
    vjepa_teacher_posterior_loss: float = 0.0
    vjepa_teacher_delta_loss: float = 0.0
    vjepa_teacher_prior_valid_count: float = 0.0
    vjepa_teacher_posterior_valid_count: float = 0.0
    vjepa_teacher_delta_valid_count: float = 0.0
    wm_lora_grad_norm: float = 0.0
    transition_aux_grad_norm: float = 0.0
    reward_head_grad_norm: float = 0.0
    observation_anchor_grad_norm: float = 0.0
    vjepa_teacher_grad_norm: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total,
            "loss/dynamics": self.dynamics,
            "loss/reward": self.reward,
            "loss/open_dynamics": self.open_dynamics,
            "loss/open_prior_reward": self.prior_reward,
            "loss/sigreg_ep": self.sigreg_ep,
            "loss/sigreg_var": self.sigreg_var,
            "curriculum/horizon": float(self.horizon),
            "train/num_windows": float(self.num_windows),
            "metric/cosine_sim": self.cosine_sim,
            "metric/posterior_prior_l2": self.posterior_prior_l2,
            "metric/grad_norm": self.grad_norm,
            "metric/grad_norm_wm_lora": self.wm_lora_grad_norm,
            "metric/grad_norm_transition_aux": self.transition_aux_grad_norm,
            "metric/grad_norm_reward_head": self.reward_head_grad_norm,
            "metric/grad_norm_observation_anchor": self.observation_anchor_grad_norm,
            "metric/grad_norm_vjepa_teacher": self.vjepa_teacher_grad_norm,
            "metric/effective_rank": self.effective_rank,
            "metric/belief_std_mean": self.belief_std_mean,
            "metric/belief_std_min": self.belief_std_min,
            "metric/belief_norm_mean": self.belief_norm_mean,
            "metric/reward_acc_post": self.posterior_reward_acc,
            "metric/reward_acc_pri": self.prior_reward_acc,
            "metric/reward_recall_post": self.posterior_reward_recall,
            "metric/reward_recall_pri": self.prior_reward_recall,
            "metric/reward_precision_post": self.posterior_reward_precision,
            "metric/reward_precision_pri": self.prior_reward_precision,
            "metric/reward_f1_post": self.posterior_reward_f1,
            "metric/reward_f1_pri": self.prior_reward_f1,
            "metric/reward_auroc_post": self.posterior_reward_auroc,
            "metric/reward_auroc_pri": self.prior_reward_auroc,
            "metric/reward_brier_post": self.posterior_reward_brier,
            "metric/reward_brier_pri": self.prior_reward_brier,
            "metric/valid_transitions": self.valid_transitions,
            "sigreg/num_samples": self.sigreg_num_samples,
            "sigreg/ep_unscaled": self.sigreg_ep_unscaled,
            "sigreg/var_unscaled": self.sigreg_var_unscaled,
            "sigreg/cov_unscaled": self.sigreg_cov_unscaled,
            "metric/posterior_valid_count": self.posterior_valid_count,
            "metric/reward_positive_rate": self.reward_positive_rate,
            "metric/reward_logit_mean_post": self.reward_logit_mean_post,
            "metric/reward_logit_mean_pri": self.reward_logit_mean_pri,
            "metric/reward_logit_gap": self.reward_logit_gap,
            "metric/reward_post_minus_pri_on_positive": self.reward_post_minus_pri_on_positive,
            "metric/reward_valid_count": self.reward_valid_count,
            "metric/reward_positive_count": self.reward_positive_count,
            "metric/reward_true_positive_count_post": self.posterior_reward_true_positive_count,
            "metric/reward_true_positive_count_pri": self.prior_reward_true_positive_count,
            "metric/reward_predicted_positive_count_post": self.posterior_reward_predicted_positive_count,
            "metric/reward_predicted_positive_count_pri": self.prior_reward_predicted_positive_count,
            "open_loop/horizon": self.open_loop_horizon,
            "open_loop/dynamics_valid_count": self.open_dynamics_valid_count,
            "open_loop/reward_valid_count": self.open_reward_valid_count,
            "open_loop/prior_reward_accuracy": self.open_prior_reward_acc,
            "open_loop/prior_reward_positive_rate": self.open_prior_reward_positive_rate,
            "loss/delta_cosine": self.delta_cosine_loss,
            "loss/inverse_action": self.inverse_action_loss,
            "metric/inverse_action_acc_prior": self.inverse_action_acc_prior,
            "metric/inverse_action_acc_post": self.inverse_action_acc_post,
            "metric/action_aux_valid_count": self.action_aux_valid_count,
            "loss/observation_anchor": self.observation_anchor_loss,
            "loss/observation_delta_anchor": self.observation_delta_anchor_loss,
            "metric/observation_anchor_valid_count": self.observation_anchor_valid_count,
            "metric/observation_delta_anchor_valid_count": self.observation_delta_anchor_valid_count,
            "loss/vjepa_teacher_prior": self.vjepa_teacher_prior_loss,
            "loss/vjepa_teacher_posterior": self.vjepa_teacher_posterior_loss,
            "loss/vjepa_teacher_delta": self.vjepa_teacher_delta_loss,
            "metric/vjepa_teacher_prior_valid_count": self.vjepa_teacher_prior_valid_count,
            "metric/vjepa_teacher_posterior_valid_count": self.vjepa_teacher_posterior_valid_count,
            "metric/vjepa_teacher_delta_valid_count": self.vjepa_teacher_delta_valid_count,
        }


class Phase1Trainer:
    """Thin orchestration layer for Phase 1 world-model training."""

    def __init__(
        self,
        *,
        config: Config,
        world_model: WorldModel,
        data_source: DataSource,
        device: torch.device | str,
        logger: Logger | None = None,
    ) -> None:
        self.config = config
        self.world_model = world_model
        # The trainer configuration is authoritative for loss construction.
        # This matters in Phase 2, where a shared WorldModel is optimized by a
        # trainer-specific refresh view (open-loop coefficients, horizon, and
        # optimizer settings) rather than by its original Phase-1 config.
        # Model architecture has already been constructed, so replacing this
        # dataclass reference changes objective/runtime settings only.
        self.world_model.config = config
        self.data_source = data_source
        self.logger = logger
        self.device = torch.device(device)
        self.curriculum = CurriculumManager(config)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.world_model.to(self.device)
        self._wandb_run = None
        self._best_validation_dynamics: float | None = None
        self._validation_guard_failures = 0

    def _sigreg_components(self, terms: SIGRegTerms) -> tuple[Tensor, Tensor]:
        """Return the EP/variance terms that actually enter the objective."""
        mode = getattr(self.config.training, "sigreg_scale_mode", "n_scaled")
        if mode == "mean":
            return terms.ep_unscaled, terms.var_unscaled
        if mode == "n_scaled":
            return terms.ep, terms.var
        raise ValueError(f"Unsupported SIGReg scale mode: {mode}")

    def _validation_guard(self, metrics: dict[str, float]) -> str | None:
        """Track persistent WM validation failures and return a stop reason."""
        tc = self.config.training
        if not getattr(tc, "validation_guard_enabled", False):
            return None

        valid = metrics.get("val/metric/reward_valid_count", 0.0)
        predicted = metrics.get(
            "val/metric/reward_predicted_positive_count_post", 0.0
        )
        positive_rate = metrics.get("val/metric/reward_positive_rate", 0.0)
        predicted_rate = predicted / max(valid, 1.0)
        brier = metrics.get("val/metric/reward_brier_post", 0.0)
        constant_brier = positive_rate * (1.0 - positive_rate)
        collapsed_reward = (
            valid > 0
            and predicted_rate
            >= getattr(tc, "validation_reward_predicted_positive_max", 0.98)
            and brier > constant_brier
        )

        dynamics = metrics.get("val/loss/dynamics")
        degraded_dynamics = False
        if dynamics is not None:
            if self._best_validation_dynamics is None:
                self._best_validation_dynamics = dynamics
            else:
                factor = getattr(
                    tc, "validation_dynamics_degradation_factor", 2.0
                )
                degraded_dynamics = dynamics > factor * max(
                    self._best_validation_dynamics, 1e-12
                )
                self._best_validation_dynamics = min(
                    self._best_validation_dynamics, dynamics
                )

        failed = collapsed_reward or degraded_dynamics
        self._validation_guard_failures = (
            self._validation_guard_failures + 1 if failed else 0
        )
        metrics["val/guard/predicted_positive_rate"] = predicted_rate
        metrics["val/guard/constant_brier"] = constant_brier
        metrics["val/guard/reward_collapsed"] = float(collapsed_reward)
        metrics["val/guard/dynamics_degraded"] = float(degraded_dynamics)
        metrics["val/guard/consecutive_failures"] = float(
            self._validation_guard_failures
        )

        patience = max(1, getattr(tc, "validation_guard_patience", 2))
        if self._validation_guard_failures < patience:
            return None
        reasons = []
        if collapsed_reward:
            reasons.append(
                f"reward predicted-positive={predicted_rate:.3f}, "
                f"Brier={brier:.4f}>constant={constant_brier:.4f}"
            )
        if degraded_dynamics:
            reasons.append(
                f"dynamics={dynamics:.6f}>"
                f"{getattr(tc, 'validation_dynamics_degradation_factor', 2.0):.1f}x best"
            )
        return "; ".join(reasons) or "validation quality guard failed"

    def _build_optimizer(self) -> torch.optim.AdamW:
        """Use full WM LR for adapters/heads and a reduced LR for Qwen base.

        ``base_lr_factor`` is specifically a safeguard for full VLM weights;
        applying it to every non-LoRA tensor would also slow the reward head
        and small transition modules by 100x, defeating supervised refresh.
        """
        tc = self.config.training
        adaptation_params = []
        vlm_base_params = []
        inverse_action_params = []
        for name, param in self.world_model.named_parameters():
            if not param.requires_grad:
                continue
            lower_name = name.lower()
            if lower_name.startswith("inverse_action_head."):
                inverse_action_params.append(param)
                continue
            is_vlm_backbone = lower_name.startswith("transition.backbone.")
            if is_vlm_backbone and "lora" not in lower_name:
                vlm_base_params.append(param)
            else:
                # WM LoRA plus reward/transition/readout parameters all need
                # the configured supervised learning rate.
                adaptation_params.append(param)

        param_groups = []
        if inverse_action_params:
            param_groups.append({
                "params": inverse_action_params,
                "lr": tc.inverse_action_lr or tc.lr,
            })
        if adaptation_params:
            param_groups.append({"params": adaptation_params, "lr": tc.lr})
        if vlm_base_params:
            param_groups.append({
                "params": vlm_base_params,
                "lr": tc.lr * tc.base_lr_factor,
            })
        if not param_groups:
            raise ValueError("World model has no trainable parameters")

        return torch.optim.AdamW(
            param_groups,
            weight_decay=tc.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup scheduler matching main branch."""
        configured_warmup = int(self.config.training.warmup_steps)
        if configured_warmup <= 0:
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda _step: 1.0,
            )
        warmup = configured_warmup
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup),
        )

    def _init_wandb(self) -> None:
        """Lazy-init wandb if enabled."""
        if self._wandb_run is not None:
            return
        wc = self.config.wandb
        if not wc.enabled:
            return
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=wc.project,
                entity=wc.entity,
                name=wc.run_name,
                config=asdict(self.config),
            )
        except ImportError:
            pass

    def train(
        self,
        *,
        start_step: int = 0,
        checkpoint_dir: str | Path | None = None,
        val_source: DataSource | None = None,
        val_every: int = 100,
        val_batches: int = 5,
    ) -> None:
        """Run the Phase 1 training loop."""

        self._init_wandb()

        checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)

        for global_step in range(start_step, self.config.training.total_steps):
            did_switch = global_step > 0 and self.curriculum.did_switch(
                global_step - 1, global_step
            )
            if did_switch and self.config.sigreg.flush_on_curriculum_switch:
                self.world_model.flush_sigreg_buffer()
            if did_switch:
                # Dynamics losses from different rollout horizons are not
                # directly comparable; establish a fresh held-out baseline
                # for each curriculum stage.
                self._best_validation_dynamics = None
                self._validation_guard_failures = 0

            batch = self.data_source.sample_batch(self.config.training.episodes_per_step)
            metrics = self.train_one_step(global_step=global_step, batch=batch)
            if self.logger is not None:
                self.logger.log_scalars(global_step, metrics.as_dict())
            self._log_wandb(global_step, metrics)

            stop_reason = None
            # Periodic validation
            if val_source is not None and (global_step + 1) % val_every == 0:
                val_metrics = self.evaluate(
                    val_source=val_source,
                    num_batches=val_batches,
                    global_step=global_step,
                )
                stop_reason = self._validation_guard(val_metrics)
                if self.logger is not None:
                    self.logger.log_scalars(global_step, val_metrics)
                self._log_wandb_dict(global_step, val_metrics)

            if checkpoint_root is not None and self._should_save_checkpoint(global_step + 1):
                checkpoint_path = self._checkpoint_path(
                    checkpoint_root, step=global_step + 1
                )
                self.save_checkpoint(checkpoint_path, step=global_step + 1)
                self._update_latest_checkpoint(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}", flush=True)
            if stop_reason is not None:
                if checkpoint_root is not None:
                    emergency = checkpoint_root / "validation_guard_stop.pt"
                    self.save_checkpoint(emergency, step=global_step + 1)
                    print(
                        f"Validation guard checkpoint saved: {emergency}",
                        flush=True,
                    )
                raise RuntimeError(
                    "Phase-1 validation quality guard stopped training after "
                    f"step {global_step + 1}: {stop_reason}"
                )

    def train_one_step(
        self,
        *,
        global_step: int,
        batch: SequenceBatch,
    ) -> TrainStepMetrics:
        """Run one optimizer step with step-level SIGReg and single backward.

        Key design (matching main branch semantics):
        - Supervision losses (dynamics + reward) are accumulated across all
          windows, then divided by total_valid_transitions.
        - Regularization (SIGReg / EMA variance) is computed ONCE on all
          valid posterior beliefs collected across all windows (step-level).
        - Total loss = supervision_mean + regularization (NOT reg / total_valid).
        - Single backward() call at step end.
        - SIGReg buffer updated once at step end (for monitoring only).
        """

        if self.config.training.isolated_prior_repair:
            # Gradients do not require train() mode.  Keep the frozen released
            # posterior/default adapter deterministic while optimizing only
            # the independent prior LoRA (whose dropout is configured to 0).
            self.world_model.eval()
        else:
            self.world_model.train()
        batch = self._move_sequence_batch(batch)
        horizon = self.curriculum.horizon_at(global_step)
        total_windows = num_windows(batch, horizon)

        self.optimizer.zero_grad(set_to_none=True)

        prev_belief: BeliefState = self.world_model.get_initial_belief(
            batch.batch_size,
            dtype=batch.obs_tokens.dtype,
        )
        window_outputs: list[Phase1Outputs] = []
        all_valid_beliefs: list[Tensor] = []  # WITH gradients
        total_sup_sum: Tensor | None = None
        total_dyn_sum: Tensor | None = None
        total_rew_sum: Tensor | None = None
        total_open_dyn_sum: Tensor | None = None
        total_prior_rew_sum: Tensor | None = None
        total_open_dyn_weight: Tensor | None = None
        total_prior_rew_weight: Tensor | None = None
        total_transition_count = 0.0
        total_delta_cosine_sum: Tensor | None = None
        total_inverse_action_sum: Tensor | None = None
        total_action_aux_weight: Tensor | None = None
        total_inverse_prior_correct: Tensor | None = None
        total_inverse_post_correct: Tensor | None = None
        total_observation_anchor_sum: Tensor | None = None
        total_observation_anchor_weight: Tensor | None = None
        total_observation_delta_sum: Tensor | None = None
        total_observation_delta_weight: Tensor | None = None
        total_vjepa_prior_sum: Tensor | None = None
        total_vjepa_prior_weight: Tensor | None = None
        total_vjepa_posterior_sum: Tensor | None = None
        total_vjepa_posterior_weight: Tensor | None = None
        total_vjepa_delta_sum: Tensor | None = None
        total_vjepa_delta_weight: Tensor | None = None

        lam_rew = self.config.training.lambda_reward

        for window_index in range(total_windows):
            window = slice_training_window(
                batch,
                window_index=window_index,
                horizon=horizon,
                config=self.config,
                prev_belief=prev_belief,
            )
            outputs = self.world_model.compute_phase1_outputs(window)

            starts = torch.cat(
                [
                    window.prev_belief.slots.unsqueeze(1),
                    outputs.rollout.posterior.beliefs[:, :-1],
                ],
                dim=1,
            )
            transition_actions = torch.cat(
                [window.prev_actions.unsqueeze(1), window.actions[:, :-1]],
                dim=1,
            )
            action_aux = self.world_model.compute_action_auxiliaries(
                start_beliefs=starts,
                prior_beliefs=outputs.rollout.prior.beliefs,
                posterior_beliefs=outputs.rollout.posterior.beliefs,
                actions=transition_actions,
                mask=outputs.dynamics_mask,
            )
            observation_anchor = self.world_model.compute_observation_anchor(
                posterior_beliefs=outputs.rollout.posterior.beliefs,
                observation_tokens=window.obs_tokens,
                posterior_mask=outputs.posterior_valid_mask,
            )
            vjepa_teacher = self.world_model.compute_vjepa_teacher_losses(
                prior_beliefs=outputs.rollout.prior.beliefs,
                posterior_beliefs=outputs.rollout.posterior.beliefs,
                start_beliefs=starts,
                teacher_tokens=window.semantic_teacher_tokens,
                teacher_mask=window.semantic_teacher_mask,
                prev_teacher_tokens=window.prev_semantic_teacher_tokens,
                prev_teacher_mask=window.prev_semantic_teacher_mask,
                dynamics_mask=outputs.dynamics_mask,
                posterior_mask=outputs.posterior_valid_mask,
            )
            total_observation_anchor_sum = (
                observation_anchor.state_sum
                if total_observation_anchor_sum is None
                else total_observation_anchor_sum + observation_anchor.state_sum
            )
            total_observation_anchor_weight = (
                observation_anchor.state_weight
                if total_observation_anchor_weight is None
                else total_observation_anchor_weight + observation_anchor.state_weight
            )
            total_observation_delta_sum = (
                observation_anchor.delta_sum
                if total_observation_delta_sum is None
                else total_observation_delta_sum + observation_anchor.delta_sum
            )
            total_observation_delta_weight = (
                observation_anchor.delta_weight
                if total_observation_delta_weight is None
                else total_observation_delta_weight + observation_anchor.delta_weight
            )
            total_vjepa_prior_sum = (
                vjepa_teacher.prior_sum
                if total_vjepa_prior_sum is None
                else total_vjepa_prior_sum + vjepa_teacher.prior_sum
            )
            total_vjepa_prior_weight = (
                vjepa_teacher.prior_weight
                if total_vjepa_prior_weight is None
                else total_vjepa_prior_weight + vjepa_teacher.prior_weight
            )
            total_vjepa_posterior_sum = (
                vjepa_teacher.posterior_sum
                if total_vjepa_posterior_sum is None
                else total_vjepa_posterior_sum + vjepa_teacher.posterior_sum
            )
            total_vjepa_posterior_weight = (
                vjepa_teacher.posterior_weight
                if total_vjepa_posterior_weight is None
                else total_vjepa_posterior_weight + vjepa_teacher.posterior_weight
            )
            total_vjepa_delta_sum = (
                vjepa_teacher.delta_sum
                if total_vjepa_delta_sum is None
                else total_vjepa_delta_sum + vjepa_teacher.delta_sum
            )
            total_vjepa_delta_weight = (
                vjepa_teacher.delta_weight
                if total_vjepa_delta_weight is None
                else total_vjepa_delta_weight + vjepa_teacher.delta_weight
            )
            total_delta_cosine_sum = (
                action_aux.delta_cosine_sum if total_delta_cosine_sum is None
                else total_delta_cosine_sum + action_aux.delta_cosine_sum
            )
            total_inverse_action_sum = (
                action_aux.inverse_action_sum if total_inverse_action_sum is None
                else total_inverse_action_sum + action_aux.inverse_action_sum
            )
            total_action_aux_weight = (
                action_aux.weight_sum if total_action_aux_weight is None
                else total_action_aux_weight + action_aux.weight_sum
            )
            total_inverse_prior_correct = (
                action_aux.prior_correct if total_inverse_prior_correct is None
                else total_inverse_prior_correct + action_aux.prior_correct
            )
            total_inverse_post_correct = (
                action_aux.posterior_correct if total_inverse_post_correct is None
                else total_inverse_post_correct + action_aux.posterior_correct
            )

            # Accumulate supervision sums (don't backward yet)
            dyn_sum = outputs.losses.dynamics_sum
            rew_sum = outputs.losses.reward_sum
            sup_sum = dyn_sum + lam_rew * rew_sum
            total_sup_sum = sup_sum if total_sup_sum is None else total_sup_sum + sup_sum
            total_dyn_sum = dyn_sum if total_dyn_sum is None else total_dyn_sum + dyn_sum
            total_rew_sum = rew_sum if total_rew_sum is None else total_rew_sum + rew_sum
            open_dyn_sum = outputs.losses.open_dynamics_sum
            prior_rew_sum = outputs.losses.prior_reward_sum
            open_dyn_weight = outputs.losses.open_dynamics_weight_sum
            prior_rew_weight = outputs.losses.prior_reward_weight_sum
            total_open_dyn_sum = (
                open_dyn_sum if total_open_dyn_sum is None
                else total_open_dyn_sum + open_dyn_sum
            )
            total_prior_rew_sum = (
                prior_rew_sum if total_prior_rew_sum is None
                else total_prior_rew_sum + prior_rew_sum
            )
            total_open_dyn_weight = (
                open_dyn_weight if total_open_dyn_weight is None
                else total_open_dyn_weight + open_dyn_weight
            )
            total_prior_rew_weight = (
                prior_rew_weight if total_prior_rew_weight is None
                else total_prior_rew_weight + prior_rew_weight
            )
            total_transition_count += float(outputs.dynamics_mask.sum().item())

            # Collect valid posterior beliefs WITH gradients for step-level reg
            # Use posterior_valid_mask (not dynamics_mask) — SIGReg needs all
            # valid posteriors including Z_0, which may differ from dynamics scope.
            valid = self.world_model._extract_valid_beliefs(
                outputs.rollout.posterior.beliefs, outputs.posterior_valid_mask,
            )
            all_valid_beliefs.append(valid)

            window_outputs.append(outputs)
            prev_belief = self._terminal_posterior_belief(outputs).detach()

        # --- Step-level regularization (computed ONCE, not per-window) ---
        step_sigreg: SIGRegTerms | None = None
        step_beliefs = torch.cat(all_valid_beliefs, dim=0) if all_valid_beliefs else None
        cov_unscaled = total_sup_sum.new_zeros(()) if total_sup_sum is not None else torch.tensor(0.0, device=self.device)
        reg_loss = total_sup_sum.new_zeros(()) if total_sup_sum is not None else torch.tensor(0.0, device=self.device)

        if (
            self.config.anti_collapse.use_sigreg
            and self.world_model.sigreg is not None
            and step_beliefs is not None
            and step_beliefs.shape[0] > 0
        ):
            step_sigreg = self.world_model.sigreg(step_beliefs, update_buffer=False)
            cov_unscaled = covariance_loss(
                step_beliefs.reshape(-1, step_beliefs.shape[-1]).to(dtype=torch.float32),
            )
            sigreg_ep, sigreg_var = self._sigreg_components(step_sigreg)
            reg_loss = (
                self.config.sigreg.lambda_ep * sigreg_ep
                + self.config.sigreg.lambda_var * sigreg_var
                + self.config.sigreg.lambda_cov * cov_unscaled
            )
            # Update buffer once at step end (detached, for monitoring only)
            self.world_model.sigreg.running_buffer.push(step_beliefs.detach())
        if (
            self.config.anti_collapse.use_ema_variance
            and step_beliefs is not None
            and step_beliefs.shape[0] > 0
        ):
            from ..ema import ema_variance_loss
            var_loss = ema_variance_loss(step_beliefs)
            reg_loss = reg_loss + self.config.ema.lambda_var * var_loss

        # Main branch semantics: supervision_mean + regularization
        # SIGReg is NOT divided by total_valid — it's added directly
        assert total_sup_sum is not None
        assert total_dyn_sum is not None
        assert total_rew_sum is not None
        assert total_open_dyn_sum is not None
        assert total_prior_rew_sum is not None
        assert total_open_dyn_weight is not None
        assert total_prior_rew_weight is not None
        assert total_delta_cosine_sum is not None
        assert total_inverse_action_sum is not None
        assert total_action_aux_weight is not None
        assert total_inverse_prior_correct is not None
        assert total_inverse_post_correct is not None
        assert total_observation_anchor_sum is not None
        assert total_observation_anchor_weight is not None
        assert total_observation_delta_sum is not None
        assert total_observation_delta_weight is not None
        assert total_vjepa_prior_sum is not None
        assert total_vjepa_prior_weight is not None
        assert total_vjepa_posterior_sum is not None
        assert total_vjepa_posterior_weight is not None
        assert total_vjepa_delta_sum is not None
        assert total_vjepa_delta_weight is not None
        normalizer = max(total_transition_count, 1.0)
        dynamics_loss_value = total_dyn_sum / normalizer
        reward_loss_value = total_rew_sum / normalizer
        open_dynamics_loss_value = (
            total_open_dyn_sum / total_open_dyn_weight.clamp_min(1e-8)
        )
        prior_reward_loss_value = (
            total_prior_rew_sum / total_prior_rew_weight.clamp_min(1e-8)
        )
        action_normalizer = total_action_aux_weight.clamp_min(1.0)
        delta_cosine_loss_value = total_delta_cosine_sum / action_normalizer
        inverse_action_loss_value = total_inverse_action_sum / action_normalizer
        observation_anchor_loss_value = (
            total_observation_anchor_sum
            / total_observation_anchor_weight.clamp_min(1.0)
        )
        observation_delta_loss_value = (
            total_observation_delta_sum
            / total_observation_delta_weight.clamp_min(1.0)
        )
        vjepa_prior_loss_value = (
            total_vjepa_prior_sum / total_vjepa_prior_weight.clamp_min(1.0)
        )
        vjepa_posterior_loss_value = (
            total_vjepa_posterior_sum
            / total_vjepa_posterior_weight.clamp_min(1.0)
        )
        vjepa_delta_loss_value = (
            total_vjepa_delta_sum / total_vjepa_delta_weight.clamp_min(1.0)
        )
        total_loss = (
            dynamics_loss_value
            + lam_rew * reward_loss_value
            + self.config.training.open_dynamics_coef * open_dynamics_loss_value
            + lam_rew
            * self.config.training.prior_reward_coef
            * prior_reward_loss_value
            + self.config.training.delta_cosine_coef * delta_cosine_loss_value
            + self.config.training.inverse_action_coef * inverse_action_loss_value
            + self.config.training.observation_anchor_coef
            * observation_anchor_loss_value
            + self.config.training.observation_delta_anchor_coef
            * observation_delta_loss_value
            + self.config.training.vjepa_teacher_prior_coef
            * vjepa_prior_loss_value
            + self.config.training.vjepa_teacher_posterior_coef
            * vjepa_posterior_loss_value
            + self.config.training.vjepa_teacher_delta_coef
            * vjepa_delta_loss_value
            + reg_loss
        )
        total_loss.backward()

        grouped_grad_norms = self._grouped_gradient_norms()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            max_norm=self.config.training.grad_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.world_model.update_ema()

        return self._aggregate_metrics(
            window_outputs,
            horizon=horizon,
            grad_norm=float(grad_norm),
            step_sigreg=step_sigreg,
            all_valid_beliefs=[b.detach() for b in all_valid_beliefs],
            total_loss=float(total_loss.detach().item()),
            dynamics_loss=float(dynamics_loss_value.detach().item()),
            reward_loss=float(reward_loss_value.detach().item()),
            open_dynamics_loss=float(open_dynamics_loss_value.detach().item()),
            prior_reward_loss=float(prior_reward_loss_value.detach().item()),
            cov_unscaled=float(cov_unscaled.detach().item()),
            delta_cosine_loss=float(delta_cosine_loss_value.detach().item()),
            inverse_action_loss=float(inverse_action_loss_value.detach().item()),
            inverse_action_acc_prior=float(
                (total_inverse_prior_correct / action_normalizer).detach().item()
            ),
            inverse_action_acc_post=float(
                (total_inverse_post_correct / action_normalizer).detach().item()
            ),
            action_aux_valid_count=float(total_action_aux_weight.detach().item()),
            observation_anchor_loss=float(
                observation_anchor_loss_value.detach().item()
            ),
            observation_delta_anchor_loss=float(
                observation_delta_loss_value.detach().item()
            ),
            vjepa_teacher_prior_loss=float(vjepa_prior_loss_value.detach().item()),
            vjepa_teacher_posterior_loss=float(
                vjepa_posterior_loss_value.detach().item()
            ),
            vjepa_teacher_delta_loss=float(vjepa_delta_loss_value.detach().item()),
            wm_lora_grad_norm=grouped_grad_norms["wm_lora"],
            transition_aux_grad_norm=grouped_grad_norms["transition_aux"],
            reward_head_grad_norm=grouped_grad_norms["reward_head"],
            observation_anchor_grad_norm=grouped_grad_norms["observation_anchor"],
            vjepa_teacher_grad_norm=grouped_grad_norms["vjepa_teacher"],
            observation_anchor_valid_count=float(
                total_observation_anchor_weight.detach().item()
            ),
            observation_delta_anchor_valid_count=float(
                total_observation_delta_weight.detach().item()
            ),
            vjepa_teacher_prior_valid_count=float(
                total_vjepa_prior_weight.detach().item()
            ),
            vjepa_teacher_posterior_valid_count=float(
                total_vjepa_posterior_weight.detach().item()
            ),
            vjepa_teacher_delta_valid_count=float(
                total_vjepa_delta_weight.detach().item()
            ),
        )

    def _grouped_gradient_norms(self) -> dict[str, float]:
        """Report pre-clipping gradient flow for the trainable WM components."""

        squared = {
            "wm_lora": 0.0,
            "transition_aux": 0.0,
            "reward_head": 0.0,
            "observation_anchor": 0.0,
            "vjepa_teacher": 0.0,
        }
        for name, parameter in self.world_model.named_parameters():
            if parameter.grad is None:
                continue
            lower_name = name.lower()
            if lower_name.startswith("transition.backbone.") and "lora" in lower_name:
                group = "wm_lora"
            elif lower_name.startswith("reward_head."):
                group = "reward_head"
            elif lower_name.startswith("observation_anchor_head."):
                group = "observation_anchor"
            elif lower_name.startswith("vjepa_teacher_head."):
                group = "vjepa_teacher"
            elif lower_name.startswith("transition."):
                group = "transition_aux"
            else:
                continue
            grad_norm = parameter.grad.detach().float().norm().item()
            squared[group] += grad_norm * grad_norm
        return {name: value ** 0.5 for name, value in squared.items()}

    def _log_wandb(self, step: int, metrics: TrainStepMetrics) -> None:
        if self._wandb_run is None:
            return
        import wandb
        log_data = metrics.as_dict()
        log_data["lr"] = self.scheduler.get_last_lr()[0]
        wandb.log(log_data, step=step)

    def _log_wandb_dict(self, step: int, metrics: dict[str, float]) -> None:
        if self._wandb_run is None or not metrics:
            return
        import wandb
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish wandb run if active."""
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
            self._wandb_run = None

    @torch.no_grad()
    def evaluate(
        self,
        *,
        val_source: DataSource,
        num_batches: int = 5,
        global_step: int,
        run_probing: bool = True,
    ) -> dict[str, float]:
        """Run validation and return metrics dict."""
        self.world_model.eval()
        horizon = self.curriculum.horizon_at(global_step)

        all_metrics: list[TrainStepMetrics] = []
        diagnostic_prior_confusion = torch.zeros(
            self.config.env.num_actions,
            self.config.env.num_actions,
            dtype=torch.float64,
        )
        diagnostic_post_confusion = torch.zeros_like(diagnostic_prior_confusion)
        for _ in range(num_batches):
            batch = val_source.sample_batch(self.config.training.episodes_per_step)
            batch = self._move_sequence_batch(batch)
            total_win = num_windows(batch, horizon)
            prev_belief: BeliefState = self.world_model.get_initial_belief(
                batch.batch_size, dtype=batch.obs_tokens.dtype,
            )
            window_outputs: list[Phase1Outputs] = []
            val_valid_beliefs: list[Tensor] = []
            total_dyn_sum: Tensor | None = None
            total_rew_sum: Tensor | None = None
            total_open_dyn_sum: Tensor | None = None
            total_prior_rew_sum: Tensor | None = None
            total_open_dyn_weight: Tensor | None = None
            total_prior_rew_weight: Tensor | None = None
            total_anchor_sum: Tensor | None = None
            total_anchor_weight: Tensor | None = None
            total_delta_anchor_sum: Tensor | None = None
            total_delta_anchor_weight: Tensor | None = None
            total_vjepa_prior_sum: Tensor | None = None
            total_vjepa_prior_weight: Tensor | None = None
            total_vjepa_posterior_sum: Tensor | None = None
            total_vjepa_posterior_weight: Tensor | None = None
            total_vjepa_delta_sum: Tensor | None = None
            total_vjepa_delta_weight: Tensor | None = None
            total_delta_cosine_sum: Tensor | None = None
            total_inverse_action_sum: Tensor | None = None
            total_action_aux_weight: Tensor | None = None
            total_inverse_prior_correct: Tensor | None = None
            total_inverse_post_correct: Tensor | None = None
            total_transition_count = 0.0
            for wi in range(total_win):
                window = slice_training_window(
                    batch,
                    window_index=wi,
                    horizon=horizon,
                    config=self.config,
                    prev_belief=prev_belief,
                )
                outputs = self.world_model.compute_phase1_outputs(window)
                starts = torch.cat(
                    [
                        window.prev_belief.slots.unsqueeze(1),
                        outputs.rollout.posterior.beliefs[:, :-1],
                    ],
                    dim=1,
                )
                transition_actions = torch.cat(
                    [window.prev_actions.unsqueeze(1), window.actions[:, :-1]],
                    dim=1,
                )
                action_aux = self.world_model.compute_action_auxiliaries(
                    start_beliefs=starts,
                    prior_beliefs=outputs.rollout.prior.beliefs,
                    posterior_beliefs=outputs.rollout.posterior.beliefs,
                    actions=transition_actions,
                    mask=outputs.dynamics_mask,
                )
                diagnostic_prior_confusion += action_aux.prior_confusion.detach().cpu().double()
                diagnostic_post_confusion += action_aux.posterior_confusion.detach().cpu().double()
                total_delta_cosine_sum = (
                    action_aux.delta_cosine_sum if total_delta_cosine_sum is None
                    else total_delta_cosine_sum + action_aux.delta_cosine_sum
                )
                total_inverse_action_sum = (
                    action_aux.inverse_action_sum if total_inverse_action_sum is None
                    else total_inverse_action_sum + action_aux.inverse_action_sum
                )
                total_action_aux_weight = (
                    action_aux.weight_sum if total_action_aux_weight is None
                    else total_action_aux_weight + action_aux.weight_sum
                )
                total_inverse_prior_correct = (
                    action_aux.prior_correct if total_inverse_prior_correct is None
                    else total_inverse_prior_correct + action_aux.prior_correct
                )
                total_inverse_post_correct = (
                    action_aux.posterior_correct if total_inverse_post_correct is None
                    else total_inverse_post_correct + action_aux.posterior_correct
                )
                anchor = self.world_model.compute_observation_anchor(
                    posterior_beliefs=outputs.rollout.posterior.beliefs,
                    observation_tokens=window.obs_tokens,
                    posterior_mask=outputs.posterior_valid_mask,
                )
                vjepa_teacher = self.world_model.compute_vjepa_teacher_losses(
                    prior_beliefs=outputs.rollout.prior.beliefs,
                    posterior_beliefs=outputs.rollout.posterior.beliefs,
                    start_beliefs=starts,
                    teacher_tokens=window.semantic_teacher_tokens,
                    teacher_mask=window.semantic_teacher_mask,
                    prev_teacher_tokens=window.prev_semantic_teacher_tokens,
                    prev_teacher_mask=window.prev_semantic_teacher_mask,
                    dynamics_mask=outputs.dynamics_mask,
                    posterior_mask=outputs.posterior_valid_mask,
                )
                total_anchor_sum = (
                    anchor.state_sum if total_anchor_sum is None
                    else total_anchor_sum + anchor.state_sum
                )
                total_anchor_weight = (
                    anchor.state_weight if total_anchor_weight is None
                    else total_anchor_weight + anchor.state_weight
                )
                total_delta_anchor_sum = (
                    anchor.delta_sum if total_delta_anchor_sum is None
                    else total_delta_anchor_sum + anchor.delta_sum
                )
                total_delta_anchor_weight = (
                    anchor.delta_weight if total_delta_anchor_weight is None
                    else total_delta_anchor_weight + anchor.delta_weight
                )
                total_vjepa_prior_sum = (
                    vjepa_teacher.prior_sum
                    if total_vjepa_prior_sum is None
                    else total_vjepa_prior_sum + vjepa_teacher.prior_sum
                )
                total_vjepa_prior_weight = (
                    vjepa_teacher.prior_weight
                    if total_vjepa_prior_weight is None
                    else total_vjepa_prior_weight + vjepa_teacher.prior_weight
                )
                total_vjepa_posterior_sum = (
                    vjepa_teacher.posterior_sum
                    if total_vjepa_posterior_sum is None
                    else total_vjepa_posterior_sum + vjepa_teacher.posterior_sum
                )
                total_vjepa_posterior_weight = (
                    vjepa_teacher.posterior_weight
                    if total_vjepa_posterior_weight is None
                    else total_vjepa_posterior_weight + vjepa_teacher.posterior_weight
                )
                total_vjepa_delta_sum = (
                    vjepa_teacher.delta_sum
                    if total_vjepa_delta_sum is None
                    else total_vjepa_delta_sum + vjepa_teacher.delta_sum
                )
                total_vjepa_delta_weight = (
                    vjepa_teacher.delta_weight
                    if total_vjepa_delta_weight is None
                    else total_vjepa_delta_weight + vjepa_teacher.delta_weight
                )
                window_outputs.append(outputs)
                prev_belief = self._terminal_posterior_belief(outputs).detach()
                dyn_sum = outputs.losses.dynamics_sum
                rew_sum = outputs.losses.reward_sum
                total_dyn_sum = dyn_sum if total_dyn_sum is None else total_dyn_sum + dyn_sum
                total_rew_sum = rew_sum if total_rew_sum is None else total_rew_sum + rew_sum
                open_dyn_sum = outputs.losses.open_dynamics_sum
                prior_rew_sum = outputs.losses.prior_reward_sum
                open_dyn_weight = outputs.losses.open_dynamics_weight_sum
                prior_rew_weight = outputs.losses.prior_reward_weight_sum
                total_open_dyn_sum = (
                    open_dyn_sum if total_open_dyn_sum is None
                    else total_open_dyn_sum + open_dyn_sum
                )
                total_prior_rew_sum = (
                    prior_rew_sum if total_prior_rew_sum is None
                    else total_prior_rew_sum + prior_rew_sum
                )
                total_open_dyn_weight = (
                    open_dyn_weight if total_open_dyn_weight is None
                    else total_open_dyn_weight + open_dyn_weight
                )
                total_prior_rew_weight = (
                    prior_rew_weight if total_prior_rew_weight is None
                    else total_prior_rew_weight + prior_rew_weight
                )
                total_transition_count += float(outputs.dynamics_mask.sum().item())
                valid = self.world_model._extract_valid_beliefs(
                    outputs.rollout.posterior.beliefs, outputs.posterior_valid_mask,
                )
                val_valid_beliefs.append(valid)
            if window_outputs:
                # Compute step-level SIGReg for val metrics
                val_sigreg = None
                val_reg = window_outputs[0].rollout.posterior.beliefs.new_zeros(())
                val_beliefs_cat = (
                    torch.cat(val_valid_beliefs, dim=0) if val_valid_beliefs else None
                )
                if (
                    self.config.anti_collapse.use_sigreg
                    and self.world_model.sigreg is not None
                    and val_beliefs_cat is not None
                    and val_beliefs_cat.shape[0] > 0
                ):
                    val_sigreg = self.world_model.sigreg(
                        val_beliefs_cat, update_buffer=False,
                    )
                    val_sigreg_ep, val_sigreg_var = self._sigreg_components(
                        val_sigreg
                    )
                    val_reg = (
                        self.config.sigreg.lambda_ep * val_sigreg_ep
                        + self.config.sigreg.lambda_var * val_sigreg_var
                    )
                if (
                    self.config.anti_collapse.use_ema_variance
                    and val_beliefs_cat is not None
                    and val_beliefs_cat.shape[0] > 0
                ):
                    from ..ema import ema_variance_loss

                    val_reg = val_reg + self.config.ema.lambda_var * ema_variance_loss(val_beliefs_cat)

                if total_dyn_sum is None:
                    total_dyn_sum = val_reg.new_zeros(())
                if total_rew_sum is None:
                    total_rew_sum = val_reg.new_zeros(())
                if total_open_dyn_sum is None:
                    total_open_dyn_sum = val_reg.new_zeros(())
                if total_prior_rew_sum is None:
                    total_prior_rew_sum = val_reg.new_zeros(())
                if total_open_dyn_weight is None:
                    total_open_dyn_weight = val_reg.new_zeros(())
                if total_prior_rew_weight is None:
                    total_prior_rew_weight = val_reg.new_zeros(())
                if total_anchor_sum is None:
                    total_anchor_sum = val_reg.new_zeros(())
                if total_anchor_weight is None:
                    total_anchor_weight = val_reg.new_zeros(())
                if total_delta_anchor_sum is None:
                    total_delta_anchor_sum = val_reg.new_zeros(())
                if total_delta_anchor_weight is None:
                    total_delta_anchor_weight = val_reg.new_zeros(())
                if total_vjepa_prior_sum is None:
                    total_vjepa_prior_sum = val_reg.new_zeros(())
                if total_vjepa_prior_weight is None:
                    total_vjepa_prior_weight = val_reg.new_zeros(())
                if total_vjepa_posterior_sum is None:
                    total_vjepa_posterior_sum = val_reg.new_zeros(())
                if total_vjepa_posterior_weight is None:
                    total_vjepa_posterior_weight = val_reg.new_zeros(())
                if total_vjepa_delta_sum is None:
                    total_vjepa_delta_sum = val_reg.new_zeros(())
                if total_vjepa_delta_weight is None:
                    total_vjepa_delta_weight = val_reg.new_zeros(())
                if total_delta_cosine_sum is None:
                    total_delta_cosine_sum = val_reg.new_zeros(())
                if total_inverse_action_sum is None:
                    total_inverse_action_sum = val_reg.new_zeros(())
                if total_action_aux_weight is None:
                    total_action_aux_weight = val_reg.new_zeros(())
                if total_inverse_prior_correct is None:
                    total_inverse_prior_correct = val_reg.new_zeros(())
                if total_inverse_post_correct is None:
                    total_inverse_post_correct = val_reg.new_zeros(())
                normalizer = max(total_transition_count, 1.0)
                val_dynamics = total_dyn_sum / normalizer
                val_reward = total_rew_sum / normalizer
                val_open_dynamics = (
                    total_open_dyn_sum / total_open_dyn_weight.clamp_min(1e-8)
                )
                val_prior_reward = (
                    total_prior_rew_sum / total_prior_rew_weight.clamp_min(1e-8)
                )
                val_anchor = total_anchor_sum / total_anchor_weight.clamp_min(1.0)
                val_delta_anchor = (
                    total_delta_anchor_sum
                    / total_delta_anchor_weight.clamp_min(1.0)
                )
                val_vjepa_prior = (
                    total_vjepa_prior_sum / total_vjepa_prior_weight.clamp_min(1.0)
                )
                val_vjepa_posterior = (
                    total_vjepa_posterior_sum
                    / total_vjepa_posterior_weight.clamp_min(1.0)
                )
                val_vjepa_delta = (
                    total_vjepa_delta_sum / total_vjepa_delta_weight.clamp_min(1.0)
                )
                action_normalizer = total_action_aux_weight.clamp_min(1.0)
                val_delta_cosine = total_delta_cosine_sum / action_normalizer
                val_inverse_action = total_inverse_action_sum / action_normalizer
                val_total = (
                    val_dynamics
                    + self.config.training.lambda_reward * val_reward
                    + self.config.training.open_dynamics_coef * val_open_dynamics
                    + self.config.training.lambda_reward
                    * self.config.training.prior_reward_coef
                    * val_prior_reward
                    + self.config.training.observation_anchor_coef * val_anchor
                    + self.config.training.observation_delta_anchor_coef
                    * val_delta_anchor
                    + self.config.training.delta_cosine_coef * val_delta_cosine
                    + self.config.training.inverse_action_coef * val_inverse_action
                    + self.config.training.vjepa_teacher_prior_coef
                    * val_vjepa_prior
                    + self.config.training.vjepa_teacher_posterior_coef
                    * val_vjepa_posterior
                    + self.config.training.vjepa_teacher_delta_coef
                    * val_vjepa_delta
                    + val_reg
                )

                all_metrics.append(
                    self._aggregate_metrics(
                        window_outputs,
                        horizon=horizon,
                        step_sigreg=val_sigreg,
                        all_valid_beliefs=val_valid_beliefs,
                        total_loss=float(val_total.item()),
                        dynamics_loss=float(val_dynamics.item()),
                        reward_loss=float(val_reward.item()),
                        open_dynamics_loss=float(val_open_dynamics.item()),
                        prior_reward_loss=float(val_prior_reward.item()),
                        delta_cosine_loss=float(val_delta_cosine.item()),
                        inverse_action_loss=float(val_inverse_action.item()),
                        inverse_action_acc_prior=float(
                            (total_inverse_prior_correct / action_normalizer).item()
                        ),
                        inverse_action_acc_post=float(
                            (total_inverse_post_correct / action_normalizer).item()
                        ),
                        action_aux_valid_count=float(total_action_aux_weight.item()),
                        observation_anchor_loss=float(val_anchor.item()),
                        observation_delta_anchor_loss=float(
                            val_delta_anchor.item()
                        ),
                        observation_anchor_valid_count=float(
                            total_anchor_weight.item()
                        ),
                        observation_delta_anchor_valid_count=float(
                            total_delta_anchor_weight.item()
                        ),
                        vjepa_teacher_prior_loss=float(val_vjepa_prior.item()),
                        vjepa_teacher_posterior_loss=float(
                            val_vjepa_posterior.item()
                        ),
                        vjepa_teacher_delta_loss=float(val_vjepa_delta.item()),
                        vjepa_teacher_prior_valid_count=float(
                            total_vjepa_prior_weight.item()
                        ),
                        vjepa_teacher_posterior_valid_count=float(
                            total_vjepa_posterior_weight.item()
                        ),
                        vjepa_teacher_delta_valid_count=float(
                            total_vjepa_delta_weight.item()
                        ),
                    )
                )

        self.world_model.train()

        if not all_metrics:
            return {}

        # Average across batches
        result: dict[str, float] = {}
        keys = all_metrics[0].as_dict().keys()
        for key in keys:
            vals = [m.as_dict()[key] for m in all_metrics]
            result[f"val/{key}"] = sum(vals) / len(vals)

        for true_action in range(self.config.env.num_actions):
            for predicted_action in range(self.config.env.num_actions):
                result[
                    f"val/diagnostic/prior_confusion_{true_action}_{predicted_action}"
                ] = float(diagnostic_prior_confusion[true_action, predicted_action])
                result[
                    f"val/diagnostic/post_confusion_{true_action}_{predicted_action}"
                ] = float(diagnostic_post_confusion[true_action, predicted_action])

        # Belief probing on val episodes
        if run_probing:
            probe_metrics = self._run_probing(val_source, horizon=horizon)
            result.update(probe_metrics)

        return result

    def save_checkpoint(self, path: str | Path, *, step: int) -> None:
        """Save model, optimizer, and step state."""

        from ..model.checkpoint_semantics import world_model_semantics

        checkpoint = {
            "model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "config": self.config,
            "wm_semantics": world_model_semantics(
                self.config.backbone.attention_mode
            ),
        }
        torch.save(checkpoint, Path(path))

    def load_checkpoint(
        self, path: str | Path, *, model_only: bool = False
    ) -> int:
        """Load model, optimizer, and return the saved global step."""

        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        from ..model.checkpoint_semantics import validate_world_model_semantics
        validate_world_model_semantics(
            checkpoint,
            attention_mode=self.config.backbone.attention_mode,
            context=f"Phase-1 checkpoint {path}",
        )
        if model_only:
            incompatible = self.world_model.load_state_dict(
                checkpoint["model"], strict=False
            )
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            allowed_missing = {
                "inverse_action_head.weight", "inverse_action_head.bias",
                "observation_anchor_head.weight",
                "vjepa_teacher_head.weight",
            }
            disallowed_missing = [
                key for key in missing
                if key not in allowed_missing
                and not key.startswith("ema_teacher.")
                and ".wm_prior." not in key
                and not key.startswith("transition.prior_residual.")
            ]
            disallowed_unexpected = [
                key for key in unexpected
                if not key.startswith("ema_teacher.online.")
            ]
            if disallowed_unexpected or disallowed_missing:
                raise RuntimeError(
                    "Model-only resume has incompatible state: "
                    f"missing={disallowed_missing}, "
                    f"unexpected={disallowed_unexpected}"
                )
            if self.world_model.ema_teacher is not None:
                # The source checkpoint may predate EMA. Shadows constructed
                # before loading still reflect random initialization, so they
                # must be synchronized to the now-loaded step checkpoint.
                self.world_model.ema_teacher.reset_to_online()
            print(
                "Model-only resume: fresh optimizer/scheduler; "
                f"new architecture keys={len(missing)} "
                f"(anchor={int('observation_anchor_head.weight' in missing)}, "
                f"vjepa_teacher={int('vjepa_teacher_head.weight' in missing)}, "
                f"ema={sum(key.startswith('ema_teacher.') for key in missing)})",
                flush=True,
            )
        else:
            self.world_model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        return int(checkpoint["step"])

    def _move_sequence_batch(self, batch: SequenceBatch) -> SequenceBatch:
        if batch.obs_tokens.device == self.device:
            return batch
        return SequenceBatch(
            obs_tokens=batch.obs_tokens.to(self.device),
            actions=batch.actions.to(self.device),
            rewards=batch.rewards.to(self.device),
            episode_lengths=batch.episode_lengths.to(self.device),
            env_ids=None if batch.env_ids is None else batch.env_ids.to(self.device),
            episode_success=(
                None
                if batch.episode_success is None
                else batch.episode_success.to(self.device)
            ),
            semantic_teacher_tokens=(
                None
                if batch.semantic_teacher_tokens is None
                else batch.semantic_teacher_tokens.to(self.device)
            ),
            semantic_teacher_mask=(
                None
                if batch.semantic_teacher_mask is None
                else batch.semantic_teacher_mask.to(self.device)
            ),
        )

    def _terminal_posterior_belief(self, outputs: Phase1Outputs) -> BeliefState:
        terminal = outputs.rollout.posterior.beliefs[:, -1]
        return BeliefState(slots=terminal)

    @torch.no_grad()
    def _run_probing(
        self,
        val_source: DataSource,
        *,
        horizon: int,
        num_episodes: int = 50,
        probe_epochs: int = 30,
    ) -> dict[str, float]:
        """Train and evaluate linear probes on val beliefs.

        Extracts frozen posterior beliefs from a subset of val episodes,
        pairs them with structured state_labels from episode metadata,
        then trains linear probes (player_pos, box_pos, etc.) and reports
        accuracy vs majority-class baseline.

        Note: This method re-enables gradients for probe training even when
        called from @torch.no_grad() contexts (e.g. evaluate()).
        """
        # Check if val_source has accessible episodes with state_labels
        if not hasattr(val_source, "dataset"):
            return {}
        episodes = val_source.dataset.episodes  # type: ignore[attr-defined]
        if not episodes:
            return {}

        # Check if state_labels are available
        sample_meta = episodes[0].metadata
        if not sample_meta or "state_labels" not in sample_meta:
            return {}

        from .probing import build_sokoban_probes, extract_sokoban_labels

        # Episode-disjoint train/eval split. The previous implementation fit
        # and evaluated probes on the exact same beliefs, substantially
        # overstating semantic generalization.
        subset = episodes[:num_episodes]
        split_at = max(1, min(len(subset) - 1, int(0.7 * len(subset))))
        probe_train_episodes = subset[:split_at]
        probe_test_episodes = subset[split_at:]

        def extract(episode_subset):
            beliefs = []
            state_labels = []
            for ep in episode_subset:
                if "state_labels" not in ep.metadata:
                    continue
                ep_labels = ep.metadata["state_labels"]
                t_seq = ep.episode_length + 1
                obs = ep.obs_tokens[:t_seq].unsqueeze(0).to(self.device)
                acts = ep.actions[:t_seq].unsqueeze(0).to(
                    self.device, dtype=torch.long
                )
                belief = self.world_model.get_initial_belief(1, dtype=obs.dtype)
                prev_action = torch.tensor(
                    [self.config.env.null_action_id],
                    dtype=torch.long,
                    device=self.device,
                )
                for timestep in range(t_seq):
                    belief = self.world_model.posterior_step(
                        prev_belief=belief,
                        prev_actions=prev_action,
                        observation_tokens=obs[:, timestep],
                    )
                    if (
                        timestep < len(ep_labels)
                        and ep_labels[timestep].get("player_pos") is not None
                    ):
                        beliefs.append(belief.slots.detach().cpu().squeeze(0))
                        state_labels.append(ep_labels[timestep])
                    if timestep < t_seq - 1:
                        prev_action = acts[:, timestep]
            return beliefs, state_labels

        self.world_model.eval()
        train_beliefs, train_state_labels = extract(probe_train_episodes)
        test_beliefs, test_state_labels = extract(probe_test_episodes)
        self.world_model.train()

        if not train_beliefs or not test_beliefs:
            return {}

        train_n = min(len(train_beliefs), len(train_state_labels))
        test_n = min(len(test_beliefs), len(test_state_labels))
        train_beliefs_tensor = torch.stack(train_beliefs[:train_n])
        test_beliefs_tensor = torch.stack(test_beliefs[:test_n])
        train_labels = extract_sokoban_labels(train_state_labels[:train_n])
        test_labels = extract_sokoban_labels(test_state_labels[:test_n])

        # Build and train probes (re-enable grad for probe training)
        with torch.enable_grad():
            probe = build_sokoban_probes(
                hidden_dim=self.config.hidden_dim,
                num_belief_slots=self.config.belief.num_slots,
                device="cpu",
            )
            probe.fit(
                train_beliefs_tensor, train_labels,
                epochs=probe_epochs, lr=1e-3,
            )
            probe_results = probe.evaluate(test_beliefs_tensor, test_labels)

        # Format results
        result: dict[str, float] = {}
        result["val/probe/train_samples"] = float(train_n)
        result["val/probe/test_samples"] = float(test_n)
        for pm in probe_results:
            result[f"val/probe/{pm.name}_acc"] = pm.accuracy
            result[f"val/probe/{pm.name}_baseline"] = pm.baseline_acc
            result[f"val/probe/{pm.name}_delta"] = pm.accuracy - pm.baseline_acc
        return result

    def _aggregate_metrics(
        self,
        outputs: list[Phase1Outputs],
        *,
        horizon: int,
        grad_norm: float = 0.0,
        step_sigreg: SIGRegTerms | None = None,
        all_valid_beliefs: list[Tensor] | None = None,
        total_loss: float | None = None,
        dynamics_loss: float | None = None,
        reward_loss: float | None = None,
        open_dynamics_loss: float | None = None,
        prior_reward_loss: float | None = None,
        cov_unscaled: float = 0.0,
        delta_cosine_loss: float = 0.0,
        inverse_action_loss: float = 0.0,
        inverse_action_acc_prior: float = 0.0,
        inverse_action_acc_post: float = 0.0,
        action_aux_valid_count: float = 0.0,
        observation_anchor_loss: float = 0.0,
        observation_delta_anchor_loss: float = 0.0,
        observation_anchor_valid_count: float = 0.0,
        observation_delta_anchor_valid_count: float = 0.0,
        vjepa_teacher_prior_loss: float = 0.0,
        vjepa_teacher_posterior_loss: float = 0.0,
        vjepa_teacher_delta_loss: float = 0.0,
        vjepa_teacher_prior_valid_count: float = 0.0,
        vjepa_teacher_posterior_valid_count: float = 0.0,
        vjepa_teacher_delta_valid_count: float = 0.0,
        wm_lora_grad_norm: float = 0.0,
        transition_aux_grad_norm: float = 0.0,
        reward_head_grad_norm: float = 0.0,
        observation_anchor_grad_norm: float = 0.0,
        vjepa_teacher_grad_norm: float = 0.0,
    ) -> TrainStepMetrics:
        if not outputs:
            raise ValueError("train_one_step requires at least one window output.")

        def mean_scalar(values: list[Tensor]) -> float:
            stacked = torch.stack([value.detach() for value in values])
            return float(stacked.mean().item())

        # --- Collect valid-only beliefs and priors for diagnostics ---
        all_valid_post_mean = []
        all_valid_prior_mean = []
        all_valid_post_pri_diff = []
        total_transition_count = 0.0
        total_posterior_count = 0.0
        total_dyn_sum = outputs[0].losses.dynamics_sum.new_zeros(())
        total_rew_sum = outputs[0].losses.reward_sum.new_zeros(())
        total_open_dyn_sum = outputs[0].losses.open_dynamics_sum.new_zeros(())
        total_prior_rew_sum = outputs[0].losses.prior_reward_sum.new_zeros(())
        total_open_dyn_weight = outputs[0].losses.open_dynamics_weight_sum.new_zeros(())
        total_prior_rew_weight = outputs[0].losses.prior_reward_weight_sum.new_zeros(())
        open_dynamics_valid_count = 0.0
        open_reward_valid_count = 0.0
        open_loop_horizon = 0.0

        for out in outputs:
            post_b = out.rollout.posterior.beliefs.detach()  # (B, T, K, D)
            pri_b = out.rollout.prior.beliefs.detach()       # (B, T, K, D)
            dynamics_mask = out.dynamics_mask.detach()
            posterior_mask = out.posterior_valid_mask.detach()
            total_transition_count += float(dynamics_mask.sum().item())
            total_posterior_count += float(posterior_mask.sum().item())
            total_dyn_sum = total_dyn_sum + out.losses.dynamics_sum.detach()
            total_rew_sum = total_rew_sum + out.losses.reward_sum.detach()
            total_open_dyn_sum = total_open_dyn_sum + out.losses.open_dynamics_sum.detach()
            total_prior_rew_sum = total_prior_rew_sum + out.losses.prior_reward_sum.detach()
            total_open_dyn_weight = (
                total_open_dyn_weight + out.losses.open_dynamics_weight_sum.detach()
            )
            total_prior_rew_weight = (
                total_prior_rew_weight + out.losses.prior_reward_weight_sum.detach()
            )
            if out.open_loop is not None:
                open_loop_horizon = max(
                    open_loop_horizon, float(out.open_loop.prior.horizon)
                )
                open_dynamics_valid_count += float(
                    out.open_loop.dynamics_mask.sum().item()
                )
                open_reward_valid_count += float(
                    out.open_loop.reward_mask.sum().item()
                )

            # Prior-vs-posterior agreement is only meaningful on real transitions.
            post_mean = post_b.mean(dim=2)   # (B, T, D)
            prior_mean = pri_b.mean(dim=2)   # (B, T, D)
            if dynamics_mask.sum() > 0:
                all_valid_post_mean.append(post_mean[dynamics_mask])   # (V, D)
                all_valid_prior_mean.append(prior_mean[dynamics_mask])
                diff = (post_b - pri_b).pow(2).mean(dim=(-2, -1))  # (B, T)
                all_valid_post_pri_diff.append(diff[dynamics_mask])

        # Cosine similarity (valid only)
        if all_valid_post_mean:
            valid_post = torch.cat(all_valid_post_mean, dim=0)
            valid_prior = torch.cat(all_valid_prior_mean, dim=0)
            cos_sim = torch.nn.functional.cosine_similarity(
                valid_post, valid_prior, dim=-1,
            ).mean().item()
            post_pri_l2 = torch.cat(all_valid_post_pri_diff, dim=0).mean().item()
        else:
            cos_sim = 0.0
            post_pri_l2 = 0.0

        # Effective rank + belief norm — ONLY on valid beliefs (not padding)
        if all_valid_beliefs and len(all_valid_beliefs) > 0:
            from .sigreg import flatten_posterior_beliefs
            valid_flat = torch.cat(
                [flatten_posterior_beliefs(b) for b in all_valid_beliefs], dim=0,
            )
        else:
            # Fallback: extract valid beliefs from outputs
            valid_parts = []
            for out in outputs:
                post_b = out.rollout.posterior.beliefs.detach()
                mask = out.posterior_valid_mask.detach()
                valid = self.world_model._extract_valid_beliefs(post_b, mask)
                from .sigreg import flatten_posterior_beliefs
                valid_parts.append(flatten_posterior_beliefs(valid))
            valid_flat = torch.cat(valid_parts, dim=0) if valid_parts else torch.empty(0)

        if dynamics_loss is None or reward_loss is None:
            normalizer = max(total_transition_count, 1.0)
            dynamics_loss = float((total_dyn_sum / normalizer).item())
            reward_loss = float((total_rew_sum / normalizer).item())
        if open_dynamics_loss is None:
            open_dynamics_loss = float(
                (total_open_dyn_sum / total_open_dyn_weight.clamp_min(1e-8)).item()
            )
        if prior_reward_loss is None:
            prior_reward_loss = float(
                (total_prior_rew_sum / total_prior_rew_weight.clamp_min(1e-8)).item()
            )

        if valid_flat.shape[0] > 0:
            belief_norm_mean = valid_flat.norm(dim=-1).mean().item()

            # Effective rank with collapse guard
            sample = valid_flat
            if sample.shape[0] > 2048:
                idx = torch.randperm(sample.shape[0])[:2048]
                sample = sample[idx]
            centered = sample - sample.mean(dim=0, keepdim=True)
            centered_rms = centered.norm() / max(centered.shape[0], 1) ** 0.5
            if centered_rms < 1e-4:
                effective_rank = 1.0
            else:
                try:
                    s = torch.linalg.svdvals(centered.float())
                    p = s / s.sum().clamp_min(1e-8)
                    p = p.clamp_min(1e-8)
                    entropy = -(p * p.log()).sum()
                    effective_rank = entropy.exp().item()
                except Exception:
                    effective_rank = 0.0
        else:
            belief_norm_mean = 0.0
            effective_rank = 0.0

        # SIGReg stats from step-level computation
        if step_sigreg is not None:
            sigreg_num_samples = float(step_sigreg.num_current)
            belief_std_mean = step_sigreg.mean_std.item()
            belief_std_min = step_sigreg.min_std.item()
            sigreg_ep_unscaled = step_sigreg.ep_unscaled.item()
            sigreg_var_unscaled = step_sigreg.var_unscaled.item()
            effective_ep, effective_var = self._sigreg_components(step_sigreg)
            sigreg_ep = effective_ep.item()
            sigreg_var = effective_var.item()
        else:
            # EMA or no-sigreg: compute variance stats directly on valid beliefs
            sigreg_num_samples = float(valid_flat.shape[0]) if valid_flat.shape[0] > 0 else 0.0
            if valid_flat.shape[0] > 0:
                from .sigreg import vicreg_variance_loss
                _, mean_std, min_std = vicreg_variance_loss(valid_flat)
                belief_std_mean = mean_std.item()
                belief_std_min = min_std.item()
            else:
                belief_std_mean = 0.0
                belief_std_min = 0.0
            sigreg_ep_unscaled = 0.0
            sigreg_var_unscaled = 0.0
            sigreg_ep = mean_scalar([out.losses.sigreg_ep for out in outputs])
            sigreg_var = mean_scalar([out.losses.sigreg_var for out in outputs])

        # Reward accuracy — only on valid reward steps
        all_post_logits = []
        all_prior_logits = []
        all_targets = []
        all_reward_masks = []
        all_open_reward_logits = []
        all_open_reward_targets = []
        all_open_reward_masks = []
        for out in outputs:
            all_post_logits.append(out.posterior_reward_logits.detach())
            all_prior_logits.append(out.prior_reward_logits.detach())
            all_targets.append(out.reward_targets.detach())
            all_reward_masks.append(out.reward_mask.detach())
            if out.open_loop is not None:
                all_open_reward_logits.append(out.open_loop.reward_logits.detach())
                all_open_reward_targets.append(out.open_loop.reward_targets.detach())
                all_open_reward_masks.append(out.open_loop.reward_mask.detach())

        def pad_time_2d(tensors: list[Tensor], max_t: int) -> Tensor:
            padded = []
            for t in tensors:
                if t.shape[1] < max_t:
                    pad_shape = list(t.shape)
                    pad_shape[1] = max_t - t.shape[1]
                    t = torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=1)
                padded.append(t)
            return torch.cat(padded, dim=0)

        max_T_rew = max(t.shape[1] for t in all_post_logits) if all_post_logits else 1
        post_logits = pad_time_2d(all_post_logits, max_T_rew)
        prior_logits = pad_time_2d(all_prior_logits, max_T_rew)
        targets = pad_time_2d(all_targets, max_T_rew)
        reward_mask = pad_time_2d(all_reward_masks, max_T_rew).bool()

        post_reward_acc = 0.0
        prior_reward_acc = 0.0
        post_reward_recall = 0.0
        prior_reward_recall = 0.0
        post_reward_precision = 0.0
        prior_reward_precision = 0.0
        post_reward_f1 = 0.0
        prior_reward_f1 = 0.0
        post_reward_auroc = 0.0
        prior_reward_auroc = 0.0
        post_reward_brier = 0.0
        prior_reward_brier = 0.0

        reward_positive_rate = 0.0
        reward_logit_mean_post = 0.0
        reward_logit_mean_pri = 0.0
        reward_logit_gap = 0.0
        reward_post_minus_pri_on_positive = 0.0
        reward_valid_count = 0.0
        reward_positive_count = 0.0
        post_reward_true_positive_count = 0.0
        prior_reward_true_positive_count = 0.0
        post_reward_predicted_positive_count = 0.0
        prior_reward_predicted_positive_count = 0.0
        open_prior_reward_acc = 0.0
        open_prior_reward_positive_rate = 0.0

        if reward_mask.sum() > 0:
            valid_post_logits = post_logits[reward_mask]
            valid_prior_logits = prior_logits[reward_mask]
            valid_post_preds = (post_logits[reward_mask] > 0).float()
            valid_prior_preds = (prior_logits[reward_mask] > 0).float()
            valid_targets = targets[reward_mask]
            reward_valid_count = float(valid_targets.numel())
            post_reward_acc = (valid_post_preds == valid_targets).float().mean().item()
            prior_reward_acc = (valid_prior_preds == valid_targets).float().mean().item()

            # Reward diagnostics
            reward_positive_rate = valid_targets.mean().item()
            reward_logit_mean_post = valid_post_logits.mean().item()
            reward_logit_mean_pri = valid_prior_logits.mean().item()
            reward_logit_gap = (valid_post_logits - valid_prior_logits).abs().mean().item()
            post_reward_brier = (
                (valid_post_logits.sigmoid() - valid_targets).pow(2).mean().item()
            )
            prior_reward_brier = (
                (valid_prior_logits.sigmoid() - valid_targets).pow(2).mean().item()
            )

            # Precision / Recall on positive class
            pos_mask = valid_targets == 1.0
            num_pos = pos_mask.sum().item()
            reward_positive_count = float(num_pos)
            if num_pos > 0:
                post_reward_true_positive_count = float(
                    (valid_post_preds[pos_mask] == 1.0).sum().item()
                )
                prior_reward_true_positive_count = float(
                    (valid_prior_preds[pos_mask] == 1.0).sum().item()
                )
                post_reward_recall = (valid_post_preds[pos_mask] == 1.0).float().mean().item()
                prior_reward_recall = (valid_prior_preds[pos_mask] == 1.0).float().mean().item()
                reward_post_minus_pri_on_positive = (
                    valid_post_logits[pos_mask] - valid_prior_logits[pos_mask]
                ).mean().item()
            post_pred_pos = valid_post_preds == 1.0
            prior_pred_pos = valid_prior_preds == 1.0
            post_reward_predicted_positive_count = float(post_pred_pos.sum().item())
            prior_reward_predicted_positive_count = float(prior_pred_pos.sum().item())
            if post_pred_pos.sum() > 0:
                post_reward_precision = (valid_targets[post_pred_pos] == 1.0).float().mean().item()
            if prior_pred_pos.sum() > 0:
                prior_reward_precision = (valid_targets[prior_pred_pos] == 1.0).float().mean().item()

            post_reward_f1 = self._binary_f1(post_reward_precision, post_reward_recall)
            prior_reward_f1 = self._binary_f1(prior_reward_precision, prior_reward_recall)
            post_reward_auroc = self._binary_auroc(valid_post_logits, valid_targets)
            prior_reward_auroc = self._binary_auroc(valid_prior_logits, valid_targets)

        if all_open_reward_logits:
            max_t_open = max(t.shape[1] for t in all_open_reward_logits)
            open_logits = pad_time_2d(all_open_reward_logits, max_t_open)
            open_targets = pad_time_2d(all_open_reward_targets, max_t_open)
            open_mask = pad_time_2d(all_open_reward_masks, max_t_open).bool()
            if open_mask.sum() > 0:
                valid_open_logits = open_logits[open_mask]
                valid_open_targets = open_targets[open_mask]
                open_prior_reward_acc = (
                    ((valid_open_logits > 0).float() == valid_open_targets)
                    .float()
                    .mean()
                    .item()
                )
                open_prior_reward_positive_rate = valid_open_targets.mean().item()

        if total_loss is None:
            reg_total = 0.0
            if step_sigreg is not None:
                effective_ep, effective_var = self._sigreg_components(step_sigreg)
                reg_total = (
                    self.config.sigreg.lambda_ep * effective_ep.item()
                    + self.config.sigreg.lambda_var * effective_var.item()
                )
            total_loss = (
                dynamics_loss
                + self.config.training.open_dynamics_coef * open_dynamics_loss
                + self.config.training.lambda_reward * reward_loss
                + self.config.training.lambda_reward
                * self.config.training.prior_reward_coef
                * prior_reward_loss
                + reg_total
            )

        return TrainStepMetrics(
            total=total_loss,
            dynamics=dynamics_loss,
            reward=reward_loss,
            open_dynamics=open_dynamics_loss,
            prior_reward=prior_reward_loss,
            sigreg_ep=sigreg_ep,
            sigreg_var=sigreg_var,
            horizon=horizon,
            num_windows=len(outputs),
            cosine_sim=cos_sim,
            posterior_prior_l2=post_pri_l2,
            grad_norm=grad_norm,
            effective_rank=effective_rank,
            belief_std_mean=belief_std_mean,
            belief_std_min=belief_std_min,
            belief_norm_mean=belief_norm_mean,
            posterior_reward_acc=post_reward_acc,
            prior_reward_acc=prior_reward_acc,
            posterior_reward_recall=post_reward_recall,
            prior_reward_recall=prior_reward_recall,
            posterior_reward_precision=post_reward_precision,
            prior_reward_precision=prior_reward_precision,
            posterior_reward_f1=post_reward_f1,
            prior_reward_f1=prior_reward_f1,
            posterior_reward_auroc=post_reward_auroc,
            prior_reward_auroc=prior_reward_auroc,
            posterior_reward_brier=post_reward_brier,
            prior_reward_brier=prior_reward_brier,
            valid_transitions=total_transition_count,
            sigreg_num_samples=sigreg_num_samples,
            sigreg_ep_unscaled=sigreg_ep_unscaled,
            sigreg_var_unscaled=sigreg_var_unscaled,
            sigreg_cov_unscaled=cov_unscaled,
            posterior_valid_count=total_posterior_count,
            reward_positive_rate=reward_positive_rate,
            reward_logit_mean_post=reward_logit_mean_post,
            reward_logit_mean_pri=reward_logit_mean_pri,
            reward_logit_gap=reward_logit_gap,
            reward_post_minus_pri_on_positive=reward_post_minus_pri_on_positive,
            reward_valid_count=reward_valid_count,
            reward_positive_count=reward_positive_count,
            posterior_reward_true_positive_count=post_reward_true_positive_count,
            prior_reward_true_positive_count=prior_reward_true_positive_count,
            posterior_reward_predicted_positive_count=post_reward_predicted_positive_count,
            prior_reward_predicted_positive_count=prior_reward_predicted_positive_count,
            open_loop_horizon=open_loop_horizon,
            open_dynamics_valid_count=open_dynamics_valid_count,
            open_reward_valid_count=open_reward_valid_count,
            open_prior_reward_acc=open_prior_reward_acc,
            open_prior_reward_positive_rate=open_prior_reward_positive_rate,
            delta_cosine_loss=delta_cosine_loss,
            inverse_action_loss=inverse_action_loss,
            inverse_action_acc_prior=inverse_action_acc_prior,
            inverse_action_acc_post=inverse_action_acc_post,
            action_aux_valid_count=action_aux_valid_count,
            observation_anchor_loss=observation_anchor_loss,
            observation_delta_anchor_loss=observation_delta_anchor_loss,
            observation_anchor_valid_count=observation_anchor_valid_count,
            observation_delta_anchor_valid_count=observation_delta_anchor_valid_count,
            vjepa_teacher_prior_loss=vjepa_teacher_prior_loss,
            vjepa_teacher_posterior_loss=vjepa_teacher_posterior_loss,
            vjepa_teacher_delta_loss=vjepa_teacher_delta_loss,
            vjepa_teacher_prior_valid_count=vjepa_teacher_prior_valid_count,
            vjepa_teacher_posterior_valid_count=vjepa_teacher_posterior_valid_count,
            vjepa_teacher_delta_valid_count=vjepa_teacher_delta_valid_count,
            wm_lora_grad_norm=wm_lora_grad_norm,
            transition_aux_grad_norm=transition_aux_grad_norm,
            reward_head_grad_norm=reward_head_grad_norm,
            observation_anchor_grad_norm=observation_anchor_grad_norm,
            vjepa_teacher_grad_norm=vjepa_teacher_grad_norm,
        )

    def _should_save_checkpoint(self, next_step: int) -> bool:
        every = self.config.training.checkpoint_every
        if every <= 0:
            return next_step == self.config.training.total_steps
        if next_step == self.config.training.total_steps:
            return True
        return next_step % every == 0

    @staticmethod
    def _checkpoint_path(root: Path, *, step: int) -> Path:
        return root / f"step_{step:06d}.pt"

    @staticmethod
    def _update_latest_checkpoint(checkpoint_path: Path) -> None:
        """Atomically point latest.pt at an immutable periodic checkpoint."""
        latest = checkpoint_path.parent / "latest.pt"
        temporary = checkpoint_path.parent / ".latest.pt.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(checkpoint_path.name)
        temporary.replace(latest)

    @staticmethod
    def _binary_f1(precision: float, recall: float) -> float:
        denom = precision + recall
        if denom <= 0.0:
            return 0.0
        return 2.0 * precision * recall / denom

    @staticmethod
    def _binary_auroc(logits: Tensor, targets: Tensor) -> float:
        """Compute AUROC from raw logits and binary targets without extra deps."""
        if logits.numel() == 0:
            return 0.0
        targets = targets.to(dtype=torch.float32)
        num_pos = int((targets == 1.0).sum().item())
        num_neg = int((targets == 0.0).sum().item())
        if num_pos == 0 or num_neg == 0:
            return 0.0

        order = torch.argsort(logits, stable=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, logits.numel() + 1, device=logits.device, dtype=torch.float32)
        pos_rank_sum = ranks[targets == 1.0].sum()
        auc = (pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
        return float(auc.item())
