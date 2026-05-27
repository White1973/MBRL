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

    def as_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total,
            "loss/dynamics": self.dynamics,
            "loss/reward": self.reward,
            "loss/sigreg_ep": self.sigreg_ep,
            "loss/sigreg_var": self.sigreg_var,
            "curriculum/horizon": float(self.horizon),
            "train/num_windows": float(self.num_windows),
            "metric/cosine_sim": self.cosine_sim,
            "metric/posterior_prior_l2": self.posterior_prior_l2,
            "metric/grad_norm": self.grad_norm,
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
        self.data_source = data_source
        self.logger = logger
        self.device = torch.device(device)
        self.curriculum = CurriculumManager(config)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.world_model.to(self.device)
        self._wandb_run = None

    def _build_optimizer(self) -> torch.optim.AdamW:
        """Build optimizer with separate LR for LoRA vs base VLM weights."""
        tc = self.config.training
        # Try to separate LoRA params from base params
        lora_params = []
        base_params = []
        for name, param in self.world_model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            else:
                base_params.append(param)

        if base_params and tc.base_lr_factor != 1.0:
            param_groups = [
                {"params": lora_params, "lr": tc.lr},
                {"params": base_params, "lr": tc.lr * tc.base_lr_factor},
            ]
        else:
            param_groups = [{"params": lora_params + base_params, "lr": tc.lr}]

        return torch.optim.AdamW(
            param_groups,
            weight_decay=tc.weight_decay,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup scheduler matching main branch."""
        warmup = max(self.config.training.warmup_steps, 1)
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
            if (
                global_step > 0
                and self.config.sigreg.flush_on_curriculum_switch
                and self.curriculum.did_switch(global_step - 1, global_step)
            ):
                self.world_model.flush_sigreg_buffer()

            batch = self.data_source.sample_batch(self.config.training.episodes_per_step)
            metrics = self.train_one_step(global_step=global_step, batch=batch)
            if self.logger is not None:
                self.logger.log_scalars(global_step, metrics.as_dict())
            self._log_wandb(global_step, metrics)

            # Periodic validation
            if val_source is not None and (global_step + 1) % val_every == 0:
                val_metrics = self.evaluate(
                    val_source=val_source,
                    num_batches=val_batches,
                    global_step=global_step,
                )
                self._log_wandb_dict(global_step, val_metrics)

            if checkpoint_root is not None and self._should_save_checkpoint(global_step + 1):
                self.save_checkpoint(
                    self._checkpoint_path(checkpoint_root, step=global_step + 1),
                    step=global_step + 1,
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
        total_transition_count = 0.0

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

            # Accumulate supervision sums (don't backward yet)
            dyn_sum = outputs.losses.dynamics_sum
            rew_sum = outputs.losses.reward_sum
            sup_sum = dyn_sum + lam_rew * rew_sum
            total_sup_sum = sup_sum if total_sup_sum is None else total_sup_sum + sup_sum
            total_dyn_sum = dyn_sum if total_dyn_sum is None else total_dyn_sum + dyn_sum
            total_rew_sum = rew_sum if total_rew_sum is None else total_rew_sum + rew_sum
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
            reg_loss = (
                self.config.sigreg.lambda_ep * step_sigreg.ep
                + self.config.sigreg.lambda_var * step_sigreg.var
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
        normalizer = max(total_transition_count, 1.0)
        dynamics_loss_value = total_dyn_sum / normalizer
        reward_loss_value = total_rew_sum / normalizer
        total_loss = total_sup_sum / normalizer + reg_loss
        total_loss.backward()

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
            cov_unscaled=float(cov_unscaled.detach().item()),
        )

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
    ) -> dict[str, float]:
        """Run validation and return metrics dict."""
        self.world_model.eval()
        horizon = self.curriculum.horizon_at(global_step)

        all_metrics: list[TrainStepMetrics] = []
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
                window_outputs.append(outputs)
                prev_belief = self._terminal_posterior_belief(outputs).detach()
                dyn_sum = outputs.losses.dynamics_sum
                rew_sum = outputs.losses.reward_sum
                total_dyn_sum = dyn_sum if total_dyn_sum is None else total_dyn_sum + dyn_sum
                total_rew_sum = rew_sum if total_rew_sum is None else total_rew_sum + rew_sum
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
                    val_reg = (
                        self.config.sigreg.lambda_ep * val_sigreg.ep
                        + self.config.sigreg.lambda_var * val_sigreg.var
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
                normalizer = max(total_transition_count, 1.0)
                val_dynamics = total_dyn_sum / normalizer
                val_reward = total_rew_sum / normalizer
                val_total = val_dynamics + self.config.training.lambda_reward * val_reward + val_reg

                all_metrics.append(
                    self._aggregate_metrics(
                        window_outputs,
                        horizon=horizon,
                        step_sigreg=val_sigreg,
                        all_valid_beliefs=val_valid_beliefs,
                        total_loss=float(val_total.item()),
                        dynamics_loss=float(val_dynamics.item()),
                        reward_loss=float(val_reward.item()),
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

        # Belief probing on val episodes
        probe_metrics = self._run_probing(val_source, horizon=horizon)
        result.update(probe_metrics)

        return result

    def save_checkpoint(self, path: str | Path, *, step: int) -> None:
        """Save model, optimizer, and step state."""

        checkpoint = {
            "model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }
        torch.save(checkpoint, Path(path))

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model, optimizer, and return the saved global step."""

        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
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

        # Select a subset of episodes
        subset = episodes[:num_episodes]

        # Extract beliefs by running forward pass on each episode
        self.world_model.eval()
        all_beliefs = []  # list of (K, D) tensors — one per valid posterior step
        all_state_labels = []  # list of state_label dicts

        for ep in subset:
            if "state_labels" not in ep.metadata:
                continue
            ep_labels = ep.metadata["state_labels"]
            T_seq = ep.episode_length + 1  # T_env + 1

            # Build a single-episode batch
            obs = ep.obs_tokens[:T_seq].unsqueeze(0).to(self.device)  # (1, T, K, D)
            acts = ep.actions[:T_seq].unsqueeze(0).to(self.device, dtype=torch.long)
            rews = ep.rewards[:T_seq].unsqueeze(0).to(self.device, dtype=torch.float32)
            lengths = torch.tensor([T_seq], dtype=torch.long, device=self.device)

            belief = self.world_model.get_initial_belief(1, dtype=obs.dtype)

            # Step through the episode collecting posterior beliefs
            null_action = torch.tensor(
                [self.config.env.null_action_id], dtype=torch.long, device=self.device,
            )
            prev_action = null_action

            for t in range(T_seq):
                belief = self.world_model.posterior_step(
                    prev_belief=belief,
                    prev_actions=prev_action,
                    observation_tokens=obs[:, t],
                )

                # Collect belief slots and matching label
                slots = belief.slots.detach().cpu()  # (1, K, D)
                if t < len(ep_labels) and ep_labels[t].get("player_pos") is not None:
                    all_beliefs.append(slots.squeeze(0))  # (K, D)
                    all_state_labels.append(ep_labels[t])

                if t < T_seq - 1:
                    prev_action = acts[:, t]

        self.world_model.train()

        if not all_beliefs or not all_state_labels:
            return {}

        # Align counts
        N = min(len(all_beliefs), len(all_state_labels))
        beliefs_tensor = torch.stack(all_beliefs[:N])  # (N, K, D)
        labels = extract_sokoban_labels(all_state_labels[:N])

        # Build and train probes (re-enable grad for probe training)
        with torch.enable_grad():
            probe = build_sokoban_probes(
                hidden_dim=self.config.hidden_dim,
                num_belief_slots=self.config.belief.num_slots,
                device="cpu",
            )
            probe.fit(beliefs_tensor, labels, epochs=probe_epochs, lr=1e-3)
            probe_results = probe.evaluate(beliefs_tensor, labels)

        # Format results
        result: dict[str, float] = {}
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
        cov_unscaled: float = 0.0,
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

        for out in outputs:
            post_b = out.rollout.posterior.beliefs.detach()  # (B, T, K, D)
            pri_b = out.rollout.prior.beliefs.detach()       # (B, T, K, D)
            dynamics_mask = out.dynamics_mask.detach()
            posterior_mask = out.posterior_valid_mask.detach()
            total_transition_count += float(dynamics_mask.sum().item())
            total_posterior_count += float(posterior_mask.sum().item())
            total_dyn_sum = total_dyn_sum + out.losses.dynamics_sum.detach()
            total_rew_sum = total_rew_sum + out.losses.reward_sum.detach()

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
            sigreg_ep = step_sigreg.ep.item()
            sigreg_var = step_sigreg.var.item()
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
        for out in outputs:
            all_post_logits.append(out.posterior_reward_logits.detach())
            all_prior_logits.append(out.prior_reward_logits.detach())
            all_targets.append(out.reward_targets.detach())
            all_reward_masks.append(out.reward_mask.detach())

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

        reward_positive_rate = 0.0
        reward_logit_mean_post = 0.0
        reward_logit_mean_pri = 0.0
        reward_logit_gap = 0.0
        reward_post_minus_pri_on_positive = 0.0

        if reward_mask.sum() > 0:
            valid_post_logits = post_logits[reward_mask]
            valid_prior_logits = prior_logits[reward_mask]
            valid_post_preds = (post_logits[reward_mask] > 0).float()
            valid_prior_preds = (prior_logits[reward_mask] > 0).float()
            valid_targets = targets[reward_mask]
            post_reward_acc = (valid_post_preds == valid_targets).float().mean().item()
            prior_reward_acc = (valid_prior_preds == valid_targets).float().mean().item()

            # Reward diagnostics
            reward_positive_rate = valid_targets.mean().item()
            reward_logit_mean_post = valid_post_logits.mean().item()
            reward_logit_mean_pri = valid_prior_logits.mean().item()
            reward_logit_gap = (valid_post_logits - valid_prior_logits).abs().mean().item()

            # Precision / Recall on positive class
            pos_mask = valid_targets == 1.0
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                post_reward_recall = (valid_post_preds[pos_mask] == 1.0).float().mean().item()
                prior_reward_recall = (valid_prior_preds[pos_mask] == 1.0).float().mean().item()
                reward_post_minus_pri_on_positive = (
                    valid_post_logits[pos_mask] - valid_prior_logits[pos_mask]
                ).mean().item()
            post_pred_pos = valid_post_preds == 1.0
            prior_pred_pos = valid_prior_preds == 1.0
            if post_pred_pos.sum() > 0:
                post_reward_precision = (valid_targets[post_pred_pos] == 1.0).float().mean().item()
            if prior_pred_pos.sum() > 0:
                prior_reward_precision = (valid_targets[prior_pred_pos] == 1.0).float().mean().item()

            post_reward_f1 = self._binary_f1(post_reward_precision, post_reward_recall)
            prior_reward_f1 = self._binary_f1(prior_reward_precision, prior_reward_recall)
            post_reward_auroc = self._binary_auroc(valid_post_logits, valid_targets)
            prior_reward_auroc = self._binary_auroc(valid_prior_logits, valid_targets)

        if total_loss is None:
            reg_total = 0.0
            if step_sigreg is not None:
                reg_total = (
                    self.config.sigreg.lambda_ep * step_sigreg.ep.item()
                    + self.config.sigreg.lambda_var * step_sigreg.var.item()
                )
            total_loss = (
                dynamics_loss
                + self.config.training.lambda_reward * reward_loss
                + reg_total
            )

        return TrainStepMetrics(
            total=total_loss,
            dynamics=dynamics_loss,
            reward=reward_loss,
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
        return root / "latest.pt"

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
