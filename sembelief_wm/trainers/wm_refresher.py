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

from dataclasses import dataclass
from typing import Any

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

    grad_clip: float = 1.0
    """Max gradient norm."""


@dataclass
class RefreshMetrics:
    """Metrics from one refresh cycle."""

    num_steps: int
    avg_dynamics_loss: float
    avg_reward_loss: float
    avg_total_loss: float
    refresh_step: int


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

    def __init__(self, phase1_trainer: Any, config: WMRefreshConfig) -> None:
        """
        Args:
            phase1_trainer: An instance of sembelief_wm.train.trainer.Phase1Trainer.
                           We use Any to avoid importing it (keeps this module
                           decoupled from the full training infrastructure).
            config: Refresh configuration.
        """
        self._trainer = phase1_trainer
        self._config = config
        self._refresh_step = 0

    @property
    def refresh_step(self) -> int:
        return self._refresh_step

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

        # Enable gradients for supervised update
        wm.requires_grad_(True)
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
            wm.requires_grad_(False)
            wm.eval()

        n = max(1, self._config.updates_per_refresh)
        return RefreshMetrics(
            num_steps=n,
            avg_dynamics_loss=totals.get("loss/dynamics", 0.0) / n,
            avg_reward_loss=totals.get("loss/reward", 0.0) / n,
            avg_total_loss=totals.get("loss/total", 0.0) / n,
            refresh_step=self._refresh_step,
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
