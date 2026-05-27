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
    checkpoint_every: int = 50

    # Imagined rollout
    rollout_batch_size: int = 256
    rollout_horizon: int = 8

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

        for update in range(start_update, cfg.total_updates):
            t0 = time.time()
            current_update = update + 1
            metrics: dict[str, float] = {
                "update": float(current_update),
                "progress": float(current_update) / float(cfg.total_updates),
            }

            # --- 1. Collect real episodes (periodic) ---
            if self._should_collect(current_update):
                t_collect = time.time()
                assert self.real_collector is not None
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

            # --- 2. World model refresh (periodic) ---
            if self._should_refresh_wm(current_update):
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
                elif isinstance(refresh_result, dict):
                    metrics.update(
                        {f"wm_refresh/{k}": v for k, v in refresh_result.items()}
                    )
                metrics["timing/wm_refresh_sec"] = time.time() - t_refresh

            # --- 3. Imagined rollout ---
            t_rollout = time.time()
            start_beliefs = self.sample_beliefs_fn(cfg.rollout_batch_size)
            trajectory = self.imagine_fn(start_beliefs)
            metrics["timing/rollout_sec"] = time.time() - t_rollout

            # Trajectory diagnostics
            metrics.update(self._trajectory_metrics(trajectory))

            # --- 4. PPO update ---
            t_ppo = time.time()
            ppo_batch = trajectory_to_ppo_batch(
                trajectory,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            ppo_metrics = self.ppo_updater.update(ppo_batch)
            metrics.update(self._ppo_metrics_to_dict(ppo_metrics))
            metrics["timing/ppo_sec"] = time.time() - t_ppo

            # --- 5. Real-env evaluation (periodic) ---
            if self._should_eval(current_update):
                t_eval = time.time()
                if self.evaluator is not None:
                    eval_metrics = self.evaluator.evaluate(cfg.eval_episodes)
                    metrics.update(eval_metrics)
                metrics["timing/eval_sec"] = time.time() - t_eval

            # --- Logging ---
            elapsed = time.time() - t0
            metrics["timing/total_sec"] = elapsed

            if self.logger is not None:
                self.logger.log_scalars(current_update, metrics)

            # Console summary
            rew = metrics.get("rollout/reward_mean", 0)
            ploss = metrics.get("ppo/policy_loss", 0)
            sr = metrics.get("eval/success_rate", -1)
            sr_str = f" | eval_sr={sr:.2f}" if sr >= 0 else ""
            print(
                f"[update {current_update}/{cfg.total_updates}] "
                f"rollout={metrics.get('timing/rollout_sec', 0):.1f}s "
                f"ppo={metrics.get('timing/ppo_sec', 0):.1f}s "
                f"total={elapsed:.1f}s "
                f"rew={rew:.4f} ploss={ploss:.4f}{sr_str}",
                flush=True,
            )

            # --- Checkpoint ---
            if checkpoint_root is not None and self._should_checkpoint(current_update):
                self._save_checkpoint(
                    checkpoint_root / "latest.pt", current_update
                )

    # -----------------------------------------------------------------------
    # Schedule helpers
    # -----------------------------------------------------------------------

    def _should_eval(self, update: int) -> bool:
        return self.config.eval_every > 0 and update % self.config.eval_every == 0

    def _should_collect(self, update: int) -> bool:
        return (
            self.config.collect_every > 0
            and update % self.config.collect_every == 0
            and self.real_collector is not None
        )

    def _should_refresh_wm(self, update: int) -> bool:
        return (
            self.config.wm_refresh_every > 0
            and update % self.config.wm_refresh_every == 0
            and self.wm_refresher is not None
        )

    def _should_checkpoint(self, update: int) -> bool:
        return self.config.checkpoint_every > 0 and update % self.config.checkpoint_every == 0

    # -----------------------------------------------------------------------
    # Metrics extraction (no algorithm logic, just reading fields)
    # -----------------------------------------------------------------------

    @staticmethod
    def _trajectory_metrics(traj: Trajectory) -> dict[str, float]:
        return {
            "rollout/reward_mean": float(traj.rewards.mean().item()),
            "rollout/reward_std": float(traj.rewards.std().item()),
            "rollout/reward_nonzero_rate": float(
                (traj.rewards != 0).float().mean().item()
            ),
        }

    @staticmethod
    def _ppo_metrics_to_dict(m: PPOMetrics) -> dict[str, float]:
        return {
            "ppo/policy_loss": m.policy_loss,
            "ppo/value_loss": m.value_loss,
            "ppo/entropy": m.entropy,
            "ppo/clip_fraction": m.clip_fraction,
            "ppo/kl_divergence": m.kl_divergence,
            "ppo/explained_variance": m.explained_variance,
        }

    # -----------------------------------------------------------------------
    # Checkpoint (pipeline-level, delegates to components)
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, path: Path, update: int) -> None:
        checkpoint: dict[str, Any] = {
            "update": update,
            "policy": self.policy.state_dict(),
            "ppo_optimizer": self.ppo_updater.state_dict(),
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
        self.policy.load_state_dict(ckpt["policy"])
        self.ppo_updater.load_state_dict(ckpt["ppo_optimizer"])
        if self.wm_refresher is not None and "wm_refresher" in ckpt:
            self.wm_refresher.load_state_dict(ckpt["wm_refresher"])
        # Restore world model weights (critical for alternating mode)
        if self.world_model is not None and "world_model" in ckpt:
            self.world_model.load_state_dict(ckpt["world_model"])
        return int(ckpt.get("update", 0))
