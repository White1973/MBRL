"""PPO updater.

Takes a PPOBatch + a Policy, runs clipped PPO updates.
Does not know where the batch came from (real env, imagined rollout, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .trajectory import PPOBatch
from .policy import Policy


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    epochs: int = 4
    minibatch_size: int = 512
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.0
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    lr: float = 3e-4
    weight_decay: float = 0.0


@dataclass
class PPOMetrics:
    """Metrics from one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    kl_divergence: float
    explained_variance: float
    value_delta: float


class PPOUpdater:
    """Stateful PPO updater that owns the optimizer.

    Uses a single optimizer over all policy.parameters() plus any
    extra_params (e.g., shared backbone LoRA parameters that are not
    registered as submodules of the policy).

    Usage:
        updater = PPOUpdater(config, policy)
        metrics = updater.update(batch)

    For shared backbone setups:
        shared_params = backbone.trainable_parameters()
        updater = PPOUpdater(config, policy, extra_params=shared_params)
    """

    def __init__(
        self,
        config: PPOConfig,
        policy: Policy,
        extra_params: list[nn.Parameter] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy

        # Collect all parameters to optimize: policy's own + any extras
        all_params = list(policy.parameters())
        if extra_params:
            # Deduplicate by data_ptr to avoid issues with overlapping param sets
            seen = {p.data_ptr() for p in all_params}
            for p in extra_params:
                if p.data_ptr() not in seen:
                    all_params.append(p)
                    seen.add(p.data_ptr())
        self._all_trainable = [p for p in all_params if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            self._all_trainable,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def update(self, batch: PPOBatch) -> PPOMetrics:
        """Run multi-epoch minibatch PPO on a flattened batch."""

        cfg = self.config
        states = batch.states
        actions = batch.actions
        advantages = batch.advantages.clone()
        returns = batch.returns
        old_log_probs = batch.old_log_probs
        old_values = batch.old_values

        N = actions.shape[0]
        if N == 0:
            return PPOMetrics(
                policy_loss=0.0, value_loss=0.0, entropy=0.0,
                clip_fraction=0.0, kl_divergence=0.0,
                explained_variance=0.0, value_delta=0.0,
            )

        if cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)

        minibatch_size = min(cfg.minibatch_size, N)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_kl = 0.0
        num_minibatches = 0

        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=states.device)
            for start in range(0, N, minibatch_size):
                idx = perm[start : start + minibatch_size]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_old_log_probs = old_log_probs[idx]

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    mb_states, mb_actions
                )
                entropy_mean = entropy.mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                )
                surr = torch.min(ratio * mb_advantages, clipped * mb_advantages)
                policy_loss = -surr.mean()

                value_loss = (values - mb_returns).pow(2).mean()
                clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float().mean()
                kl_div = (mb_old_log_probs - new_log_probs).mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                    + cfg.kl_coef * kl_div
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._all_trainable,
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_clipfrac += clipfrac.item()
                total_kl += kl_div.item()
                num_minibatches += 1

        n = max(1, num_minibatches)

        # Post-update diagnostics
        with torch.no_grad():
            updated_values = self.policy.evaluate_actions(states, actions)[2]
            ev = _explained_variance(updated_values, returns)
            vdelta = (updated_values - old_values).abs().mean().item()

        return PPOMetrics(
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / n,
            entropy=total_entropy / n,
            clip_fraction=total_clipfrac / n,
            kl_divergence=total_kl / n,
            explained_variance=ev,
            value_delta=vdelta,
        )

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])


def _explained_variance(values: Tensor, targets: Tensor) -> float:
    target_var = torch.var(targets)
    if not torch.isfinite(target_var) or target_var.item() < 1e-8:
        return 0.0
    residual = torch.var(targets - values)
    return float((1.0 - residual / target_var).item())
