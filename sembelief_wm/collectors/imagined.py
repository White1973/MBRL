"""Imagined trajectory collector.

Rolls out a policy in latent space using a dynamics model and reward
predictor. Produces Trajectory objects that can be directly fed to
GAE / PPO via trajectory_to_ppo_batch().

This module does not know about PPO, optimizers, or training.
It only produces data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor

from ..rl.trajectory import Trajectory
from ..rl.policy import Policy


@dataclass
class ImaginedCollectorConfig:
    """Configuration for imagined rollout."""

    horizon: int = 8
    batch_size: int = 256
    bootstrap_with_value: bool = False  # False = zero bootstrap (finite-horizon)
    # ``fixed_horizon`` is required when the reward head predicts whether the
    # H-step endpoint succeeds (terminal_success).  Such a classifier is not a
    # per-step termination model and therefore must never truncate a rollout.
    # ``predicted_success`` is reserved for dense, per-transition heads whose
    # probability really does mean "the episode ended at this step".
    termination_mode: Literal["fixed_horizon", "predicted_success"] = (
        "predicted_success"
    )
    success_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.termination_mode not in {"fixed_horizon", "predicted_success"}:
            raise ValueError(
                "termination_mode must be 'fixed_horizon' or "
                "'predicted_success'"
            )
        if not 0.0 < self.success_threshold < 1.0:
            raise ValueError("success_threshold must be strictly between 0 and 1")


class ImaginedCollector:
    """Collect imagined trajectories in latent space.

    Dependencies are injected as callables/protocols so this class
    does not import WorldModel, Config, or any concrete types.

    Args:
        policy: act(states, **kwargs) -> (action, log_prob, entropy, value)
        dynamics_step: (belief, action, **kwargs) -> next_belief
        predict_reward: (belief) -> reward_logits (raw)
        reward_transform: (reward_logits, **kwargs) -> scalar reward
        get_belief_slots: (belief) -> Tensor (B, K, D) — extract tensor from belief object
        config: rollout configuration
    """

    def __init__(
        self,
        *,
        policy: Policy,
        dynamics_step: Callable[..., Any],
        predict_reward: Callable[[Any], Tensor],
        reward_transform: Callable[..., Tensor],
        relative_action_value: Callable[[Any, Tensor], Tensor] | None = None,
        get_belief_slots: Callable[[Any], Tensor],
        config: ImaginedCollectorConfig,
    ) -> None:
        self.policy = policy
        self.dynamics_step = dynamics_step
        self.predict_reward = predict_reward
        self.reward_transform = reward_transform
        self.relative_action_value = relative_action_value
        self.get_belief_slots = get_belief_slots
        self.config = config

    @torch.no_grad()
    def collect(
        self,
        start_beliefs: Any,
        **kwargs: Any,
    ) -> Trajectory:
        """Roll out from start_beliefs for H steps.

        Args:
            start_beliefs: initial belief states, any type accepted by
                           dynamics_step and policy.act.
            **kwargs: extra context (e.g. env_ids) forwarded to dynamics_step
                      and reward_transform.

        Returns:
            Trajectory with shape (B, H), a validity mask for padding after
            predicted success, and done flags that cut GAE bootstrapping.
        """
        # Critic multi-continuation targets estimate Q(s, a), so replicas must
        # condition on the *same* first action and vary only the continuation.
        # Consume this collector-only argument instead of forwarding it to the
        # dynamics/reward callables.
        forced_first_actions = kwargs.pop("forced_first_actions", None)
        forced_action_sequences = kwargs.pop("forced_action_sequences", None)
        if forced_first_actions is not None and forced_action_sequences is not None:
            raise ValueError(
                "forced_first_actions and forced_action_sequences are mutually exclusive"
            )
        H = self.config.horizon
        belief = start_beliefs
        stabilize = getattr(
            self.policy,
            "set_deterministic_forward_mode",
            None,
        )
        if callable(stabilize):
            stabilize()

        all_states: list[Tensor] = []
        all_actions: list[Tensor] = []
        all_log_probs: list[Tensor] = []
        all_rewards: list[Tensor] = []
        all_reward_logits: list[Tensor] = []
        all_base_rewards: list[Tensor] = []
        all_shaping_rewards: list[Tensor] = []
        relative_diagnostics: dict[str, list[Tensor]] = {}
        all_values: list[Tensor] = []
        per_step_dones: list[Tensor] = []
        per_step_mask: list[Tensor] = []
        finished: Tensor | None = None  # lazily init from first reward_logits

        for step_index in range(H):
            # Extract flat state representation for policy
            state_tensor = self.get_belief_slots(belief)
            action, log_prob, _entropy, value = self.policy.act(
                state_tensor, **kwargs
            )
            forced_action = None
            if forced_action_sequences is not None:
                sequence = torch.as_tensor(
                    forced_action_sequences, device=action.device, dtype=action.dtype
                )
                if sequence.ndim != 2 or sequence.shape != (len(action), H):
                    raise ValueError(
                        "forced_action_sequences must have shape (batch, horizon)"
                    )
                forced_action = sequence[:, step_index]
            elif step_index == 0 and forced_first_actions is not None:
                forced_action = torch.as_tensor(
                    forced_first_actions,
                    device=action.device,
                    dtype=action.dtype,
                )
            if forced_action is not None:
                action = forced_action
                if action.shape != log_prob.shape:
                    raise ValueError(
                        "forced actions must match the rollout batch shape"
                    )
                log_prob, _entropy, value = self.policy.evaluate_actions(
                    state_tensor, action, **kwargs
                )

            # Dynamics step in latent space
            next_belief = self.dynamics_step(belief, action, **kwargs)

            # Reward prediction on next_belief
            reward_logits = self.predict_reward(next_belief)
            base_reward = self.reward_transform(
                reward_logits, step_index=step_index, horizon=H, **kwargs
            )
            reward = base_reward
            shaping_reward = torch.zeros_like(base_reward)
            if self.relative_action_value is not None:
                relative_result = self.relative_action_value(belief, action)
                if isinstance(relative_result, tuple):
                    shaping_reward, step_diagnostics = relative_result
                    for name, value in step_diagnostics.items():
                        relative_diagnostics.setdefault(name, []).append(value.detach())
                else:
                    shaping_reward = relative_result
                reward = reward + shaping_reward

            # Ensure reward is (B,) — some reward heads return (B, 1)
            if reward.ndim > 1:
                reward = reward.squeeze(-1)
            if base_reward.ndim > 1:
                base_reward = base_reward.squeeze(-1)
            if shaping_reward.ndim > 1:
                shaping_reward = shaping_reward.squeeze(-1)
            diagnostic_logits = reward_logits
            if diagnostic_logits.ndim > 1:
                diagnostic_logits = diagnostic_logits.squeeze(-1)

            # Lazy-init finished from first reward_logits shape
            if finished is None:
                finished = torch.zeros_like(reward, dtype=torch.bool)

            if self.config.termination_mode == "predicted_success":
                # Only a head trained as a per-transition termination model may
                # decide that later imagined steps are padding.
                success_prob = torch.sigmoid(reward_logits)
                if success_prob.ndim > 1:
                    success_prob = success_prob.squeeze(-1)
                is_success = (
                    (success_prob >= self.config.success_threshold) & (~finished)
                )
            else:
                # A fixed-horizon endpoint classifier supplies reward, not
                # termination.  Keep every imagined step valid and let the
                # finite-horizon zero bootstrap end the return calculation.
                is_success = torch.zeros_like(finished)

            # done = success at this step OR already finished (padding)
            step_done = is_success | finished
            per_step_dones.append(step_done)

            # mask = 1 for samples not yet finished before this step (valid step)
            step_mask = (~finished).to(dtype=reward.dtype)
            per_step_mask.append(step_mask)

            # A finished sample is padding from this point onward.  Keep its
            # stored rewards exactly zero so diagnostics and any downstream
            # consumer agree with the validity mask.
            reward = reward * step_mask
            base_reward = base_reward * step_mask
            shaping_reward = shaping_reward * step_mask

            all_states.append(state_tensor.detach())
            all_actions.append(action.detach())
            all_log_probs.append(log_prob.detach())
            all_rewards.append(reward.detach())
            all_base_rewards.append(base_reward.detach())
            all_shaping_rewards.append(shaping_reward.detach())
            all_reward_logits.append(diagnostic_logits.detach())
            all_values.append(value.detach())

            # Update finished: once success, stay finished for rest of rollout
            finished = finished | is_success
            belief = next_belief

        # Bootstrap value at the end of the horizon
        if self.config.bootstrap_with_value:
            final_state = self.get_belief_slots(belief)
            bootstrap_fn = getattr(self.policy, "bootstrap_value", None)
            if callable(bootstrap_fn):
                bootstrap_value = bootstrap_fn(final_state).detach()
            else:
                _, _, _, bootstrap_value = self.policy.act(final_state, **kwargs)
                bootstrap_value = bootstrap_value.detach()
            # True predicted terminals have no continuation value.  Only a
            # non-terminal fragment truncation may bootstrap.
            assert finished is not None
            bootstrap_value = torch.where(
                finished, torch.zeros_like(bootstrap_value), bootstrap_value
            )
        else:
            bootstrap_value = torch.zeros_like(all_values[0])

        # Stack: (B, H) for scalars, (B, H, ...) for states
        values_with_bootstrap = torch.stack(all_values + [bootstrap_value], dim=1)

        B = all_actions[0].shape[0]
        stacked_dones = torch.stack(per_step_dones, dim=1).to(dtype=all_rewards[0].dtype)  # (B, H)
        stacked_mask = torch.stack(per_step_mask, dim=1)  # (B, H)

        return Trajectory(
            states=torch.stack(all_states, dim=1),        # (B, H, *state_shape)
            actions=torch.stack(all_actions, dim=1),       # (B, H)
            rewards=torch.stack(all_rewards, dim=1),       # (B, H)
            dones=stacked_dones,                           # (B, H) — 1 on success, cuts GAE bootstrap
            log_probs=torch.stack(all_log_probs, dim=1),   # (B, H)
            values=values_with_bootstrap,                   # (B, H+1)
            mask=stacked_mask,  # (B, H) — 0 for steps after success (padding)
            reward_logits=torch.stack(all_reward_logits, dim=1),  # (B, H)
            base_rewards=torch.stack(all_base_rewards, dim=1),
            shaping_rewards=torch.stack(all_shaping_rewards, dim=1),
            relative_score_gap=(
                torch.stack(relative_diagnostics["score_gap"], dim=1)
                if "score_gap" in relative_diagnostics else None
            ),
            relative_top1_top2_margin=(
                torch.stack(relative_diagnostics["top1_top2_margin"], dim=1)
                if "top1_top2_margin" in relative_diagnostics else None
            ),
            relative_selected_rank=(
                torch.stack(relative_diagnostics["selected_rank"], dim=1)
                if "selected_rank" in relative_diagnostics else None
            ),
            relative_selected_is_top1=(
                torch.stack(relative_diagnostics["selected_is_top1"], dim=1)
                if "selected_is_top1" in relative_diagnostics else None
            ),
        )
