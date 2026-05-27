"""Imagined trajectory collector.

Rolls out a policy in latent space using a dynamics model and reward
predictor. Produces Trajectory objects that can be directly fed to
GAE / PPO via trajectory_to_ppo_batch().

This module does not know about PPO, optimizers, or training.
It only produces data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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
        get_belief_slots: Callable[[Any], Tensor],
        config: ImaginedCollectorConfig,
    ) -> None:
        self.policy = policy
        self.dynamics_step = dynamics_step
        self.predict_reward = predict_reward
        self.reward_transform = reward_transform
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
            Trajectory with shape (B, H) — no mask (all steps valid in
            imagined rollout), no dones (no early termination).
        """
        H = self.config.horizon
        belief = start_beliefs

        all_states: list[Tensor] = []
        all_actions: list[Tensor] = []
        all_log_probs: list[Tensor] = []
        all_rewards: list[Tensor] = []
        all_values: list[Tensor] = []

        for _ in range(H):
            # Extract flat state representation for policy
            state_tensor = self.get_belief_slots(belief)
            action, log_prob, _entropy, value = self.policy.act(
                state_tensor, **kwargs
            )

            # Dynamics step in latent space
            next_belief = self.dynamics_step(belief, action, **kwargs)

            # Reward prediction on next_belief
            reward_logits = self.predict_reward(next_belief)
            reward = self.reward_transform(reward_logits, **kwargs)

            # Ensure reward is (B,) — some reward heads return (B, 1)
            if reward.ndim > 1:
                reward = reward.squeeze(-1)

            all_states.append(state_tensor.detach())
            all_actions.append(action.detach())
            all_log_probs.append(log_prob.detach())
            all_rewards.append(reward.detach())
            all_values.append(value.detach())

            belief = next_belief

        # Bootstrap value at the end of the horizon
        if self.config.bootstrap_with_value:
            final_state = self.get_belief_slots(belief)
            _, _, _, bootstrap_value = self.policy.act(final_state, **kwargs)
            bootstrap_value = bootstrap_value.detach()
        else:
            bootstrap_value = torch.zeros_like(all_values[0])

        # Stack: (B, H) for scalars, (B, H, ...) for states
        values_with_bootstrap = torch.stack(all_values + [bootstrap_value], dim=1)

        B = all_actions[0].shape[0]
        return Trajectory(
            states=torch.stack(all_states, dim=1),        # (B, H, *state_shape)
            actions=torch.stack(all_actions, dim=1),       # (B, H)
            rewards=torch.stack(all_rewards, dim=1),       # (B, H)
            dones=torch.zeros(B, H, device=all_actions[0].device),  # no termination
            log_probs=torch.stack(all_log_probs, dim=1),   # (B, H)
            values=values_with_bootstrap,                   # (B, H+1)
            mask=None,  # all steps valid in imagined rollout
        )
