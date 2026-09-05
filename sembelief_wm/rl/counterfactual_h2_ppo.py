"""Exact finite-H2 counterfactual targets for an action-conditioned Critic.

The generic GAE path assumes that trajectory values are scalar ``V(s)``.
The deployed slotwise Critic instead returns ``Q(s, a)``.  Feeding those
selected-action Q values to GAE turns a correct action-value function into an
almost-zero TD residual and removes the action-ranking signal seen by PPO.

This module builds the Actor and Critic targets directly from an exhaustive
4x4 two-step panel.  It intentionally contains no Sokoban-specific logic: the
collector remains responsible for dynamics, termination and rewards.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .trajectory import PPOBatch, Trajectory


NUM_ACTIONS = 4
SEQUENCES_PER_STATE = NUM_ACTIONS * NUM_ACTIONS


def ordered_h2_action_sequences(
    num_states: int, *, device: torch.device,
) -> Tensor:
    """Return ``[00, 01, ..., 33]`` for every input state."""
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    actions = torch.arange(NUM_ACTIONS, device=device, dtype=torch.long)
    pairs = torch.cartesian_prod(actions, actions)
    return pairs.repeat(num_states, 1)


def counterfactual_h2_ppo_batch(
    trajectory: Trajectory,
    *,
    gamma: float,
) -> tuple[PPOBatch, dict[str, float]]:
    """Build exact Q targets and policy-weighted PPO advantages.

    ``trajectory`` must contain all 16 ordered ``(a0, a1)`` sequences for
    each unique start state.  For every first action, its finite-H2 target is

        Q_H2(s,a0) = E_{a1~pi_old(.|s1)}[r0 + gamma*r1].

    The four first actions are enumerated rather than sampled.  Consequently
    their PPO advantages are importance weighted by ``4*pi_old(a0|s)`` so the
    uniform four-action average has the same policy-gradient expectation as
    sampling ``a0`` from the old policy.
    """
    if trajectory.horizon != 2:
        raise ValueError(
            "counterfactual H2 PPO requires trajectory horizon exactly two"
        )
    total = trajectory.batch_size
    if total % SEQUENCES_PER_STATE != 0:
        raise ValueError(
            "counterfactual H2 trajectory size must be divisible by 16"
        )
    groups = total // SEQUENCES_PER_STATE
    actions = trajectory.actions.long().reshape(groups, NUM_ACTIONS, NUM_ACTIONS, 2)
    expected = ordered_h2_action_sequences(groups, device=actions.device).reshape(
        groups, NUM_ACTIONS, NUM_ACTIONS, 2
    )
    if not torch.equal(actions, expected):
        raise RuntimeError(
            "counterfactual H2 trajectory lost ordered 4x4 action sequences"
        )

    state_shape = trajectory.states.shape[2:]
    states = trajectory.states.reshape(
        groups, NUM_ACTIONS, NUM_ACTIONS, 2, *state_shape
    )
    log_probs = trajectory.log_probs.float().reshape(
        groups, NUM_ACTIONS, NUM_ACTIONS, 2
    )
    old_values = trajectory.values[:, :2].float().reshape(
        groups, NUM_ACTIONS, NUM_ACTIONS, 2
    )
    rewards = trajectory.rewards.float().reshape(
        groups, NUM_ACTIONS, NUM_ACTIONS, 2
    )
    if trajectory.mask is None:
        mask = torch.ones_like(rewards)
    else:
        mask = trajectory.mask.float().reshape_as(rewards)

    # All a1 branches sharing (s,a0) must have the same s0, log pi(a0|s),
    # and Q_old(s,a0).  Taking branch zero is therefore exact, not sampling.
    first_states = states[:, :, 0, 0]
    first_log_probs = log_probs[:, :, 0, 0]
    first_old_values = old_values[:, :, 0, 0]
    first_probabilities = first_log_probs.exp()
    first_probabilities = first_probabilities / first_probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)

    second_probabilities = log_probs[:, :, :, 1].exp()
    second_probabilities = second_probabilities / second_probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    sequence_returns = (
        rewards[..., 0] * mask[..., 0]
        + float(gamma) * rewards[..., 1] * mask[..., 1]
    )
    q_targets = (second_probabilities * sequence_returns).sum(dim=-1)
    state_values = (first_probabilities * q_targets).sum(
        dim=-1, keepdim=True
    )
    centered_advantages = q_targets - state_values

    # The PPO batch contains every action uniformly.  Multiplication by
    # |A|*pi_old converts that uniform empirical expectation back into the
    # on-policy expectation without losing guaranteed four-action coverage.
    ppo_advantages = (
        float(NUM_ACTIONS) * first_probabilities * centered_advantages
    )
    weighted_zero_mean = (
        first_probabilities * centered_advantages
    ).sum(dim=-1)
    zero_error = float(weighted_zero_mean.abs().max())
    if zero_error > 1e-5:
        raise RuntimeError(
            "counterfactual H2 policy-centered advantage failed its "
            f"zero-mean invariant: max_error={zero_error:.3e}"
        )

    batch = PPOBatch(
        states=first_states.flatten(0, 1),
        actions=torch.arange(
            NUM_ACTIONS, device=actions.device, dtype=torch.long
        ).repeat(groups),
        advantages=ppo_advantages.flatten(),
        returns=q_targets.flatten(),
        old_log_probs=first_log_probs.flatten(),
        old_values=first_old_values.flatten(),
    )
    metrics = {
        "counterfactual_h2/base_states": float(groups),
        "counterfactual_h2/sequences": float(total),
        "counterfactual_h2/action_coverage": float(
            torch.unique(batch.actions).numel()
        ),
        "counterfactual_h2/weighted_advantage_zero_error": float(
            zero_error
        ),
        "counterfactual_h2/raw_advantage_abs_mean": float(
            centered_advantages.abs().mean()
        ),
        "counterfactual_h2/q_margin_mean": float(
            q_targets.topk(2, dim=-1).values.diff(dim=-1).neg().mean()
        ),
    }
    return batch, metrics
