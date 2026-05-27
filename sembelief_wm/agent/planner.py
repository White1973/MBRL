"""Lookahead planner for latent-space action selection.

This module implements a parameterized lookahead planning wrapper that can
sit on top of any policy family (MLP or VLM freeform). It is inference-time
only — it does not affect PPO gradient computation.

The planner works by:
1. Expanding candidate actions at the current belief state (exhaustive or top-K)
2. For each candidate, rolling out H steps using the base policy in imagination
3. Scoring each branch by cumulative discounted imagined reward + value bootstrap
4. Selecting the first action of the highest-scoring branch

Parameterization:
  - expansion_depth: how many layers of exhaustive expansion (depth=1 is simplest)
  - expansion_width: how many actions to try per expansion node (num_actions = exhaustive)
  - rollout_horizon: how many policy-rollout steps after expansion for value estimation
  - discount: gamma for cumulative reward scoring
  - use_value_bootstrap: whether to add V(Z_leaf) at the end

Example configurations:
  - depth=1, width=4: try all 4 actions, rollout each, pick best
  - depth=2, width=4: try all 4 actions, then all 4 again, 16 branches
  - depth=2, width=2: try top-2 actions, then top-2 again, 4 branches
"""
from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from ..config import Config, LookaheadConfig
from ..types import BeliefState


class SimulatorInterface(Protocol):
    """Minimal interface the planner needs from the frozen world model."""

    def prior_step(
        self,
        prev_belief: BeliefState,
        prev_actions: Tensor,
        env_ids: Tensor | None = None,
    ) -> BeliefState: ...

    def predict_reward(self, belief: BeliefState | Tensor) -> Tensor: ...


class PolicyInterface(Protocol):
    """Minimal interface the planner needs from the actor-critic."""

    def action_scores(
        self,
        belief: BeliefState | Tensor,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        """Return action scores (B, num_actions).

        Works for any policy family:
        - MLP: returns policy_logits(belief)
        - VLM freeform: returns freeform_policy.action_logits(belief, env_ids)
        """
        ...

    def value(self, belief: BeliefState | Tensor) -> Tensor:
        """Return scalar state value (B,)."""
        ...


class LookaheadPlanner:
    """Parameterized lookahead planner for latent-space action selection.

    This is a stateless planning wrapper. It does not own parameters and
    does not participate in gradient computation.
    """

    def __init__(
        self,
        config: Config,
        simulator: SimulatorInterface,
        policy: PolicyInterface,
        reward_mapper: object,  # callable(logits, env_ids, config) -> scalar rewards
    ) -> None:
        self.la_cfg = config.phase2.lookahead
        self.config = config
        self.simulator = simulator
        self.policy = policy
        self.reward_mapper = reward_mapper  # type: ignore[assignment]
        self.num_actions = config.env.num_actions

    @property
    def enabled(self) -> bool:
        return self.la_cfg.enabled

    @torch.no_grad()
    def select_actions(
        self,
        belief: BeliefState,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        """Select actions via lookahead planning.

        Args:
            belief: Current belief state, shape (B, K, D).
            env_ids: Optional environment IDs, shape (B,).

        Returns:
            Best first actions, shape (B,) int64.
        """
        la_cfg = self.la_cfg
        batch_size = belief.batch_size
        device = belief.slots.device

        # --- Phase 1: Expansion ---
        # Build the expansion tree. After expansion we have a set of branches,
        # each defined by a sequence of actions taken during expansion.
        # The first action of each branch is what we ultimately want to select.

        # Start: (B, K, D) belief, expand into (B * num_branches, K, D)
        # where num_branches = width^depth

        width = min(la_cfg.expansion_width, self.num_actions)
        depth = la_cfg.expansion_depth

        # current_beliefs: (B * current_branches, K, D)
        current_beliefs = belief
        current_env_ids = env_ids  # (B,) or None

        # Track which first-action each branch belongs to.
        # After depth=1: first_actions is (B * width,) with each group of `width`
        # sharing the same batch element but different first actions.
        first_actions: Tensor | None = None
        current_branches = 1  # branches per batch element

        for d in range(depth):
            B_expanded = batch_size * current_branches
            K = current_beliefs.slots.shape[-2]
            D = current_beliefs.slots.shape[-1]

            # Determine which actions to expand
            if width >= self.num_actions:
                # Exhaustive: try all actions
                expand_actions = torch.arange(
                    self.num_actions, device=device
                ).unsqueeze(0).expand(B_expanded, -1)  # (B_expanded, num_actions)
                actual_width = self.num_actions
            else:
                # Selective: pick top-K by action scores
                logits = self.policy.action_scores(
                    current_beliefs, current_env_ids,
                )  # (B_expanded, num_actions)
                _, top_indices = logits.topk(width, dim=-1)  # (B_expanded, width)
                expand_actions = top_indices
                actual_width = width

            # Replicate beliefs for each candidate action
            # (B_expanded, K, D) -> (B_expanded, actual_width, K, D) -> (B_expanded * actual_width, K, D)
            rep_slots = current_beliefs.slots.unsqueeze(1).expand(
                -1, actual_width, -1, -1
            ).reshape(B_expanded * actual_width, K, D)

            # Flatten actions: (B_expanded * actual_width,)
            flat_actions = expand_actions.reshape(-1)

            # Replicate env_ids
            if current_env_ids is not None:
                rep_env_ids = current_env_ids.unsqueeze(1).expand(
                    -1, actual_width
                ).reshape(-1)
            else:
                rep_env_ids = None

            # Step the simulator
            next_beliefs = self.simulator.prior_step(
                prev_belief=BeliefState(slots=rep_slots),
                prev_actions=flat_actions,
                env_ids=rep_env_ids,
            )

            # Track first actions
            if d == 0:
                # first_actions: (B * actual_width,) — the action we took at depth 0
                # For batch element b, positions [b*actual_width : (b+1)*actual_width]
                # hold the candidate first actions
                first_actions = flat_actions  # (batch_size * actual_width,)
            else:
                # Each existing branch spawns `actual_width` children.
                # Repeat first_actions accordingly.
                assert first_actions is not None
                first_actions = first_actions.unsqueeze(1).expand(
                    -1, actual_width
                ).reshape(-1)

            current_beliefs = next_beliefs
            current_env_ids = rep_env_ids
            current_branches *= actual_width

        assert first_actions is not None
        total_branches = current_branches  # width^depth per batch element

        # --- Phase 2: Policy rollout from each leaf ---
        # current_beliefs: (B * total_branches, K, D)
        # Roll out `rollout_horizon` steps using the base policy to accumulate reward.

        cumulative_reward = torch.zeros(
            batch_size * total_branches, device=device, dtype=current_beliefs.slots.dtype
        )
        discount_factor = 1.0

        for _ in range(la_cfg.rollout_horizon):
            # Get greedy policy action at current belief
            logits = self.policy.action_scores(
                current_beliefs, current_env_ids,
            )  # (B*total_branches, num_actions)
            rollout_action = logits.argmax(dim=-1)  # greedy for evaluation

            # Step simulator
            next_belief = self.simulator.prior_step(
                prev_belief=current_beliefs,
                prev_actions=rollout_action,
                env_ids=current_env_ids,
            )

            # Accumulate reward
            reward_logits = self.simulator.predict_reward(next_belief)
            reward = self.reward_mapper(reward_logits, current_env_ids, self.config)  # type: ignore[operator]
            cumulative_reward += discount_factor * reward
            discount_factor *= la_cfg.discount

            current_beliefs = next_belief

        # Optional value bootstrap at the leaf
        if la_cfg.use_value_bootstrap:
            leaf_value = self.policy.value(current_beliefs)  # (B * total_branches,)
            cumulative_reward += discount_factor * leaf_value

        # --- Phase 3: Select best first action per batch element ---
        # Reshape to (B, total_branches) and pick argmax
        scores = cumulative_reward.view(batch_size, total_branches)
        best_branch = scores.argmax(dim=-1)  # (B,)

        # Map branch index back to first action
        first_actions_matrix = first_actions.view(batch_size, total_branches)
        best_actions = first_actions_matrix.gather(
            dim=1, index=best_branch.unsqueeze(1)
        ).squeeze(1)

        return best_actions


def make_planner(
    config: Config,
    simulator: SimulatorInterface,
    policy: PolicyInterface,
    reward_mapper: object,
) -> LookaheadPlanner | None:
    """Create a planner if lookahead is enabled, otherwise return None."""
    if not config.phase2.lookahead.enabled:
        return None
    return LookaheadPlanner(config, simulator, policy, reward_mapper)
