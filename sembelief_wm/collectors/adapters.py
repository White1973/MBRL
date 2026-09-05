"""Adapters that bridge existing concrete classes to collector protocols.

These thin wrappers adapt WorldModel and LatentActorCritic to the
interfaces expected by ImaginedCollector, without modifying the
original classes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..rl.policy import Policy
from ..types import BeliefState


class LatentActorCriticAdapter:
    """Adapts LatentActorCritic to the rl.Policy protocol.

    LatentActorCritic.act() returns 5 values (extra None);
    this adapter drops the 5th and converts BeliefState slots
    to flat state tensors for the policy protocol.
    """

    def __init__(self, actor_critic: Any) -> None:
        self._ac = actor_critic

    def act(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        belief = self._to_belief(states)
        action, log_prob, entropy, value, _ = self._ac.act(
            belief, deterministic=deterministic, **kwargs
        )
        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        belief = self._to_belief(states)
        return self._ac.evaluate_actions(belief, actions, **kwargs)

    def parameters(self) -> Any:
        return self._ac.parameters()

    @staticmethod
    def _to_belief(states: Tensor) -> BeliefState:
        """Convert state tensor to BeliefState.

        Requires 3D input (B, K, D) — BeliefState.slots must be 3D
        because BeliefReadout validates ndim == 3.
        """
        if states.ndim != 3:
            raise ValueError(
                f"LatentActorCriticAdapter requires 3D states (B, K, D), "
                f"got shape {tuple(states.shape)}. If you have pooled 2D "
                f"states, use a different policy adapter."
            )
        return BeliefState(slots=states)


def make_dynamics_step(world_model: Any, *, action_id_offset: int = 0):
    """Create a dynamics_step callable from a WorldModel instance.

    Returns a function: (belief, action, **kwargs) -> next_belief

    ``action_id_offset=1`` is a compatibility shim for legacy Sokoban WM
    checkpoints trained from tokenized environment action ids 1..4. Policy
    actions remain canonical 0..3 everywhere else, including real evaluation.
    """
    if action_id_offset < 0:
        raise ValueError("action_id_offset must be non-negative")

    def dynamics_step(belief: Any, action: Tensor, **kwargs: Any) -> Any:
        wm_action = action + action_id_offset
        return world_model.prior_step(
            prev_belief=belief,
            prev_actions=wm_action,
            **kwargs,
        )
    return dynamics_step


def make_predict_reward(world_model: Any):
    """Create a predict_reward callable from a WorldModel instance."""
    def predict_reward(belief: Any) -> Tensor:
        return world_model.predict_reward(belief)
    return predict_reward


def make_get_belief_slots():
    """Create a get_belief_slots callable that extracts belief.slots."""
    def get_belief_slots(belief: Any) -> Tensor:
        return belief.slots
    return get_belief_slots
