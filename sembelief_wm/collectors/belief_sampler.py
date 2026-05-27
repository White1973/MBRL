"""Sample starting beliefs for imagined rollout from tokenized episode data.

This bridges the data layer (SequenceBatch) and the collectors layer
(ImaginedCollector needs start beliefs as Tensors). It depends on the world
model's posterior_step to ground observations into belief states.

Design:
  - Takes a SequenceBatch and a posterior grounding function.
  - For each episode in the batch, picks a random non-terminal timestep.
  - Rolls the posterior forward from t=0 to the chosen timestep.
  - Returns a (B, K, D) belief tensor ready for ImaginedCollector.

This replaces the old _ground_initial_belief + _sample_rollout_start_batch
methods that were embedded in Phase2Trainer.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from sembelief_wm.types import SequenceBatch


class PosteriorGrounder(Protocol):
    """Protocol for a world model's posterior step."""

    def get_initial_belief(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> object:
        """Return initial belief state with .slots attribute of shape (B, K, D)."""
        ...

    def posterior_step(
        self,
        *,
        prev_belief: object,
        prev_actions: Tensor,
        observation_tokens: Tensor,
        env_ids: Tensor | None,
    ) -> object:
        """Posterior update. Returns belief with .slots attribute."""
        ...


def sample_start_beliefs(
    batch: SequenceBatch,
    grounder: PosteriorGrounder,
    *,
    device: torch.device,
    dtype: torch.dtype,
    null_action_id: int = 0,
) -> Tensor:
    """Ground beliefs at random timesteps from a batch of tokenized episodes.

    Args:
        batch: Tokenized episode batch (B, T, K, D).
        grounder: World model with get_initial_belief() and posterior_step().
        device: Target device.
        dtype: Target dtype for beliefs.
        null_action_id: Action id used at t=0 (before any real action).

    Returns:
        Tensor of shape (B, K, D) — one grounded belief per episode.
    """
    batch_size = batch.obs_tokens.shape[0]
    ep_lens = batch.episode_lengths  # (B,)

    # Per-sample random timestep: valid indices are [0, ep_len - 2]
    # (ep_len - 1 is the terminal observation, not a valid rollout start)
    max_valid_t = (ep_lens - 2).clamp(min=0)

    rand_t = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        upper = int(max_valid_t[i].item()) + 1
        rand_t[i] = torch.randint(0, max(upper, 1), (1,)).item()

    # Null action for t=0 posterior step
    null_action = torch.full((1,), null_action_id, device=device, dtype=torch.long)

    # Ground each sample by rolling posterior forward
    ref_belief = grounder.get_initial_belief(1, device=device, dtype=dtype)
    belief_slots = torch.empty(
        batch_size,
        *ref_belief.slots.shape[1:],
        device=device,
        dtype=dtype,
    )

    obs_tokens = batch.obs_tokens.to(device=device, dtype=dtype)
    actions = batch.actions.to(device=device)
    env_ids = batch.env_ids.to(device=device) if batch.env_ids is not None else None

    for i in range(batch_size):
        t_i = int(rand_t[i].item())
        belief_i = grounder.get_initial_belief(1, device=device, dtype=dtype)
        for t in range(t_i + 1):
            act_t = null_action if t == 0 else actions[i : i + 1, t - 1]
            belief_i = grounder.posterior_step(
                prev_belief=belief_i,
                prev_actions=act_t,
                observation_tokens=obs_tokens[i : i + 1, t],
                env_ids=env_ids[i : i + 1] if env_ids is not None else None,
            )
        belief_slots[i] = belief_i.slots[0]

    return belief_slots
