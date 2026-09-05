"""Sample starting beliefs for imagined rollout from tokenized episode data.

This bridges the data layer (SequenceBatch) and the collectors layer
(ImaginedCollector needs start beliefs as Tensors). It depends on the world
model's posterior_step to ground observations into belief states.

Design:
  - Takes a SequenceBatch and a posterior grounding function.
  - For each episode in the batch, picks a random timestep with enough real
    transitions remaining for the requested imagined-rollout horizon.
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
    rollout_horizon: int,
    null_action_id: int,
) -> Tensor:
    """Ground beliefs at random timesteps from a batch of tokenized episodes.

    Args:
        batch: Tokenized episode batch (B, T, K, D).
        grounder: World model with get_initial_belief() and posterior_step().
        device: Target device.
        dtype: Target dtype for beliefs.
        rollout_horizon: Number of imagined transitions that will be rolled
            out. Every input episode must contain at least this many real
            transitions after every sampled start.
        null_action_id: Action id used at t=0 (before any real action).

    Returns:
        Tensor of shape (B, K, D) — one grounded belief per episode.
    """
    belief_slots, _ = _ground_random_start_beliefs(
        batch,
        grounder,
        device=device,
        dtype=dtype,
        rollout_horizon=rollout_horizon,
        null_action_id=null_action_id,
    )
    return belief_slots


def sample_start_beliefs_and_actions(
    batch: SequenceBatch,
    grounder: PosteriorGrounder,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rollout_horizon: int,
    null_action_id: int,
    early_state_probability: float = 0.0,
    early_state_max_t: int = 2,
) -> tuple[Tensor, Tensor]:
    """Ground offline posterior beliefs and return their logged actions.

    This is the Phase-2 behavior-cloning counterpart of
    :func:`sample_start_beliefs`.  The sampled action is exactly ``a_t`` from
    the same offline episode/timestep that produced posterior belief ``b_t``;
    no environment interaction and no prior-imagined label is involved.
    """
    belief_slots, start_timesteps = _ground_random_start_beliefs(
        batch,
        grounder,
        device=device,
        dtype=dtype,
        rollout_horizon=rollout_horizon,
        null_action_id=null_action_id,
        early_state_probability=early_state_probability,
        early_state_max_t=early_state_max_t,
    )
    actions = batch.actions.to(device=device, dtype=torch.long)
    rows = torch.arange(actions.shape[0], device=device)
    logged_actions = actions[rows, start_timesteps.to(device=device)]
    return belief_slots, logged_actions


@torch.no_grad()
def _ground_random_start_beliefs(
    batch: SequenceBatch,
    grounder: PosteriorGrounder,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rollout_horizon: int,
    null_action_id: int,
    early_state_probability: float = 0.0,
    early_state_max_t: int = 2,
) -> tuple[Tensor, Tensor]:
    """Shared frozen-WM grounding implementation.

    Phase-2 grounding is data preparation, not an optimization path.  The
    explicit ``no_grad`` is essential even when the caller intends the WM to
    be frozen: otherwise a stray trainable shared-LoRA/input can retain every
    per-sample, per-timestep Qwen graph through the assembled ``belief_slots``
    tensor and exhaust GPU memory on the next BC/PPO batch.
    """
    if rollout_horizon <= 0:
        raise ValueError("rollout_horizon must be positive")

    batch_size = batch.obs_tokens.shape[0]
    ep_lens = batch.episode_lengths  # (B,)

    # episode_lengths counts observations/posterior steps.  An episode with
    # T_env transitions has T_env + 1 observations, so an H-step rollout from
    # observation t is valid iff t + H <= T_env:
    #     t <= episode_length - 1 - H.
    max_valid_t = ep_lens - 1 - rollout_horizon
    ineligible = max_valid_t < 0
    if bool(ineligible.any()):
        count = int(ineligible.sum().item())
        raise ValueError(
            f"{count}/{batch_size} episodes are shorter than "
            f"rollout_horizon={rollout_horizon}; filter or resample them "
            "before grounding beliefs"
        )

    early_probability = float(early_state_probability)
    early_max_t = max(0, int(early_state_max_t))
    if not 0.0 <= early_probability <= 1.0:
        raise ValueError("early_state_probability must be in [0, 1]")
    rand_t = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        upper = int(max_valid_t[i].item()) + 1
        if early_probability > 0.0 and torch.rand(()).item() < early_probability:
            upper = min(upper, early_max_t + 1)
        rand_t[i] = torch.randint(0, upper, (1,)).item()

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

    return belief_slots, rand_t
