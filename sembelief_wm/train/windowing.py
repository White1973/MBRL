"""Window slicing utilities for Phase 1 truncated BPTT."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.belief import zero_belief
from ..config import Config
from ..types import BeliefState, SequenceBatch, TrainingWindow


@dataclass(frozen=True)
class WindowCursor:
    """Metadata describing one training window's position inside an episode."""

    start: int
    horizon: int
    is_initial_window: bool


def num_windows(sequence: SequenceBatch, horizon: int) -> int:
    """Return the number of non-overlapping BPTT windows for the batch."""

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    seq_len = sequence.seq_len
    return (seq_len + horizon - 1) // horizon


def window_cursor_at(window_index: int, horizon: int) -> WindowCursor:
    """Resolve the `(start, horizon)` metadata for one non-overlapping window."""

    if window_index < 0:
        raise ValueError(f"window_index must be non-negative, got {window_index}.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    start = window_index * horizon
    return WindowCursor(
        start=start,
        horizon=horizon,
        is_initial_window=(window_index == 0),
    )


def slice_training_window(
    sequence: SequenceBatch,
    *,
    window_index: int,
    horizon: int,
    config: Config,
    prev_belief: BeliefState | None = None,
) -> TrainingWindow:
    """Slice one non-overlapping BPTT window from an episode batch.

    The caller owns the sequential belief dependency:
    - pass `prev_belief=None` for the first window to get the fallback zero-init
    - pass the previous window's terminal posterior belief for later windows
    """

    cursor = window_cursor_at(window_index, horizon)
    start = cursor.start
    end = min(start + horizon, sequence.seq_len)
    if start >= sequence.seq_len:
        raise IndexError(
            f"window_index={window_index} starts at {start}, past seq_len={sequence.seq_len}."
        )

    obs_tokens = sequence.obs_tokens[:, start:end]
    actions = sequence.actions[:, start:end]
    rewards = sequence.rewards[:, start:end]
    valid_lengths = (sequence.episode_lengths - start).clamp(min=0, max=end - start)
    # SequenceBatch.episode_lengths is T_env + 1 posterior states.  Z_T is
    # therefore inside this window exactly when its exclusive index lies in
    # (start, end].  This must not be inferred from episode-level success.
    terminal_mask = (sequence.episode_lengths > start) & (sequence.episode_lengths <= end)

    semantic_teacher_tokens = (
        None
        if sequence.semantic_teacher_tokens is None
        else sequence.semantic_teacher_tokens[:, start:end]
    )
    semantic_teacher_mask = (
        None
        if sequence.semantic_teacher_mask is None
        else sequence.semantic_teacher_mask[:, start:end]
    )

    if cursor.is_initial_window:
        prev_actions = sequence.actions.new_full(
            (sequence.batch_size,),
            fill_value=config.env.null_action_id,
        )
        prev_rewards = sequence.rewards.new_zeros((sequence.batch_size,))
        has_prev_reward = sequence.episode_lengths.new_zeros(
            (sequence.batch_size,),
            dtype=bool,
        )
        prev_semantic_teacher_tokens = (
            None
            if sequence.semantic_teacher_tokens is None
            else sequence.semantic_teacher_tokens.new_zeros(
                (
                    sequence.batch_size,
                    sequence.semantic_teacher_tokens.shape[-2],
                    sequence.semantic_teacher_tokens.shape[-1],
                )
            )
        )
        prev_semantic_teacher_mask = sequence.episode_lengths.new_zeros(
            (sequence.batch_size,), dtype=torch.bool
        )
        if prev_belief is None:
            prev_belief = zero_belief(
                batch_size=sequence.batch_size,
                config=config,
                device=sequence.obs_tokens.device,
                dtype=sequence.obs_tokens.dtype,
            )
    else:
        prev_actions = sequence.actions[:, start - 1]
        prev_rewards = sequence.rewards[:, start - 1]
        has_prev_reward = sequence.episode_lengths > start
        if prev_belief is None:
            raise ValueError(
                "prev_belief must be provided for non-initial windows."
            )
        prev_semantic_teacher_tokens = (
            None
            if sequence.semantic_teacher_tokens is None
            else sequence.semantic_teacher_tokens[:, start - 1]
        )
        prev_semantic_teacher_mask = (
            sequence.episode_lengths > start
            if sequence.semantic_teacher_mask is None
            else sequence.semantic_teacher_mask[:, start - 1]
        )

    return TrainingWindow(
        obs_tokens=obs_tokens,
        actions=actions,
        rewards=rewards,
        prev_actions=prev_actions,
        prev_rewards=prev_rewards,
        has_prev_reward=has_prev_reward,
        valid_lengths=valid_lengths,
        terminal_mask=terminal_mask,
        prev_belief=prev_belief,
        start_index=start,
        is_initial_window=cursor.is_initial_window,
        env_ids=sequence.env_ids,
        episode_success=sequence.episode_success,
        semantic_teacher_tokens=semantic_teacher_tokens,
        semantic_teacher_mask=semantic_teacher_mask,
        prev_semantic_teacher_tokens=prev_semantic_teacher_tokens,
        prev_semantic_teacher_mask=prev_semantic_teacher_mask,
    )
