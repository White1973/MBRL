"""Core data types for SemBelief-WM.

These typed containers define the data contracts between modules.
All time semantics follow the convention in the architecture doc (§5.1):

    posterior_step(o_t, Z_{t-1}, a_{t-1}) -> Z_t
    prior_step(Z_{t-1}, a_{t-1}) -> Z_t_pri
    Z_t predicts reward for transition (t-1) -> t, i.e. target = 1[r_{t-1} > 0]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """Latent belief state.

    Shape: (batch, K_belief, D)

    This is the core state representation of the world model.
    It contains only the algorithmic state — no runtime metadata.
    Time indexing is maintained by the external loop, not by this object.
    """
    slots: Tensor  # (B, K_belief, D)

    def detach(self) -> BeliefState:
        """Detach from computation graph (used at BPTT boundaries)."""
        return BeliefState(slots=self.slots.detach())

    @property
    def batch_size(self) -> int:
        return self.slots.shape[0]

    @property
    def num_slots(self) -> int:
        return self.slots.shape[-2]

    @property
    def hidden_dim(self) -> int:
        return self.slots.shape[-1]


@dataclass
class BeliefTrajectory:
    """A sequence of beliefs produced by rollout_prior.

    beliefs: (B, T, K_belief, D) — one predicted belief per action step.
    actions: (B, T) — the action sequence that produced this trajectory.
    """
    beliefs: Tensor  # (B, T, K_belief, D)
    actions: Tensor  # (B, T)

    @property
    def horizon(self) -> int:
        return self.beliefs.shape[1]


# ---------------------------------------------------------------------------
# Observation & Action batches
# ---------------------------------------------------------------------------

@dataclass
class ObservationTokens:
    """Pre-computed visual tokens from the frozen encoder.

    tokens: (B, K_vis, D) — visual token embeddings.
    """
    tokens: Tensor  # (B, K_vis, D)


@dataclass
class ActionBatch:
    """Discrete actions.

    actions: (B,) — integer action ids.
    Action ids may include the reserved null action used for a_{-1}
    or truncated-window starts.
    """
    actions: Tensor  # (B,) int64


# ---------------------------------------------------------------------------
# Training batch
# ---------------------------------------------------------------------------

@dataclass
class SequenceBatch:
    """A batch of episode sequences for Phase 1 training.

    All tensors have a shared (B, T) prefix where:
        B = batch size (number of episodes)
        T = sequence length

    Time semantics:
        obs_tokens[b, t]  = observation at time t
        actions[b, t]     = action taken at time t (a_t)
        rewards[b, t]     = reward received at time t (r_t, from executing a_t)

    The world model will construct its own alignment:
        Z_t is produced from (o_t, Z_{t-1}, a_{t-1})
        reward target for Z_t is 1[r_{t-1} > 0]
        Z_0 does not participate in reward supervision (no r_{-1}).
        First valid reward label is for Z_1, target = 1[r_0 > 0].

    `episode_lengths` counts real posterior timesteps, not real transitions.
    For an episode with `T_env` real transitions:
        obs_tokens length = T_env + 1
        episode_lengths   = T_env + 1
        real dynamics/reward supervision count = T_env

    This is the raw episode-level container. Truncated BPTT windows should use
    TrainingWindow, which carries the window-start previous action explicitly.
    """
    obs_tokens: Tensor   # (B, T, K_vis, D) — precomputed visual tokens
    actions: Tensor      # (B, T) int64 — actions taken
    rewards: Tensor      # (B, T) float — raw environment rewards
    episode_lengths: Tensor  # (B,) int64 — valid length of each episode
    env_ids: Tensor | None = None  # (B,) int64 — environment id index for multi-env WM
    # Per-episode terminal-success label (1 if the episode reached success,
    # 0 otherwise). Carried into training windows for the terminal-step
    # auxiliary reward BCE so the reward head gets a clean success/failure
    # signal instead of the all-negative-window-diluted per-step label.
    episode_success: Tensor | None = None  # (B,) float32
    # Optional frozen V-JEPA spatial-semantic targets. These are intentionally
    # not aliases of obs_tokens: Qwen tokens condition the WM, while V-JEPA
    # tokens supervise the predicted future belief.
    semantic_teacher_tokens: Tensor | None = None  # (B, T, K_teacher, D_teacher)
    semantic_teacher_mask: Tensor | None = None    # (B, T) bool

    @property
    def batch_size(self) -> int:
        return self.obs_tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.obs_tokens.shape[1]


@dataclass
class TrainingWindow:
    """A truncated BPTT window with explicit window-start alignment.

    Shapes:
        obs_tokens:   (B, H, K_vis, D)
        actions:      (B, H)       where actions[:, t] = a_t inside the window
        rewards:      (B, H)       where rewards[:, t] = r_t inside the window
        prev_actions: (B,)         where prev_actions[b] = a_{t0-1} for window start
        prev_rewards: (B,)         where prev_rewards[b] = r_{t0-1} for window start
        has_prev_reward: (B,)      whether prev_rewards is a valid label for step 0
        valid_lengths: (B,)        number of valid posterior timesteps inside the window
        terminal_mask: (B,)        whether this window contains the episode-terminal posterior
        prev_belief: BeliefState   belief carried into the window start
        start_index: int           global time index of the window start
        is_initial_window: bool    whether this window starts at t=0

    Time semantics:
        If the window starts at episode index t0, then:
            posterior_step(o_{t0}, Z_{t0-1}, prev_actions) -> Z_{t0}
            posterior_step(o_{t0+1}, Z_{t0}, actions[:, 0]) -> Z_{t0+1}

    Reward supervision follows:
        Z_{t0}     -> 1[r_{t0-1} > 0] if has_prev_reward is true
        Z_{t0+1}   -> 1[r_{t0} > 0]

    The first window of an episode must set prev_actions to the reserved
    null-action id from EnvironmentConfig and has_prev_reward to false.
    """
    obs_tokens: Tensor       # (B, H, K_vis, D)
    actions: Tensor          # (B, H) int64
    rewards: Tensor          # (B, H) float
    prev_actions: Tensor     # (B,) int64
    prev_rewards: Tensor     # (B,) float
    has_prev_reward: Tensor  # (B,) bool
    valid_lengths: Tensor    # (B,) int64 — valid length within this window
    terminal_mask: Tensor    # (B,) bool — window contains Z_T for this episode
    prev_belief: BeliefState
    start_index: int
    is_initial_window: bool
    env_ids: Tensor | None = None  # (B,) int64
    # Retained for episode-level metrics/backward compatibility. Reward-head
    # supervision must use the aligned per-transition rewards, not this label.
    episode_success: Tensor | None = None  # (B,) float32
    semantic_teacher_tokens: Tensor | None = None  # (B, H, K_teacher, D_teacher)
    semantic_teacher_mask: Tensor | None = None    # (B, H) bool
    # Teacher target and availability for the state immediately before the
    # window. These make the first transition of every noninitial BPTT window
    # eligible for semantic-delta supervision.
    prev_semantic_teacher_tokens: Tensor | None = None  # (B, K_teacher, D_teacher)
    prev_semantic_teacher_mask: Tensor | None = None    # (B,) bool

    @property
    def batch_size(self) -> int:
        return self.obs_tokens.shape[0]

    @property
    def horizon(self) -> int:
        return self.obs_tokens.shape[1]
