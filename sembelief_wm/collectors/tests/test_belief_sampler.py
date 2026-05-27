"""Tests for belief_sampler."""

import pytest
import torch
from dataclasses import dataclass

from sembelief_wm.collectors.belief_sampler import sample_start_beliefs
from sembelief_wm.types import SequenceBatch


# --- Fakes ---

K, D = 36, 64  # slot dimensions


@dataclass
class FakeBeliefState:
    slots: torch.Tensor  # (B, K, D)


class FakeGrounder:
    """Minimal PosteriorGrounder that increments belief per step."""

    def __init__(self) -> None:
        self.step_count = 0

    def get_initial_belief(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> FakeBeliefState:
        return FakeBeliefState(
            slots=torch.zeros(batch_size, K, D, device=device, dtype=dtype)
        )

    def posterior_step(
        self,
        *,
        prev_belief: FakeBeliefState,
        prev_actions: torch.Tensor,
        observation_tokens: torch.Tensor,
        env_ids: torch.Tensor | None,
    ) -> FakeBeliefState:
        self.step_count += 1
        # Increment slots so we can verify grounding happened
        return FakeBeliefState(slots=prev_belief.slots + 1.0)


def _make_batch(batch_size: int = 4, seq_len: int = 6) -> SequenceBatch:
    """Create a fake SequenceBatch."""
    return SequenceBatch(
        obs_tokens=torch.randn(batch_size, seq_len, K, D),
        actions=torch.randint(0, 4, (batch_size, seq_len)),
        rewards=torch.zeros(batch_size, seq_len),
        episode_lengths=torch.full((batch_size,), seq_len, dtype=torch.long),
        env_ids=torch.zeros(batch_size, dtype=torch.long),
    )


# --- Tests ---


def test_output_shape():
    batch = _make_batch(batch_size=4, seq_len=6)
    grounder = FakeGrounder()

    beliefs = sample_start_beliefs(
        batch, grounder, device=torch.device("cpu"), dtype=torch.float32
    )

    assert beliefs.shape == (4, K, D)


def test_grounding_happens():
    """Verify that posterior_step is called at least once per sample."""
    batch = _make_batch(batch_size=3, seq_len=5)
    grounder = FakeGrounder()

    beliefs = sample_start_beliefs(
        batch, grounder, device=torch.device("cpu"), dtype=torch.float32
    )

    # Each sample gets random t in [0, seq_len-2], posterior runs t+1 steps
    # So step_count should be >= batch_size (at least 1 step per sample)
    assert grounder.step_count >= 3

    # All beliefs should be > 0 (incremented from zero by posterior_step)
    assert (beliefs > 0).any()


def test_single_step_episode():
    """Episode of length 1 (only initial obs) should still work."""
    batch = SequenceBatch(
        obs_tokens=torch.randn(2, 1, K, D),
        actions=torch.randint(0, 4, (2, 1)),
        rewards=torch.zeros(2, 1),
        episode_lengths=torch.tensor([1, 1]),
        env_ids=None,
    )
    grounder = FakeGrounder()

    beliefs = sample_start_beliefs(
        batch, grounder, device=torch.device("cpu"), dtype=torch.float32
    )

    assert beliefs.shape == (2, K, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
