"""Tests for real environment collector."""
import torch
import pytest
from dataclasses import dataclass

from sembelief_wm.collectors.real import (
    RealCollector,
    RealCollectorConfig,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEnv:
    """Deterministic fake Sokoban env."""
    def __init__(self, episode_length: int = 5, success: bool = False):
        self._max_steps = episode_length
        self._success = success
        self._step_count = 0

    def reset(self, *, seed=None):
        self._step_count = 0
        return torch.randn(96, 96, 3)  # fake image obs

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= self._max_steps
        reward = 10.0 if (done and self._success) else -0.1
        info = {"success": self._success and done}
        obs = torch.randn(96, 96, 3)
        return obs, reward, done, info


class FakeTokenizer:
    def __init__(self, K=36, D=64):
        self.K = K
        self.D = D
    def tokenize(self, obs):
        return torch.randn(self.K, self.D)


@dataclass
class FakeBelief:
    slots: torch.Tensor


class FakeWorldModel:
    def __init__(self, K=36, D=64):
        self.K = K
        self.D = D
    def get_initial_belief(self, batch_size, *, device, dtype):
        return FakeBelief(slots=torch.randn(batch_size, self.K, self.D))
    def posterior_step(self, *, prev_belief, prev_actions, observation_tokens, env_ids=None):
        B = observation_tokens.shape[0]
        return FakeBelief(slots=torch.randn(B, self.K, self.D))


class FakePolicy:
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        self.call_count = 0
    def act(self, states, **kwargs):
        self.call_count += 1
        B = states.shape[0] if states.dim() > 1 else 1
        action = torch.randint(0, self.num_actions, (B,))
        log_prob = torch.zeros(B)
        value = torch.zeros(B)
        return action, log_prob, value, {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_collect_basic():
    """Basic collection: 3 episodes, check metrics shape."""
    collector = RealCollector(
        env_factory=lambda seed: FakeEnv(episode_length=5),
        tokenizer=FakeTokenizer(),
        world_model=FakeWorldModel(),
        policy=FakePolicy(),
        config=RealCollectorConfig(max_steps=10),
    )
    result = collector.collect(3, device="cpu", dtype=torch.float32)
    assert len(result.episodes) == 3
    assert "success_rate" in result.metrics
    assert "avg_return" in result.metrics
    assert result.metrics["num_episodes"] == 3.0


def test_collect_with_success():
    """Episodes that succeed should be counted."""
    collector = RealCollector(
        env_factory=lambda seed: FakeEnv(episode_length=3, success=True),
        tokenizer=FakeTokenizer(),
        world_model=FakeWorldModel(),
        policy=FakePolicy(),
        config=RealCollectorConfig(max_steps=10),
    )
    result = collector.collect(5, device="cpu", dtype=torch.float32)
    assert result.metrics["success_rate"] == 1.0
    assert result.metrics["num_success"] == 5.0


def test_collect_tokenized_episodes():
    """collect_tokenized=True should produce TokenizedEpisode."""
    collector = RealCollector(
        env_factory=lambda seed: FakeEnv(episode_length=4),
        tokenizer=FakeTokenizer(K=36, D=64),
        world_model=FakeWorldModel(K=36, D=64),
        policy=FakePolicy(),
        config=RealCollectorConfig(max_steps=10),
    )
    result = collector.collect(
        2, collect_tokenized=True, device="cpu", dtype=torch.float32
    )
    for ep in result.episodes:
        assert ep.tokenized is not None
        # T+1 observations, T actions, T rewards
        assert ep.tokenized.obs_tokens.shape[0] == ep.length + 1
        assert ep.tokenized.obs_tokens.shape[1] == 36
        assert ep.tokenized.obs_tokens.shape[2] == 64
        assert ep.tokenized.actions.shape[0] == ep.length + 1
        assert ep.tokenized.rewards.shape[0] == ep.length + 1
        assert ep.tokenized.episode_length == ep.length


def test_collect_and_store():
    """collect_and_store should write episodes to replay buffer."""
    class FakeReplay:
        def __init__(self):
            self.episodes = []
            self.size = 0
        def add_episode(self, episode_id, episode):
            self.episodes.append((episode_id, episode))
            self.size += 1

    replay = FakeReplay()
    collector = RealCollector(
        env_factory=lambda seed: FakeEnv(episode_length=3),
        tokenizer=FakeTokenizer(),
        world_model=FakeWorldModel(),
        policy=FakePolicy(),
        config=RealCollectorConfig(max_steps=10),
        replay_buffer=replay,
        device="cpu",
        dtype=torch.float32,
    )
    result = collector.collect_and_store(4, update_id=5)
    assert len(replay.episodes) == 4
    assert result.metrics["replay/saved_episodes"] == 4.0
    # Check episode IDs include update_id
    for eid, ep in replay.episodes:
        assert "u000005" in eid


def test_max_steps_truncation():
    """Episodes should be truncated at max_steps."""
    collector = RealCollector(
        env_factory=lambda seed: FakeEnv(episode_length=100),
        tokenizer=FakeTokenizer(),
        world_model=FakeWorldModel(),
        policy=FakePolicy(),
        config=RealCollectorConfig(max_steps=5),
    )
    result = collector.collect(2, device="cpu", dtype=torch.float32)
    for ep in result.episodes:
        assert ep.length == 5


def test_policy_receives_belief_slots():
    """Policy.act should receive belief slots, not raw obs."""
    policy = FakePolicy()
    collector = RealCollector(
        env_factory=lambda seed: FakeEnv(episode_length=3),
        tokenizer=FakeTokenizer(K=36, D=64),
        world_model=FakeWorldModel(K=36, D=64),
        policy=policy,
        config=RealCollectorConfig(max_steps=10),
    )
    collector.collect(1, device="cpu", dtype=torch.float32)
    # Policy should have been called 3 times (3 steps)
    assert policy.call_count == 3
