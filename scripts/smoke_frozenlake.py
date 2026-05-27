#!/usr/bin/env python3
"""End-to-end smoke test: FrozenLake adapter → collect → tokenize → store → DataSource → trainer.

This validates the full data-layer pipeline with the simplest possible
environment. No GPU required; uses a mock backbone and small dimensions.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sembelief_wm.config import Config, EnvironmentConfig
from sembelief_wm.data.adapters.frozenlake import FrozenLakeAdapter
from sembelief_wm.data.collector import collect_episodes
from sembelief_wm.data.datasource import OfflineDataSource, TokenizedEpisodeDataset
from sembelief_wm.data.schema import CollectorConfig, StrategySpec, TokenizedEpisode
from sembelief_wm.data.storage import (
    existing_episode_ids,
    load_tokenized_episode,
    read_manifest,
    save_raw_episode,
    save_tokenized_episode,
)
from sembelief_wm.data.tokenizers.discrete_state import DiscreteStateTokenizer
from sembelief_wm.model import TransitionBackbone, WorldModel
from sembelief_wm.train import Phase1Trainer

# ---------------------------------------------------------------------------
# Small-scale config for testing
# ---------------------------------------------------------------------------
D = 32  # tiny hidden dim
K = 4   # tiny token count (instead of 36, for speed)
N_STATES = 16  # FrozenLake 4x4


def make_test_config() -> Config:
    return Config(
        hidden_dim=D,
        env=EnvironmentConfig(
            num_actions=4,
            null_action_id=4,
            env_ids=["frozenlake"],
            unified_obs_tokens=K,
        ),
    )


class MockBackbone(TransitionBackbone):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(tokens)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_1_collect(adapter: FrozenLakeAdapter, tmpdir: Path) -> list[str]:
    """Collect episodes with both strategies and save raw."""
    print("[1/5] Collecting episodes...")
    config = CollectorConfig(
        num_episodes=10,
        max_steps=25,
        strategies=[
            StrategySpec(name="random", weight=1),
            StrategySpec(name="value_iteration_expert", weight=1),
        ],
        seed=42,
        split="train",
    )
    episodes = collect_episodes(adapter, config)
    assert len(episodes) == 10, f"Expected 10 episodes, got {len(episodes)}"

    episode_ids = []
    for i, ep in enumerate(episodes):
        eid = f"ep_{i:04d}"
        save_raw_episode(ep, tmpdir, episode_id=eid, split="train")
        episode_ids.append(eid)
        assert len(ep.observations) == len(ep.actions) + 1

    # Check manifest
    raw_manifest = read_manifest(tmpdir / "raw" / "manifest.jsonl")
    assert len(raw_manifest) == 10
    print(f"  Collected {len(episodes)} episodes, "
          f"expert success: {sum(1 for e in episodes if e.metadata.get('strategy') == 'value_iteration_expert')}/5")
    return episode_ids


def step_2_tokenize(
    adapter: FrozenLakeAdapter,
    tokenizer: DiscreteStateTokenizer,
    tmpdir: Path,
    episode_ids: list[str],
) -> None:
    """Tokenize raw episodes and save as tokenized .pt files."""
    print("[2/5] Tokenizing episodes...")
    raw_manifest = read_manifest(tmpdir / "raw" / "manifest.jsonl")

    # Check resume: skip already-tokenized episodes
    tok_manifest_path = tmpdir / "tokenized" / "manifest.jsonl"
    already_done: set[str] = set()
    if tok_manifest_path.exists():
        already_done = existing_episode_ids(read_manifest(tok_manifest_path))

    for entry in raw_manifest:
        eid = entry["episode_id"]
        if eid in already_done:
            continue

        raw_data = torch.load(tmpdir / "raw" / f"{eid}.pt", map_location="cpu", weights_only=False)
        observations = raw_data["observations"]
        actions = raw_data["model_actions"]
        rewards = raw_data["rewards"]

        # Tokenize all observations (T+1 frames)
        obs_tokens = tokenizer.batch_tokenize(observations)  # (T+1, K, D)

        # Build TokenizedEpisode with T_seq = T_env + 1
        T_env = len(actions)
        T_seq = T_env + 1
        null_action = adapter.action_spec().null_action_id
        actions_tensor = torch.tensor(actions + [null_action], dtype=torch.long)
        rewards_tensor = torch.tensor(rewards + [0.0], dtype=torch.float32)  # dummy reward

        episode = TokenizedEpisode(
            obs_tokens=obs_tokens,
            actions=actions_tensor,
            rewards=rewards_tensor,
            episode_length=T_seq,
            env_id=entry["env_id"],
            split=entry.get("split", "train"),
            metadata={"strategy": entry.get("strategy")},
        )
        save_tokenized_episode(
            episode, tmpdir, episode_id=eid,
            tokenizer="discrete_state",
            config_hash=tokenizer.config_hash,
        )

    tok_manifest = read_manifest(tok_manifest_path)
    assert len(tok_manifest) == 10
    print(f"  Tokenized {len(tok_manifest)} episodes, "
          f"token shape per obs: ({K}, {D})")


def step_3_datasource(tmpdir: Path, config: Config) -> OfflineDataSource:
    """Load tokenized episodes into DataSource."""
    print("[3/5] Building DataSource...")
    dataset = TokenizedEpisodeDataset.from_manifest(
        tmpdir / "tokenized" / "manifest.jsonl",
        split="train",
    )
    assert len(dataset) == 10
    source = OfflineDataSource(dataset, config)
    batch = source.sample_batch(batch_size=4)
    assert batch.obs_tokens.ndim == 4, f"Expected 4D obs_tokens, got {batch.obs_tokens.ndim}D"
    assert batch.env_ids is not None
    assert batch.env_ids.shape == (4,)
    print(f"  DataSource ready, batch shape: obs_tokens={tuple(batch.obs_tokens.shape)}, "
          f"actions={tuple(batch.actions.shape)}")
    return source


def step_4_trainer(source: OfflineDataSource, config: Config) -> None:
    """Run one training step through the trainer."""
    print("[4/5] Running trainer step...")
    backbone = MockBackbone(config.hidden_dim)
    model = WorldModel(config, backbone)
    trainer = Phase1Trainer(
        config=config,
        world_model=model,
        data_source=source,
        device="cpu",
    )
    batch = source.sample_batch(batch_size=2)
    metrics = trainer.train_one_step(global_step=0, batch=batch)
    assert torch.isfinite(torch.tensor(metrics.total)), f"Loss is not finite: {metrics.total}"
    print(f"  Loss: total={metrics.total:.4f}, dynamics={metrics.dynamics:.4f}, "
          f"reward={metrics.reward:.4f}")


def step_5_strategy_validation(adapter: FrozenLakeAdapter) -> None:
    """Verify strategy registry validates correctly."""
    print("[5/5] Validating strategy registry...")

    # Should fail for unsupported strategy
    bad_config = CollectorConfig(
        num_episodes=1,
        max_steps=10,
        strategies=[StrategySpec(name="nonexistent_strategy", weight=1)],
        seed=0,
    )
    try:
        collect_episodes(adapter, bad_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_strategy" in str(e)
        print(f"  Strategy validation works: {e}")

    # Expert should find a path
    expert_config = CollectorConfig(
        num_episodes=3,
        max_steps=25,
        strategies=[StrategySpec(name="value_iteration_expert", weight=1)],
        seed=0,
    )
    expert_episodes = collect_episodes(adapter, expert_config)
    successes = sum(1 for ep in expert_episodes if ep.dones[-1] and ep.rewards[-1] > 0)
    print(f"  Expert success rate: {successes}/{len(expert_episodes)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("FrozenLake End-to-End Smoke Test")
    print("=" * 60)

    config = make_test_config()
    adapter = FrozenLakeAdapter()
    tokenizer = DiscreteStateTokenizer(
        num_states=N_STATES,
        output_dim=D,
        output_tokens=K,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        episode_ids = step_1_collect(adapter, tmpdir_path)
        step_2_tokenize(adapter, tokenizer, tmpdir_path, episode_ids)
        source = step_3_datasource(tmpdir_path, config)
        step_4_trainer(source, config)

    step_5_strategy_validation(adapter)

    print("=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
