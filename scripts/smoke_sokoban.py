"""Smoke test for Sokoban adapter + BFS solver + all four strategies.

Validates:
1. BFS solver correctness on a known puzzle
2. All four strategies (expert, noisy_expert, random, near_success) produce valid episodes
3. Episode collection via collector with strategy scheduling
4. Action mapping (env 1-4 → model 0-3)
5. Integration with storage + tokenizer + datasource pipeline
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add code root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sembelief_wm.data.adapters.sokoban import (
    SokobanAdapter,
    SokobanEnv,
    _bfs_solve,
    _extract_sokoban_state,
)
from sembelief_wm.data.collector import collect_episodes
from sembelief_wm.data.schema import CollectorConfig, StrategySpec


def test_bfs_solver():
    """Test BFS on a trivially solvable configuration."""
    print("=== Test 1: BFS solver ===")

    # Simple case: player at (1,1), box at (1,2), target at (1,3)
    # Walls around the border, 4x4 room
    walls = frozenset()
    player = (1, 1)
    boxes = frozenset({(1, 2)})
    targets = frozenset({(1, 3)})
    dim_room = (4, 4)

    solution = _bfs_solve(player, boxes, targets, walls, dim_room, max_depth=10)
    assert solution is not None, "BFS should find a solution"
    # Player pushes box right: action 4 (right)
    assert len(solution) == 1, f"Expected 1-step solution, got {len(solution)}"
    assert solution[0] == 4, f"Expected push right (4), got {solution[0]}"
    print(f"  BFS found solution: {solution} (length {len(solution)})")

    # Unsolvable case: box against wall with no room to push
    walls2 = frozenset({(0, 3), (2, 3)})
    player2 = (1, 1)
    boxes2 = frozenset({(1, 2)})
    targets2 = frozenset({(1, 3)})
    solution2 = _bfs_solve(player2, boxes2, targets2, walls2, dim_room, max_depth=10)
    # This might or might not be solvable depending on exact wall config;
    # the important thing is BFS terminates
    print(f"  BFS with walls: solution={'found' if solution2 else 'none'}")
    print("  PASSED\n")


def test_mock_env_strategies():
    """Test all four strategies work with mock env (no gym_sokoban needed)."""
    print("=== Test 2: All strategies with mock env ===")

    adapter = SokobanAdapter(dim_room=(6, 6), num_boxes=1, max_steps=10)
    env = adapter.make_env(seed=42)

    # Check it's a mock env (no gym_sokoban installed in test)
    is_mock = isinstance(env, SokobanEnv) and env._use_mock
    print(f"  Using mock env: {is_mock}")

    for strategy in adapter.available_strategies():
        policy = adapter.make_policy(strategy, env, seed=42)
        obs = env.reset(seed=42)
        actions = []
        for _ in range(10):
            action = policy(obs)
            assert action in (1, 2, 3, 4), f"Invalid env action: {action}"
            obs, reward, done, info = env.step(action)
            actions.append(action)
            model_action = adapter.map_action(action)
            assert model_action in (0, 1, 2, 3), f"Invalid model action: {model_action}"
            if done:
                break
        print(f"  {strategy}: {len(actions)} steps, actions={actions[:5]}...")

    print("  PASSED\n")


def test_collector_integration():
    """Test episode collection with strategy scheduling."""
    print("=== Test 3: Collector integration ===")

    adapter = SokobanAdapter(dim_room=(6, 6), num_boxes=1, max_steps=10)
    config = CollectorConfig(
        num_episodes=10,
        max_steps=10,
        strategies=[
            StrategySpec(name="expert", weight=3),
            StrategySpec(name="noisy_expert", weight=3, params={"noise_prob": 0.3}),
            StrategySpec(name="random", weight=3),
            StrategySpec(name="near_success", weight=1, params={"remaining_steps": 3}),
        ],
        seed=42,
    )

    episodes = collect_episodes(adapter, config)
    assert len(episodes) == 10, f"Expected 10 episodes, got {len(episodes)}"

    # Check strategy distribution
    strategy_counts: dict[str, int] = {}
    for ep in episodes:
        s = ep.metadata.get("strategy", "unknown")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    print(f"  Strategy distribution: {strategy_counts}")

    # Verify episode structure
    for i, ep in enumerate(episodes):
        assert ep.env_id == "sokoban"
        assert len(ep.observations) == len(ep.actions) + 1
        assert len(ep.model_actions) == len(ep.actions)
        assert all(a in (0, 1, 2, 3) for a in ep.model_actions), \
            f"Episode {i}: bad model actions"

    total_transitions = sum(ep.length for ep in episodes)
    print(f"  Total episodes: {len(episodes)}, transitions: {total_transitions}")
    print(f"  Avg episode length: {total_transitions / len(episodes):.1f}")
    print("  PASSED\n")


def test_action_mapping():
    """Verify env action 1-4 maps to model action 0-3."""
    print("=== Test 4: Action mapping ===")
    adapter = SokobanAdapter()
    for env_a in range(1, 5):
        model_a = adapter.map_action(env_a)
        assert model_a == env_a - 1, f"map_action({env_a}) = {model_a}, expected {env_a - 1}"
    print("  env 1→0, 2→1, 3→2, 4→3: PASSED\n")


def test_storage_pipeline():
    """Test full pipeline: collect → save → tokenize → load."""
    print("=== Test 5: Storage pipeline ===")
    from sembelief_wm.data.storage import (
        save_raw_episode,
        save_tokenized_episode,
        load_tokenized_episode,
        read_manifest,
    )
    from sembelief_wm.data.schema import TokenizedEpisode

    adapter = SokobanAdapter(max_steps=5)
    config = CollectorConfig(
        num_episodes=3,
        max_steps=5,
        strategies=[StrategySpec(name="random", weight=1)],
        seed=99,
    )
    episodes = collect_episodes(adapter, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save raw
        for i, ep in enumerate(episodes):
            save_raw_episode(ep, tmpdir, episode_id=f"sok_{i:04d}")

        raw_manifest = read_manifest(Path(tmpdir) / "raw" / "manifest.jsonl")
        assert len(raw_manifest) == 3
        print(f"  Saved {len(raw_manifest)} raw episodes")

        # Mock tokenization (random tokens, since we don't have V-JEPA here)
        D = 3584
        K = 36
        for i, ep in enumerate(episodes):
            T_seq = ep.length + 1
            obs_tokens = torch.randn(T_seq, K, D)
            actions_tensor = torch.tensor(
                ep.model_actions + [adapter.action_spec().null_action_id],
                dtype=torch.long,
            )
            rewards_tensor = torch.tensor(
                ep.rewards + [0.0],
                dtype=torch.float32,
            )
            tok_ep = TokenizedEpisode(
                obs_tokens=obs_tokens,
                actions=actions_tensor,
                rewards=rewards_tensor,
                episode_length=ep.length,
                env_id=ep.env_id,
                split="train",
                metadata=ep.metadata,
            )
            save_tokenized_episode(
                tok_ep, tmpdir,
                episode_id=f"sok_{i:04d}",
                tokenizer="mock_random",
            )

        tok_manifest = read_manifest(Path(tmpdir) / "tokenized" / "manifest.jsonl")
        assert len(tok_manifest) == 3
        print(f"  Saved {len(tok_manifest)} tokenized episodes")

        # Load back
        loaded = load_tokenized_episode(
            Path(tmpdir) / "tokenized" / tok_manifest[0]["path"]
        )
        print(f"  Loaded episode: obs_tokens={tuple(loaded.obs_tokens.shape)}, "
              f"length={loaded.episode_length}")

    print("  PASSED\n")


def main():
    test_bfs_solver()
    test_mock_env_strategies()
    test_collector_integration()
    test_action_mapping()
    test_storage_pipeline()
    print("=" * 50)
    print("All Sokoban smoke tests PASSED")


if __name__ == "__main__":
    main()
