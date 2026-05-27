"""Collect Sokoban episodes with configurable strategy mix.

Usage:
    python scripts/collect_sokoban.py --num_episodes 300 --output_dir data/sokoban_300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sembelief_wm.data.adapters.sokoban import SokobanAdapter
from sembelief_wm.data.collector import collect_one_episode, strategy_schedule, validate_strategies
from sembelief_wm.data.schema import CollectorConfig, StrategySpec
from sembelief_wm.data.storage import save_raw_episode, read_manifest, existing_episode_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Sokoban episodes")
    parser.add_argument("--num_episodes", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="data/sokoban_300")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dim_room", type=int, nargs=2, default=[6, 6])
    parser.add_argument("--num_boxes", type=int, default=1)
    parser.add_argument("--bfs_max_depth", type=int, default=20)
    parser.add_argument("--noise_prob", type=float, default=0.3)
    parser.add_argument("--near_success_remaining", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = SokobanAdapter(
        dim_room=tuple(args.dim_room),
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        bfs_max_depth=args.bfs_max_depth,
    )

    strategies = [
        StrategySpec(name="expert", weight=3, params={"bfs_max_depth": args.bfs_max_depth}),
        StrategySpec(name="noisy_expert", weight=3, params={"noise_prob": args.noise_prob, "bfs_max_depth": args.bfs_max_depth}),
        StrategySpec(name="random", weight=3),
        StrategySpec(name="near_success", weight=1, params={"remaining_steps": args.near_success_remaining, "bfs_max_depth": args.bfs_max_depth}),
    ]

    config = CollectorConfig(
        strategies=strategies,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        split=args.split,
    )
    validate_strategies(adapter, config)

    # Resume support: check existing manifest
    raw_manifest = output_dir / "raw" / "manifest.jsonl"
    existing_ids: set[str] = set()
    if raw_manifest.exists():
        try:
            existing_ids = existing_episode_ids(read_manifest(raw_manifest))
            print(f"Resuming: found {len(existing_ids)} existing episodes")
        except Exception:
            pass

    schedule = strategy_schedule(strategies)
    from itertools import cycle
    strategy_iter = cycle(schedule)

    t0 = time.time()
    collected = 0
    skipped = 0
    stats = {"expert": 0, "noisy_expert": 0, "random": 0, "near_success": 0}
    success_count = 0
    total_steps = 0
    total_rewards = 0.0
    all_episodes = []

    for i in range(args.num_episodes):
        strategy = next(strategy_iter)
        episode_id = f"sokoban_{i:05d}"

        if episode_id in existing_ids:
            skipped += 1
            continue

        seed = args.seed + i
        ep = collect_one_episode(
            adapter=adapter,
            strategy=strategy,
            max_steps=args.max_steps,
            seed=seed,
            split=args.split,
        )

        save_raw_episode(ep, output_dir, episode_id=episode_id, split=args.split)
        all_episodes.append(ep)
        collected += 1
        stats[strategy.name] = stats.get(strategy.name, 0) + 1
        if ep.metadata.get("success"):
            success_count += 1
        total_steps += ep.length
        total_rewards += sum(ep.rewards)

        if collected % 50 == 0:
            elapsed = time.time() - t0
            eps_per_sec = collected / elapsed if elapsed > 0 else 0
            print(
                f"  [{collected}/{args.num_episodes}] "
                f"{eps_per_sec:.1f} ep/s | "
                f"success={success_count}/{collected} | "
                f"avg_len={total_steps/collected:.1f} | "
                f"avg_reward={total_rewards/collected:.2f}"
            )

    elapsed = time.time() - t0
    print(f"\nDone: {collected} episodes collected in {elapsed:.1f}s")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Strategy distribution: {stats}")
    print(f"  Success rate: {success_count}/{collected} ({100*success_count/max(collected,1):.1f}%)")
    print(f"  Avg episode length: {total_steps/max(collected,1):.1f}")
    print(f"  Avg total reward: {total_rewards/max(collected,1):.2f}")
    print(f"  Output: {output_dir}")

    # --- Structured metadata diagnostics ---
    if all_episodes:
        from collections import Counter
        import numpy as np

        print("\n=== Structured Metadata Diagnostics ===")

        # Check presence of structured data
        sample = all_episodes[0]
        has_initial = "initial_room_state" in sample.metadata
        has_step_labels = bool(sample.infos) and "state_labels" in sample.infos[0]
        print(f"  initial_room_state present: {has_initial}")
        print(f"  step state_labels present: {has_step_labels}")
        if has_step_labels:
            print(f"  state_labels fields: {sorted(sample.infos[0]['state_labels'].keys())}")

        # Optimal solution length distribution
        opt_lens = [ep.metadata.get("optimal_solution_length", -1) for ep in all_episodes]
        valid_opts = [x for x in opt_lens if x >= 0]
        if valid_opts:
            print(f"\n  Optimal solution lengths: mean={np.mean(valid_opts):.1f}, "
                  f"min={min(valid_opts)}, max={max(valid_opts)}")
        unsolvable_init = sum(1 for x in opt_lens if x < 0)
        if unsolvable_init:
            print(f"  Unsolvable initial states: {unsolvable_init}")

        # Action effect distribution
        effect_counts: Counter[str] = Counter()
        for ep in all_episodes:
            for info in ep.infos:
                labels = info.get("state_labels", {})
                effect_counts[labels.get("action_effect", "unknown")] += 1
        if effect_counts:
            print(f"\n  Action effects: {dict(effect_counts)}")

        # Solvability across steps
        solvable = sum(
            1 for ep in all_episodes for info in ep.infos
            if info.get("state_labels", {}).get("is_solvable", False)
        )
        total_steps_check = sum(len(ep.infos) for ep in all_episodes)
        if total_steps_check:
            print(f"  Solvable states: {solvable}/{total_steps_check} "
                  f"({100*solvable/total_steps_check:.1f}%)")

        # Success rate by strategy
        print("\n  Success by strategy:")
        for strat in sorted(stats):
            strat_eps = [ep for ep in all_episodes if ep.metadata["strategy"] == strat]
            strat_success = sum(1 for ep in strat_eps if ep.metadata.get("success"))
            if strat_eps:
                print(f"    {strat}: {strat_success}/{len(strat_eps)} "
                      f"({100*strat_success/len(strat_eps):.1f}%)")


if __name__ == "__main__":
    main()
