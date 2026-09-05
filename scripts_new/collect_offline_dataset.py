#!/usr/bin/env python3
"""Collect raw RGB episodes through the shared EnvironmentAdapter contract."""
from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

from sembelief_wm.data.adapters import make_default_adapter
from sembelief_wm.data.collector import collect_one_episode
from sembelief_wm.data.schema import StrategySpec
from sembelief_wm.data.storage import (
    existing_episode_ids,
    read_manifest,
    save_raw_episode,
)


def _parse_strategy(value: str) -> StrategySpec:
    name, separator, raw_weight = value.partition(":")
    weight = int(raw_weight) if separator else 1
    if not name or weight <= 0:
        raise argparse.ArgumentTypeError("strategy must be NAME[:POSITIVE_WEIGHT]")
    return StrategySpec(name=name, weight=weight)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic RGB episode collection for the shared MBRL stack"
    )
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--strategy",
        action="append",
        type=_parse_strategy,
        dest="strategies",
        help="Repeatable NAME[:WEIGHT]; defaults to random:1",
    )
    args = parser.parse_args()

    if args.num_episodes <= 0 or args.max_steps <= 0:
        parser.error("num-episodes and max-steps must be positive")

    adapter = make_default_adapter(args.env_id)
    strategies = args.strategies or [StrategySpec(name="random", weight=1)]
    available = set(adapter.available_strategies())
    unsupported = sorted({item.name for item in strategies} - available)
    if unsupported:
        parser.error(
            f"unsupported strategies {unsupported}; available={sorted(available)}"
        )

    schedule = [item for spec in strategies for item in [spec] * spec.weight]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "raw" / "manifest.jsonl"
    completed = (
        existing_episode_ids(read_manifest(manifest)) if manifest.is_file() else set()
    )

    collected = skipped = successes = 0
    for index, strategy in zip(
        range(args.num_episodes), cycle(schedule), strict=False
    ):
        seed = args.seed_start + index
        episode_id = f"{args.env_id}_{seed:05d}"
        if episode_id in completed:
            skipped += 1
            continue
        episode = collect_one_episode(
            adapter=adapter,
            strategy=strategy,
            max_steps=args.max_steps,
            seed=seed,
            split=args.split,
        )
        save_raw_episode(
            episode, output, episode_id=episode_id, split=args.split
        )
        collected += 1
        successes += int(bool(episode.metadata.get("success")))
        if collected % 100 == 0 or index + 1 == args.num_episodes:
            print(
                f"[{index + 1}/{args.num_episodes}] collected={collected} "
                f"skipped={skipped} success={successes / max(collected, 1):.4f}",
                flush=True,
            )

    print(
        f"Done: env={args.env_id} collected={collected} skipped={skipped} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
