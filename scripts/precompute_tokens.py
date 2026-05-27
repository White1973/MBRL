"""Precompute visual tokens for raw Sokoban episodes using V-JEPA 2.

Usage:
    python scripts/precompute_tokens.py \
        --raw-dir data/sokoban_300_v2 \
        --out-dir data/sokoban_300_v2_tokenized \
        --batch-size 64 \
        --device cuda

Reads raw episodes from --raw-dir (manifest.jsonl + .pt files),
runs each frame through V-JEPA 2 + compression + projection,
saves tokenized episodes to --out-dir.

Supports resume: already-tokenized episodes (present in output manifest)
are skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sembelief_wm.config import Config, EncoderConfig
from sembelief_wm.data.tokenizers.image import ImageTokenizer
from sembelief_wm.data.schema import TokenizedEpisode
from sembelief_wm.data.storage import save_tokenized_episode, read_manifest, existing_episode_ids


def load_raw_manifest(raw_dir: Path) -> list[dict]:
    """Load raw episode manifest entries."""
    manifest_path = raw_dir / "raw" / "manifest.jsonl"
    if not manifest_path.exists():
        # Fallback: check top-level manifest
        manifest_path = raw_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.jsonl in {raw_dir}")
    return read_manifest(manifest_path)


def load_existing_tokenized(out_dir: Path) -> set[str]:
    """Load set of already-tokenized episode IDs for resume support."""
    manifest_path = out_dir / "tokenized" / "manifest.jsonl"
    if not manifest_path.exists():
        return set()
    return existing_episode_ids(read_manifest(manifest_path))


def main():
    parser = argparse.ArgumentParser(description="Precompute V-JEPA 2 visual tokens")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory with raw episodes")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for tokenized episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="Frames per V-JEPA 2 batch")
    parser.add_argument("--device", type=str, default="cuda", help="Device for V-JEPA 2 inference")
    parser.add_argument("--hidden-dim", type=int, default=3584, help="World model hidden dim (D_model)")
    parser.add_argument("--vjepa-model", type=str, default="facebook/vjepa2-vitg-fpc64-384-ssv2")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    raw_entries = load_raw_manifest(raw_dir)
    done_ids = load_existing_tokenized(out_dir)
    pending = [e for e in raw_entries if e.get("episode_id", "") not in done_ids]
    print(f"Total episodes: {len(raw_entries)}, already done: {len(done_ids)}, pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Build config for tokenizer
    config = Config(
        hidden_dim=args.hidden_dim,
        encoder=EncoderConfig(vjepa2_model=args.vjepa_model),
    )

    # Load tokenizer (this loads V-JEPA 2 — takes a minute)
    print(f"Loading V-JEPA 2 from {args.vjepa_model} on {args.device}...")
    t0 = time.time()
    tokenizer = ImageTokenizer(config, device=args.device)
    print(f"V-JEPA 2 loaded in {time.time() - t0:.1f}s")

    # Save tokenizer weights for reproducibility
    tokenizer.save_weights(str(out_dir / "tokenizer_weights.pt"))

    # Process episodes
    total_frames = 0
    t_start = time.time()

    for i, entry in enumerate(pending):
        # Load raw episode
        raw_subdir = raw_dir / "raw"
        ep_path = raw_subdir / entry["path"] if (raw_subdir / entry["path"]).exists() else raw_dir / entry["path"]
        ep_data = torch.load(ep_path, map_location="cpu", weights_only=False)

        # Extract RGB frames from observations
        observations = ep_data["observations"]  # list of np.ndarray (H, W, 3) uint8
        actions = ep_data["actions"]  # list of int
        rewards = ep_data["rewards"]  # list of float

        # Tokenize all frames in batches
        all_tokens = []
        for batch_start in range(0, len(observations), args.batch_size):
            batch_frames = observations[batch_start:batch_start + args.batch_size]
            tokens = tokenizer.batch_tokenize(batch_frames)  # (B, K, D)
            all_tokens.append(tokens.cpu())
            total_frames += len(batch_frames)

        obs_tokens = torch.cat(all_tokens, dim=0)  # (T+1, K, D) — all observations including initial

        # Build tokenized episode: T_seq = T_env + 1, pad last action/reward
        episode_id = entry.get("episode_id", f"ep_{i}")
        env_id = entry.get("env_id", "sokoban")
        split = entry.get("split", "train")
        metadata = ep_data.get("metadata", {})

        T_env = len(actions)
        # Extract per-step state_labels from infos (for belief probing)
        # Build a T_seq-length list aligned with obs/beliefs:
        #   index 0 = initial_state_labels (before any action)
        #   index t = infos[t-1]["state_labels"] (state after action t-1)
        raw_infos = ep_data.get("infos", [])
        initial_labels = metadata.get("initial_state_labels", {})
        if raw_infos and isinstance(raw_infos[0], dict) and "state_labels" in raw_infos[0]:
            per_step_labels = [initial_labels] + [info["state_labels"] for info in raw_infos]
            # Pad to T_seq with empty dict (dummy last step)
            while len(per_step_labels) < T_env + 1:
                per_step_labels.append({})
            metadata["state_labels"] = per_step_labels[:T_env + 1]
        T_seq = T_env + 1  # includes initial observation

        # Pad actions and rewards to T_seq (last entry is dummy)
        actions_tensor = torch.tensor(actions + [0], dtype=torch.long)  # null/dummy action
        rewards_tensor = torch.tensor(rewards + [0.0], dtype=torch.float32)

        episode = TokenizedEpisode(
            obs_tokens=obs_tokens,
            actions=actions_tensor,
            rewards=rewards_tensor,
            episode_length=T_env,
            env_id=env_id,
            split=split,
            metadata=metadata,
        )

        save_tokenized_episode(
            episode=episode,
            root=out_dir,
            episode_id=episode_id,
            tokenizer="vjepa2",
            config_hash=None,
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            elapsed = time.time() - t_start
            fps = total_frames / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(pending)}] {total_frames} frames, {fps:.0f} frames/s")

    elapsed = time.time() - t_start
    print(f"\nDone. {len(pending)} episodes, {total_frames} frames in {elapsed:.1f}s ({total_frames/elapsed:.0f} fps)")


if __name__ == "__main__":
    main()
