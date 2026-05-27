"""Patch existing tokenized episodes to add per-step state_labels from raw data.

Much faster than re-tokenizing (no V-JEPA 2 needed).
Reads raw episode infos, extracts state_labels, and updates tokenized episodes.

Usage:
    python scripts/patch_state_labels.py \
        --raw-dir /path/to/sokoban_10k \
        --tokenized-dir /path/to/sokoban_10k/tokenized
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Patch tokenized episodes with state_labels")
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--tokenized-dir", type=str, required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) / "raw"
    tok_dir = Path(args.tokenized_dir)

    # Load raw manifest
    raw_manifest_path = raw_dir / "manifest.jsonl"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(f"No manifest.jsonl in {raw_dir}")

    with open(raw_manifest_path) as f:
        raw_entries = [json.loads(line) for line in f if line.strip()]

    # Build episode_id -> raw_path mapping
    raw_map = {}
    for entry in raw_entries:
        eid = entry.get("episode_id", "")
        raw_map[eid] = raw_dir / entry["path"]

    # Load tokenized manifest
    tok_manifest_path = tok_dir / "manifest.jsonl"
    if not tok_manifest_path.exists():
        raise FileNotFoundError(f"No manifest.jsonl in {tok_dir}")

    with open(tok_manifest_path) as f:
        tok_entries = [json.loads(line) for line in f if line.strip()]

    print(f"Raw episodes: {len(raw_entries)}, Tokenized episodes: {len(tok_entries)}")

    patched = 0
    skipped = 0
    t0 = time.time()

    for i, entry in enumerate(tok_entries):
        eid = entry.get("episode_id", "")
        tok_path = tok_dir / entry["path"]

        # Load tokenized episode
        tok_data = torch.load(tok_path, map_location="cpu", weights_only=False)
        metadata = tok_data.get("metadata", {})

        # Skip if already has state_labels
        if "state_labels" in metadata and isinstance(metadata["state_labels"], list) and len(metadata["state_labels"]) > 1:
            skipped += 1
            continue

        # Load raw episode to get infos
        if eid not in raw_map:
            print(f"  Warning: {eid} not found in raw data, skipping")
            skipped += 1
            continue

        raw_data = torch.load(raw_map[eid], map_location="cpu", weights_only=False)
        raw_infos = raw_data.get("infos", [])

        if not raw_infos or not isinstance(raw_infos[0], dict) or "state_labels" not in raw_infos[0]:
            skipped += 1
            continue

        # Build aligned state_labels list
        initial_labels = metadata.get("initial_state_labels", {})
        T_env = len(raw_data.get("actions", []))
        per_step_labels = [initial_labels] + [info["state_labels"] for info in raw_infos]
        # Pad to T_seq = T_env + 1
        while len(per_step_labels) < T_env + 1:
            per_step_labels.append({})
        metadata["state_labels"] = per_step_labels[:T_env + 1]

        # Save back
        tok_data["metadata"] = metadata
        torch.save(tok_data, tok_path)
        patched += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(tok_entries):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(tok_entries)}] patched={patched}, skipped={skipped}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone. patched={patched}, skipped={skipped}, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
